"""Artist agent — personalised, context-aware artist conversation.

``ArtistService.get_info()`` is the entry point.  For a given artist name it:

1. Searches Spotify for the artist's top tracks.
2. Queries Brave Search for recent news / activity.
3. Retrieves the user's history with this artist from Weaviate.
4. Synthesises all three into a warm, personalised response via the persona LLM.

The response is designed to feel like a knowledgeable friend who has done
their homework done, not a Wikipedia article.

Section 5 prompt injection (from the spec)::

    User's history with {artist_name}: {user_artist_memory}
    Artist's recent activity (from web): {brave_results}
    Artist's top tracks: {top_tracks}

    Respond as Gia would: as a knowledgeable friend, not a Wikipedia article.
    Reference the user's personal history with this artist.
    If the recent news is interesting or funny, react to it genuinely.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from crewai import Agent

from backend.app.config import Settings
from backend.app.interfaces import SpotifyClientProtocol
from backend.app.memory.embeddings import embed
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.observability.logging import get_logger
from backend.app.prompts import PromptRegistry, get_registry
from backend.app.providers.llm import get_llm
from backend.app.schemas.artist import ArtistInfoResponse, BraveResult
from backend.app.tools.brave import BraveSearchClient

logger = get_logger(__name__)

AGENT_KEY = "agents.artist"


# Phrases that introduce an artist name ("tell me about <name>"). Matched on a
# lower-cased copy; the name is sliced from the *original* text to keep its case.
_ARTIST_TRIGGERS = [
    r"\btell me about\s+",
    r"\btalk (?:to me )?about\s+",
    r"\bwhat do you know about\s+",
    r"\bwho(?:'s| is| are)\s+",
    r"\bwhat(?:'s| has| did)?\s+(?=.+\b(?:been up to|up to|done|released|dropped|new)\b)",
    r"\bwhat about\s+",
    r"\bhow about\s+",
    r"\banything (?:new )?(?:from|on|about)\s+",
]

# Trailing clauses to drop from a captured name ("Tems lately" → "Tems").
_NAME_TAIL = re.compile(
    r"\b(been up to|up to|lately|these days|right now|please|for me|recently|"
    r"new|doing|releasing|dropping|up)\b.*$",
    re.IGNORECASE,
)

# Words that disqualify a *bare* message from being read as an artist name, so
# small talk ("whats the weather like") is never looked up as an artist.
_NOT_ARTIST = {
    "weather", "mood", "moods", "pattern", "patterns", "you", "your", "yourself",
    "gia", "help", "what", "whats", "what's", "how", "why", "when", "where",
    "hey", "hi", "hello", "yo", "sup", "thanks", "thank", "please", "music",
    "song", "songs", "play", "find", "recommend", "something", "vibe", "vibes",
    "chill", "hype", "tired", "bored", "happy", "sad", "okay", "ok", "yes", "no",
    "the", "a", "an", "is", "are", "do", "can", "could", "would", "i", "im",
    "i'm", "me", "we", "they", "this", "that", "today", "now",
    # Conversational affirmations / fillers — "yeah sure", "nah", "cool" are
    # replies, never artist names. Without these, a bare "yeah sure" answer to
    # "want me to play him?" gets looked up as an artist and hallucinates.
    "yeah", "yea", "yep", "yup", "nah", "nope", "maybe", "cool", "nice", "wow",
    "lol", "haha", "hmm", "fine", "alright", "aight", "right", "whatever",
    "dunno", "idk", "huh", "sure", "really", "totally", "exactly", "absolutely",
    "good", "great", "awesome", "perfect", "sorry", "wait", "stop",
}

# A plausible artist name: letters/digits/spaces and a few name punctuation marks.
_NAME_SHAPE = re.compile(r"^[A-Za-z0-9 '&.\-]+$")


def extract_artist_name(message: str) -> str:
    """Pull the artist name out of an artist-info message, or return ``""``.

    Two strategies, in order:

    1. **Trigger phrase** — "tell me about <name>", "who is <name>", "what has
       <name> been up to" → the text after the trigger, with trailing filler
       ("lately", "been up to") stripped.
    2. **Bare name** — a short message (≤ 4 words) that is *only* a name, with no
       small-talk / question words → the message itself.

    Anything else (greetings, weather, generic chatter) yields ``""`` so the
    caller skips the artist agent instead of looking up a nonsense "artist" like
    *"whats the weather like"*.

    Args:
        message: The user's raw message.

    Returns:
        The extracted artist name, or ``""`` when none is present.
    """
    text = message.strip().strip("\"'").rstrip("?.!").strip()
    if not text:
        return ""
    lower = text.lower()

    for trigger in _ARTIST_TRIGGERS:
        m = re.search(trigger, lower)
        if m:
            name = text[m.end():].strip().strip("\"'")
            name = _NAME_TAIL.sub("", name).strip().rstrip(",")
            return name if name and _NAME_SHAPE.match(name) else ""

    words = text.split()
    if (
        1 <= len(words) <= 4
        and _NAME_SHAPE.match(text)
        and not any(w.lower().strip(",'") in _NOT_ARTIST for w in words)
    ):
        return text
    return ""


def build_artist_agent(cfg: Settings, registry: PromptRegistry | None = None) -> Agent:
    """Construct the CrewAI Artist agent from the externalised prompt registry.

    Args:
        cfg:      Application settings.
        registry: Prompt registry for the agent identity; defaults to the
                  process-wide singleton.

    Returns:
        A configured ``crewai.Agent`` for artist-focused conversation.
    """
    prompt = (registry or get_registry()).get(AGENT_KEY)
    return Agent(
        role=prompt.render("role"),
        goal=prompt.render("goal"),
        backstory=prompt.render("backstory"),
        llm=get_llm(cfg),
        verbose=False,
        allow_delegation=False,
    )


@dataclass
class ArtistService:
    """Orchestrates the three-source artist context assembly and LLM synthesis.

    Attributes:
        spotify: Spotify client for top-tracks lookup.
        brave:   Brave Search client for recent news.
        store:   Weaviate memory store for user history (``None`` = skip).
        cfg:     Application settings.
    """

    spotify: SpotifyClientProtocol
    brave: BraveSearchClient
    cfg: Settings
    store: WeaviateMemoryStore | None = field(default=None)
    registry: PromptRegistry = field(default_factory=get_registry)

    async def get_info(
        self,
        artist_name: str,
        user_id: str | None = None,
    ) -> ArtistInfoResponse:
        """Build a personalised artist response from three data sources.

        Fetches run in parallel where independent.  Individual failures are
        caught and degraded gracefully so a missing Brave key or empty Weaviate
        history does not block the response.

        Args:
            artist_name: The artist to research.
            user_id:     Optional user UUID for history lookup.

        Returns:
            ``ArtistInfoResponse`` with Gia's narrative, top tracks, and news.
        """
        # ── Parallel fetches ─────────────────────────────────────────────────
        top_tracks_coro = self.spotify.search_tracks(artist_name, limit=5)
        # Freshness is essential here: a bare "{artist} {year}" query returns
        # evergreen tour/ticket pages and misses new releases, so the model falls
        # back to its stale memory ("latest album" → a year-old answer). A
        # past-month window surfaces the actual recent activity (e.g. a new album
        # released last week) for the prompt to ground on.
        year = datetime.now(UTC).year
        brave_coro = self.brave.search(f"{artist_name} {year}", count=5, freshness="pm")

        top_tracks_raw, brave_raw = await asyncio.gather(
            top_tracks_coro,
            brave_coro,
            return_exceptions=True,
        )

        if isinstance(top_tracks_raw, Exception):
            logger.warning("artist_spotify_error", error=str(top_tracks_raw))
            top_tracks_raw = []
        if isinstance(brave_raw, Exception):
            logger.warning("artist_brave_error", error=str(brave_raw))
            brave_raw = []

        top_tracks: list[dict] = top_tracks_raw  # type: ignore[assignment]
        brave_results: list[dict] = brave_raw  # type: ignore[assignment]

        # ── User history from Weaviate ────────────────────────────────────────
        user_artist_memory = "No personal history with this artist yet."
        if user_id and self.store:
            try:
                query = f"{artist_name} music"
                query_vector = await embed(query)
                memories = await self.store.search(user_id, query_vector, "preference", k=5)
                relevant = [
                    m for m in memories
                    if artist_name.lower() in m.text.lower()
                ]
                if relevant:
                    user_artist_memory = "\n".join(f"- {m.text}" for m in relevant)
            except Exception as exc:  # noqa: BLE001
                logger.warning("artist_weaviate_error", error=str(exc))

        # ── Assemble LLM prompt ───────────────────────────────────────────────
        tracks_text = "\n".join(
            f"- {t.get('name', '?')} ({t.get('artist', '?')})"
            for t in top_tracks[:5]
        ) or "No tracks available."

        brave_text = "\n".join(
            f"- {r.get('title', '')}"
            + (f" ({r['age']})" if r.get("age") else "")
            + f": {r.get('description', '')[:200]}"
            for r in brave_results[:5]
        ) or "No recent news found."

        prompt = self.registry.get(AGENT_KEY).render(
            "task",
            persona=self.registry.get("persona.gia").render(),
            artist_name=artist_name,
            user_artist_memory=user_artist_memory,
            brave_text=brave_text,
            tracks_text=tracks_text,
        )

        llm = get_llm(self.cfg)
        try:
            response_text = await asyncio.to_thread(
                llm.call, [{"role": "user", "content": prompt}]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("artist_llm_error", error=str(exc))
            response_text = f"I'd love to talk about {artist_name} — ask me again in a moment."

        logger.info(
            "artist_info_done",
            artist=artist_name,
            user_id=user_id,
            tracks=len(top_tracks),
            news=len(brave_results),
        )

        return ArtistInfoResponse(
            artist_name=artist_name,
            response=response_text.strip(),
            top_tracks=top_tracks,
            recent_news=[
                BraveResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    description=r.get("description", ""),
                )
                for r in brave_results
            ],
        )
