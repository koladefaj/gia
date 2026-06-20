"""DJ agent — track discovery, audio-feature-aware recommendation, crossfade queuing.

``DJService.recommend()`` is the single entry point.  It:

1. Searches Spotify for tracks matching the user's query.
2. Fetches audio features for all candidates.
3. Builds a Camelot-compatible crossfade queue from the best seed track.
4. Generates a warm, grounded recommendation via the persona LLM.
5. Optionally starts Spotify playback immediately.

The CrewAI ``Agent`` (from ``build_dj_agent``) is returned for composition
into multi-agent crews.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from crewai import Agent

from backend.app.config import Settings
from backend.app.interfaces import SpotifyClientProtocol
from backend.app.observability.logging import get_logger
from backend.app.prompts import PromptRegistry, get_registry
from backend.app.providers.llm import get_llm
from backend.app.schemas.dj import CrossfadeQueue, DJResponse, TrackItem
from backend.app.tools.crossfade import build_key_matched_sequence, track_from_dict

logger = get_logger(__name__)

AGENT_KEY = "agents.dj"


def build_dj_agent(cfg: Settings, registry: PromptRegistry | None = None) -> Agent:
    """Construct the CrewAI DJ agent from the externalised prompt registry.

    Args:
        cfg:      Application settings (LLM provider / model).
        registry: Prompt registry to read the agent identity from; defaults to
                  the process-wide singleton.

    Returns:
        A configured ``crewai.Agent`` ready to be composed into a crew.
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
class DJService:
    """Orchestrates track search, crossfade sequencing, and LLM recommendation.

    Attributes:
        spotify: Spotify client (live or fake).
        cfg:     Application settings.
    """

    spotify: SpotifyClientProtocol
    cfg: Settings
    registry: PromptRegistry = field(default_factory=get_registry)

    async def recommend(
        self,
        query: str,
        user_context_text: str = "",
        start_playback: bool = False,
        n: int = 4,
    ) -> DJResponse:
        """Search for tracks, build a crossfade queue, and generate a recommendation.

        Args:
            query:             Natural-language request from the user.
            user_context_text: Rendered ``UserContext.to_prompt_text()`` string
                               to inject into the LLM prompt for personalisation.
                               Pass ``""`` when no context is available.
            start_playback:    If ``True``, immediately start the seed track.
            n:                 Queue depth (number of tracks after the seed).

        Returns:
            A ``DJResponse`` with the recommendation, seed track, and queue.

        Raises:
            ValueError: If no tracks are found for *query*.
        """
        # ── 1. Search for candidate tracks ───────────────────────────────────
        raw_tracks = await self.spotify.search_tracks(query, limit=20)
        if not raw_tracks:
            raise ValueError(f"No tracks found for query: {query!r}")

        # ── 2. Fetch audio features ───────────────────────────────────────────
        uris = [t["uri"] for t in raw_tracks if t.get("uri")]
        features = await self.spotify.get_audio_features(uris)

        # Merge name/artist from search result into audio feature dict
        uri_to_meta: dict[str, dict] = {t["uri"]: t for t in raw_tracks if t.get("uri")}
        candidates: list[TrackItem] = []
        for feat in features:
            uri = feat.get("uri", "")
            meta = uri_to_meta.get(uri, {})
            merged = {**feat, "name": meta.get("name", ""), "artist": meta.get("artist", "")}
            candidates.append(track_from_dict(merged))

        if not candidates:
            raise ValueError(f"No audio features available for query: {query!r}")

        # ── 3. Pick seed (first candidate) and sequence the rest ─────────────
        seed = candidates[0]
        rest = [c for c in candidates[1:] if c.uri != seed.uri]
        queue_tracks = build_key_matched_sequence(seed, rest, n=n)

        # ── 4. Generate natural language recommendation ───────────────────────
        camelot = seed.camelot_key or "?"
        queued_names = ", ".join(f"{t.name} by {t.artist}" for t in queue_tracks[:3]) or "none"

        prompt = self.registry.get(AGENT_KEY).render(
            "task",
            persona=self.registry.get("persona.gia").render(),
            user_context=user_context_text,
            query=query,
            seed_name=seed.name,
            seed_artist=seed.artist,
            energy=seed.energy,
            valence=seed.valence,
            camelot=camelot,
            queued_names=queued_names,
            start_playback=start_playback,
        )

        llm = get_llm(self.cfg)
        try:
            recommendation = await asyncio.to_thread(
                llm.call, [{"role": "user", "content": prompt}]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("dj_llm_error", error=str(exc))
            recommendation = f"Here's {seed.name} by {seed.artist} — should fit the vibe."

        # ── 5. Optionally start playback ──────────────────────────────────────
        playback_started = False
        if start_playback:
            await self.spotify.start_playback(seed.uri)
            playback_started = True
            logger.info("dj_playback_started", uri=seed.uri)

        logger.info(
            "dj_recommend_done",
            query=query,
            seed=seed.name,
            queue_depth=len(queue_tracks),
        )

        return DJResponse(
            recommendation=recommendation.strip(),
            primary_track=seed,
            queue=CrossfadeQueue(
                seed_uri=seed.uri,
                tracks=queue_tracks,
                crossfade_ms=3000,
            ),
            playback_started=playback_started,
        )
