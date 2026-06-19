"""Router agent — intent classification for the Gia crew.

The Router is the first agent to run on every turn.  It decides which
downstream agents are needed (DJ, Artist, Mood, or a combination) so the
crew does not run unnecessary work.

Design principles (from Section 3):
  - "Mostly deterministic — fast and traceable beats smart and slow."
  - Keyword heuristics handle 80 % of intents in < 1 ms.
  - An LLM call is made only when the keywords conflict or are absent.

Intent types::

    MUSIC_FIND   — "find me something chill", "recommend a song"
    MUSIC_QUEUE  — "add to queue", "save this", "build me a playlist"
    ARTIST_INFO  — "tell me about Odumodublvck", "who is Tems"
    MOOD_CHECK   — "what's my mood", "how am I listening lately"
    MIXED        — message references both music discovery and an artist,
                   or music and a mood check
"""

from __future__ import annotations

import asyncio
import re
import time

from crewai import Agent

from backend.app.config import Settings
from backend.app.observability.logging import get_logger
from backend.app.providers.llm import get_fast_llm
from backend.app.schemas.chat import IntentType

logger = get_logger(__name__)

# ── Keyword heuristics ────────────────────────────────────────────────────────

_ARTIST_PATTERNS = [
    r"\btell me about\b",
    r"\bwho is\b",
    r"\bwhat about\b",
    r"\bwhat('s| has| did)\b.*(done|released|dropped|up to)",
    r"\bartist\b",
    r"\balbum\b",
    r"\bdiscography\b",
    r"\bsingle\b",
]

_QUEUE_KEYWORDS = [
    "add", "save", "queue", "playlist", "like this",
    "next up", "skip", "build me a playlist",
]

_MOOD_KEYWORDS = [
    "mood", "pattern", "patterns", "usually", "how am i",
    "what do i normally", "what do i usually",
]

_MUSIC_KEYWORDS = [
    "find", "play", "recommend", "something", "song", "songs",
    "music", "vibe", "chill", "hype", "listen", "listening",
]


def _word_match(text: str, keywords: list[str]) -> bool:
    """Return ``True`` when any keyword matches as a complete word in *text*."""
    return any(re.search(r"\b" + re.escape(kw) + r"\b", text) for kw in keywords)


def _keyword_classify(message: str) -> IntentType | None:
    """Return an ``IntentType`` when keyword evidence is unambiguous.

    Queue intent takes precedence over music intent when both are present
    (e.g. "save this track" → MUSIC_QUEUE, not MIXED).

    Args:
        message: Raw user message (any case).

    Returns:
        An ``IntentType`` or ``None`` when the keywords conflict or are absent.
    """
    lower = message.lower()

    is_artist = any(re.search(p, lower) for p in _ARTIST_PATTERNS)
    is_queue = _word_match(lower, _QUEUE_KEYWORDS)
    is_mood = _word_match(lower, _MOOD_KEYWORDS)
    is_music = _word_match(lower, _MUSIC_KEYWORDS)

    # Queue + music without artist/mood → queue intent dominates
    if is_queue and is_music and not is_artist and not is_mood:
        return IntentType.MUSIC_QUEUE

    # Mood + music without artist/queue → mood intent dominates
    # e.g. "what are my listening patterns?" is a mood check, not music discovery
    if is_mood and is_music and not is_artist and not is_queue:
        return IntentType.MOOD_CHECK

    signals = sum([is_artist, is_queue, is_mood, is_music])

    if signals > 1:
        return IntentType.MIXED
    if is_artist:
        return IntentType.ARTIST_INFO
    if is_queue:
        return IntentType.MUSIC_QUEUE
    if is_mood:
        return IntentType.MOOD_CHECK
    if is_music:
        return IntentType.MUSIC_FIND
    return None


_ROUTER_PROMPT = """\
Classify the user's intent into exactly one of:
  MUSIC_FIND   — user wants to discover or play music
  MUSIC_QUEUE  — user wants to save, queue, or organise tracks
  ARTIST_INFO  — user wants to talk about a specific artist
  MOOD_CHECK   — user wants mood analysis or patterns
  MIXED        — message touches more than one category

User message: "{message}"

Reply with ONLY the intent label, nothing else.
"""


async def classify_intent(
    message: str,
    cfg: Settings,
) -> tuple[IntentType, float]:
    """Classify user intent — heuristic first, LLM fallback.

    Args:
        message: User's raw message text.
        cfg:     Settings (used for the fallback LLM call).

    Returns:
        ``(IntentType, confidence)`` where confidence is 1.0 for keyword
        matches and 0.8 for LLM classifications.
    """
    heuristic = _keyword_classify(message)
    if heuristic is not None:
        logger.debug("router_heuristic_hit", intent=heuristic.value)
        return heuristic, 1.0

    llm = get_fast_llm(cfg)
    try:
        prompt = _ROUTER_PROMPT.format(message=message)
        raw = await asyncio.to_thread(
            llm.call, [{"role": "user", "content": prompt}]
        )
        raw = raw.strip().upper()
        intent = IntentType(raw)
        logger.debug("router_llm_hit", intent=intent.value, raw=raw)
        return intent, 0.8
    except Exception as exc:  # noqa: BLE001
        logger.warning("router_llm_fallback", error=str(exc))
        return IntentType.MUSIC_FIND, 0.5


def build_router_agent(cfg: Settings) -> Agent:
    """Construct the CrewAI Router agent.

    The Router is used for multi-agent crew composition starting Day 6.
    For direct intent classification, prefer ``classify_intent`` directly.

    Args:
        cfg: Application settings.

    Returns:
        Configured ``crewai.Agent``.
    """
    return Agent(
        role="Router",
        goal=(
            "Classify the user's intent with precision so the right downstream "
            "agents are activated — and only those agents."
        ),
        backstory=(
            "You are the traffic controller for Gia's crew. You read each message "
            "and decide: is this person looking for music, asking about an artist, "
            "checking their patterns, or some mix? You are fast and deliberate — "
            "you never guess when you can reason from the words."
        ),
        llm=get_fast_llm(cfg),
        verbose=False,
        allow_delegation=False,
    )
