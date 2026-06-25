"""Router agent — intent classification for the Gia crew.

The Router is the first agent to run on every turn.  It decides which
downstream agents are needed (DJ, Artist, Mood, or a combination) so the
crew does not run unnecessary work.

Design principles:
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

from backend.app.config import Settings
from backend.app.observability.logging import get_logger
from backend.app.prompts import PromptRegistry, get_registry
from backend.app.providers.llm import get_fast_llm
from backend.app.schemas.chat import IntentType

logger = get_logger(__name__)

AGENT_KEY = "agents.router"

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

_GREETING_KEYWORDS = [
    "hey", "hello", "hi", "sup", "yo", "morning", "evening",
    "what's up", "how are you", "how's it going", "good morning",
    "good evening", "good afternoon", "what can you do",
    "who are you", "help",
]

# Non-music small talk (weather, how-are-you, your-name) — routed to GENERAL so
# Gia answers conversationally instead of the LLM guessing MIXED and treating
# the whole sentence as an artist to look up.
_SMALLTALK_KEYWORDS = [
    "weather", "raining", "sunny",
    "how are things", "whats up", "your name", "what do you do",
    "thanks", "thank you", "tired", "bored", "stressed",
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
    has_signal = is_artist or is_queue or is_mood or is_music

    # Greetings / small talk that carry no music, artist, queue, or mood signal →
    # GENERAL (conversational). No word-count cap: "just thought I'd say hi, how
    # have you been" is a greeting even at nine words. A greeting that DOES carry
    # a real ask ("hey, play something") keeps its true intent below.
    if not has_signal and _word_match(lower, _GREETING_KEYWORDS + _SMALLTALK_KEYWORDS):
        return IntentType.GENERAL

    # Queue + music without artist/mood → queue intent dominates
    if is_queue and is_music and not is_artist and not is_mood:
        return IntentType.MUSIC_QUEUE

    # Mood + music without artist/queue → mood intent dominates
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


async def classify_intent(
    message: str,
    cfg: Settings,
    registry: PromptRegistry | None = None,
) -> tuple[IntentType, float]:
    """Classify user intent — heuristic first, LLM fallback.

    Args:
        message:  User's raw message text.
        cfg:      Settings (used for the fallback LLM call).
        registry: Prompt registry for the LLM-fallback prompt; defaults to the
                  process-wide singleton.

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
        prompt = (registry or get_registry()).get(AGENT_KEY).render("classify", message=message)
        raw = await asyncio.to_thread(
            llm.call, [{"role": "user", "content": prompt}]
        )
        raw = raw.strip().upper()
        intent = IntentType(raw)
        logger.debug("router_llm_hit", intent=intent.value, raw=raw)
        return intent, 0.8
    except Exception as exc:  # noqa: BLE001
        logger.warning("router_llm_unavailable", error=str(exc))
        raise RuntimeError(
            f"LLM service unavailable — could not classify intent: {exc}"
        ) from exc


