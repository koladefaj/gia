"""LLM mood labeling from play history.

Replaces the dead (energy, valence) quadrant classifier: instead of audio
features Spotify no longer provides, an LLM reads the *track and artist names*
the user actually plays and returns one label from the closed
:data:`backend.app.mood.classifier.MOOD_LABELS` vocabulary.

The call is best-effort — any failure (or an empty set) degrades to
``"neutral"`` so mood analysis never blocks a turn.
"""

from __future__ import annotations

import asyncio

from backend.app.config import Settings
from backend.app.mood.classifier import MOOD_LABELS, coerce_label
from backend.app.observability.logging import get_logger
from backend.app.prompts import PromptRegistry, get_registry
from backend.app.providers.llm import get_fast_llm

logger = get_logger(__name__)

AGENT_KEY = "agents.mood"

# Cap how many tracks we describe to the labeler — enough to read a vibe, not so
# many the prompt bloats.
_MAX_TRACKS = 15


async def label_mood(
    tracks: list[dict],
    cfg: Settings,
    registry: PromptRegistry | None = None,
) -> str:
    """Return one :data:`MOOD_LABELS` label for a set of *tracks*.

    Args:
        tracks:   Track dicts with ``name`` / ``artist`` (from search / recently
                  played / ``ListeningEvent`` rows mapped to the same shape).
        cfg:      Application settings (fast-tier model).
        registry: Prompt registry; defaults to the process-wide singleton.

    Returns:
        A label from :data:`MOOD_LABELS` (``"neutral"`` when unknown / on error).
    """
    named = [t for t in tracks if (t.get("name") or t.get("artist"))]
    if not named:
        return "neutral"

    reg = registry or get_registry()
    lines = "\n".join(
        f"- {t.get('name') or '?'} — {t.get('artist') or '?'}" for t in named[:_MAX_TRACKS]
    )
    prompt = reg.get(AGENT_KEY).render("label", labels=", ".join(MOOD_LABELS), tracks=lines)

    llm = get_fast_llm(cfg)
    try:
        raw = await asyncio.to_thread(llm.call, [{"role": "user", "content": prompt}])
    except Exception as exc:  # noqa: BLE001
        logger.warning("mood_label_error", error=str(exc))
        return "neutral"
    return coerce_label(raw)
