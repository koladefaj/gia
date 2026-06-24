"""Distilled local router — tier between the keyword fast-path and the LLM.

A frozen MiniLM encoder + small linear heads (trained in ``ml/router``) predict
the categorical ``RouterDecision`` fields in ~20-40ms on CPU, taking the ~1.4s
gpt-4o-mini call off the critical path for the confident, no-query-resolution
turns ("I'm doing good", "what's my mood lately", "who do I usually play").

Two guardrails keep it safe:

* **Confidence gate** — only used when the top intent probability clears a
  threshold; below it the turn falls back to the LLM, so net accuracy is the
  teacher's, not the student's.
* **Safe intents only** — it returns a decision *only* for intents that need no
  ``search_query`` / ``track_titles`` / ``start_playback`` (the free-form fields
  it doesn't model). Music / artist / news / mixed always go to the LLM.

Degrades to a no-op (returns ``None`` → LLM path) when the deps or the trained
model aren't present, so the app runs identically without it.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from backend.app.observability.logging import get_logger
from backend.app.schemas.chat import IntentType
from backend.app.schemas.router import EngagementMode, RouterDecision, Tone

logger = get_logger(__name__)

_MODEL_DIR = Path(__file__).resolve().parents[3] / "ml" / "router"
_BOOL_FIELDS = ("needs_search", "needs_memory", "needs_music", "needs_artist_lookup")

# Intents safe to answer from the local heads: they need none of the free-form
# fields (search_query/track_titles/start_playback). Music/artist/news/mixed are
# excluded — they need the LLM's query resolution.
_LOCAL_SAFE = {
    IntentType.GENERAL_CHAT,
    IntentType.MOOD_CHECK,
    IntentType.MEMORY_QUERY,
}

_lock = threading.Lock()
_state: Any = None  # (encoder, heads, labelmaps) | False (unavailable) | None (unloaded)


def _load() -> Any:
    """Lazy-load the encoder + heads once; cache ``False`` if unavailable."""
    global _state  # noqa: PLW0603
    if _state is not None:
        return _state
    with _lock:
        if _state is not None:
            return _state
        try:
            import joblib  # noqa: PLC0415
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            heads = joblib.load(_MODEL_DIR / "heads.joblib")
            labelmaps = json.loads((_MODEL_DIR / "labelmaps.json").read_text(encoding="utf-8"))
            encoder = SentenceTransformer(heads["encoder"])
            _state = (encoder, heads, labelmaps)
            logger.info("router_local_loaded", encoder=heads["encoder"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("router_local_unavailable", error=str(exc))
            _state = False
    return _state


def _bool_head(head: Any, x: Any) -> bool:
    """A needs_* head is either a fitted classifier or a constant int."""
    if isinstance(head, int):
        return bool(head)
    return bool(head.predict(x)[0])


def classify_local(message: str, threshold: float = 0.75) -> RouterDecision | None:
    """Return a confident local ``RouterDecision`` for a safe intent, else ``None``.

    Synchronous (encode is blocking CPU work) — call via ``asyncio.to_thread``.
    """
    state = _load()
    if not state:
        return None
    encoder, heads, labelmaps = state

    x = encoder.encode([message], normalize_embeddings=True)
    proba = heads["intent"].predict_proba(x)[0]
    j = int(proba.argmax())
    confidence = float(proba[j])
    intent = IntentType(labelmaps["intent"][j])

    if confidence < threshold or intent not in _LOCAL_SAFE:
        return None  # uncertain or needs query resolution → let the LLM handle it

    tone = Tone(labelmaps["tone"][int(heads["tone"].predict(x)[0])])
    engagement = EngagementMode(
        labelmaps["engagement_mode"][int(heads["engagement_mode"].predict(x)[0])]
    )
    needs = {f: _bool_head(heads[f], x) for f in _BOOL_FIELDS}
    # A safe-intent turn never plays/searches music, so force those off regardless.
    needs["needs_music"] = False
    needs["needs_artist_lookup"] = False

    return RouterDecision(
        intent=intent,
        tone=tone,
        confidence=round(confidence, 2),
        engagement_mode=engagement,
        **needs,
    )
