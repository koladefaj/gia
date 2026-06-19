"""Planner — turns a user message into an execution plan for the crew.

Where the Router answers "what is this one thing?", the Planner answers "what
do I need to do, and which real-world signals would make the answer better?".
It is the architectural step that makes Gia feel like a companion rather than a
music search box: one message can fan out to several agents plus context
signals (weather today, more as the toolset grows), and the crew executes the
plan and synthesises a single reply.

Design (inherited from the Router):
  - Mostly deterministic. The intent comes from the existing
    ``classify_intent`` (keyword heuristics first, LLM fallback).
  - Signal detection is pure keyword matching — fast, traceable, free.

Keeping the planner thin and deterministic is deliberate: fast and predictable
beats clever and slow on a voice turn.
"""

from __future__ import annotations

import re

from backend.app.agents.router import classify_intent
from backend.app.config import Settings
from backend.app.observability.logging import get_logger
from backend.app.prompts import PromptRegistry
from backend.app.schemas.chat import ExecutionPlan, IntentType

logger = get_logger(__name__)

# Which agents run for each primary intent.  MIXED fans out to discovery +
# artist talk (matching the prior /chat behaviour); mood is surfaced separately
# via the proactive draft and the explicit MOOD_CHECK path.
_INTENT_STEPS: dict[IntentType, list[str]] = {
    IntentType.MUSIC_FIND: ["dj"],
    IntentType.MUSIC_QUEUE: ["dj"],
    IntentType.ARTIST_INFO: ["artist"],
    IntentType.MOOD_CHECK: ["mood"],
    IntentType.MIXED: ["dj", "artist"],
}

# Contexts where the current weather meaningfully changes a music pick.
_WEATHER_KEYWORDS = [
    "run", "running", "jog", "jogging", "workout", "work out", "gym",
    "exercise", "training", "drive", "driving", "commute", "commuting",
    "walk", "walking", "outside", "outdoor", "outdoors", "weather",
    "hot", "cold", "rain", "raining", "sunny",
]


def _wants_weather(message: str, steps: list[str]) -> bool:
    """Return ``True`` when weather context would improve this turn.

    Weather only helps when we are actually recommending music (a ``dj`` step)
    *and* the message mentions an activity or condition the weather bears on.

    Args:
        message: Raw user message (any case).
        steps:   The agent steps already chosen for this turn.

    Returns:
        Whether to add the ``"weather"`` signal to the plan.
    """
    if "dj" not in steps:
        return False
    lower = message.lower()
    return any(
        re.search(r"\b" + re.escape(kw) + r"\b", lower) for kw in _WEATHER_KEYWORDS
    )


async def build_plan(
    message: str,
    cfg: Settings,
    registry: PromptRegistry | None = None,
) -> ExecutionPlan:
    """Build an :class:`ExecutionPlan` for *message*.

    Classifies intent (heuristic → LLM fallback), maps it to the agents that
    should run, and detects whether real-world signals (weather) would help.

    Args:
        message:  The user's raw message text.
        cfg:      Application settings (for the fallback LLM call).
        registry: Prompt registry forwarded to ``classify_intent``.

    Returns:
        A populated ``ExecutionPlan``.
    """
    intent, confidence = await classify_intent(message, cfg, registry)
    steps = list(_INTENT_STEPS.get(intent, ["dj"]))

    signals: list[str] = []
    if _wants_weather(message, steps):
        signals.append("weather")

    plan = ExecutionPlan(
        intent=intent, steps=steps, signals=signals, confidence=confidence
    )
    logger.debug(
        "plan_built",
        intent=intent.value,
        steps=steps,
        signals=signals,
        confidence=confidence,
    )
    return plan
