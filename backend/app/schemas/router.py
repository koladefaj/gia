"""Schema for the structured (hybrid) router's decision.

The router replaces keyword classification with a single small-model call that
returns *everything the turn needs to know* as JSON: what the user wants
(``intent``), how Gia should sound (``tone``), how sure we are (``confidence``),
whether to react/clarify/confirm/execute (``engagement_mode``), and which
retrievers to fire (``needs_*``).

Keeping this as a validated Pydantic model means a malformed model response is
caught at the boundary and degraded to a safe default, never propagated as a
half-parsed dict deep into the turn.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from backend.app.schemas.chat import IntentType


class Tone(str, Enum):
    """Abstract delivery tone — provider-neutral.

    The router emits one of these; the ``VoiceAdapter`` maps it to provider tags.
    The router must NEVER emit provider tags like ``[light laugh]`` directly.
    """

    CURIOUS = "curious"
    SURPRISED = "surprised"
    WARM = "warm"
    PLAYFUL = "playful"
    THOUGHTFUL = "thoughtful"
    EXCITED = "excited"
    EMPATHETIC = "empathetic"
    CONFIDENT = "confident"


class EngagementMode(str, Enum):
    """How Gia engages this turn — react, clarify, confirm, or just execute."""

    DIRECT_EXECUTE = "direct_execute"
    REACT_THEN_EXECUTE = "react_then_execute"
    CLARIFY = "clarify"
    CONFIRM_ACTION = "confirm_action"


class RouterDecision(BaseModel):
    """The structured output of one router call.

    Attributes:
        intent:          Primary classified intent.
        tone:            Abstract delivery tone for the acknowledgment + reply.
        confidence:      Router confidence in ``intent`` (0–1). Below the
                         configured threshold the turn escalates to the Planner.
        engagement_mode: Whether to react/clarify/confirm/execute this turn.
        needs_search:    Web/news search (Brave) would help.
        needs_memory:    Personal memory/RAG lookup would help.
        needs_music:     A music search/recommendation is wanted.
        needs_artist_lookup: A specific artist's info is wanted.
    """

    intent: IntentType
    tone: Tone = Tone.WARM
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    engagement_mode: EngagementMode = EngagementMode.DIRECT_EXECUTE
    needs_search: bool = False
    needs_memory: bool = False
    needs_music: bool = False
    needs_artist_lookup: bool = False
    # A clean search query with pronouns RESOLVED from the conversation
    # ("play it now" after discussing Fortworth → "Fortworth Drake"). Populated
    # for music/search turns; ``None`` otherwise. The DJ/search uses this instead
    # of the raw message so references and filler ("just play … now") don't leak
    # into Spotify queries.
    search_query: str | None = None
    # Specific song TITLES the user named, in the order named (title only, no
    # artist) — e.g. ``["So Will I", "Promises"]``. Empty for vibe/genre requests
    # ("something chill") or when no track is named. The first is the primary
    # (now-playing) target; the DJ checks it against what Spotify actually
    # returned and surfaces a "did you mean…?" instead of silently playing the
    # wrong track.
    track_titles: list[str] = Field(default_factory=list)
    # Whether the user wants playback to START now (vs. only queue/discover).
    start_playback: bool = False

    @property
    def primary_title(self) -> str | None:
        """The first named track title (the now-playing target), or ``None``."""
        return self.track_titles[0] if self.track_titles else None

    @property
    def reacts(self) -> bool:
        """Whether this turn should speak an acknowledgment before the answer."""
        return self.engagement_mode in (
            EngagementMode.REACT_THEN_EXECUTE,
            EngagementMode.CLARIFY,
            EngagementMode.CONFIRM_ACTION,
        )

    @property
    def executes(self) -> bool:
        """Whether retrieval / the answer should run this turn (vs wait for reply)."""
        return self.engagement_mode in (
            EngagementMode.DIRECT_EXECUTE,
            EngagementMode.REACT_THEN_EXECUTE,
        )


# Safe default used when the router model errors or returns unparseable output:
# a warm, conversational turn that fires no retrieval and just lets Gia respond.
def safe_default_decision() -> RouterDecision:
    """Return the fallback decision for a failed/garbled router call."""
    return RouterDecision(
        intent=IntentType.GENERAL_CHAT,
        tone=Tone.WARM,
        confidence=0.0,
        engagement_mode=EngagementMode.DIRECT_EXECUTE,
    )
