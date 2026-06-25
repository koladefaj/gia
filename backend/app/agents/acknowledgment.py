"""Acknowledgment selection — the instant, LLM-free reaction layer.

While retrieval and the conversation model run, Gia needs to *say something*
within ~1s so the user never hears silence.  That something is a pre-written
acknowledgment chosen by intent+tone — no model call, just a dictionary lookup
and a random pick that avoids the last few used so she never repeats herself.

Templates carry NO provider voice tags; the ``VoiceAdapter`` prepends the tone
tag at speak time, keeping the templates provider-neutral.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

from backend.app.observability.logging import get_logger
from backend.app.schemas.chat import IntentType
from backend.app.schemas.router import EngagementMode, RouterDecision, Tone

logger = get_logger(__name__)

_TABLE_PATH = Path(__file__).resolve().parent.parent / "data" / "acknowledgements.json"
# With a ~30-line filler pool, avoid the last 8 so a given "okay" doesn't recur
# for several turns running.
_AVOID_LAST = 8


@lru_cache(maxsize=1)
def _load_table() -> dict[str, dict[str, list[str]]]:
    """Load and cache the acknowledgment template table."""
    data: dict[str, dict[str, list[str]]] = json.loads(
        _TABLE_PATH.read_text(encoding="utf-8")
    )
    return data


class AcknowledgmentSelector:
    """Selects a non-repeating acknowledgment per session.

    Keeps a per-session ring of the last ``_AVOID_LAST`` lines used so the same
    reaction never lands twice in a row.  State is in-process (a turn is cheap;
    there is no need to round-trip Redis on the <10ms hot path).
    """

    def __init__(self, table: dict[str, dict[str, list[str]]] | None = None) -> None:
        self._table = table if table is not None else _load_table()
        self._recent: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=_AVOID_LAST))

    def _candidates(self, intent: IntentType, tone: Tone) -> list[str]:
        """Resolve the candidate lines for *intent*+*tone* with graceful fallback."""
        by_intent = self._table.get(intent.value) or self._table["_default"]
        return (
            by_intent.get(tone.value)
            or by_intent.get("default")
            or self._table["_default"]["default"]
        )

    def select_filler(self, session_id: str) -> str:
        """Return a neutral, intent-agnostic filler line, avoiding recent repeats.

        Spoken on the *fast* path — before the router's real decision exists — so
        it must claim nothing specific. "One sec." can front a recommendation, a
        clarifying question, or pure chat without ever contradicting the reply,
        whereas an intent-specific line ("Let me organize the tracks") promises
        work a conversational turn never does. The keyword pre-classifier only
        decides *whether* to react; the content stays safe regardless of outcome.

        Args:
            session_id: Conversation session id (scopes its own no-repeat ring).

        Returns:
            A single neutral filler line (no voice tags).
        """
        candidates = self._table["_default"]["default"]
        recent = self._recent[f"filler:{session_id}"]
        pool = [c for c in candidates if c not in recent] or candidates
        choice = random.choice(pool)
        recent.append(choice)
        return choice

    def select(self, intent: IntentType, tone: Tone, session_id: str) -> str:
        """Return an acknowledgment for the turn, avoiding the session's last few.

        Args:
            intent:     The router's classified intent.
            tone:       The router's chosen tone.
            session_id: Conversation session id (scopes the no-repeat ring).

        Returns:
            A single acknowledgment line (no voice tags).
        """
        candidates = self._candidates(intent, tone)
        recent = self._recent[session_id]
        pool = [c for c in candidates if c not in recent] or candidates
        choice = random.choice(pool)
        recent.append(choice)
        return choice


def should_acknowledge(decision: RouterDecision) -> bool:
    """Whether the turn warrants an immediate spoken acknowledgment.

    We acknowledge when there is real background work that would otherwise leave
    the user in silence:

    - ``react_then_execute`` always reacts (the personality moment).
    - ``direct_execute`` only when a retriever will run (search / music / artist
      lookup) — so "play Tems" gets a brief "On it.", but pure chit-chat doesn't
      get an "On it." in front of an already-fast conversational reply.
    - ``clarify`` / ``confirm_action`` speak a question/confirmation instead,
      handled by the engagement logic rather than a generic ack.
    """
    if decision.engagement_mode == EngagementMode.REACT_THEN_EXECUTE:
        return True
    if decision.engagement_mode == EngagementMode.DIRECT_EXECUTE:
        return (
            decision.needs_search
            or decision.needs_music
            or decision.needs_artist_lookup
        )
    return False


# Process-wide singleton so the no-repeat ring persists across requests.
_selector: AcknowledgmentSelector | None = None


def get_selector() -> AcknowledgmentSelector:
    """Return the shared ``AcknowledgmentSelector`` (built on first use)."""
    global _selector
    if _selector is None:
        _selector = AcknowledgmentSelector()
    return _selector
