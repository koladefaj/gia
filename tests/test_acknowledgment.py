"""Tests for the acknowledgment selector + voice adapter."""

from __future__ import annotations

from backend.app.agents.acknowledgment import (
    AcknowledgmentSelector,
    should_acknowledge,
)
from backend.app.schemas.chat import IntentType
from backend.app.schemas.router import EngagementMode, RouterDecision, Tone
from backend.app.voice.adapter import VoiceAdapter


def _decision(mode: EngagementMode, **needs: bool) -> RouterDecision:
    return RouterDecision(
        intent=IntentType.NEWS_QUERY, tone=Tone.SURPRISED,
        engagement_mode=mode, **needs,
    )


class TestSelector:
    def test_selects_intent_tone_line(self) -> None:
        # Picks from the exact intent+tone bucket (assert against the live table
        # so growing the line list never breaks this test).
        from backend.app.agents.acknowledgment import _load_table

        sel = AcknowledgmentSelector()
        line = sel.select(IntentType.NEWS_QUERY, Tone.SURPRISED, "s1")
        assert line in _load_table()["NEWS_QUERY"]["surprised"]

    def test_falls_back_to_intent_default(self) -> None:
        # PLAYFUL has no NEWS_QUERY bucket → falls back to that intent's default.
        from backend.app.agents.acknowledgment import _load_table

        sel = AcknowledgmentSelector()
        line = sel.select(IntentType.NEWS_QUERY, Tone.PLAYFUL, "s1")
        assert line in _load_table()["NEWS_QUERY"]["default"]

    def test_select_filler_uses_neutral_default_bucket(self) -> None:
        # The fast-path filler must be intent-agnostic — drawn only from the
        # neutral _default bucket, never an intent-specific (retrieval) line.
        from backend.app.agents.acknowledgment import _load_table

        sel = AcknowledgmentSelector()
        neutral = _load_table()["_default"]["default"]
        picks = [sel.select_filler("s1") for _ in range(5)]
        assert all(p in neutral for p in picks)
        assert len(set(picks)) == len(picks)  # no repeats within the avoid window

    def test_global_default_for_unknown_intent(self) -> None:
        sel = AcknowledgmentSelector()
        line = sel.select(IntentType.MIXED, Tone.EMPATHETIC, "s1")
        assert line  # MIXED has no empathetic → MIXED default → still a string

    def test_avoids_recent_until_exhausted(self) -> None:
        sel = AcknowledgmentSelector()
        seen = [sel.select(IntentType.NEWS_QUERY, Tone.SURPRISED, "s1") for _ in range(3)]
        # 3 surprised lines exist; all 3 should be distinct before any repeat.
        assert len(set(seen)) == 3

    def test_sessions_are_independent(self) -> None:
        sel = AcknowledgmentSelector()
        a = sel.select(IntentType.NEWS_QUERY, Tone.SURPRISED, "sA")
        b = sel.select(IntentType.NEWS_QUERY, Tone.SURPRISED, "sB")
        # Different sessions don't share the no-repeat ring; both are valid lines.
        assert a and b


class TestShouldAcknowledge:
    def test_react_then_execute_always_acks(self) -> None:
        assert should_acknowledge(_decision(EngagementMode.REACT_THEN_EXECUTE)) is True

    def test_direct_execute_acks_only_with_retrieval(self) -> None:
        assert should_acknowledge(_decision(EngagementMode.DIRECT_EXECUTE)) is False
        assert should_acknowledge(
            _decision(EngagementMode.DIRECT_EXECUTE, needs_music=True)
        ) is True

    def test_clarify_and_confirm_do_not_generic_ack(self) -> None:
        assert should_acknowledge(_decision(EngagementMode.CLARIFY)) is False
        assert should_acknowledge(_decision(EngagementMode.CONFIRM_ACTION)) is False


class TestVoiceAdapter:
    def test_known_tone_maps_to_tag(self) -> None:
        assert VoiceAdapter().convert_tone_to_tags("surprised") == "[surprised]"

    def test_unknown_tone_maps_to_empty(self) -> None:
        assert VoiceAdapter().convert_tone_to_tags("nonsense") == ""

    def test_apply_prepends_tag(self) -> None:
        assert VoiceAdapter().apply("warm", "Hey there") == "[warmly] Hey there"

    def test_apply_noop_without_tag(self) -> None:
        assert VoiceAdapter().apply("nonsense", "Hey there") == "Hey there"
