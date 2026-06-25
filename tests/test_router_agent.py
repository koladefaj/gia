"""Tests for the Router agent intent classifier."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from backend.app.schemas.chat import IntentType


class TestKeywordClassify:
    """Pure unit tests for the heuristic classifier (no LLM, no network)."""

    def _classify(self, msg: str) -> IntentType | None:
        from backend.app.agents.router import _keyword_classify
        return _keyword_classify(msg)

    def test_music_find_keywords(self) -> None:
        assert self._classify("find me something chill") == IntentType.MUSIC_FIND

    def test_music_find_play(self) -> None:
        assert self._classify("play some Afrobeats") == IntentType.MUSIC_FIND

    def test_music_queue_add(self) -> None:
        assert self._classify("add this to my queue") == IntentType.MUSIC_QUEUE

    def test_music_queue_save(self) -> None:
        assert self._classify("save this track") == IntentType.MUSIC_QUEUE

    def test_music_queue_playlist(self) -> None:
        assert self._classify("build me a playlist") == IntentType.MUSIC_QUEUE

    def test_artist_info_tell_me_about(self) -> None:
        assert self._classify("tell me about Odumodublvck") == IntentType.ARTIST_INFO

    def test_artist_info_who_is(self) -> None:
        assert self._classify("who is Tems?") == IntentType.ARTIST_INFO

    def test_mood_check_mood(self) -> None:
        assert self._classify("what's my mood right now?") == IntentType.MOOD_CHECK

    def test_mood_check_pattern(self) -> None:
        assert self._classify("what are my listening patterns?") == IntentType.MOOD_CHECK

    def test_mixed_music_and_artist(self) -> None:
        result = self._classify("play something from Tems — maybe a song from her album")
        assert result == IntentType.MIXED

    def test_returns_none_for_empty_message(self) -> None:
        assert self._classify("") is None

    def test_greeting_is_general(self) -> None:
        # Bare greetings are fast-pathed to a conversational GENERAL reply.
        assert self._classify("hey") == IntentType.GENERAL

    def test_smalltalk_is_general(self) -> None:
        # Non-music small talk (weather) → GENERAL, never an artist lookup.
        assert self._classify("whats the weather like") == IntentType.GENERAL

    def test_long_greeting_is_general(self) -> None:
        # A greeting stays GENERAL even past the old 8-word cap (no music signal).
        assert self._classify("just thought I'd say hi how have you been") == IntentType.GENERAL

    def test_greeting_with_music_keeps_music_intent(self) -> None:
        # A greeting that also asks for music must not be swallowed by GENERAL.
        assert self._classify("hey play me something chill") == IntentType.MUSIC_FIND

    def test_returns_none_for_ambiguous(self) -> None:
        # No keyword, not a greeting/small talk → defer to the LLM (None here).
        assert self._classify("you decide") is None


@pytest.mark.asyncio
async def test_classify_intent_heuristic_path(test_settings) -> None:
    """Heuristic matches short-circuit the LLM call."""
    from backend.app.agents.router import classify_intent

    intent, confidence = await classify_intent("find me Afrobeats", test_settings)
    assert intent == IntentType.MUSIC_FIND
    assert confidence == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_classify_intent_llm_fallback(test_settings) -> None:
    """Ambiguous messages fall back to the LLM."""
    from backend.app.agents.router import classify_intent

    with patch("backend.app.agents.router.asyncio.to_thread", new=AsyncMock(return_value="ARTIST_INFO")):
        intent, confidence = await classify_intent("you decide", test_settings)

    assert intent == IntentType.ARTIST_INFO
    assert confidence == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_classify_intent_llm_error_raises(test_settings) -> None:
    """An unreachable LLM on an ambiguous message surfaces as a RuntimeError.

    The planner/chat layer catches this to emit a friendly "thinking service
    unavailable" reply rather than silently guessing an intent.
    """
    from backend.app.agents.router import classify_intent

    with patch("backend.app.agents.router.asyncio.to_thread", new=AsyncMock(side_effect=RuntimeError("LLM down"))), \
         pytest.raises(RuntimeError):
        await classify_intent("you decide", test_settings)


