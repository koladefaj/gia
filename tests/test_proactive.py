"""Tests for proactive mood shift detection."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stored pattern in the format inference writes: "... {bucket}: {label}. ..."
_PATTERN = "Mood pattern for sunday_evening: chill. Often plays Tems, Wizkid. Based on 8 plays."


def _mem(text: str) -> MagicMock:
    m = MagicMock()
    m.id = "mem-1"
    m.text = text
    return m


def test_parse_pattern_extracts_label() -> None:
    from backend.app.mood.proactive import _parse_pattern

    assert _parse_pattern(_PATTERN) == "chill"
    assert _parse_pattern("no marker here") == "neutral"


@pytest.mark.asyncio
async def test_check_and_draft_proactive_produces_draft_on_shift() -> None:
    """A current mood different from the bucket's pattern drafts a nudge."""
    from backend.app.mood.proactive import check_and_draft_proactive

    fake_redis = AsyncMock()
    fake_redis.setex = AsyncMock()

    with patch("backend.app.mood.proactive.get_pattern_for_now",
               new=AsyncMock(return_value=_mem(_PATTERN))):
        draft = await check_and_draft_proactive("user-1", "hype", MagicMock(), fake_redis)

    assert draft is not None
    assert "chill" in draft and "hype" in draft
    fake_redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_check_and_draft_no_shift_returns_none() -> None:
    """No draft when the current mood matches the pattern."""
    from backend.app.mood.proactive import check_and_draft_proactive

    fake_redis = AsyncMock()

    with patch("backend.app.mood.proactive.get_pattern_for_now",
               new=AsyncMock(return_value=_mem(_PATTERN))):
        draft = await check_and_draft_proactive("user-1", "chill", MagicMock(), fake_redis)

    assert draft is None
    fake_redis.setex.assert_not_called()


@pytest.mark.asyncio
async def test_check_and_draft_neutral_current_returns_none() -> None:
    """A neutral current label is not a meaningful shift — no draft."""
    from backend.app.mood.proactive import check_and_draft_proactive

    fake_redis = AsyncMock()

    with patch("backend.app.mood.proactive.get_pattern_for_now",
               new=AsyncMock(return_value=_mem(_PATTERN))):
        draft = await check_and_draft_proactive("user-1", "neutral", MagicMock(), fake_redis)

    assert draft is None


@pytest.mark.asyncio
async def test_check_and_draft_no_pattern_returns_none() -> None:
    """No draft when no pattern exists for the current bucket."""
    from backend.app.mood.proactive import check_and_draft_proactive

    fake_redis = AsyncMock()

    with patch("backend.app.mood.proactive.get_pattern_for_now", new=AsyncMock(return_value=None)):
        draft = await check_and_draft_proactive("user-1", "hype", MagicMock(), fake_redis)

    assert draft is None


@pytest.mark.asyncio
async def test_pop_proactive_draft_returns_and_clears() -> None:
    """``pop_proactive_draft`` returns the stored draft and deletes the key."""
    from backend.app.mood.proactive import pop_proactive_draft

    fake_redis = AsyncMock()
    fake_redis.get = AsyncMock(return_value="[thoughtful] You're usually on something softer.")
    fake_redis.delete = AsyncMock()

    result = await pop_proactive_draft("user-1", fake_redis)

    assert result == "[thoughtful] You're usually on something softer."
    fake_redis.delete.assert_called_once_with("proactive:user-1")


@pytest.mark.asyncio
async def test_pop_proactive_draft_none_when_empty() -> None:
    """``pop_proactive_draft`` returns ``None`` when no draft is pending."""
    from backend.app.mood.proactive import pop_proactive_draft

    fake_redis = AsyncMock()
    fake_redis.get = AsyncMock(return_value=None)

    result = await pop_proactive_draft("user-1", fake_redis)
    assert result is None
