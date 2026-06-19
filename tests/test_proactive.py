"""Tests for proactive mood shift detection."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _mem(text: str) -> MagicMock:
    m = MagicMock()
    m.id = "mem-1"
    m.text = text
    return m


@pytest.mark.asyncio
async def test_check_and_draft_proactive_produces_draft_on_deviation() -> None:
    """A significant energy deviation produces a draft stored in Redis."""
    from backend.app.mood.proactive import check_and_draft_proactive

    pattern_text = (
        "Mood pattern for sunday_evening: wind-down. "
        "avg energy=0.31, avg valence=0.72, avg tempo=90 BPM. "
        "Based on 8 sessions (consistency σ=0.045)."
    )
    fake_store = MagicMock()
    fake_store.search = AsyncMock(return_value=[_mem(pattern_text)])
    fake_redis = AsyncMock()
    fake_redis.setex = AsyncMock()

    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "backend.app.mood.proactive.embed", new=AsyncMock(return_value=[0.0] * 768)
    ), __import__("unittest.mock", fromlist=["patch"]).patch(
        "backend.app.mood.proactive.datetime"
    ) as mock_dt:
        from datetime import datetime, timezone
        mock_dt.now.return_value = datetime(2026, 6, 21, 20, 0, tzinfo=timezone.utc)  # Sunday 20h
        mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)

        draft = await check_and_draft_proactive(
            user_id="user-1",
            current_energy=0.8,  # High energy — deviates from pattern 0.31
            current_valence=0.5,
            store=fake_store,
            redis=fake_redis,
        )

    assert draft is not None
    assert "pattern" in draft.lower() or "usually" in draft.lower() or "thoughtful" in draft.lower()
    fake_redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_check_and_draft_no_deviation_returns_none() -> None:
    """No draft when current features match the pattern."""
    from backend.app.mood.proactive import check_and_draft_proactive

    pattern_text = (
        "Mood pattern for sunday_evening: wind-down. "
        "avg energy=0.35, avg valence=0.70, avg tempo=90 BPM. "
        "Based on 8 sessions."
    )
    fake_store = MagicMock()
    fake_store.search = AsyncMock(return_value=[_mem(pattern_text)])
    fake_redis = AsyncMock()

    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "backend.app.mood.proactive.embed", new=AsyncMock(return_value=[0.0] * 768)
    ), __import__("unittest.mock", fromlist=["patch"]).patch(
        "backend.app.mood.proactive.datetime"
    ) as mock_dt:
        from datetime import datetime, timezone
        mock_dt.now.return_value = datetime(2026, 6, 21, 20, 0, tzinfo=timezone.utc)
        mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)

        draft = await check_and_draft_proactive(
            user_id="user-1",
            current_energy=0.33,  # Close to pattern 0.35 — no deviation
            current_valence=0.68,
            store=fake_store,
            redis=fake_redis,
        )

    assert draft is None
    fake_redis.setex.assert_not_called()


@pytest.mark.asyncio
async def test_check_and_draft_no_pattern_returns_none() -> None:
    """No draft when no pattern exists for the current bucket."""
    from backend.app.mood.proactive import check_and_draft_proactive

    fake_store = MagicMock()
    fake_store.search = AsyncMock(return_value=[])  # No patterns stored
    fake_redis = AsyncMock()

    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "backend.app.mood.proactive.embed", new=AsyncMock(return_value=[0.0] * 768)
    ):
        draft = await check_and_draft_proactive(
            user_id="user-1",
            current_energy=0.8,
            current_valence=0.5,
            store=fake_store,
            redis=fake_redis,
        )

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
