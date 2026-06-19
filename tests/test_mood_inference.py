"""Tests for mood inference — time-series grouping and pattern detection."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.mood.classifier import time_bucket
from backend.app.mood.inference import (
    _stddev,
    group_by_time_bucket,
)


def _mock_event(energy: float, valence: float, hour: int = 20, weekday: int = 6) -> MagicMock:
    evt = MagicMock()
    evt.energy = energy
    evt.valence = valence
    evt.tempo = 100.0
    evt.played_at = datetime(2026, 6, 15 + weekday, hour, 0, tzinfo=timezone.utc)
    return evt


class TestStddev:
    def test_zero_for_single_value(self) -> None:
        assert _stddev([0.5]) == pytest.approx(0.0)

    def test_zero_for_identical_values(self) -> None:
        assert _stddev([0.3, 0.3, 0.3]) == pytest.approx(0.0)

    def test_correct_deviation(self) -> None:
        values = [0.0, 0.2, 0.4, 0.6, 0.8]
        result = _stddev(values)
        assert 0.28 < result < 0.30

    def test_empty_list(self) -> None:
        assert _stddev([]) == pytest.approx(0.0)


class TestGroupByTimeBucket:
    def test_groups_by_weekday_and_hour(self) -> None:
        events = [
            _mock_event(0.3, 0.7, hour=20, weekday=6),  # sunday_evening
            _mock_event(0.3, 0.7, hour=21, weekday=6),  # sunday_evening
            _mock_event(0.8, 0.4, hour=8, weekday=0),   # monday_morning
        ]
        buckets = group_by_time_bucket(events)
        assert "sunday_evening" in buckets
        assert "monday_morning" in buckets
        assert len(buckets["sunday_evening"]) == 2
        assert len(buckets["monday_morning"]) == 1

    def test_empty_events(self) -> None:
        assert group_by_time_bucket([]) == {}

    def test_all_in_same_bucket(self) -> None:
        events = [_mock_event(0.3 + i * 0.01, 0.7, hour=20, weekday=6) for i in range(5)]
        buckets = group_by_time_bucket(events)
        assert len(buckets) == 1
        assert len(buckets["sunday_evening"]) == 5


@pytest.mark.asyncio
async def test_infer_mood_patterns_detects_consistent_pattern() -> None:
    """A consistent bucket with >= 5 samples produces a stored pattern."""
    from backend.app.mood.inference import infer_mood_patterns

    events = [_mock_event(0.31 + i * 0.005, 0.72, hour=20, weekday=6) for i in range(6)]

    fake_db = MagicMock()
    fake_db.execute = AsyncMock()
    fake_db.execute.return_value.scalars.return_value.all.return_value = events

    fake_store = MagicMock()
    fake_store.search = AsyncMock(return_value=[])
    fake_store.upsert_memory = AsyncMock(return_value="mem-id")
    fake_store.delete_by_id = AsyncMock()

    with patch("backend.app.mood.inference.get_listening_events", new=AsyncMock(return_value=events)), \
         patch("backend.app.mood.inference.embed", new=AsyncMock(return_value=[0.0] * 768)):
        stored = await infer_mood_patterns("user-1", fake_db, fake_store)

    assert "sunday_evening" in stored
    fake_store.upsert_memory.assert_called_once()


@pytest.mark.asyncio
async def test_infer_mood_patterns_skips_low_sample_bucket() -> None:
    """Buckets with fewer than MIN_SAMPLE_SIZE events are skipped."""
    from backend.app.mood.inference import infer_mood_patterns

    events = [_mock_event(0.31, 0.72, hour=20, weekday=6) for _ in range(3)]  # only 3

    fake_db = MagicMock()
    fake_store = MagicMock()
    fake_store.search = AsyncMock(return_value=[])
    fake_store.upsert_memory = AsyncMock()

    with patch("backend.app.mood.inference.get_listening_events", new=AsyncMock(return_value=events)), \
         patch("backend.app.mood.inference.embed", new=AsyncMock(return_value=[0.0] * 768)):
        stored = await infer_mood_patterns("user-1", fake_db, fake_store)

    assert stored == []
    fake_store.upsert_memory.assert_not_called()


@pytest.mark.asyncio
async def test_infer_mood_patterns_skips_inconsistent_bucket() -> None:
    """High-variance buckets (σ >= 0.15) are not stored as patterns."""
    from backend.app.mood.inference import infer_mood_patterns

    events = [
        _mock_event(0.1, 0.7, hour=20, weekday=6),
        _mock_event(0.9, 0.7, hour=20, weekday=6),
        _mock_event(0.1, 0.7, hour=20, weekday=6),
        _mock_event(0.9, 0.7, hour=20, weekday=6),
        _mock_event(0.5, 0.7, hour=20, weekday=6),
    ]
    fake_db = MagicMock()
    fake_store = MagicMock()
    fake_store.search = AsyncMock(return_value=[])
    fake_store.upsert_memory = AsyncMock()

    with patch("backend.app.mood.inference.get_listening_events", new=AsyncMock(return_value=events)), \
         patch("backend.app.mood.inference.embed", new=AsyncMock(return_value=[0.0] * 768)):
        stored = await infer_mood_patterns("user-1", fake_db, fake_store)

    assert stored == []


@pytest.mark.asyncio
async def test_infer_mood_patterns_supersedes_existing() -> None:
    """Existing pattern for the same bucket is deleted before inserting."""
    from backend.app.mood.inference import infer_mood_patterns
    from backend.app.schemas.memory import MemoryEntry

    events = [_mock_event(0.31 + i * 0.005, 0.72, hour=20, weekday=6) for i in range(6)]
    existing_mem = MagicMock(spec=MemoryEntry)
    existing_mem.id = "old-mem-id"
    existing_mem.text = "Mood pattern for sunday_evening: wind-down."

    fake_db = MagicMock()
    fake_store = MagicMock()
    fake_store.search = AsyncMock(return_value=[existing_mem])
    fake_store.delete_by_id = AsyncMock()
    fake_store.upsert_memory = AsyncMock(return_value="new-mem-id")

    with patch("backend.app.mood.inference.get_listening_events", new=AsyncMock(return_value=events)), \
         patch("backend.app.mood.inference.embed", new=AsyncMock(return_value=[0.0] * 768)):
        stored = await infer_mood_patterns("user-1", fake_db, fake_store)

    fake_store.delete_by_id.assert_called_once_with("old-mem-id")
    assert "sunday_evening" in stored
