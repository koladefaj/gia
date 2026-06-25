"""Tests for mood inference — time bucketing + LLM-labeled patterns."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.mood.inference import group_by_time_bucket


def _mock_event(name: str = "Free Mind", artist: str = "Tems", hour: int = 20, weekday: int = 6) -> MagicMock:
    evt = MagicMock()
    evt.track_name = name
    evt.artist_name = artist
    evt.played_at = datetime(2026, 6, 15 + weekday, hour, 0, tzinfo=UTC)
    return evt


class TestGroupByTimeBucket:
    def test_groups_by_weekday_and_hour(self) -> None:
        events = [
            _mock_event(hour=20, weekday=6),  # sunday_evening
            _mock_event(hour=21, weekday=6),  # sunday_evening
            _mock_event(hour=8, weekday=0),   # monday_morning
        ]
        buckets = group_by_time_bucket(events)
        assert len(buckets["sunday_evening"]) == 2
        assert len(buckets["monday_morning"]) == 1

    def test_empty_events(self) -> None:
        assert group_by_time_bucket([]) == {}


@pytest.mark.asyncio
async def test_infer_mood_patterns_labels_and_stores(test_settings) -> None:
    """A busy bucket is LLM-labeled and stored as a mood_pattern memory."""
    from backend.app.mood.inference import infer_mood_patterns

    events = [_mock_event(hour=20, weekday=6) for _ in range(6)]

    fake_store = MagicMock()
    fake_store.search = AsyncMock(return_value=[])
    fake_store.upsert_memory = AsyncMock(return_value="mem-id")
    fake_store.delete_by_id = AsyncMock()

    with patch("backend.app.mood.inference.get_listening_events", new=AsyncMock(return_value=events)), \
         patch("backend.app.mood.inference.label_mood", new=AsyncMock(return_value="chill")), \
         patch("backend.app.mood.inference.embed", new=AsyncMock(return_value=[0.0] * 768)):
        stored = await infer_mood_patterns("user-1", MagicMock(), fake_store, test_settings)

    assert "sunday_evening" in stored
    fake_store.upsert_memory.assert_called_once()
    # The stored pattern text carries the bucket + label so it parses back out.
    entry = fake_store.upsert_memory.call_args.args[1]
    assert "sunday_evening" in entry.text and "chill" in entry.text


@pytest.mark.asyncio
async def test_infer_mood_patterns_skips_low_sample_bucket(test_settings) -> None:
    """Buckets with fewer than MIN_SAMPLE_SIZE events are skipped (no labeling)."""
    from backend.app.mood.inference import infer_mood_patterns

    events = [_mock_event(hour=20, weekday=6) for _ in range(3)]  # only 3

    fake_store = MagicMock()
    fake_store.upsert_memory = AsyncMock()
    label = AsyncMock(return_value="chill")

    with patch("backend.app.mood.inference.get_listening_events", new=AsyncMock(return_value=events)), \
         patch("backend.app.mood.inference.label_mood", new=label):
        stored = await infer_mood_patterns("user-1", MagicMock(), fake_store, test_settings)

    assert stored == []
    label.assert_not_called()
    fake_store.upsert_memory.assert_not_called()


@pytest.mark.asyncio
async def test_infer_mood_patterns_skips_neutral_label(test_settings) -> None:
    """A 'neutral' label means no real signal — nothing is stored."""
    from backend.app.mood.inference import infer_mood_patterns

    events = [_mock_event(hour=20, weekday=6) for _ in range(6)]

    fake_store = MagicMock()
    fake_store.search = AsyncMock(return_value=[])
    fake_store.upsert_memory = AsyncMock()

    with patch("backend.app.mood.inference.get_listening_events", new=AsyncMock(return_value=events)), \
         patch("backend.app.mood.inference.label_mood", new=AsyncMock(return_value="neutral")):
        stored = await infer_mood_patterns("user-1", MagicMock(), fake_store, test_settings)

    assert stored == []
    fake_store.upsert_memory.assert_not_called()


@pytest.mark.asyncio
async def test_infer_mood_patterns_supersedes_existing(test_settings) -> None:
    """An existing pattern for the same bucket is deleted before inserting."""
    from backend.app.mood.inference import infer_mood_patterns
    from backend.app.schemas.memory import MemoryEntry

    events = [_mock_event(hour=20, weekday=6) for _ in range(6)]
    existing_mem = MagicMock(spec=MemoryEntry)
    existing_mem.id = "old-mem-id"
    existing_mem.text = "Mood pattern for sunday_evening: hype."

    fake_store = MagicMock()
    fake_store.search = AsyncMock(return_value=[existing_mem])
    fake_store.delete_by_id = AsyncMock()
    fake_store.upsert_memory = AsyncMock(return_value="new-mem-id")

    with patch("backend.app.mood.inference.get_listening_events", new=AsyncMock(return_value=events)), \
         patch("backend.app.mood.inference.label_mood", new=AsyncMock(return_value="chill")), \
         patch("backend.app.mood.inference.embed", new=AsyncMock(return_value=[0.0] * 768)):
        stored = await infer_mood_patterns("user-1", MagicMock(), fake_store, test_settings)

    fake_store.delete_by_id.assert_called_once_with("old-mem-id")
    assert "sunday_evening" in stored
