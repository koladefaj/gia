"""Tests for the memory decay and conflict-resolution module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.schemas.memory import MemoryEntry

_UID = "00000000-0000-0000-0000-000000000001"
_NOW = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)


def _entry(text: str, offset_days: int = 0, uid: str = _UID) -> MemoryEntry:
    return MemoryEntry(
        id=uid,
        type="preference",
        text=text,
        confidence=0.8,
        created_at=_NOW + timedelta(days=offset_days),
    )


def test_prefer_recent_sorts_newest_first() -> None:
    """``prefer_recent`` returns memories newest-first."""
    from backend.app.memory.decay import prefer_recent

    memories = [
        _entry("oldest", offset_days=-10),
        _entry("newest", offset_days=0),
        _entry("middle", offset_days=-5),
    ]
    result = prefer_recent(memories)
    assert [m.text for m in result] == ["newest", "middle", "oldest"]


def test_prefer_recent_empty_list() -> None:
    """``prefer_recent`` on an empty list returns an empty list."""
    from backend.app.memory.decay import prefer_recent

    assert prefer_recent([]) == []


def test_prefer_recent_single_item() -> None:
    """``prefer_recent`` on a single item returns that item."""
    from backend.app.memory.decay import prefer_recent

    m = _entry("only")
    assert prefer_recent([m]) == [m]


def test_deduplicate_keeps_newest_copy() -> None:
    """``deduplicate`` keeps only the newest occurrence of duplicate texts."""
    from backend.app.memory.decay import deduplicate

    old = _entry("same text", offset_days=-5, uid="00000000-0000-0000-0000-000000000001")
    new = _entry("same text", offset_days=0, uid="00000000-0000-0000-0000-000000000002")
    unique = _entry("unique text", offset_days=-2, uid="00000000-0000-0000-0000-000000000003")

    result = deduplicate([old, new, unique])
    texts = {m.text for m in result}
    assert texts == {"same text", "unique text"}

    same_text_entry = next(m for m in result if m.text == "same text")
    assert same_text_entry.id == new.id


def test_deduplicate_no_duplicates_unchanged() -> None:
    """``deduplicate`` returns all items when there are no duplicates."""
    from backend.app.memory.decay import deduplicate

    memories = [
        _entry("text A", uid="00000000-0000-0000-0000-000000000001"),
        _entry("text B", uid="00000000-0000-0000-0000-000000000002"),
    ]
    result = deduplicate(memories)
    assert {m.text for m in result} == {"text A", "text B"}


@pytest.mark.asyncio
async def test_apply_supersede_deletes_existing() -> None:
    """``apply_supersede`` deletes the old memory when it exists."""
    from backend.app.memory.decay import apply_supersede

    old_entry = _entry("old preference")
    mock_store = MagicMock()
    mock_store.get_by_id = AsyncMock(return_value=old_entry)
    mock_store.delete_by_id = AsyncMock()

    await apply_supersede(mock_store, _UID)

    mock_store.delete_by_id.assert_called_once_with(_UID)


@pytest.mark.asyncio
async def test_apply_supersede_skips_if_not_found() -> None:
    """``apply_supersede`` is a no-op when the target memory does not exist."""
    from backend.app.memory.decay import apply_supersede

    mock_store = MagicMock()
    mock_store.get_by_id = AsyncMock(return_value=None)
    mock_store.delete_by_id = AsyncMock()

    await apply_supersede(mock_store, _UID)

    mock_store.delete_by_id.assert_not_called()
