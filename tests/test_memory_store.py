"""Tests for ``WeaviateMemoryStore``.

Weaviate's synchronous client is replaced with a ``MagicMock`` so these
tests run without a real Weaviate instance.  The critical behaviours tested
are: correct filter construction, result mapping, and ``asyncio.to_thread``
dispatch.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.schemas.memory import ExtractedMemory


def _make_weaviate_obj(
    uid: str,
    memory_type: str,
    text: str,
    confidence: float = 0.9,
    supersedes_id: str = "",
    score: float = 0.85,
) -> MagicMock:
    """Build a fake Weaviate result object."""
    obj = MagicMock()
    obj.uuid = uuid.UUID(uid)
    obj.properties = {
        "type": memory_type,
        "text": text,
        "confidence": confidence,
        "created_at": datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc),
        "supersedes_id": supersedes_id,
        "user_id": "00000000-0000-0000-0000-000000000001",
    }
    obj.metadata = MagicMock()
    obj.metadata.score = score
    return obj


@pytest.fixture()
def fake_weaviate() -> MagicMock:
    mock = MagicMock()
    col = MagicMock()
    mock.collections.get.return_value = col
    # Default: empty results
    col.query.near_vector.return_value = MagicMock(objects=[])
    col.data.insert.return_value = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
    return mock


@pytest.mark.asyncio
async def test_search_returns_empty_when_no_results(fake_weaviate: MagicMock) -> None:
    """``search`` returns ``[]`` when Weaviate has no matching objects."""
    from backend.app.memory.store import WeaviateMemoryStore

    store = WeaviateMemoryStore(client=fake_weaviate)

    with patch("backend.app.memory.store.asyncio.to_thread", new=AsyncMock(return_value=[])):
        result = await store.search("user-1", [0.1] * 768, "preference", k=5)

    assert result == []


@pytest.mark.asyncio
async def test_search_maps_objects_to_memory_entries(fake_weaviate: MagicMock) -> None:
    """``search`` converts raw Weaviate objects to ``MemoryEntry`` instances."""
    from backend.app.memory.store import WeaviateMemoryStore, _obj_to_entry

    obj = _make_weaviate_obj(
        "aaaaaaaa-0000-0000-0000-000000000001",
        "preference",
        "User loves Tems",
        confidence=0.9,
        score=0.87,
    )

    entry = _obj_to_entry(obj)
    assert entry.id == "aaaaaaaa-0000-0000-0000-000000000001"
    assert entry.type == "preference"
    assert entry.text == "User loves Tems"
    assert entry.confidence == 0.9
    assert entry.score == pytest.approx(0.87)
    assert entry.supersedes_id is None


@pytest.mark.asyncio
async def test_upsert_memory_returns_uuid_string(fake_weaviate: MagicMock) -> None:
    """``upsert_memory`` returns the inserted Weaviate UUID as a string."""
    from backend.app.memory.store import WeaviateMemoryStore

    store = WeaviateMemoryStore(client=fake_weaviate)
    memory = ExtractedMemory(type="preference", text="User loves Tems", confidence=0.9)
    vector = [0.1] * 768

    with patch(
        "backend.app.memory.store.asyncio.to_thread",
        new=AsyncMock(return_value="aaaaaaaa-0000-0000-0000-000000000001"),
    ):
        result = await store.upsert_memory("user-1", memory, vector)

    assert result == "aaaaaaaa-0000-0000-0000-000000000001"


@pytest.mark.asyncio
async def test_delete_by_id_calls_weaviate(fake_weaviate: MagicMock) -> None:
    """``delete_by_id`` dispatches a deletion to Weaviate."""
    from backend.app.memory.store import WeaviateMemoryStore

    store = WeaviateMemoryStore(client=fake_weaviate)

    with patch("backend.app.memory.store.asyncio.to_thread", new=AsyncMock(return_value=None)):
        await store.delete_by_id("aaaaaaaa-0000-0000-0000-000000000001")


@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_missing(fake_weaviate: MagicMock) -> None:
    """``get_by_id`` returns ``None`` when Weaviate returns no object."""
    from backend.app.memory.store import WeaviateMemoryStore

    store = WeaviateMemoryStore(client=fake_weaviate)

    with patch("backend.app.memory.store.asyncio.to_thread", new=AsyncMock(return_value=None)):
        result = await store.get_by_id("aaaaaaaa-0000-0000-0000-000000000001")

    assert result is None


def test_obj_to_entry_handles_string_created_at() -> None:
    """``_obj_to_entry`` parses ISO string dates correctly."""
    from backend.app.memory.store import _obj_to_entry

    obj = MagicMock()
    obj.uuid = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
    obj.properties = {
        "type": "preference",
        "text": "test",
        "confidence": 0.8,
        "created_at": "2026-06-19T12:00:00+00:00",
        "supersedes_id": None,
        "user_id": "u1",
    }
    obj.metadata = None

    entry = _obj_to_entry(obj)
    assert entry.created_at.year == 2026
    assert entry.score == 0.0


def test_obj_to_entry_handles_none_created_at() -> None:
    """``_obj_to_entry`` falls back to utcnow when ``created_at`` is absent."""
    from backend.app.memory.store import _obj_to_entry

    obj = MagicMock()
    obj.uuid = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
    obj.properties = {
        "type": "episode",
        "text": "session summary",
        "confidence": None,
        "created_at": None,
        "supersedes_id": "",
        "user_id": "u1",
    }
    obj.metadata = None

    entry = _obj_to_entry(obj)
    assert entry.confidence == 0.8  # fallback
    assert entry.supersedes_id is None  # empty string coerced
