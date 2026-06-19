"""Tests for ``WeaviateMemoryStore.hybrid_search`` (Weaviate client mocked)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from backend.app.memory.store import WeaviateMemoryStore


def _obj(uid: str, text: str, score: float) -> MagicMock:
    obj = MagicMock()
    obj.uuid = uuid.UUID(uid)
    obj.properties = {
        "type": "preference",
        "text": text,
        "confidence": 0.9,
        "created_at": datetime(2026, 6, 19, tzinfo=timezone.utc),
        "supersedes_id": "",
        "user_id": "00000000-0000-0000-0000-000000000001",
    }
    obj.metadata = MagicMock()
    obj.metadata.score = score
    return obj


@pytest.mark.asyncio
async def test_hybrid_search_maps_results_and_passes_params() -> None:
    col = MagicMock()
    col.query.hybrid.return_value = MagicMock(
        objects=[
            _obj("11111111-0000-0000-0000-000000000001", "loves Tems", 0.91),
            _obj("22222222-0000-0000-0000-000000000002", "likes Burna", 0.80),
        ]
    )
    client = MagicMock()
    client.collections.get.return_value = col

    store = WeaviateMemoryStore(client=client)
    results = await store.hybrid_search(
        "user-1", "tems song", [0.1] * 768, "preference", k=5, alpha=0.4
    )

    assert [m.text for m in results] == ["loves Tems", "likes Burna"]
    assert results[0].score == pytest.approx(0.91)

    # The hybrid query received both legs (keyword text + dense vector) and alpha.
    _, kwargs = col.query.hybrid.call_args
    assert kwargs["query"] == "tems song"
    assert kwargs["vector"] == [0.1] * 768
    assert kwargs["alpha"] == 0.4
    assert kwargs["limit"] == 5


@pytest.mark.asyncio
async def test_hybrid_search_empty() -> None:
    col = MagicMock()
    col.query.hybrid.return_value = MagicMock(objects=[])
    client = MagicMock()
    client.collections.get.return_value = col

    store = WeaviateMemoryStore(client=client)
    results = await store.hybrid_search("u", "q", [0.0] * 768, "preference")
    assert results == []
