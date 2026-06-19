"""Tests for ``GET /memory/{user_id}/context`` and ``POST /memory/{user_id}/extract``."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from backend.app.schemas.memory import MemoryEntry, UserContext

_USER_ID = "00000000-0000-0000-0000-000000000001"
_NOW = datetime(2026, 6, 19, tzinfo=timezone.utc)


def _fake_context() -> UserContext:
    return UserContext(
        user_id=_USER_ID,
        profile={"timezone": "UTC", "preferred_genres": ["afrobeats"], "preferred_volume": 0.7},
        preferences=[
            MemoryEntry(
                id="00000000-0000-0000-0000-000000000011",
                type="preference",
                text="Loves Tems",
                confidence=0.9,
                created_at=_NOW,
            )
        ],
    )


@pytest.mark.asyncio
async def test_get_context_returns_user_context(client: AsyncClient) -> None:
    """``GET /memory/{user_id}/context`` returns a ``UserContext`` for a valid UUID."""
    with patch(
        "backend.app.api.memory.build_user_context",
        new=AsyncMock(return_value=_fake_context()),
    ):
        response = await client.get(f"/memory/{_USER_ID}/context")

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == _USER_ID
    assert len(data["preferences"]) == 1
    assert data["preferences"][0]["text"] == "Loves Tems"


@pytest.mark.asyncio
async def test_get_context_invalid_uuid_returns_400(client: AsyncClient) -> None:
    """``GET /memory/{user_id}/context`` returns 400 for an invalid UUID."""
    response = await client.get("/memory/not-a-uuid/context")
    assert response.status_code == 400
    assert "not a valid UUID" in response.json()["detail"]


@pytest.mark.asyncio
async def test_extract_endpoint_returns_extraction_response(client: AsyncClient) -> None:
    """``POST /memory/{user_id}/extract`` runs extraction and returns stored count."""
    with patch(
        "backend.app.api.memory.MemoryService",
    ) as mock_service_cls:
        instance = MagicMock()
        instance.run_extraction = AsyncMock(return_value=["mem-id-1"])
        mock_service_cls.return_value = instance

        response = await client.post(
            f"/memory/{_USER_ID}/extract",
            json={"transcript": "I loved Free Mind by Tems"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == _USER_ID
    assert data["stored"] == 1
    assert data["memory_ids"] == ["mem-id-1"]


@pytest.mark.asyncio
async def test_extract_endpoint_invalid_uuid_returns_400(client: AsyncClient) -> None:
    """``POST /memory/{user_id}/extract`` returns 400 for an invalid UUID."""
    response = await client.post(
        "/memory/bad-id/extract",
        json={"transcript": "some text"},
    )
    assert response.status_code == 400
    assert "not a valid UUID" in response.json()["detail"]


@pytest.mark.asyncio
async def test_extract_endpoint_no_memories_returns_zero(client: AsyncClient) -> None:
    """When extraction finds nothing durable, ``stored`` is 0."""
    with patch("backend.app.api.memory.MemoryService") as mock_service_cls:
        instance = MagicMock()
        instance.run_extraction = AsyncMock(return_value=[])
        mock_service_cls.return_value = instance

        response = await client.post(
            f"/memory/{_USER_ID}/extract",
            json={"transcript": "play it at 8pm"},
        )

    assert response.status_code == 200
    assert response.json()["stored"] == 0
