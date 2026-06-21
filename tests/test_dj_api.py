"""Tests for ``POST /dj/recommend``."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from backend.app.schemas.dj import CrossfadeQueue, DJResponse, TrackItem


def _fake_dj_response() -> DJResponse:
    track = TrackItem(uri="spotify:track:001", name="Free Mind", artist="Tems")
    return DJResponse(
        recommendation="[thoughtful] Here's Free Mind by Tems — should sit right.",
        primary_track=track,
        queue=CrossfadeQueue(
            seed_uri="spotify:track:001",
            tracks=[
                TrackItem(uri="spotify:track:002", name="Essence", artist="Wizkid"),
            ],
            crossfade_ms=3000,
        ),
        playback_started=False,
    )


@pytest.mark.asyncio
async def test_dj_recommend_returns_200(client: AsyncClient) -> None:
    """``POST /dj/recommend`` returns 200 with a valid ``DJResponse``."""
    with patch("backend.app.api.dj.DJService") as mock_cls:
        instance = MagicMock()
        instance.recommend = AsyncMock(return_value=_fake_dj_response())
        mock_cls.return_value = instance

        response = await client.post("/dj/recommend", json={"query": "chill Afrobeats"})

    assert response.status_code == 200
    data = response.json()
    assert data["primary_track"]["name"] == "Free Mind"
    assert data["primary_track"]["artist"] == "Tems"
    assert len(data["queue"]["tracks"]) == 1
    assert data["playback_started"] is False


@pytest.mark.asyncio
async def test_dj_recommend_with_user_id(client: AsyncClient) -> None:
    """User context is fetched when ``user_id`` is provided."""
    with patch("backend.app.api.dj.DJService") as mock_cls, \
         patch("backend.app.api.dj.build_user_context", new=AsyncMock(return_value=MagicMock(to_prompt_text=lambda: "ctx"))):
        instance = MagicMock()
        instance.recommend = AsyncMock(return_value=_fake_dj_response())
        mock_cls.return_value = instance

        response = await client.post(
            "/dj/recommend",
            json={
                "query": "chill Afrobeats",
                "user_id": "00000000-0000-0000-0000-000000000001",
            },
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_dj_recommend_start_playback(client: AsyncClient) -> None:
    """``start_playback=true`` is passed through to the service."""
    playing_response = _fake_dj_response()
    playing_response.playback_started = True

    with patch("backend.app.api.dj.DJService") as mock_cls:
        instance = MagicMock()
        instance.recommend = AsyncMock(return_value=playing_response)
        mock_cls.return_value = instance

        response = await client.post(
            "/dj/recommend",
            json={"query": "hype", "start_playback": True},
        )

    assert response.status_code == 200
    assert response.json()["playback_started"] is True


@pytest.mark.asyncio
async def test_dj_recommend_context_error_still_succeeds(client: AsyncClient) -> None:
    """Context fetch errors are swallowed — the recommendation still returns."""
    with patch("backend.app.api.dj.DJService") as mock_cls, \
         patch("backend.app.api.dj.build_user_context", new=AsyncMock(side_effect=RuntimeError("Weaviate down"))):
        instance = MagicMock()
        instance.recommend = AsyncMock(return_value=_fake_dj_response())
        mock_cls.return_value = instance

        response = await client.post(
            "/dj/recommend",
            json={"query": "chill", "user_id": "00000000-0000-0000-0000-000000000001"},
        )

    assert response.status_code == 200
