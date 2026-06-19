"""Tests for ``POST /artist/info``."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from backend.app.schemas.artist import ArtistInfoResponse, BraveResult


def _fake_artist_response(artist_name: str = "Odumodublvck") -> ArtistInfoResponse:
    return ArtistInfoResponse(
        artist_name=artist_name,
        response=(
            "[curious] Odumodublvck has been on a tear this year — he just won Rap Album "
            "of the Year at the Headies. His flow on Declan is still unmatched."
        ),
        top_tracks=[
            {"uri": "spotify:track:a01", "name": "Declan", "artist": "Odumodublvck"},
            {"uri": "spotify:track:a02", "name": "Greek God", "artist": "Odumodublvck"},
        ],
        recent_news=[
            BraveResult(
                title="Odumodublvck wins at Headies 2026",
                url="https://example.com/headies",
                description="The Abuja-born rapper took home Rap Album of the Year.",
            )
        ],
    )


@pytest.mark.asyncio
async def test_artist_info_returns_200(client: AsyncClient) -> None:
    """``POST /artist/info`` returns 200 with a valid ``ArtistInfoResponse``."""
    with patch("backend.app.api.artist.ArtistService") as mock_cls:
        instance = MagicMock()
        instance.get_info = AsyncMock(return_value=_fake_artist_response())
        mock_cls.return_value = instance

        response = await client.post("/artist/info", json={"artist_name": "Odumodublvck"})

    assert response.status_code == 200
    data = response.json()
    assert data["artist_name"] == "Odumodublvck"
    assert "Headies" in data["response"]
    assert len(data["top_tracks"]) == 2
    assert len(data["recent_news"]) == 1


@pytest.mark.asyncio
async def test_artist_info_with_user_id(client: AsyncClient) -> None:
    """``user_id`` is passed through to ``ArtistService.get_info``."""
    user_id = "00000000-0000-0000-0000-000000000001"

    with patch("backend.app.api.artist.ArtistService") as mock_cls:
        instance = MagicMock()
        instance.get_info = AsyncMock(return_value=_fake_artist_response())
        mock_cls.return_value = instance

        response = await client.post(
            "/artist/info",
            json={"artist_name": "Odumodublvck", "user_id": user_id},
        )

        instance.get_info.assert_called_once_with(artist_name="Odumodublvck", user_id=user_id)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_artist_info_without_user_id(client: AsyncClient) -> None:
    """Request without ``user_id`` succeeds (anonymous mode)."""
    with patch("backend.app.api.artist.ArtistService") as mock_cls:
        instance = MagicMock()
        instance.get_info = AsyncMock(return_value=_fake_artist_response())
        mock_cls.return_value = instance

        response = await client.post("/artist/info", json={"artist_name": "Burna Boy"})

        instance.get_info.assert_called_once_with(artist_name="Burna Boy", user_id=None)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_artist_info_different_artists(client: AsyncClient) -> None:
    """Endpoint correctly passes the artist name to the service."""
    artists = ["Tems", "Wizkid", "Rema"]
    for name in artists:
        with patch("backend.app.api.artist.ArtistService") as mock_cls:
            instance = MagicMock()
            instance.get_info = AsyncMock(return_value=_fake_artist_response(name))
            mock_cls.return_value = instance

            response = await client.post("/artist/info", json={"artist_name": name})

        assert response.status_code == 200
        assert response.json()["artist_name"] == name
