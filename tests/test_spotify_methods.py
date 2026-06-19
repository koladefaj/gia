"""Additional ``SpotifyMCPClient`` method coverage tests.

Verifies that every MCP tool method forwards the correct tool name and
arguments to ``_call``, satisfying the remaining uncovered lines in
``backend/app/tools/spotify.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.app.config import Settings
from backend.app.tools.spotify import SpotifyMCPClient


@pytest.fixture()
def client_with_mock_http(test_settings: Settings) -> tuple[SpotifyMCPClient, list[dict]]:
    """Return a client whose HTTP layer records all ``POST /tools/call`` calls."""
    captured: list[dict] = []

    async def fake_post(url: str, *, json: dict, **_: object) -> MagicMock:  # noqa: A002
        captured.append(json)
        resp = MagicMock()
        resp.json.return_value = {"status": "ok"}
        resp.raise_for_status = MagicMock()
        return resp

    mock_http = AsyncMock()
    mock_http.post = fake_post
    mock_http.is_closed = False

    client = SpotifyMCPClient(cfg=test_settings)
    client._http = mock_http
    return client, captured


@pytest.mark.asyncio
async def test_get_currently_playing_tool_name(
    client_with_mock_http: tuple[SpotifyMCPClient, list[dict]],
) -> None:
    """``get_currently_playing`` sends tool name ``get_currently_playing``."""
    client, captured = client_with_mock_http
    await client.get_currently_playing()
    assert captured[0]["name"] == "get_currently_playing"


@pytest.mark.asyncio
async def test_get_top_artists_tool_name_and_args(
    client_with_mock_http: tuple[SpotifyMCPClient, list[dict]],
) -> None:
    """``get_top_artists`` forwards ``time_range`` and ``limit``."""
    client, captured = client_with_mock_http
    await client.get_top_artists(time_range="short_term", limit=5)
    assert captured[0]["name"] == "get_top_artists"
    assert captured[0]["arguments"]["time_range"] == "short_term"
    assert captured[0]["arguments"]["limit"] == 5


@pytest.mark.asyncio
async def test_search_tracks_forwards_query(
    client_with_mock_http: tuple[SpotifyMCPClient, list[dict]],
) -> None:
    """``search_tracks`` forwards the query string and limit."""
    client, captured = client_with_mock_http
    await client.search_tracks("afrobeats chill", limit=3)
    assert captured[0]["name"] == "search_tracks"
    assert captured[0]["arguments"]["query"] == "afrobeats chill"
    assert captured[0]["arguments"]["limit"] == 3


@pytest.mark.asyncio
async def test_save_track_forwards_uri(
    client_with_mock_http: tuple[SpotifyMCPClient, list[dict]],
) -> None:
    """``save_track`` forwards the track URI."""
    client, captured = client_with_mock_http
    await client.save_track("spotify:track:001")
    assert captured[0]["name"] == "save_track"
    assert captured[0]["arguments"]["uri"] == "spotify:track:001"


@pytest.mark.asyncio
async def test_add_to_queue_forwards_uri(
    client_with_mock_http: tuple[SpotifyMCPClient, list[dict]],
) -> None:
    """``add_to_queue`` forwards the track URI."""
    client, captured = client_with_mock_http
    await client.add_to_queue("spotify:track:002")
    assert captured[0]["name"] == "add_to_queue"
    assert captured[0]["arguments"]["uri"] == "spotify:track:002"


@pytest.mark.asyncio
async def test_create_playlist_forwards_name_and_description(
    client_with_mock_http: tuple[SpotifyMCPClient, list[dict]],
) -> None:
    """``create_playlist`` forwards both name and description."""
    client, captured = client_with_mock_http
    await client.create_playlist("Sunday Wind Down", "chill afrobeats")
    assert captured[0]["name"] == "create_playlist"
    assert captured[0]["arguments"]["name"] == "Sunday Wind Down"
    assert captured[0]["arguments"]["description"] == "chill afrobeats"


@pytest.mark.asyncio
async def test_add_tracks_to_playlist_forwards_playlist_id_and_uris(
    client_with_mock_http: tuple[SpotifyMCPClient, list[dict]],
) -> None:
    """``add_tracks_to_playlist`` forwards playlist ID and URI list."""
    client, captured = client_with_mock_http
    await client.add_tracks_to_playlist("pl-123", ["spotify:track:001", "spotify:track:002"])
    assert captured[0]["name"] == "add_tracks_to_playlist"
    assert captured[0]["arguments"]["playlist_id"] == "pl-123"
    assert len(captured[0]["arguments"]["uris"]) == 2


@pytest.mark.asyncio
async def test_get_artist_info_forwards_artist_id(
    client_with_mock_http: tuple[SpotifyMCPClient, list[dict]],
) -> None:
    """``get_artist_info`` forwards the artist ID."""
    client, captured = client_with_mock_http
    await client.get_artist_info("artist-abc")
    assert captured[0]["name"] == "get_artist_info"
    assert captured[0]["arguments"]["artist_id"] == "artist-abc"


@pytest.mark.asyncio
async def test_get_artist_top_tracks_forwards_artist_id(
    client_with_mock_http: tuple[SpotifyMCPClient, list[dict]],
) -> None:
    """``get_artist_top_tracks`` forwards the artist ID."""
    client, captured = client_with_mock_http
    await client.get_artist_top_tracks("artist-xyz")
    assert captured[0]["name"] == "get_artist_top_tracks"
    assert captured[0]["arguments"]["artist_id"] == "artist-xyz"
