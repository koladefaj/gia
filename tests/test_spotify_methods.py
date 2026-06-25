"""``SpotifyMCPClient`` method-mapping coverage (MCP stdio).

Verifies each Protocol method maps to the correct MCP tool name + arguments via
the bridge, and that the unsupported operations (save_track, artist info/tracks)
degrade gracefully without an MCP call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.app.config import Settings
from backend.app.tools.spotify import SpotifyMCPClient


@pytest.fixture()
def client(test_settings: Settings) -> SpotifyMCPClient:
    """Client whose MCP bridge records ``(tool, arguments)`` calls."""
    c = SpotifyMCPClient(cfg=test_settings)
    c._bridge = MagicMock()
    c._bridge.call = AsyncMock(return_value="")
    return c


def _last_call(client: SpotifyMCPClient) -> tuple[str, dict]:
    args = client._bridge.call.call_args
    return args[0][0], args[0][1]


@pytest.mark.asyncio
async def test_get_currently_playing_tool_name(client: SpotifyMCPClient) -> None:
    await client.get_currently_playing()
    tool, _ = _last_call(client)
    assert tool == "getNowPlaying"


@pytest.mark.asyncio
async def test_get_top_artists_tool_name_and_args(client: SpotifyMCPClient) -> None:
    await client.get_top_artists(time_range="short_term", limit=5)
    tool, args = _last_call(client)
    assert tool == "getTopArtists"
    assert args == {"timeRange": "short_term", "limit": 5}


@pytest.mark.asyncio
async def test_search_tracks_forwards_query(client: SpotifyMCPClient) -> None:
    await client.search_tracks("afrobeats chill", limit=3)
    tool, args = _last_call(client)
    assert tool == "searchSpotify"
    assert args == {"query": "afrobeats chill", "type": "track", "limit": 3}


@pytest.mark.asyncio
async def test_add_to_queue_forwards_uri(client: SpotifyMCPClient) -> None:
    await client.add_to_queue("spotify:track:002")
    tool, args = _last_call(client)
    assert tool == "addToQueue"
    assert args == {"uri": "spotify:track:002"}


@pytest.mark.asyncio
async def test_create_playlist_forwards_name_and_description(client: SpotifyMCPClient) -> None:
    await client.create_playlist("Sunday Wind Down", "chill afrobeats")
    tool, args = _last_call(client)
    assert tool == "createPlaylist"
    assert args == {"name": "Sunday Wind Down", "description": "chill afrobeats"}


@pytest.mark.asyncio
async def test_add_tracks_to_playlist_forwards_playlist_id_and_uris(client: SpotifyMCPClient) -> None:
    await client.add_tracks_to_playlist("pl-123", ["spotify:track:001", "spotify:track:002"])
    tool, args = _last_call(client)
    assert tool == "addTracksToPlaylist"
    assert args["playlistId"] == "pl-123"
    assert len(args["trackUris"]) == 2


@pytest.mark.asyncio
async def test_save_track_unsupported_no_call(client: SpotifyMCPClient) -> None:
    """No save-single-track tool exists; returns a status without an MCP call."""
    res = await client.save_track("spotify:track:001")
    assert res["status"] == "unsupported"
    client._bridge.call.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_artist_info_placeholder_no_call(client: SpotifyMCPClient) -> None:
    res = await client.get_artist_info("artist-abc")
    assert res["id"] == "artist-abc"
    client._bridge.call.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_artist_top_tracks_empty_no_call(client: SpotifyMCPClient) -> None:
    res = await client.get_artist_top_tracks("artist-xyz")
    assert res == []
    client._bridge.call.assert_not_awaited()
