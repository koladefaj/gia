"""Tests for ``SpotifyMCPClient`` and ``FakeSpotifyClient``.

Unit tests validate the real client's tool-mapping and text-parsing logic with
the MCP bridge replaced by a mock — no live MCP server is spawned.

The ``FakeSpotifyClient`` is also tested to ensure it stays aligned with the
``SpotifyClientProtocol`` interface — so tests that use the fake remain
meaningful as the interface evolves.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.app.config import Settings
from backend.app.interfaces import SpotifyClientProtocol
from backend.app.tools.spotify import SpotifyMCPClient
from tests.conftest import FakeSpotifyClient

# ── FakeSpotifyClient contract tests ─────────────────────────────────────────


def test_fake_spotify_satisfies_protocol() -> None:
    """``FakeSpotifyClient`` must structurally satisfy ``SpotifyClientProtocol``."""
    assert isinstance(FakeSpotifyClient(), SpotifyClientProtocol)


@pytest.mark.asyncio
async def test_fake_get_currently_playing_returns_track() -> None:
    """Fake client returns a track dict with ``is_playing: True``."""
    client = FakeSpotifyClient()
    result = await client.get_currently_playing()
    assert result is not None
    assert result["is_playing"] is True
    assert "uri" in result
    assert "energy" in result


@pytest.mark.asyncio
async def test_fake_get_recently_played_respects_limit() -> None:
    """``get_recently_played`` returns at most ``limit`` tracks."""
    client = FakeSpotifyClient()
    result = await client.get_recently_played(limit=1)
    assert len(result) == 1


@pytest.mark.asyncio
async def test_fake_start_playback_records_uri() -> None:
    """``start_playback`` records the URI and returns a confirmation dict."""
    client = FakeSpotifyClient()
    result = await client.start_playback("spotify:track:001")
    assert result["status"] == "playing"
    assert "spotify:track:001" in client.playback_started


@pytest.mark.asyncio
async def test_fake_save_track_records_uri() -> None:
    """``save_track`` records the URI in ``client.saved_tracks``."""
    client = FakeSpotifyClient()
    result = await client.save_track("spotify:track:002")
    assert result["status"] == "saved"
    assert "spotify:track:002" in client.saved_tracks


@pytest.mark.asyncio
async def test_fake_create_playlist_records_and_returns_id() -> None:
    """``create_playlist`` returns playlist metadata and records the creation."""
    client = FakeSpotifyClient()
    result = await client.create_playlist("Sunday Wind Down", "chill afrobeats")
    assert result["name"] == "Sunday Wind Down"
    assert len(client.created_playlists) == 1


@pytest.mark.asyncio
async def test_fake_add_to_queue_records_uri() -> None:
    """``add_to_queue`` records the URI in ``client.queued_tracks``."""
    client = FakeSpotifyClient()
    await client.add_to_queue("spotify:track:003")
    assert "spotify:track:003" in client.queued_tracks


@pytest.mark.asyncio
async def test_fake_get_audio_features_maps_uris() -> None:
    """``get_audio_features`` returns features in the same order as the input URIs."""
    client = FakeSpotifyClient()
    uris = ["spotify:track:001", "spotify:track:002"]
    features = await client.get_audio_features(uris)
    assert len(features) == 2
    assert features[0]["uri"] == "spotify:track:001"
    assert features[1]["uri"] == "spotify:track:002"


@pytest.mark.asyncio
async def test_fake_get_audio_features_unknown_uri_returns_fallback() -> None:
    """Unknown URIs fall back to the first track rather than raising."""
    client = FakeSpotifyClient()
    features = await client.get_audio_features(["spotify:track:UNKNOWN"])
    assert len(features) == 1
    assert "uri" in features[0]


# ── SpotifyMCPClient (real, MCP stdio) unit tests ─────────────────────────────


@pytest.fixture()
def mcp_client(test_settings: Settings) -> SpotifyMCPClient:
    """Return a ``SpotifyMCPClient`` whose MCP bridge is replaced by an AsyncMock."""
    client = SpotifyMCPClient(cfg=test_settings)
    client._bridge = MagicMock()
    client._bridge.call = AsyncMock(return_value="")
    client._bridge.stop = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_search_tracks_maps_tool_and_parses(mcp_client: SpotifyMCPClient) -> None:
    """``search_tracks`` calls ``searchSpotify`` and parses the text into dicts."""
    mcp_client._bridge.call = AsyncMock(return_value=(
        '# Search results for "tems" (type: track)\n\n'
        '1. "Free Mind" by Tems (4:08) - ID: 2mzM4Y0Rnx2BDZqRnhQ5Q6\n'
    ))
    out = await mcp_client.search_tracks("tems", limit=5)
    name, args = mcp_client._bridge.call.call_args
    assert name[0] == "searchSpotify"
    assert name[1] == {"query": "tems", "type": "track", "limit": 5}
    assert out == [{
        "uri": "spotify:track:2mzM4Y0Rnx2BDZqRnhQ5Q6",
        "id": "2mzM4Y0Rnx2BDZqRnhQ5Q6", "name": "Free Mind", "artist": "Tems",
    }]


@pytest.mark.asyncio
async def test_get_recently_played_maps_tool(mcp_client: SpotifyMCPClient) -> None:
    await mcp_client.get_recently_played(limit=3)
    name, _ = mcp_client._bridge.call.call_args
    assert name[0] == "getRecentlyPlayed"
    assert name[1] == {"limit": 3}


@pytest.mark.asyncio
async def test_get_audio_features_returns_neutral_without_calling(mcp_client: SpotifyMCPClient) -> None:
    """No audio-features tool exists; return neutral placeholders, no MCP call."""
    uris = ["spotify:track:001", "spotify:track:002"]
    features = await mcp_client.get_audio_features(uris)
    assert [f["uri"] for f in features] == uris
    assert all("energy" in f and "key" in f for f in features)
    mcp_client._bridge.call.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_playback_maps_to_play_music_with_device(mcp_client: SpotifyMCPClient) -> None:
    res = await mcp_client.start_playback("spotify:track:001", device_id="dev-abc")
    name, _ = mcp_client._bridge.call.call_args
    assert name[0] == "playMusic"
    assert name[1] == {"uri": "spotify:track:001", "deviceId": "dev-abc"}
    assert res["status"] == "playing"


@pytest.mark.asyncio
async def test_close_stops_bridge(mcp_client: SpotifyMCPClient) -> None:
    await mcp_client.close()
    mcp_client._bridge.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_prewarm_noop_when_path_unset(test_settings: Settings) -> None:
    """With no server path configured, prewarm does not start the bridge."""
    cfg = test_settings.model_copy(update={"spotify_mcp_server_path": ""})
    client = SpotifyMCPClient(cfg=cfg)
    client._bridge.start = AsyncMock()
    await client.prewarm()
    client._bridge.start.assert_not_awaited()
