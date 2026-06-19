"""Tests for ``SpotifyMCPClient`` and ``FakeSpotifyClient``.

Unit tests validate the real client's HTTP call logic using a mocked
``httpx.AsyncClient``.  Integration concerns (MCP server actually running)
are out of scope here; those live in ``tests/integration/``.

The ``FakeSpotifyClient`` is also tested to ensure it stays aligned with the
``SpotifyClientProtocol`` interface — so tests that use the fake remain
meaningful as the interface evolves.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

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


# ── SpotifyMCPClient (real) unit tests ────────────────────────────────────────


@pytest.fixture()
def spotify_cfg(test_settings: Settings) -> Settings:
    """Settings with a test MCP URL."""
    return test_settings


@pytest.fixture()
def mcp_client(spotify_cfg: Settings) -> SpotifyMCPClient:
    """Return a ``SpotifyMCPClient`` with a mocked HTTP layer."""
    return SpotifyMCPClient(cfg=spotify_cfg)


def _mock_http_response(data: object) -> AsyncMock:
    """Build a mock ``httpx.Response`` that returns *data* as JSON."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


@pytest.mark.asyncio
async def test_mcp_client_calls_correct_tool_name(mcp_client: SpotifyMCPClient) -> None:
    """``_call`` sends the correct ``name`` field to the MCP server."""
    captured: list[dict] = []

    async def fake_post(url: str, *, json: dict, **_: object) -> MagicMock:  # noqa: A002
        captured.append(json)
        resp = MagicMock()
        resp.json.return_value = [{"uri": "spotify:track:001", "name": "Free Mind"}]
        resp.raise_for_status = MagicMock()
        return resp

    mock_http = AsyncMock()
    mock_http.post = fake_post
    mock_http.is_closed = False
    mcp_client._http = mock_http

    await mcp_client.get_recently_played(limit=5)
    assert captured[0]["name"] == "get_recently_played"
    assert captured[0]["arguments"]["limit"] == 5


@pytest.mark.asyncio
async def test_mcp_client_get_audio_features_passes_uris(mcp_client: SpotifyMCPClient) -> None:
    """``get_audio_features`` forwards the URI list in ``arguments``."""
    captured: list[dict] = []

    async def fake_post(url: str, *, json: dict, **_: object) -> MagicMock:  # noqa: A002
        captured.append(json)
        resp = MagicMock()
        resp.json.return_value = [{"uri": "spotify:track:001", "energy": 0.38}]
        resp.raise_for_status = MagicMock()
        return resp

    mock_http = AsyncMock()
    mock_http.post = fake_post
    mock_http.is_closed = False
    mcp_client._http = mock_http

    uris = ["spotify:track:001"]
    await mcp_client.get_audio_features(uris)
    assert captured[0]["arguments"]["uris"] == uris


@pytest.mark.asyncio
async def test_mcp_client_close_closes_http(mcp_client: SpotifyMCPClient) -> None:
    """``close()`` closes the underlying HTTP client."""
    mock_http = AsyncMock()
    mock_http.is_closed = False
    mcp_client._http = mock_http

    await mcp_client.close()
    mock_http.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_client_close_is_idempotent(mcp_client: SpotifyMCPClient) -> None:
    """``close()`` is safe to call when the HTTP client is already closed."""
    mock_http = AsyncMock()
    mock_http.is_closed = True
    mcp_client._http = mock_http

    await mcp_client.close()  # should not raise
    mock_http.aclose.assert_not_called()


@pytest.mark.asyncio
async def test_mcp_client_start_playback_passes_device_id(mcp_client: SpotifyMCPClient) -> None:
    """``start_playback`` forwards an optional ``device_id`` to the MCP call."""
    captured: list[dict] = []

    async def fake_post(url: str, *, json: dict, **_: object) -> MagicMock:  # noqa: A002
        captured.append(json)
        resp = MagicMock()
        resp.json.return_value = {"status": "playing"}
        resp.raise_for_status = MagicMock()
        return resp

    mock_http = AsyncMock()
    mock_http.post = fake_post
    mock_http.is_closed = False
    mcp_client._http = mock_http

    await mcp_client.start_playback("spotify:track:001", device_id="device-abc")
    assert captured[0]["arguments"]["device_id"] == "device-abc"
