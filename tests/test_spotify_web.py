"""Tests for the direct Spotify Web API client (playlists)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.app.config import Settings
from backend.app.tools.spotify_web import SpotifyWebClient


class _Resp:
    def __init__(self, status: int, data: dict | None = None) -> None:
        self.status_code = status
        self._data = data or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)  # type: ignore[arg-type]

    def json(self) -> dict:
        return self._data


class _FakeClient:
    def __init__(self, responses: list[_Resp]) -> None:
        self._responses = list(responses)
        self.post_urls: list[str] = []

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def post(self, url: str, **_: object) -> _Resp:
        self.post_urls.append(url)
        return self._responses.pop(0)


def _cfg() -> Settings:
    return Settings(spotify_client_id="cid", spotify_client_secret="sec")


@pytest.mark.asyncio
async def test_create_playlist_uses_new_endpoint_and_adds_tracks() -> None:
    web = SpotifyWebClient(_cfg())
    fake = _FakeClient([
        _Resp(201, {"id": "pl1", "external_urls": {"spotify": "https://open.spotify.com/playlist/pl1"}}),
        _Resp(201, {}),  # add tracks via /items
    ])

    with patch.object(SpotifyWebClient, "_access_token", new=AsyncMock(return_value="tok")), \
         patch("backend.app.tools.spotify_web.httpx.AsyncClient", return_value=fake):
        result = await web.create_playlist("Sunday Wind Down", "chill", ["spotify:track:1", "spotify:track:2"])

    assert result == {
        "id": "pl1",
        "name": "Sunday Wind Down",
        "url": "https://open.spotify.com/playlist/pl1",
        "added": 2,
    }
    # New create endpoint + current /items add endpoint
    assert fake.post_urls[0].endswith("/me/playlists")
    assert fake.post_urls[1].endswith("/playlists/pl1/items")


@pytest.mark.asyncio
async def test_add_tracks_falls_back_to_tracks_endpoint() -> None:
    web = SpotifyWebClient(_cfg())
    fake = _FakeClient([
        _Resp(201, {"id": "pl2", "external_urls": {"spotify": "u"}}),
        _Resp(404, {}),  # /items rejected
        _Resp(201, {}),  # /tracks accepted
    ])

    with patch.object(SpotifyWebClient, "_access_token", new=AsyncMock(return_value="tok")), \
         patch("backend.app.tools.spotify_web.httpx.AsyncClient", return_value=fake):
        result = await web.create_playlist("X", track_uris=["spotify:track:1"])

    assert result["added"] == 1
    assert fake.post_urls[1].endswith("/items")
    assert fake.post_urls[2].endswith("/tracks")
