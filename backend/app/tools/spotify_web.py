"""Direct Spotify Web API client for endpoints the MCP server can't do.

The ``marcelmarais/spotify-mcp-server`` SDK still calls Spotify's **deprecated**
``/v1/users/{id}/playlists`` create endpoint (→ 403 after the Feb-2026 changes).
The current endpoint, ``POST /v1/me/playlists``, works — but the server doesn't
use it.  Rather than fork the server's SDK, this thin client calls the current
endpoints directly for the gaps: **playlist creation / track add** and the
**user profile** (display name).

Auth: it reuses the OAuth tokens the MCP server already obtained (its
``spotify-config.json``), refreshing them itself when expired.  In a multi-user
frontend this would instead use each user's ``Profile`` token; for the
single-account demo, the shared config is the pragmatic source of truth.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from backend.app.config import Settings
from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

_API = "https://api.spotify.com/v1"
_TOKEN_URL = "https://accounts.spotify.com/api/token"


def _config_path(cfg: Settings) -> Path:
    """Locate the MCP server's ``spotify-config.json`` (token store)."""
    if cfg.spotify_config_path:
        return Path(cfg.spotify_config_path)
    # .../spotify-mcp-server/build/index.js → .../spotify-mcp-server/spotify-config.json
    return Path(cfg.spotify_mcp_server_path).parent.parent / "spotify-config.json"


class SpotifyWebClient:
    """Minimal current-endpoint Spotify Web API client (playlists + profile)."""

    def __init__(self, cfg: Settings) -> None:
        self._cfg = cfg

    async def _access_token(self) -> str:
        """Return a valid access token, refreshing via the saved refresh token."""
        path = _config_path(self._cfg)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("accessToken") and data.get("expiresAt", 0) / 1000 > time.time() + 60:
            return data["accessToken"]

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                _TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": data["refreshToken"],
                    "client_id": self._cfg.spotify_client_id,
                    "client_secret": self._cfg.spotify_client_secret,
                },
            )
            resp.raise_for_status()
            tok = resp.json()

        data["accessToken"] = tok["access_token"]
        data["expiresAt"] = int((time.time() + tok.get("expires_in", 3600)) * 1000)
        if tok.get("refresh_token"):
            data["refreshToken"] = tok["refresh_token"]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("spotify_web_token_refreshed")
        return data["accessToken"]

    async def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {await self._access_token()}",
            "Content-Type": "application/json",
        }

    async def get_me(self) -> dict:
        """Return the current user's profile (``display_name``, ``id``, ``product``)."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{_API}/me", headers=await self._headers())
            resp.raise_for_status()
            return resp.json()

    async def create_playlist(
        self, name: str, description: str = "", track_uris: list[str] | None = None
    ) -> dict:
        """Create a playlist (current endpoint) and optionally add *track_uris*.

        Args:
            name:        Playlist name.
            description: Playlist description.
            track_uris:  Optional ``spotify:track:…`` URIs to add (max 100).

        Returns:
            ``{"id", "name", "url", "added"}``.
        """
        headers = await self._headers()
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{_API}/me/playlists",
                headers=headers,
                json={"name": name, "description": description, "public": False},
            )
            resp.raise_for_status()
            playlist = resp.json()
            playlist_id = playlist["id"]

            added = 0
            if track_uris:
                added = await self._add_tracks(client, headers, playlist_id, track_uris)

        logger.info("spotify_web_playlist_created", playlist_id=playlist_id, added=added)
        return {
            "id": playlist_id,
            "name": name,
            "url": playlist.get("external_urls", {}).get("spotify", ""),
            "added": added,
        }

    async def _add_tracks(
        self, client: httpx.AsyncClient, headers: dict, playlist_id: str, uris: list[str]
    ) -> int:
        """Add *uris* using the current ``/items`` endpoint, falling back to ``/tracks``."""
        for endpoint in ("items", "tracks"):  # /items is current; /tracks is the old name
            resp = await client.post(
                f"{_API}/playlists/{playlist_id}/{endpoint}",
                headers=headers,
                json={"uris": uris[:100]},
            )
            if resp.status_code < 400:
                return len(uris[:100])
            logger.warning(
                "spotify_web_add_tracks_failed",
                endpoint=endpoint, status=resp.status_code,
            )
        return 0
