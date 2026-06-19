"""Playlist API — create real Spotify playlists via the current Web API.

``POST /playlist`` creates a playlist (and optionally fills it) using the
Feb-2026 ``/v1/me/playlists`` endpoint, which the MCP server can't reach (it
still calls the deprecated ``/v1/users/{id}/playlists`` → 403).  The DJ/frontend
calls this with a queue of track URIs to make the "save it as a playlist" beat
work again.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.dependencies import get_spotify_web_client
from backend.app.observability.logging import get_logger
from backend.app.tools.spotify_web import SpotifyWebClient

logger = get_logger(__name__)

router = APIRouter(prefix="/playlist", tags=["playlist"])


class CreatePlaylistRequest(BaseModel):
    """Body for ``POST /playlist``."""

    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=300)
    track_uris: list[str] = Field(default_factory=list)


@router.post("")
async def create_playlist(
    body: CreatePlaylistRequest,
    web: SpotifyWebClient = Depends(get_spotify_web_client),
) -> dict:
    """Create a Spotify playlist and optionally add *track_uris*.

    Returns:
        ``{"id", "name", "url", "added"}``.

    Raises:
        HTTPException 502: If the Spotify Web API call fails.
    """
    try:
        result = await web.create_playlist(body.name, body.description, body.track_uris)
    except Exception as exc:  # noqa: BLE001
        logger.warning("create_playlist_error", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Spotify playlist creation failed: {exc}")
    return result
