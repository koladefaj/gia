"""Artist API — personalised artist context endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from typing import Annotated
from weaviate import WeaviateClient

from backend.app.agents.artist import ArtistService
from backend.app.config import Settings
from backend.app.dependencies import (
    get_brave_client,
    get_settings,
    get_spotify_client,
    get_weaviate_client,
)
from backend.app.interfaces import SpotifyClientProtocol
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.observability.logging import get_logger
from backend.app.schemas.artist import ArtistInfoRequest, ArtistInfoResponse
from backend.app.tools.brave import BraveSearchClient

logger = get_logger(__name__)

router = APIRouter(prefix="/artist", tags=["artist"])


@router.post("/info", summary="Get personalised artist information", status_code=200, response_model=ArtistInfoResponse)
async def artist_info(
    body: ArtistInfoRequest,
    spotify: Annotated[SpotifyClientProtocol, Depends(get_spotify_client)],
    brave: Annotated[BraveSearchClient, Depends(get_brave_client)],
    weaviate: Annotated[WeaviateClient, Depends(get_weaviate_client)],
    cfg: Annotated[Settings, Depends(get_settings)],
) -> ArtistInfoResponse:
    """Return a warm, personalised take on *body.artist_name*.

    Combines:
      - Spotify top tracks for the artist.
      - Brave Search results for recent news / activity.
      - The user's Weaviate memory about this artist (when ``user_id`` is provided).

    Args:
        body: Request with ``artist_name`` and optional ``user_id``.

    Returns:
        ``ArtistInfoResponse`` with Gia's narrative, top tracks, and recent news.
    """
    store: WeaviateMemoryStore | None = None
    if body.user_id:
        store = WeaviateMemoryStore(client=weaviate)

    service = ArtistService(
        spotify=spotify,
        brave=brave,
        cfg=cfg,
        store=store,
    )
    return await service.get_info(
        artist_name=body.artist_name,
        user_id=body.user_id,
    )
