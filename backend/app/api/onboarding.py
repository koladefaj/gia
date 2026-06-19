"""Onboarding API — bootstrap a user's taste profile from Spotify.

``POST /memory/{user_id}/bootstrap`` pulls the user's real Spotify listening
(top artists/tracks + recent plays via the MCP server), distils it into durable
taste-preference memories, and stores them in Weaviate.  Call it once after a
user connects Spotify so cold-start conversations feel personalised immediately.
"""

from __future__ import annotations

import uuid as _uuid

from fastapi import APIRouter, Depends, HTTPException
from redis.asyncio import Redis as AsyncRedis
from weaviate import WeaviateClient

from backend.app.config import Settings
from backend.app.dependencies import (
    get_redis,
    get_settings,
    get_spotify_client,
    get_weaviate_client,
)
from backend.app.interfaces import SpotifyClientProtocol
from backend.app.memory.profiler import bootstrap_taste_profile
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/{user_id}/bootstrap")
async def bootstrap_profile(
    user_id: str,
    spotify: SpotifyClientProtocol = Depends(get_spotify_client),
    weaviate: WeaviateClient = Depends(get_weaviate_client),
    redis: AsyncRedis = Depends(get_redis),
    cfg: Settings = Depends(get_settings),
) -> dict:
    """Bootstrap taste-preference memories from the user's Spotify listening.

    Args:
        user_id: UUID string of the user (must already exist / be connected).

    Returns:
        ``{"stored": <count>, "memory_ids": [...]}``.

    Raises:
        HTTPException 400: If *user_id* is not a valid UUID.
    """
    try:
        _uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{user_id!r} is not a valid UUID.")

    store = WeaviateMemoryStore(client=weaviate)
    stored = await bootstrap_taste_profile(
        user_id, spotify=spotify, store=store, redis=redis, cfg=cfg
    )
    logger.info("bootstrap_profile_done", user_id=user_id, stored=len(stored))
    return {"stored": len(stored), "memory_ids": stored}
