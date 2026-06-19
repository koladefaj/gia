"""DJ API — track recommendation and crossfade queue endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from weaviate import WeaviateClient

from backend.app.agents.dj import DJService
from backend.app.config import Settings
from backend.app.dependencies import (
    get_db,
    get_redis,
    get_settings,
    get_spotify_client,
    get_weaviate_client,
)
from backend.app.interfaces import SpotifyClientProtocol
from backend.app.memory.retrieval import build_user_context
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.observability.logging import get_logger
from backend.app.schemas.dj import DJRequest, DJResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/dj", tags=["dj"])


@router.post("/recommend", response_model=DJResponse)
async def recommend(
    body: DJRequest,
    spotify: SpotifyClientProtocol = Depends(get_spotify_client),
    weaviate: WeaviateClient = Depends(get_weaviate_client),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
    cfg: Settings = Depends(get_settings),
) -> DJResponse:
    """Find tracks matching the query, build a Camelot crossfade queue, and recommend.

    If ``user_id`` is supplied the user's preferences, mood patterns, and
    current Spotify state are fetched and injected into the LLM prompt so the
    recommendation is grounded in what Gia knows about this specific person.

    Args:
        body: DJ request with query, optional user_id, playback flag, and queue depth.

    Returns:
        ``DJResponse`` with Gia's recommendation, seed track, and ordered queue.
    """
    user_context_text = ""
    if body.user_id:
        try:
            store = WeaviateMemoryStore(client=weaviate)
            ctx = await build_user_context(
                body.user_id,
                body.query,
                db=db,
                store=store,
                redis=redis,
                spotify=spotify,
                cfg=cfg,
            )
            user_context_text = ctx.to_prompt_text()
        except Exception as exc:  # noqa: BLE001
            logger.warning("dj_context_error", error=str(exc))

    service = DJService(spotify=spotify, cfg=cfg)
    result = await service.recommend(
        query=body.query,
        user_context_text=user_context_text,
        start_playback=body.start_playback,
        n=body.n,
    )
    return result
