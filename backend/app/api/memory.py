"""Memory API — context retrieval and extraction endpoints.

Used for debugging and the Day 3 done-criterion demo:

  ``GET  /memory/{user_id}/context``   — assemble and return ``UserContext``
  ``POST /memory/{user_id}/extract``   — run extraction on a transcript

Both endpoints require an open Weaviate client, Redis, Postgres session, and
Spotify client, all injected via FastAPI's ``Depends`` mechanism.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from redis.asyncio import Redis as AsyncRedis
from sqlalchemy.ext.asyncio import AsyncSession
from weaviate import WeaviateClient

from backend.app.agents.memory import MemoryService
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
from backend.app.schemas.memory import UserContext

logger = get_logger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


class ExtractionRequest:
    """Request body for the extract endpoint."""

    def __init__(self, transcript: str) -> None:
        self.transcript = transcript


from pydantic import BaseModel  # noqa: E402


class ExtractionRequestBody(BaseModel):
    """Body for ``POST /memory/{user_id}/extract``."""

    transcript: str


class ExtractionResponse(BaseModel):
    """Response from ``POST /memory/{user_id}/extract``."""

    user_id: str
    stored: int
    memory_ids: list[str]


@router.get("/{user_id}/context", summary="Get user context", status_code=200, response_model=UserContext)
async def get_user_context(
    db: Annotated[AsyncSession, Depends(get_db)],
    weaviate: Annotated[WeaviateClient, Depends(get_weaviate_client)],
    redis: Annotated[AsyncRedis, Depends(get_redis)],
    spotify: Annotated[SpotifyClientProtocol, Depends(get_spotify_client)],
    cfg: Annotated[Settings, Depends(get_settings)],
    user_id: str,
    query: str = "music preferences",
) -> UserContext:
    """Assemble and return the full ``UserContext`` for *user_id*.

    This is the same context that gets injected into every agent turn.
    The ``query`` parameter steers the Weaviate semantic search — pass the
    user's current utterance in production.

    Args:
        user_id: UUID string of the user.
        query:   Semantic search query for Weaviate (defaults to a generic
                 music preferences query for debugging).

    Returns:
        ``UserContext`` with profile, preferences, mood patterns, episodes,
        session notes, and current Spotify state.

    Raises:
        HTTPException 400: If *user_id* is not a valid UUID.
    """
    try:
        import uuid as _uuid

        _uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{user_id!r} is not a valid UUID.") from None

    store = WeaviateMemoryStore(client=weaviate)
    context = await build_user_context(
        user_id,
        query,
        db=db,
        store=store,
        redis=redis,
        spotify=spotify,
        cfg=cfg,
    )
    logger.info("context_assembled", user_id=user_id, query=query)
    return context


@router.post("/{user_id}/extract", summary="Extract memories from conversation", status_code=200, response_model=ExtractionResponse)
async def extract_memories_endpoint(
    user_id: str,
    body: ExtractionRequestBody,
    weaviate: Annotated[WeaviateClient, Depends(get_weaviate_client)],
    redis: Annotated[AsyncRedis, Depends(get_redis)],
    cfg: Annotated[Settings, Depends(get_settings)],
) -> ExtractionResponse:
    """Run the LLM memory extractor on *body.transcript* and persist results.

    This is the Day 3 done-criterion endpoint: feed a scripted exchange here,
    then call ``GET /memory/{user_id}/context`` to confirm the preference was
    stored and retrieved.

    Args:
        user_id: UUID string of the user.
        body:    Request body containing the conversation transcript.

    Returns:
        ``ExtractionResponse`` with the count and IDs of stored memories.

    Raises:
        HTTPException 400: If *user_id* is not a valid UUID.
    """
    try:
        import uuid as _uuid

        _uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{user_id!r} is not a valid UUID.") from None

    store = WeaviateMemoryStore(client=weaviate)
    service = MemoryService(store=store, redis=redis, cfg=cfg)
    memory_ids = await service.run_extraction(
        user_id=user_id,
        transcript=body.transcript,
    )
    logger.info("extract_endpoint_done", user_id=user_id, stored=len(memory_ids))
    return ExtractionResponse(
        user_id=user_id,
        stored=len(memory_ids),
        memory_ids=memory_ids,
    )
