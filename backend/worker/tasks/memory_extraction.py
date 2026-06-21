"""Celery task — extract and persist memories after a completed session.

Called at the end of each conversation session (or periodically during long
sessions).  The task is async-heavy so it uses ``asyncio.run`` to drive the
coroutine inside the synchronous Celery worker process.

The session transcript is retrieved from Redis (``session:{user_id}``), run
through the LLM extractor, and persisted to Weaviate via ``MemoryService``.
"""

from __future__ import annotations

import asyncio

from backend.worker.celery_app import celery_app


@celery_app.task(name="backend.worker.tasks.memory_extraction.extract_session_memories")
def extract_session_memories(user_id: str, session_id: str) -> dict:
    """Extract durable preferences from a completed session transcript.

    Retrieves the session transcript from Redis, runs the LLM extraction
    pass, and stores non-duplicate preferences in Weaviate.

    Args:
        user_id:    UUID string of the user whose session ended.
        session_id: Identifier of the completed session (used for logging).

    Returns:
        Dict with ``status``, ``stored`` count, and ``memory_ids`` list.
    """
    return asyncio.run(_extract_async(user_id, session_id))


async def _extract_async(user_id: str, session_id: str) -> dict:
    """Async body of the extraction task — runs inside ``asyncio.run``."""
    import redis.asyncio as aioredis  # noqa: PLC0415

    from backend.app.agents.memory import MemoryService  # noqa: PLC0415
    from backend.app.config import settings  # noqa: PLC0415
    from backend.app.db.weaviate_init import get_weaviate_client  # noqa: PLC0415
    from backend.app.memory.session_history import (  # noqa: PLC0415
        format_history,
        get_history,
    )
    from backend.app.memory.store import WeaviateMemoryStore  # noqa: PLC0415
    from backend.app.observability.logging import get_logger  # noqa: PLC0415

    logger = get_logger(__name__)
    logger.info("memory_task_start", user_id=user_id, session_id=session_id)

    redis_client = aioredis.from_url(
        settings.redis_url, encoding="utf-8", decode_responses=True
    )
    weaviate_client = await asyncio.to_thread(get_weaviate_client)

    try:
        # The real conversation lives in the per-session history ring written by
        # the chat endpoint (chat:hist:{session_id}).
        transcript = format_history(await get_history(redis_client, session_id))
        if not transcript:
            logger.info("memory_task_no_transcript", user_id=user_id)
            return {"status": "no_transcript", "stored": 0, "memory_ids": []}

        store = WeaviateMemoryStore(client=weaviate_client)
        service = MemoryService(store=store, redis=redis_client, cfg=settings)
        memory_ids = await service.run_extraction(
            user_id=user_id,
            transcript=transcript,
        )

        # New facts landed → re-synthesise this user's higher-order insights
        # (the reflection loop). Decoupled into its own task so extraction stays
        # fast and a consolidation failure never breaks extraction.
        if memory_ids:
            from backend.worker.celery_app import celery_app  # noqa: PLC0415
            celery_app.send_task(
                "backend.worker.tasks.memory_consolidation.run_consolidation",
                args=[user_id],
            )
    finally:
        await redis_client.aclose()
        await asyncio.to_thread(weaviate_client.close)

    logger.info("memory_task_done", user_id=user_id, stored=len(memory_ids))
    return {"status": "ok", "stored": len(memory_ids), "memory_ids": memory_ids}
