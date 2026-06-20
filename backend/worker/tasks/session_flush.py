"""Celery task — final memory extraction for sessions that went idle.

Runs on a Celery Beat schedule every 45 minutes.  Finds sessions that:
  1. Have been idle for at least 15 minutes (no new chat turn recorded).
  2. Have no active ``extract_throttle`` key (not mid-extraction window).
  3. Still have conversation history in Redis to extract from.

This catches the common case where a user stops chatting before the 45-minute
per-session throttle fires — without this task those tail turns would never be
distilled into Weaviate memories.

Members in ``gia:pending_flush`` are stored as ``{user_id}:{session_id}`` with
the score set to the Unix timestamp of the last chat turn.  The flush task uses
the score to determine idleness.
"""

from __future__ import annotations

import asyncio
import time

from backend.worker.celery_app import celery_app

_PENDING_SET = "gia:pending_flush"
_IDLE_THRESHOLD = 900  # 15 minutes — session considered idle after this


@celery_app.task(name="backend.worker.tasks.session_flush.flush_idle_sessions")
def flush_idle_sessions() -> dict:
    """Run final memory extraction for sessions idle for > 15 minutes."""
    return asyncio.run(_flush_async())


async def _flush_async() -> dict:
    import redis.asyncio as aioredis

    from backend.app.agents.memory import MemoryService
    from backend.app.config import settings
    from backend.app.db.weaviate_init import get_weaviate_client
    from backend.app.memory.session_history import format_history, get_history
    from backend.app.memory.store import WeaviateMemoryStore
    from backend.app.observability.logging import get_logger

    logger = get_logger(__name__)
    redis_client = aioredis.from_url(
        settings.redis_url, encoding="utf-8", decode_responses=True
    )
    weaviate_client = await asyncio.to_thread(get_weaviate_client)

    processed = skipped = errors = 0

    try:
        cutoff = time.time() - _IDLE_THRESHOLD
        members: list[str] = await redis_client.zrangebyscore(_PENDING_SET, 0, cutoff)

        if not members:
            logger.info("session_flush_nothing_idle")
            return {"status": "ok", "processed": 0, "skipped": 0, "errors": 0}

        store = WeaviateMemoryStore(client=weaviate_client)
        service = MemoryService(store=store, redis=redis_client, cfg=settings)

        for member in members:
            try:
                user_id, session_id = member.split(":", 1)
            except ValueError:
                await redis_client.zrem(_PENDING_SET, member)
                continue

            # Skip if the per-session throttle is still active.
            if await redis_client.exists(f"extract_throttle:{session_id}"):
                skipped += 1
                continue

            # Remove stale entries whose history has already expired.
            transcript = format_history(await get_history(redis_client, session_id))
            if not transcript:
                await redis_client.zrem(_PENDING_SET, member)
                skipped += 1
                continue

            try:
                memory_ids = await service.run_extraction(
                    user_id=user_id,
                    transcript=transcript,
                )
                logger.info(
                    "session_flush_extracted",
                    user_id=user_id,
                    session_id=session_id,
                    stored=len(memory_ids),
                )
                processed += 1
                await redis_client.zrem(_PENDING_SET, member)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "session_flush_error",
                    session_id=session_id,
                    error=str(exc),
                )
                errors += 1

    finally:
        await redis_client.aclose()
        await asyncio.to_thread(weaviate_client.close)

    logger.info(
        "session_flush_complete",
        processed=processed,
        skipped=skipped,
        errors=errors,
    )
    return {"status": "ok", "processed": processed, "skipped": skipped, "errors": errors}
