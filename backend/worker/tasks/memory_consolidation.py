"""Celery task — memory consolidation (the reflection loop).

Synthesises a user's raw memories into higher-order ``insight`` memories. Runs
two ways: enqueued right after a memory-extraction pass (so insights refresh as
soon as new facts land), and on a slow beat as a backstop for quiet users.

This is reflection, not a hot path — it never touches a conversation turn.
"""

from __future__ import annotations

import asyncio
import uuid

import redis as sync_redis
import weaviate as weaviate_lib

from backend.app.config import settings as cfg
from backend.app.memory.consolidation import consolidate_memories
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.observability.logging import get_logger, setup_logging
from backend.worker.celery_app import celery_app

setup_logging(cfg.log_level)
logger = get_logger(__name__)


async def _consolidate_async(user_id: str) -> dict:
    """Open a Weaviate client, consolidate one user, then close it."""
    wv_client = await asyncio.to_thread(
        weaviate_lib.connect_to_local,
        host=cfg.weaviate_url.replace("http://", "").split(":")[0],
        port=int(cfg.weaviate_url.rsplit(":", 1)[-1]) if ":" in cfg.weaviate_url else 8080,
    )
    store = WeaviateMemoryStore(client=wv_client)
    try:
        insights = await consolidate_memories(user_id, store, cfg)
        return {"status": "ok", "user_id": user_id, "insights": len(insights)}
    finally:
        await asyncio.to_thread(wv_client.close)


@celery_app.task(name="backend.worker.tasks.memory_consolidation.run_consolidation")
def run_consolidation(user_id: str) -> dict:
    """Celery task — synthesise one user's raw memories into insights."""
    return asyncio.run(_consolidate_async(user_id))


async def _consolidate_all_async() -> dict:
    """Dispatch consolidation for every active user (``session:*`` keys)."""
    r = sync_redis.from_url(cfg.redis_url, decode_responses=True)
    try:
        user_ids = [k.replace("session:", "") for k in r.keys("session:*")]
        for uid in user_ids:
            try:
                uuid.UUID(uid)
                run_consolidation.delay(uid)
            except ValueError:
                pass
        return {"status": "ok", "users_dispatched": len(user_ids)}
    finally:
        r.close()


@celery_app.task(name="backend.worker.tasks.memory_consolidation.run_consolidation_all")
def run_consolidation_all() -> dict:
    """Beat task — consolidate all active users (backstop for the inline trigger)."""
    return asyncio.run(_consolidate_all_async())
