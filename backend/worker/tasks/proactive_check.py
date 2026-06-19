"""Celery task — proactive mood pattern shift detection.

Checks whether the user is currently listening to something out of character
for the current time of day.  If a significant deviation is detected, a draft
message is cached in Redis for the next chat turn to surface naturally.
"""

from __future__ import annotations

import asyncio
import json

import redis as sync_redis
import weaviate as weaviate_lib

from backend.app.config import settings as cfg
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.mood.proactive import check_and_draft_proactive
from backend.app.observability.logging import get_logger, setup_logging
from backend.worker.celery_app import celery_app

setup_logging(cfg.log_level)
logger = get_logger(__name__)


async def _check_async(user_id: str) -> dict:
    """Check pattern deviation for one user and store draft if needed.

    Reads the user's current session energy/valence from Redis (set by the
    Spotify listening event logger).  Skips gracefully if the session key
    is absent (user is not currently active).

    Args:
        user_id: UUID string of the user.

    Returns:
        Dict with ``status``, ``user_id``, and whether a draft was produced.
    """
    r = sync_redis.from_url(cfg.redis_url, decode_responses=True)
    try:
        session_data = r.get(f"session:{user_id}")
        if not session_data:
            return {"status": "skipped", "reason": "no_active_session", "user_id": user_id}

        data = json.loads(session_data)
        energy = float(data.get("energy") or 0.5)
        valence = float(data.get("valence") or 0.5)
    finally:
        r.close()

    wv_client = await asyncio.to_thread(
        weaviate_lib.connect_to_local,
        host=cfg.weaviate_url.replace("http://", "").split(":")[0],
        port=int(cfg.weaviate_url.rsplit(":", 1)[-1]) if ":" in cfg.weaviate_url else 8080,
    )
    store = WeaviateMemoryStore(client=wv_client)

    import redis.asyncio as aioredis
    ar = aioredis.from_url(cfg.redis_url, decode_responses=True)

    try:
        draft = await check_and_draft_proactive(user_id, energy, valence, store, ar)
        return {
            "status": "ok",
            "user_id": user_id,
            "draft_produced": draft is not None,
        }
    finally:
        await asyncio.to_thread(wv_client.close)
        await ar.aclose()


@celery_app.task(name="backend.worker.tasks.proactive_check.check_pattern_shift")
def check_pattern_shift(user_id: str) -> dict:
    """Detect mood pattern deviations and draft proactive message.

    Args:
        user_id: UUID string of the user.

    Returns:
        Dict with execution summary.
    """
    return asyncio.run(_check_async(user_id))
