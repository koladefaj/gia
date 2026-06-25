"""Celery task — mood pattern inference from listening history.

Runs after each session and on a 30-minute beat timer during active listening.
Reads ``ListeningEvent`` history from Postgres, detects patterns per time
bucket, and writes ``MoodPattern`` memories to Weaviate.
"""

from __future__ import annotations

import asyncio
import uuid

import redis as sync_redis
import weaviate as weaviate_lib
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.config import settings as cfg
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.mood.inference import get_listening_events, infer_mood_patterns
from backend.app.mood.labeler import label_mood
from backend.app.mood.proactive import check_and_draft_proactive
from backend.app.observability.logging import get_logger, setup_logging
from backend.worker.celery_app import celery_app

setup_logging(cfg.log_level)
logger = get_logger(__name__)


async def _infer_async(user_id: str) -> dict:
    """Full async implementation of mood inference.

    Creates fresh DB + Weaviate + Redis clients (no long-lived connection).
    All resources are closed on return.

    Args:
        user_id: UUID string of the user to analyse.

    Returns:
        Dict with ``status``, ``user_id``, and ``patterns_stored`` count.
    """
    engine = create_async_engine(cfg.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    wv_client = await asyncio.to_thread(
        weaviate_lib.connect_to_local,
        host=cfg.weaviate_url.replace("http://", "").split(":")[0],
        port=int(cfg.weaviate_url.rsplit(":", 1)[-1]) if ":" in cfg.weaviate_url else 8080,
    )
    store = WeaviateMemoryStore(client=wv_client)

    try:
        async with session_factory() as db:
            stored = await infer_mood_patterns(user_id, db, store, cfg)
            # The API ingests recently-played into listening_events, so the most
            # recent rows are "now" — label them to check for a mood shift.
            recent = await get_listening_events(user_id, db, limit=8)

        if recent:
            tracks = [{"name": e.track_name, "artist": e.artist_name} for e in recent]
            current_label = await label_mood(tracks, cfg)

            import redis.asyncio as aioredis
            ar = aioredis.from_url(cfg.redis_url, decode_responses=True)
            try:
                await check_and_draft_proactive(user_id, current_label, store, ar)
            finally:
                await ar.aclose()

        return {"status": "ok", "user_id": user_id, "patterns_stored": len(stored)}
    finally:
        await asyncio.to_thread(wv_client.close)
        await engine.dispose()


@celery_app.task(name="backend.worker.tasks.mood_inference.run_mood_inference")
def run_mood_inference(user_id: str) -> dict:
    """Celery task — analyse one user's listening history for mood patterns.

    Args:
        user_id: UUID string of the user.

    Returns:
        Dict with execution summary.
    """
    return asyncio.run(_infer_async(user_id))


async def _infer_all_async() -> dict:
    """Fetch all active user IDs from Redis and dispatch per-user tasks.

    Active users are tracked under the key pattern ``session:*``.

    Returns:
        Dict with ``status`` and ``users_dispatched`` count.
    """
    r = sync_redis.from_url(cfg.redis_url, decode_responses=True)
    try:
        keys = r.keys("session:*")
        user_ids = [k.replace("session:", "") for k in keys]
        for uid in user_ids:
            try:
                uuid.UUID(uid)
                run_mood_inference.delay(uid)
            except ValueError:
                pass
        return {"status": "ok", "users_dispatched": len(user_ids)}
    finally:
        r.close()


@celery_app.task(name="backend.worker.tasks.mood_inference.run_mood_inference_all")
def run_mood_inference_all() -> dict:
    """Beat task — dispatch mood inference for all active users.

    Scheduled by Celery Beat every 30 minutes during active listening hours.

    Returns:
        Dict with dispatch summary.
    """
    return asyncio.run(_infer_all_async())
