"""Mood pattern inference from listening history.

Reads the ``ListeningEvent`` history from Postgres, groups events by
``(weekday, time-of-day)`` bucket, and asks an LLM to label each busy bucket's
mood from the *track and artist names* played there (Spotify audio features are
gone — see :mod:`backend.app.mood.labeler`).

Detected patterns are written to Weaviate as ``MemoryEntry`` objects with
``type="mood_pattern"``, available to ``build_user_context`` and the proactive
engine.  The Celery task wrapper in ``backend.worker.tasks.mood_inference``
bridges the sync/async boundary.
"""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import Settings
from backend.app.db.models import ListeningEvent
from backend.app.memory.embeddings import embed
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.mood.classifier import time_bucket
from backend.app.mood.labeler import label_mood
from backend.app.observability.logging import get_logger
from backend.app.schemas.memory import MemoryEntry

logger = get_logger(__name__)

MIN_SAMPLE_SIZE = 5
MAX_EVENTS = 300


async def get_listening_events(
    user_id: str, db: AsyncSession, limit: int = MAX_EVENTS
) -> list[ListeningEvent]:
    """Fetch the most recent named ``ListeningEvent`` rows for *user_id*.

    Only rows with a track name are returned — the labeler needs something to
    read. Newest first.
    """
    uid = uuid.UUID(user_id)
    result = await db.execute(
        select(ListeningEvent)
        .where(ListeningEvent.user_id == uid)
        .where(ListeningEvent.track_name.is_not(None))
        .order_by(ListeningEvent.played_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


def group_by_time_bucket(events: list[ListeningEvent]) -> dict[str, list[ListeningEvent]]:
    """Partition events into named (weekday × time-of-day) buckets."""
    buckets: dict[str, list[ListeningEvent]] = {}
    for evt in events:
        dt: datetime = evt.played_at
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        buckets.setdefault(time_bucket(dt.hour, dt.weekday()), []).append(evt)
    return buckets


async def infer_mood_patterns(
    user_id: str,
    db: AsyncSession,
    store: WeaviateMemoryStore,
    cfg: Settings,
) -> list[str]:
    """Detect per-time-bucket mood patterns and persist them to Weaviate.

    For each bucket with at least ``MIN_SAMPLE_SIZE`` plays, the bucket's tracks
    are LLM-labeled into one mood. A confident (non-neutral) label supersedes any
    existing pattern for that bucket.

    Args:
        user_id: UUID string of the user.
        db:      Async session for reading ``ListeningEvent``.
        store:   ``WeaviateMemoryStore`` for writing patterns.
        cfg:     Settings (the labeler's fast-tier model).

    Returns:
        Bucket names where a pattern was detected and stored.
    """
    events = await get_listening_events(user_id, db)
    if not events:
        logger.info("mood_inference_no_events", user_id=user_id)
        return []

    buckets = group_by_time_bucket(events)
    stored: list[str] = []

    for bucket_name, bucket_events in buckets.items():
        if len(bucket_events) < MIN_SAMPLE_SIZE:
            continue

        tracks = [
            {"name": e.track_name, "artist": e.artist_name} for e in bucket_events
        ]
        label = await label_mood(tracks, cfg)
        if label == "neutral":  # no usable signal — don't store a non-pattern
            continue

        top_artists = [
            a for a, _ in Counter(
                e.artist_name for e in bucket_events if e.artist_name
            ).most_common(3)
        ]
        pattern_text = (
            f"Mood pattern for {bucket_name}: {label}. "
            f"Often plays {', '.join(top_artists) or 'a mix'}. "
            f"Based on {len(bucket_events)} plays."
        )

        # Supersede the previous pattern for this bucket.
        existing = await store.search(
            user_id,
            await embed(f"mood pattern {bucket_name}"),
            memory_type="mood_pattern",
            k=3,
        )
        for old in existing:
            if bucket_name in old.text:
                await store.delete_by_id(old.id)
                logger.debug("mood_pattern_superseded", bucket=bucket_name, id=old.id)

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            type="mood_pattern",
            text=pattern_text,
            confidence=min(1.0, len(bucket_events) / 20),
            created_at=datetime.now(UTC),
        )
        await store.upsert_memory(user_id, entry, await embed(pattern_text))
        stored.append(bucket_name)
        logger.info(
            "mood_pattern_stored",
            user_id=user_id,
            bucket=bucket_name,
            label=label,
            samples=len(bucket_events),
        )

    return stored
