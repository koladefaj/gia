"""Mood inference — audio feature time-series analysis.

Reads the ``ListeningEvent`` history from Postgres, groups events by
``(weekday, time-of-day)`` bucket, and detects consistent mood patterns.
A pattern is "consistent" when the standard deviation of energy across
events in that bucket is below ``CONSISTENCY_THRESHOLD`` and there are
at least ``MIN_SAMPLE_SIZE`` events.

Detected patterns are written to Weaviate as ``MemoryEntry`` objects with
``type="mood_pattern"``, making them available to ``build_user_context``
for proactive mood observation.

This module contains only async functions; the Celery task wrapper in
``backend.worker.tasks.mood_inference`` calls ``asyncio.run()`` to bridge
the sync/async boundary.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import ListeningEvent
from backend.app.memory.embeddings import embed
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.mood.classifier import classify_mood, time_bucket
from backend.app.observability.logging import get_logger
from backend.app.schemas.memory import MemoryEntry

logger = get_logger(__name__)

CONSISTENCY_THRESHOLD = 0.15
MIN_SAMPLE_SIZE = 5
MAX_EVENTS = 300


async def get_listening_events(user_id: str, db: AsyncSession, limit: int = MAX_EVENTS) -> list[ListeningEvent]:
    """Fetch the most recent *limit* ``ListeningEvent`` rows for *user_id*.

    Args:
        user_id: UUID string of the user.
        db:      Async SQLAlchemy session.
        limit:   Maximum events to return (default 300).

    Returns:
        List of ``ListeningEvent`` ORM instances, newest first.
    """
    uid = uuid.UUID(user_id)
    result = await db.execute(
        select(ListeningEvent)
        .where(ListeningEvent.user_id == uid)
        .where(ListeningEvent.energy.is_not(None))
        .order_by(ListeningEvent.played_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


def _stddev(values: list[float]) -> float:
    """Return the population standard deviation of *values*."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def group_by_time_bucket(events: list[ListeningEvent]) -> dict[str, list[ListeningEvent]]:
    """Partition events into named (weekday × time-of-day) buckets.

    Args:
        events: ``ListeningEvent`` rows with non-null ``played_at``.

    Returns:
        Dict mapping bucket name → list of events in that bucket.
    """
    buckets: dict[str, list[ListeningEvent]] = {}
    for evt in events:
        dt: datetime = evt.played_at
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        bucket = time_bucket(dt.hour, dt.weekday())
        buckets.setdefault(bucket, []).append(evt)
    return buckets


async def infer_mood_patterns(
    user_id: str,
    db: AsyncSession,
    store: WeaviateMemoryStore,
) -> list[str]:
    """Detect mood patterns in the user's listening history and persist them.

    For each time bucket with enough events and sufficient consistency, a
    ``mood_pattern`` memory is upserted into Weaviate.  Existing pattern
    for the same bucket is superseded (deleted then re-inserted) to keep
    only the latest inference.

    Args:
        user_id: UUID string of the user.
        db:      Async SQLAlchemy session for reading ``ListeningEvent``.
        store:   ``WeaviateMemoryStore`` for writing patterns.

    Returns:
        List of bucket names where a pattern was detected and stored.
    """
    events = await get_listening_events(user_id, db)
    if not events:
        logger.info("mood_inference_no_events", user_id=user_id)
        return []

    buckets = group_by_time_bucket(events)
    stored: list[str] = []

    for bucket_name, bucket_events in buckets.items():
        energies = [e.energy for e in bucket_events if e.energy is not None]
        valences = [e.valence for e in bucket_events if e.valence is not None]
        tempos = [e.tempo for e in bucket_events if e.tempo is not None]

        if len(energies) < MIN_SAMPLE_SIZE:
            continue

        consistency = _stddev(energies)
        if consistency >= CONSISTENCY_THRESHOLD:
            continue

        avg_energy = sum(energies) / len(energies)
        avg_valence = sum(valences) / len(valences) if valences else 0.5
        avg_tempo = sum(tempos) / len(tempos) if tempos else 100.0
        label = classify_mood(avg_energy, avg_valence)

        pattern_text = (
            f"Mood pattern for {bucket_name}: {label}. "
            f"avg energy={avg_energy:.2f}, avg valence={avg_valence:.2f}, "
            f"avg tempo={avg_tempo:.0f} BPM. "
            f"Based on {len(energies)} sessions (consistency σ={consistency:.3f})."
        )

        # Supersede existing pattern for this bucket
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
            confidence=min(1.0, len(energies) / 20),
            created_at=datetime.now(timezone.utc),
        )
        vector = await embed(pattern_text)
        await store.upsert_memory(user_id, entry, vector)
        stored.append(bucket_name)
        logger.info(
            "mood_pattern_stored",
            user_id=user_id,
            bucket=bucket_name,
            label=label,
            samples=len(energies),
        )

    return stored
