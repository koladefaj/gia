"""Proactive mood awareness — detect pattern shifts before the user says anything.

This is the feature that makes Aria feel aware: she notices you are listening
to something out of character for this time of day and reacts naturally.

The pattern deviation check runs at the start of every crew turn.  If the
current track's audio features deviate significantly from the user's known
pattern for the current time bucket, a draft message is generated and stored
in Redis under ``proactive:{user_id}``.  The Router picks it up and injects it
into the first reply.
"""

from __future__ import annotations

from datetime import UTC, datetime

from backend.app.memory.embeddings import embed
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.mood.classifier import coerce_label, time_bucket
from backend.app.observability.logging import get_logger
from backend.app.schemas.memory import MemoryEntry

logger = get_logger(__name__)

_PROACTIVE_REDIS_TTL = 300  # 5 minutes — draft expires if not used


async def get_pattern_for_now(
    user_id: str,
    store: WeaviateMemoryStore,
) -> MemoryEntry | None:
    """Fetch the mood pattern stored for the current time bucket.

    Args:
        user_id: UUID string of the user.
        store:   ``WeaviateMemoryStore`` with indexed mood patterns.

    Returns:
        The matching ``MemoryEntry`` (type=mood_pattern), or ``None``.
    """
    now = datetime.now(UTC)
    bucket = time_bucket(now.hour, now.weekday())
    query = f"mood pattern {bucket}"
    try:
        results = await store.search(user_id, await embed(query), memory_type="mood_pattern", k=5)
        for r in results:
            if bucket in r.text:
                return r
    except Exception as exc:  # noqa: BLE001
        logger.warning("proactive_pattern_fetch_error", error=str(exc))
    return None


def _parse_pattern(text: str) -> str:
    """Extract the mood label from a stored pattern text.

    Pattern text format::

        Mood pattern for sunday_evening: chill. Often plays Tems, Wizkid. ...

    Args:
        text: The pattern ``MemoryEntry.text``.

    Returns:
        A vocabulary mood label (``"neutral"`` if it can't be parsed).
    """
    if ": " not in text:
        return "neutral"
    after = text.split(": ", 1)[1]
    return coerce_label(after.split(".", 1)[0])


async def check_and_draft_proactive(
    user_id: str,
    current_label: str,
    store: WeaviateMemoryStore,
    redis,
) -> str | None:
    """Compare the current mood label to the bucket's pattern and draft a nudge.

    When the user's current listening reads as a different mood than they usually
    play in this time bucket, a short message is drafted in Gia's voice and cached
    in Redis so the next chat turn can surface it naturally.

    Args:
        user_id:       UUID string of the user.
        current_label: Mood label for what they're playing now (from the labeler).
        store:         Weaviate memory store.
        redis:         Async Redis client for caching the draft.

    Returns:
        The proactive draft string, or ``None`` if no shift is detected.
    """
    pattern = await get_pattern_for_now(user_id, store)
    if pattern is None:
        return None

    pattern_label = _parse_pattern(pattern.text)
    if current_label == "neutral" or current_label == pattern_label:
        return None

    now = datetime.now(UTC)
    bucket = time_bucket(now.hour, now.weekday())

    draft = (
        f"[thoughtful] Hey — you're usually on {pattern_label} stuff "
        f"around {bucket.replace('_', ' ')}. "
        f"This is a bit more {current_label}. "
        f"Everything okay, or just mixing it up?"
    )

    redis_key = f"proactive:{user_id}"
    try:
        await redis.setex(redis_key, _PROACTIVE_REDIS_TTL, draft)
        logger.info(
            "proactive_draft_stored",
            user_id=user_id,
            bucket=bucket,
            pattern_label=pattern_label,
            current_label=current_label,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("proactive_redis_error", error=str(exc))

    return draft


async def pop_proactive_draft(user_id: str, redis) -> str | None:
    """Retrieve and delete the pending proactive draft from Redis.

    Called at the start of each crew turn.  Returns the draft once and
    clears it so it is not repeated in the next turn.

    Args:
        user_id: UUID string of the user.
        redis:   Async Redis client.

    Returns:
        The draft string, or ``None`` if none is pending.
    """
    key = f"proactive:{user_id}"
    try:
        draft = await redis.get(key)
        if draft:
            await redis.delete(key)
            return draft
    except Exception as exc:  # noqa: BLE001
        logger.warning("proactive_pop_error", error=str(exc))
    return None
