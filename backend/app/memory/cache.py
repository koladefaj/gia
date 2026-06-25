"""Redis-backed retrieval cache.

Vector search is not free — embedding + ANN + (optional) rerank can cost
hundreds of milliseconds, which is painful on a voice turn where the same user
often asks closely-related things in quick succession.  This cache memoises the
*result* of a retrieval (a list of ``MemoryEntry``) under a short TTL so repeat
or near-repeat turns skip the round-trip entirely.

Correctness over staleness: the cache is invalidated for a user whenever new
memories are written (``invalidate_user``), so a freshly-learned preference is
never hidden behind a stale cache entry.

Keys are namespaced ``retr:{user_id}:{type}:{query_hash}`` so a single user's
cache can be wiped with one pattern scan without touching other Redis data.
"""

from __future__ import annotations

import hashlib
import json

from redis.asyncio import Redis as AsyncRedis

from backend.app.observability.logging import get_logger
from backend.app.schemas.memory import MemoryEntry

logger = get_logger(__name__)

_KEY_PREFIX = "retr"


def cache_key(user_id: str, memory_type: str, query: str) -> str:
    """Build the Redis key for a retrieval result.

    Args:
        user_id:     UUID string of the user.
        memory_type: Memory class (``preference`` / ``mood_pattern`` / ``episode``).
        query:       The natural-language query the results were fetched for.

    Returns:
        A namespaced cache key, e.g. ``retr:abc:preference:1f3c...``.
    """
    query_hash = hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
    return f"{_KEY_PREFIX}:{user_id}:{memory_type}:{query_hash}"


async def get_cached(redis: AsyncRedis, key: str) -> list[MemoryEntry] | None:
    """Return cached ``MemoryEntry`` results for *key*, or ``None`` on a miss.

    A corrupt or unparseable cache value is treated as a miss (and not raised),
    so a bad entry degrades to a fresh fetch rather than breaking the turn.

    Args:
        redis: App-level async Redis client (``decode_responses=True``).
        key:   Key produced by :func:`cache_key`.

    Returns:
        The cached list, or ``None`` if absent/invalid.
    """
    try:
        raw = await redis.get(key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("retrieval_cache_get_error", error=str(exc))
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return [MemoryEntry.model_validate(item) for item in data]
    except Exception:  # noqa: BLE001
        logger.warning("retrieval_cache_decode_error", key=key)
        return None


async def set_cached(redis: AsyncRedis, key: str, entries: list[MemoryEntry], ttl: int) -> None:
    """Store *entries* under *key* with a *ttl* second expiry.

    A non-positive *ttl* disables caching (no-op), so the feature can be turned
    off entirely via ``settings.retrieval_cache_ttl=0``.

    Args:
        redis:   App-level async Redis client.
        key:     Key produced by :func:`cache_key`.
        entries: Retrieval results to cache.
        ttl:     Expiry in seconds.
    """
    if ttl <= 0:
        return
    try:
        payload = json.dumps([m.model_dump(mode="json") for m in entries])
        await redis.setex(key, ttl, payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("retrieval_cache_set_error", error=str(exc))


async def invalidate_user(redis: AsyncRedis, user_id: str) -> int:
    """Delete every cached retrieval for *user_id*.

    Called after a memory write so newly-learned facts are immediately visible.

    Args:
        redis:   App-level async Redis client.
        user_id: UUID string whose cache should be cleared.

    Returns:
        The number of keys deleted.
    """
    pattern = f"{_KEY_PREFIX}:{user_id}:*"
    try:
        keys = [k async for k in redis.scan_iter(match=pattern)]
        if not keys:
            return 0
        await redis.delete(*keys)
        logger.debug("retrieval_cache_invalidated", user_id=user_id, count=len(keys))
        return len(keys)
    except Exception as exc:  # noqa: BLE001
        logger.warning("retrieval_cache_invalidate_error", error=str(exc))
        return 0
