"""Embedding service — OpenAI ``text-embedding-3-small`` (1536-dim).

Memory text is embedded via the OpenAI API rather than a local model, so the
containers stay torch-free and production ships nothing heavy.  The call is
already async (OpenAI SDK), so there's no thread-pool hop.

SHA-256 deduplication is handled at the *storage* level, not here.
``text_hash`` returns the key callers use to check Redis before writing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json

from backend.app.config import settings
from backend.app.observability.logging import get_logger
from backend.app.providers.openai_client import get_async_openai

logger = get_logger(__name__)

# text-embedding-3-small native dimensionality. If this changes, the Weaviate
# collections must be recreated and the user re-seeded (vectors of different
# dimension can't be compared).
VECTOR_DIM = 1536

_EMBED_CACHE_TTL = 86400  # 24 hours — vectors are stable for the same text


async def embed(text: str, redis=None) -> list[float]:
    """Return an embedding vector for *text* via the OpenAI embeddings API.

    When *redis* is supplied the result is cached under
    ``embed_cache:{sha256(text)}`` for :data:`_EMBED_CACHE_TTL` seconds so
    identical queries within a day skip the API call entirely.  Cache errors
    are silently swallowed — a miss is always safe.

    Thin wrapper over :func:`embed_many` for the single-text callers (the
    per-turn retrieval query); behaviour is identical (cache check + one API
    call on a miss).

    Args:
        text:  The string to embed.
        redis: Optional async Redis client for the embedding cache.

    Returns:
        A ``list[float]`` of length :data:`VECTOR_DIM`, suitable for Weaviate
        ``near_vector`` / hybrid cosine queries.

    Raises:
        Exception: Propagates OpenAI/network errors — callers already degrade
            the turn on failure.
    """
    return (await embed_many([text], redis=redis))[0]


async def embed_many(texts: list[str], redis=None) -> list[list[float]]:
    """Embed several texts in ONE API call (cache-aware), preserving input order.

    The worker's extraction pass produces a handful of new memories per session;
    embedding them one HTTP round-trip at a time is wasteful. This batches every
    cache miss into a single ``embeddings.create(input=[...])`` request and
    returns the vectors aligned 1:1 with *texts*.

    Args:
        texts: Strings to embed (any already-cached ones skip the API).
        redis: Optional async Redis client for the 24h embedding cache.

    Returns:
        A list of vectors, one per input text, in the same order.

    Raises:
        Exception: Propagates OpenAI/network errors — callers degrade on failure.
    """
    if not texts:
        return []

    results: list[list[float] | None] = [None] * len(texts)
    miss_indices: list[int] = []
    miss_texts: list[str] = []

    # ── Cache lookups — only the misses go to the API ────────────────────────
    for i, text in enumerate(texts):
        if redis is not None:
            try:
                cached = await redis.get(f"embed_cache:{text_hash(text)}")
                if cached is not None:
                    results[i] = json.loads(cached)
                    continue
            except Exception:  # noqa: BLE001
                pass  # cache failure → treat as a miss
        miss_indices.append(i)
        miss_texts.append(text)

    if miss_texts:
        client = get_async_openai(settings)
        resp = await client.embeddings.create(
            model=settings.embedding_model, input=miss_texts
        )
        # The API returns items in input order, but each carries its own .index —
        # sort by it so a vector never lands against the wrong text.
        ordered = sorted(resp.data, key=lambda d: d.index)
        logger.debug("embed_batch", total=len(texts), api_calls=len(miss_texts))
        for idx, item in zip(miss_indices, ordered, strict=True):
            vector = list(item.embedding)
            results[idx] = vector
            if redis is not None:
                # Caching failure is non-fatal — a miss next time is always safe.
                with contextlib.suppress(Exception):
                    await redis.set(
                        f"embed_cache:{text_hash(texts[idx])}",
                        json.dumps(vector),
                        ex=_EMBED_CACHE_TTL,
                    )

    # Every index is either a cache hit or was filled from the batch above.
    return [vector for vector in results if vector is not None]


def text_hash(text: str) -> str:
    """Return the SHA-256 hex digest of *text*.

    Used as a Redis dedup key (``memory_hash:{hash}:{user_id}``) to skip
    embedding + Weaviate insertion when the exact same text has already been
    stored for this user.

    Args:
        text: The memory text to fingerprint.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(text.encode()).hexdigest()
