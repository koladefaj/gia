"""Embedding service — OpenAI ``text-embedding-3-small`` (1536-dim).

Memory text is embedded via the OpenAI API rather than a local model, so the
containers stay torch-free and production ships nothing heavy.  The call is
already async (OpenAI SDK), so there's no thread-pool hop.

SHA-256 deduplication is handled at the *storage* level, not here.
``text_hash`` returns the key callers use to check Redis before writing.
"""

from __future__ import annotations

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
    cache_key = f"embed_cache:{text_hash(text)}" if redis is not None else None

    if cache_key is not None:
        try:
            cached = await redis.get(cache_key)
            if cached is not None:
                logger.debug("embed_cache_hit", chars=len(text))
                return json.loads(cached)
        except Exception:  # noqa: BLE001
            pass  # cache failure → fall through to API call

    client = get_async_openai(settings)
    resp = await client.embeddings.create(model=settings.embedding_model, input=text)
    vector = list(resp.data[0].embedding)

    if cache_key is not None:
        try:
            await redis.set(cache_key, json.dumps(vector), ex=_EMBED_CACHE_TTL)
        except Exception:  # noqa: BLE001
            pass  # caching failure is non-fatal

    return vector


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
