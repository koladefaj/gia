"""BGE-base-en-v1.5 embedding service.

The model is loaded lazily on first call and cached in-process.  All
encoding runs in a thread pool (``asyncio.to_thread``) so the event loop
is never blocked by CPU-bound work.

SHA-256 deduplication is handled at the *storage* level, not here.
``text_hash`` returns the key used by callers to check Redis before writing
to Weaviate.
"""

from __future__ import annotations

import asyncio
import hashlib
from functools import lru_cache

from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

_MODEL_NAME = "BAAI/bge-base-en-v1.5"
VECTOR_DIM = 768


@lru_cache(maxsize=1)
def _get_model():  # type: ignore[return]
    """Load the BGE model once and cache it for the lifetime of the process.

    The import is intentionally deferred so unit tests that mock
    ``asyncio.to_thread`` don't trigger a full model download.

    Returns:
        A ``SentenceTransformer`` instance ready for encoding.
    """
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    logger.info("bge_model_loading", model=_MODEL_NAME)
    model = SentenceTransformer(_MODEL_NAME)
    logger.info("bge_model_ready", dim=VECTOR_DIM)
    return model


async def embed(text: str) -> list[float]:
    """Return a 768-dim normalised BGE embedding for *text*.

    Args:
        text: The string to embed.  Long texts are handled by the model's
              internal pooling, but prefer keeping inputs under 512 tokens
              for best quality.

    Returns:
        A Python ``list[float]`` of length 768, L2-normalised, suitable for
        Weaviate ``near_vector`` cosine-distance queries.
    """

    def _encode() -> list[float]:
        model = _get_model()
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    return await asyncio.to_thread(_encode)


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
