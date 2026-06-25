"""Cross-encoder reranking — the optional precision stage.

A bi-encoder (BGE embeddings) is fast but approximate: it scores query and
document independently, so a keyword overlap or a subtle relevance signal can be
lost.  A cross-encoder reads the query and each candidate *together* and scores
the pair directly — markedly better ranking, at the cost of one model forward
pass per candidate.

That cost is why this stage is **off by default** (``settings.rerank_enabled``).
On a live voice turn the latency budget (~1.2s to first word) does not justify
the extra 100-300ms; in Celery background tasks, evaluation, or a demo where
recall is the point, it is worth turning on.

Model and lifecycle mirror ``embeddings.py``: lazy ``@lru_cache`` load, all
inference on a thread so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from backend.app.observability.logging import get_logger
from backend.app.schemas.memory import MemoryEntry

logger = get_logger(__name__)


@lru_cache(maxsize=2)
def _get_cross_encoder(model_name: str):  # type: ignore[return]
    """Load and cache a ``CrossEncoder`` for *model_name*.

    Cached per model name so swapping ``settings.rerank_model`` does not reload
    a model already in memory.  Import is deferred so tests that never rerank do
    not pay the import / download cost.

    Args:
        model_name: HuggingFace cross-encoder id (e.g. ``BAAI/bge-reranker-base``).

    Returns:
        A ready ``sentence_transformers.CrossEncoder``.
    """
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    logger.info("reranker_model_loading", model=model_name)
    encoder = CrossEncoder(model_name)
    logger.info("reranker_model_ready", model=model_name)
    return encoder


async def rerank(
    query: str,
    candidates: list[MemoryEntry],
    *,
    top_k: int,
    model_name: str = "BAAI/bge-reranker-base",
) -> list[MemoryEntry]:
    """Reorder *candidates* by cross-encoder relevance to *query*; keep top-*k*.

    The cross-encoder score replaces each entry's ``score`` field so downstream
    consumers (and Langfuse telemetry) see the rerank signal rather than the
    original bi-encoder score.

    Args:
        query:      The user's query text.
        candidates: Memories from the first-stage (hybrid) retrieval.
        top_k:      Number of memories to keep after reranking.
        model_name: Cross-encoder model id.

    Returns:
        Up to *top_k* ``MemoryEntry`` ordered by descending cross-encoder score.
        Returns the input unchanged (trimmed to *top_k*) if it is empty or a
        single item, where reranking cannot change the order.
    """
    if len(candidates) <= 1:
        return candidates[:top_k]

    def _score() -> list[float]:
        encoder = _get_cross_encoder(model_name)
        pairs = [(query, c.text) for c in candidates]
        return [float(s) for s in encoder.predict(pairs)]

    try:
        scores = await asyncio.to_thread(_score)
    except Exception as exc:  # noqa: BLE001
        # Reranking is an enhancement, never a hard dependency — on failure,
        # fall back to the first-stage order rather than breaking the turn.
        logger.warning("reranker_error", error=str(exc))
        return candidates[:top_k]

    ranked = sorted(
        zip(candidates, scores, strict=False), key=lambda pair: pair[1], reverse=True
    )
    result: list[MemoryEntry] = []
    for entry, score in ranked[:top_k]:
        result.append(entry.model_copy(update={"score": score}))
    return result
