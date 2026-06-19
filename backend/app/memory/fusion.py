"""Reciprocal Rank Fusion (RRF) for combining ranked result lists.

Weaviate's ``hybrid()`` already fuses BM25 and dense *internally*.  This module
is the complementary tool: it fuses results that come from **separate queries**
— e.g. a query rewrite plus the original, or results from different collections
— where no single backend ranking exists to lean on.

RRF is the industry-standard fusion method precisely because it is rank-based,
not score-based: it never has to reconcile a BM25 score (unbounded) with a
cosine score (0–1).  Each list contributes ``1 / (k + rank)`` to every item it
contains, and items are re-sorted by the summed contribution.

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
individual Rank Learning Methods" (SIGIR 2009).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from backend.app.schemas.memory import MemoryEntry

RRF_K = 60  # standard damping constant; larger = flatter rank weighting


def reciprocal_rank_fusion[T](
    result_lists: Sequence[Iterable[T]],
    *,
    key: Callable[[T], str],
    k: int = RRF_K,
    limit: int | None = None,
) -> list[T]:
    """Fuse several ranked lists into one using Reciprocal Rank Fusion.

    Args:
        result_lists: Ranked iterables (best first).  Each item's position is
                      its rank; the same item may appear in several lists.
        key:          Function returning a stable identity for an item so the
                      same entity across lists is recognised and its scores
                      summed.
        k:            RRF damping constant (default 60).
        limit:        If given, return at most this many fused results.

    Returns:
        A new list ordered by descending fused score.  When an item appears in
        multiple lists, the first-seen object instance is kept as the
        representative.
    """
    scores: dict[str, float] = {}
    representative: dict[str, T] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            ident = key(item)
            scores[ident] = scores.get(ident, 0.0) + 1.0 / (k + rank)
            representative.setdefault(ident, item)

    ordered = sorted(representative.values(), key=lambda it: scores[key(it)], reverse=True)
    return ordered[:limit] if limit is not None else ordered


def fuse_memories(
    result_lists: Sequence[Sequence[MemoryEntry]],
    *,
    k: int = RRF_K,
    limit: int | None = None,
) -> list[MemoryEntry]:
    """RRF convenience wrapper specialised for ``MemoryEntry`` lists.

    Identity is the Weaviate ``id`` so the same memory retrieved by two queries
    is merged rather than duplicated.

    Args:
        result_lists: Ranked ``MemoryEntry`` lists to fuse.
        k:            RRF damping constant.
        limit:        Optional cap on the number of fused results.

    Returns:
        Fused, de-duplicated list of ``MemoryEntry`` ordered by RRF score.
    """
    return reciprocal_rank_fusion(result_lists, key=lambda m: m.id, k=k, limit=limit)
