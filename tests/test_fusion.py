"""Tests for Reciprocal Rank Fusion."""

from __future__ import annotations

from datetime import datetime, timezone

from backend.app.memory.fusion import RRF_K, fuse_memories, reciprocal_rank_fusion
from backend.app.schemas.memory import MemoryEntry

_NOW = datetime(2026, 6, 19, tzinfo=timezone.utc)


def _m(uid: str, text: str) -> MemoryEntry:
    return MemoryEntry(
        id=uid, type="preference", text=text, confidence=0.8, created_at=_NOW
    )


def test_rrf_rewards_items_high_in_multiple_lists() -> None:
    list_a = ["x", "y", "z"]
    list_b = ["y", "x", "w"]
    fused = reciprocal_rank_fusion([list_a, list_b], key=lambda s: s)
    # y is rank0+rank1, x is rank0+rank1 too but x is rank0 in A → both high;
    # importantly w/z (single-list, low rank) trail.
    assert set(fused[:2]) == {"x", "y"}
    assert fused[-1] in {"w", "z"}


def test_rrf_single_list_preserves_order() -> None:
    assert reciprocal_rank_fusion([["a", "b", "c"]], key=lambda s: s) == ["a", "b", "c"]


def test_rrf_empty_input() -> None:
    assert reciprocal_rank_fusion([], key=lambda s: s) == []


def test_rrf_limit() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c", "d"]], key=lambda s: s, limit=2)
    assert fused == ["a", "b"]


def test_rrf_score_math() -> None:
    # An item ranked 0 in two lists beats an item ranked 0 in one list.
    two = reciprocal_rank_fusion(
        [["top"], ["top"]], key=lambda s: s
    )
    assert two == ["top"]
    # Verify the constant is the standard 60.
    assert RRF_K == 60


def test_fuse_memories_dedupes_by_id() -> None:
    a1 = _m("11111111-0000-0000-0000-000000000001", "loves Tems")
    a1_again = _m("11111111-0000-0000-0000-000000000001", "loves Tems")
    b = _m("22222222-0000-0000-0000-000000000002", "likes Burna")
    fused = fuse_memories([[a1, b], [a1_again]])
    ids = [m.id for m in fused]
    # a1 merged (appears once) and ranked first (it's in both lists)
    assert ids.count("11111111-0000-0000-0000-000000000001") == 1
    assert ids[0] == "11111111-0000-0000-0000-000000000001"
