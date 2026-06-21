"""Tests for the cross-encoder reranker (model mocked — no download)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from backend.app.memory import reranker
from backend.app.schemas.memory import MemoryEntry

_NOW = datetime(2026, 6, 19, tzinfo=UTC)


def _m(uid: str, text: str) -> MemoryEntry:
    return MemoryEntry(
        id=uid, type="preference", text=text, confidence=0.8, created_at=_NOW, score=0.0
    )


@pytest.mark.asyncio
async def test_rerank_orders_by_cross_encoder_score() -> None:
    candidates = [
        _m("1" * 8 + "-0000-0000-0000-000000000001", "A"),
        _m("2" * 8 + "-0000-0000-0000-000000000002", "B"),
        _m("3" * 8 + "-0000-0000-0000-000000000003", "C"),
    ]
    fake_encoder = MagicMock()
    # B most relevant, then C, then A
    fake_encoder.predict.return_value = [0.1, 0.9, 0.5]

    with patch.object(reranker, "_get_cross_encoder", return_value=fake_encoder):
        out = await reranker.rerank("query", candidates, top_k=2)

    assert [m.text for m in out] == ["B", "C"]
    # cross-encoder score is written back onto the entries
    assert out[0].score == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_rerank_short_circuits_single_item() -> None:
    one = [_m("1" * 8 + "-0000-0000-0000-000000000001", "only")]
    # No model call should happen for <=1 candidate.
    with patch.object(reranker, "_get_cross_encoder", side_effect=AssertionError):
        out = await reranker.rerank("q", one, top_k=5)
    assert out == one


@pytest.mark.asyncio
async def test_rerank_falls_back_on_model_error() -> None:
    candidates = [
        _m("1" * 8 + "-0000-0000-0000-000000000001", "A"),
        _m("2" * 8 + "-0000-0000-0000-000000000002", "B"),
    ]
    fake_encoder = MagicMock()
    fake_encoder.predict.side_effect = RuntimeError("model boom")

    with patch.object(reranker, "_get_cross_encoder", return_value=fake_encoder):
        out = await reranker.rerank("q", candidates, top_k=1)

    # Falls back to first-stage order, trimmed to top_k.
    assert [m.text for m in out] == ["A"]
