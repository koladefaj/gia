"""Tests for memory consolidation (raw memories → synthesised insights)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.memory.consolidation import _parse_insights, consolidate_memories
from backend.app.schemas.memory import MemoryEntry, UserContext


def _mem(text: str) -> MagicMock:
    m = MagicMock()
    m.id = str(uuid.uuid4())
    m.text = text
    return m


class TestParseInsights:
    def test_plain_array(self) -> None:
        assert _parse_insights('["a", "b"]') == ["a", "b"]

    def test_fenced(self) -> None:
        assert _parse_insights('```json\n["a"]\n```') == ["a"]

    def test_embedded_in_prose(self) -> None:
        assert _parse_insights('Here you go: ["x"] done') == ["x"]

    def test_garbage_is_empty(self) -> None:
        assert _parse_insights("not json at all") == []

    def test_non_list_is_empty(self) -> None:
        assert _parse_insights('{"a": 1}') == []


@pytest.mark.asyncio
async def test_consolidate_synthesises_and_supersedes(test_settings) -> None:
    """Enough raw memories → insights stored, prior insights replaced."""
    raw = [_mem("Likes Tems"), _mem("Likes Wizkid"), _mem("Likes Rema"), _mem("Likes Asake")]
    old_insight = _mem("stale insight")

    async def fake_fetch(_uid, memory_type, limit=50):
        return {"preference": raw, "life_fact": [], "insight": [old_insight]}.get(memory_type, [])

    store = MagicMock()
    store.fetch_by_type = AsyncMock(side_effect=fake_fetch)
    store.delete_by_id = AsyncMock()
    store.upsert_memory = AsyncMock(return_value="new-id")

    with patch("backend.app.memory.consolidation.asyncio.to_thread",
               new=AsyncMock(return_value='["Prefers emotionally expressive Afrobeats"]')), \
         patch("backend.app.memory.consolidation.embed_many",
               new=AsyncMock(return_value=[[0.0] * 768])):
        insights = await consolidate_memories("u1", store, test_settings)

    assert insights == ["Prefers emotionally expressive Afrobeats"]
    store.delete_by_id.assert_called_once_with(old_insight.id)  # old insight superseded
    store.upsert_memory.assert_called_once()


@pytest.mark.asyncio
async def test_consolidate_skips_low_signal(test_settings) -> None:
    """Too few raw memories → no LLM call, nothing stored."""
    store = MagicMock()
    store.fetch_by_type = AsyncMock(return_value=[_mem("Likes Tems")])  # 1 per type → 2 total
    to_thread = AsyncMock()

    with patch("backend.app.memory.consolidation.asyncio.to_thread", to_thread):
        insights = await consolidate_memories("u1", store, test_settings)

    assert insights == []
    to_thread.assert_not_called()


@pytest.mark.asyncio
async def test_consolidate_empty_llm_output_stores_nothing(test_settings) -> None:
    raw = [_mem(f"Likes artist {i}") for i in range(5)]
    store = MagicMock()
    store.fetch_by_type = AsyncMock(return_value=raw)
    store.upsert_memory = AsyncMock()

    with patch("backend.app.memory.consolidation.asyncio.to_thread",
               new=AsyncMock(return_value="[]")):
        insights = await consolidate_memories("u1", store, test_settings)

    assert insights == []
    store.upsert_memory.assert_not_called()


def test_user_context_renders_insights_section() -> None:
    """Insights surface prominently in the prompt text."""
    ctx = UserContext(
        user_id="u",
        insights=[MemoryEntry(
            id="11111111-1111-1111-1111-111111111111",
            type="insight",
            text="Prefers emotionally expressive Afrobeats",
            confidence=0.9,
            created_at=datetime.now(UTC),
        )],
    )
    out = ctx.to_prompt_text()
    assert "Who they are" in out
    assert "Prefers emotionally expressive Afrobeats" in out
