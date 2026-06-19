"""Tests for ``MemoryService`` and ``build_memory_agent``."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.schemas.memory import ExtractedMemory, MemoryEntry

_USER_ID = "00000000-0000-0000-0000-000000000001"
_NOW = datetime(2026, 6, 19, tzinfo=timezone.utc)


def _entry(text: str) -> MemoryEntry:
    return MemoryEntry(
        id="00000000-0000-0000-0000-000000000011",
        type="preference",
        text=text,
        confidence=0.8,
        created_at=_NOW,
    )


@pytest.fixture()
def fake_store() -> MagicMock:
    store = MagicMock()
    store.search = AsyncMock(return_value=[])
    store.upsert_memory = AsyncMock(return_value="new-mem-id")
    store.get_by_id = AsyncMock(return_value=None)
    store.delete_by_id = AsyncMock()
    return store


@pytest.fixture()
def fake_redis() -> MagicMock:
    r = MagicMock()
    r.exists = AsyncMock(return_value=0)
    r.setex = AsyncMock()
    return r


def test_build_memory_agent_returns_crewai_agent(test_settings) -> None:
    """``build_memory_agent`` returns a properly configured CrewAI ``Agent``."""
    from crewai import Agent

    from backend.app.agents.memory import build_memory_agent

    # CrewAI's Agent validates ``llm`` — it must be a model-name string or BaseLLM.
    with patch("backend.app.agents.memory.get_fast_llm", return_value="gpt-4o-mini"):
        agent = build_memory_agent(test_settings)

    assert isinstance(agent, Agent)
    assert "Memory Curator" in agent.role
    assert agent.allow_delegation is False


@pytest.mark.asyncio
async def test_memory_service_stores_new_memories(
    fake_store, fake_redis, test_settings
) -> None:
    """``MemoryService.run_extraction`` stores a valid extracted memory."""
    from backend.app.agents.memory import MemoryService

    new_mem = ExtractedMemory(type="preference", text="Loves Tems", confidence=0.9)

    with patch("backend.app.agents.memory.build_memory_agent"), \
         patch("backend.app.agents.memory.extract_memories", new=AsyncMock(return_value=[new_mem])), \
         patch("backend.app.agents.memory.embed", new=AsyncMock(return_value=[0.0] * 768)):
        service = MemoryService(store=fake_store, redis=fake_redis, cfg=test_settings)
        ids = await service.run_extraction(_USER_ID, "I loved Free Mind by Tems")

    assert ids == ["new-mem-id"]
    fake_store.upsert_memory.assert_called_once()
    fake_redis.setex.assert_called_once()


@pytest.mark.asyncio
async def test_memory_service_skips_duplicates(
    fake_store, fake_redis, test_settings
) -> None:
    """Memories already in Redis (dedup hash present) are skipped."""
    fake_redis.exists = AsyncMock(return_value=1)  # hash exists

    from backend.app.agents.memory import MemoryService

    new_mem = ExtractedMemory(type="preference", text="Duplicate memory", confidence=0.9)

    with patch("backend.app.agents.memory.build_memory_agent"), \
         patch("backend.app.agents.memory.extract_memories", new=AsyncMock(return_value=[new_mem])), \
         patch("backend.app.agents.memory.embed", new=AsyncMock(return_value=[0.0] * 768)):
        service = MemoryService(store=fake_store, redis=fake_redis, cfg=test_settings)
        ids = await service.run_extraction(_USER_ID, "duplicate transcript")

    assert ids == []
    fake_store.upsert_memory.assert_not_called()


@pytest.mark.asyncio
async def test_memory_service_handles_supersede(
    fake_store, fake_redis, test_settings
) -> None:
    """When ``supersedes_id`` is set, the old memory is deleted before storing."""
    old_entry = _entry("old preference")
    fake_store.get_by_id = AsyncMock(return_value=old_entry)

    from backend.app.agents.memory import MemoryService

    new_mem = ExtractedMemory(
        type="preference",
        text="New preference",
        confidence=0.9,
        supersedes_id="00000000-0000-0000-0000-000000000011",
    )

    with patch("backend.app.agents.memory.build_memory_agent"), \
         patch("backend.app.agents.memory.extract_memories", new=AsyncMock(return_value=[new_mem])), \
         patch("backend.app.agents.memory.embed", new=AsyncMock(return_value=[0.0] * 768)):
        service = MemoryService(store=fake_store, redis=fake_redis, cfg=test_settings)
        await service.run_extraction(_USER_ID, "I changed my mind")

    fake_store.delete_by_id.assert_called_once_with("00000000-0000-0000-0000-000000000011")


@pytest.mark.asyncio
async def test_memory_service_no_extraction_returns_empty(
    fake_store, fake_redis, test_settings
) -> None:
    """When the LLM extracts nothing, ``run_extraction`` returns ``[]``."""
    from backend.app.agents.memory import MemoryService

    with patch("backend.app.agents.memory.build_memory_agent"), \
         patch("backend.app.agents.memory.extract_memories", new=AsyncMock(return_value=[])), \
         patch("backend.app.agents.memory.embed", new=AsyncMock(return_value=[0.0] * 768)):
        service = MemoryService(store=fake_store, redis=fake_redis, cfg=test_settings)
        ids = await service.run_extraction(_USER_ID, "play it at 8pm")

    assert ids == []
