"""Memory decay and conflict resolution.

Two mechanisms prevent stale preferences from lingering:

1. **Explicit supersede** — when the LLM extractor identifies a direct conflict
   (e.g. "actually I'm into high-energy now" contradicts "prefers low-energy"),
   it returns a ``supersedes_id``.  ``apply_supersede`` deletes the old record
   before the new one is written.

2. **Implicit recency preference** — ``prefer_recent`` sorts a result list by
   ``created_at`` descending so that, when two memories cover the same topic,
   the newer one appears first and is used by the prompt assembler.  This
   handles gradual drift without requiring explicit conflict detection.
"""

from __future__ import annotations

from backend.app.memory.store import WeaviateMemoryStore
from backend.app.observability.logging import get_logger
from backend.app.schemas.memory import MemoryEntry

logger = get_logger(__name__)


async def apply_supersede(store: WeaviateMemoryStore, old_id: str) -> None:
    """Delete *old_id* from Weaviate to resolve a direct preference conflict.

    Called by the storage layer when an ``ExtractedMemory`` carries a
    non-null ``supersedes_id``.  The old record is removed before the new
    one is inserted so the collection never contains both sides of a
    contradiction simultaneously.

    Silently skips if *old_id* does not exist (idempotent on retry).

    Args:
        store:  The ``WeaviateMemoryStore`` instance to delete from.
        old_id: Weaviate UUID string of the memory being replaced.
    """
    existing = await store.get_by_id(old_id)
    if existing is None:
        logger.debug("supersede_target_not_found", id=old_id)
        return
    await store.delete_by_id(old_id)
    logger.info("memory_superseded", old_id=old_id, old_text=existing.text[:60])


def prefer_recent(memories: list[MemoryEntry]) -> list[MemoryEntry]:
    """Return *memories* sorted newest-first.

    When the same topic appears in multiple memories (e.g. because the user
    changed their mind gradually), the most recent entry wins in the
    assembled prompt.  This is the implicit complement to the explicit
    supersede mechanism.

    Args:
        memories: Unsorted list of ``MemoryEntry`` objects.

    Returns:
        New list sorted by ``created_at`` descending (most recent first),
        with the original scores preserved.
    """
    return sorted(memories, key=lambda m: m.created_at, reverse=True)


def deduplicate(memories: list[MemoryEntry]) -> list[MemoryEntry]:
    """Remove exact-text duplicates, keeping the most recent copy.

    Weaviate dedup relies on the Redis hash check at write time, but this
    provides a read-time safety net in case the Redis key expired and the
    same text was stored twice.

    Args:
        memories: List of ``MemoryEntry`` objects, possibly containing
                  duplicate texts.

    Returns:
        Deduplicated list with the newest copy of each unique text retained.
    """
    seen: dict[str, MemoryEntry] = {}
    for m in prefer_recent(memories):
        if m.text not in seen:
            seen[m.text] = m
    return list(seen.values())
