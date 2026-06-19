"""Weaviate read/write operations for the memory engine.

All weaviate-client v4 calls are synchronous, so every public method in
``WeaviateMemoryStore`` uses ``asyncio.to_thread`` to avoid blocking the
event loop.

Collections used
----------------
``UserMemory``
    Stores learned preferences and episodic summaries.  Schema is created
    by ``weaviate_init.init_weaviate_schema`` at startup.
``MoodPattern``
    Aggregated time-bucket mood statistics (written by the mood-inference
    Celery task, read here for context assembly).
"""

from __future__ import annotations

import asyncio
import uuid as uuid_lib
from dataclasses import dataclass
from datetime import datetime, timezone

from weaviate import WeaviateClient
from weaviate.classes.query import Filter, MetadataQuery

from backend.app.observability.logging import get_logger
from backend.app.schemas.memory import ExtractedMemory, MemoryEntry

logger = get_logger(__name__)


def _obj_to_entry(obj) -> MemoryEntry:  # type: ignore[return]
    """Convert a raw Weaviate query result object to a ``MemoryEntry``.

    Args:
        obj: A ``weaviate.types.Object`` returned by ``near_vector`` or
             ``fetch_objects``.

    Returns:
        A validated ``MemoryEntry`` with all fields populated.
    """
    props = obj.properties
    created_at = props.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    if created_at is None:
        created_at = datetime.now(timezone.utc)

    score = 0.0
    if obj.metadata is not None:
        score = obj.metadata.score or 0.0

    return MemoryEntry(
        id=str(obj.uuid),
        type=str(props.get("type", "preference")),
        text=str(props.get("text", "")),
        confidence=float(props.get("confidence") or 0.8),
        created_at=created_at,
        supersedes_id=props.get("supersedes_id") or None,
        score=score,
    )


@dataclass
class WeaviateMemoryStore:
    """Async wrapper around the Weaviate v4 ``UserMemory`` collection.

    Attributes:
        client: An open ``WeaviateClient``.  The caller owns the lifecycle —
                this store does **not** close the client.
    """

    client: WeaviateClient

    async def search(
        self,
        user_id: str,
        query_vector: list[float],
        memory_type: str,
        k: int = 5,
    ) -> list[MemoryEntry]:
        """Return the top-*k* memories for *user_id* matching *query_vector*.

        Filters by ``type`` so preferences, mood patterns, and episodes do
        not bleed into each other's result sets.

        Args:
            user_id:      UUID string identifying the user.
            query_vector: 768-dim embedding to search against.
            memory_type:  One of ``"preference"``, ``"mood_pattern"``, ``"episode"``.
            k:            Maximum number of results to return.

        Returns:
            List of ``MemoryEntry`` objects ordered by semantic similarity
            (highest score first).
        """

        def _run() -> list[MemoryEntry]:
            col = self.client.collections.get("UserMemory")
            results = col.query.near_vector(
                near_vector=query_vector,
                limit=k,
                filters=(
                    Filter.by_property("user_id").equal(user_id)
                    & Filter.by_property("type").equal(memory_type)
                ),
                return_metadata=MetadataQuery(score=True),
            )
            return [_obj_to_entry(o) for o in results.objects]

        return await asyncio.to_thread(_run)

    async def upsert_memory(
        self,
        user_id: str,
        memory: ExtractedMemory,
        vector: list[float],
    ) -> str:
        """Insert a new memory into the ``UserMemory`` collection.

        Args:
            user_id: UUID string identifying the owner of this memory.
            memory:  Validated ``ExtractedMemory`` from the LLM extractor.
            vector:  Pre-computed 768-dim embedding of ``memory.text``.

        Returns:
            The Weaviate UUID string of the newly inserted object.
        """

        def _run() -> str:
            col = self.client.collections.get("UserMemory")
            weaviate_uuid = col.data.insert(
                properties={
                    "user_id": user_id,
                    "type": memory.type,
                    "text": memory.text,
                    "confidence": memory.confidence,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "supersedes_id": memory.supersedes_id or "",
                },
                vector=vector,
            )
            return str(weaviate_uuid)

        inserted_id = await asyncio.to_thread(_run)
        logger.info("memory_stored", user_id=user_id, type=memory.type, id=inserted_id)
        return inserted_id

    async def delete_by_id(self, memory_id: str) -> None:
        """Delete a memory by its Weaviate UUID.

        Used by the supersede flow to remove the old conflicting preference
        before inserting the updated one.

        Args:
            memory_id: Weaviate UUID string of the memory to remove.
        """

        def _run() -> None:
            col = self.client.collections.get("UserMemory")
            col.data.delete_by_id(uuid=uuid_lib.UUID(memory_id))

        await asyncio.to_thread(_run)
        logger.info("memory_deleted", id=memory_id)

    async def get_by_id(self, memory_id: str) -> MemoryEntry | None:
        """Fetch a single memory by UUID.

        Args:
            memory_id: Weaviate UUID string.

        Returns:
            ``MemoryEntry`` if found, ``None`` otherwise.
        """

        def _run() -> MemoryEntry | None:
            col = self.client.collections.get("UserMemory")
            obj = col.query.fetch_object_by_id(
                uuid=uuid_lib.UUID(memory_id),
                return_metadata=MetadataQuery(score=False),
            )
            if obj is None:
                return None
            return _obj_to_entry(obj)

        return await asyncio.to_thread(_run)
