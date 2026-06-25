"""Memory agent — extraction and storage orchestrator.

``MemoryService`` ties together embedding, extraction, dedup, and supersede
into a single ``run_extraction`` method that can be called from a FastAPI
route or a Celery task.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.config import Settings
from backend.app.memory.cache import invalidate_user
from backend.app.memory.decay import apply_supersede
from backend.app.memory.embeddings import embed_many, text_hash
from backend.app.memory.extractor import extract_memories
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.observability.logging import get_logger
from backend.app.schemas.memory import ExtractedMemory, MemoryEntry

logger = get_logger(__name__)

@dataclass
class MemoryService:
    """Orchestrates the full extraction → dedup → store pipeline.

    Attributes:
        store: Weaviate store bound to an open client.
        redis: App-level ``AsyncRedis`` instance.
        cfg:   Application settings.
    """

    store: WeaviateMemoryStore
    redis: object  # AsyncRedis — typed as object to avoid heavy import at module level
    cfg: Settings

    async def run_extraction(
        self,
        user_id: str,
        transcript: str,
        existing: list[MemoryEntry] | None = None,
    ) -> list[str]:
        """Extract memories from *transcript* and persist non-duplicate entries.

        Steps:
          1. Semantic search for existing relevant memories (used by extractor
             for supersede ID lookup).
          2. LLM extraction pass → list of ``ExtractedMemory``.
          3. SHA-256 Redis dedup — skip if this exact text was stored before.
          4. Supersede old memory if ``supersedes_id`` is set.
          5. Embed and insert the new memory into Weaviate.
          6. Record the hash in Redis (30-day TTL).

        Args:
            user_id:    UUID string of the user this transcript belongs to.
            transcript: Full conversation exchange text.
            existing:   Pre-fetched relevant memories (avoids a second search
                        if the caller already has them).

        Returns:
            List of Weaviate UUID strings for the memories that were stored.
        """
        if existing is None:
            from backend.app.memory.embeddings import embed as _embed  # noqa: PLC0415

            query_vector = await _embed(transcript[:500], redis=self.redis)
            existing = await self.store.search(user_id, query_vector, "preference", k=5)

        new_memories: list[ExtractedMemory] = await extract_memories(
            transcript, existing, self.cfg
        )
        stored_ids = await self.persist_memories(user_id, new_memories)
        logger.info(
            "extraction_complete",
            user_id=user_id,
            extracted=len(new_memories),
            stored=len(stored_ids),
        )
        return stored_ids

    async def persist_memories(
        self, user_id: str, memories: list[ExtractedMemory]
    ) -> list[str]:
        """Store *memories* via the dedup → supersede → embed → upsert pipeline.

        Shared by ``run_extraction`` (conversation memories) and the cold-start
        profiler (Spotify taste memories).  SHA-256 Redis dedup skips texts seen
        before; the retrieval cache is invalidated once anything new lands.

        Args:
            user_id:  UUID string of the owning user.
            memories: Validated ``ExtractedMemory`` objects to store.

        Returns:
            Weaviate UUID strings of the memories actually inserted.
        """
        # ── 1. Drop dups first (SHA seen-before in Redis, or repeated within this
        #        batch) so we only embed — and pay for — genuinely new texts. ───
        to_store: list[ExtractedMemory] = []
        seen_hashes: set[str] = set()
        for memory in memories:
            digest = text_hash(memory.text)
            if digest in seen_hashes:
                continue
            if await self.redis.exists(f"memory_hash:{digest}:{user_id}"):  # type: ignore[union-attr]
                logger.debug("memory_dedup_skip", user_id=user_id, text=memory.text[:40])
                continue
            seen_hashes.add(digest)
            to_store.append(memory)

        if not to_store:
            return []

        # ── 2. One batched embedding call for every new memory (cache-aware) ──
        vectors = await embed_many([m.text for m in to_store], redis=self.redis)

        # ── 3. Supersede + upsert + mark each as seen ────────────────────────
        stored_ids: list[str] = []
        for memory, vector in zip(to_store, vectors, strict=True):
            if memory.supersedes_id:
                await apply_supersede(self.store, memory.supersedes_id)

            mem_id = await self.store.upsert_memory(user_id, memory, vector)
            key = f"memory_hash:{text_hash(memory.text)}:{user_id}"
            await self.redis.setex(key, 86400 * 30, "1")  # type: ignore[union-attr]
            stored_ids.append(mem_id)

        # Newly-learned facts must not be hidden behind a stale retrieval cache.
        if stored_ids:
            await invalidate_user(self.redis, user_id)
        return stored_ids
