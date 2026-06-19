"""CrewAI Memory agent — extraction and storage orchestrator.

``MemoryAgent`` ties together embedding, extraction, dedup, and supersede
into a single ``run_extraction`` method that can be called from a FastAPI
route, a Celery task, or composed into a larger CrewAI crew.

The agent itself is deliberately thin: all business logic lives in the
``memory.*`` modules so it can be tested independently of CrewAI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from crewai import Agent

from backend.app.config import Settings
from backend.app.memory.cache import invalidate_user
from backend.app.memory.decay import apply_supersede
from backend.app.memory.embeddings import embed, text_hash
from backend.app.memory.extractor import extract_memories
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.observability.logging import get_logger
from backend.app.prompts import PromptRegistry, get_registry
from backend.app.providers.llm import get_fast_llm
from backend.app.schemas.memory import ExtractedMemory, MemoryEntry

logger = get_logger(__name__)

AGENT_KEY = "agents.memory"


def build_memory_agent(cfg: Settings, registry: PromptRegistry | None = None) -> Agent:
    """Construct the CrewAI ``Agent`` instance for memory curation.

    The agent's role is injected into other agents' system prompts so they
    know who surfaced the user context they received.

    Args:
        cfg:      Application settings (provides LLM provider / model).
        registry: Prompt registry for the agent identity; defaults to the
                  process-wide singleton.

    Returns:
        A configured ``crewai.Agent`` ready to be added to a ``Crew``.
    """
    prompt = (registry or get_registry()).get(AGENT_KEY)
    return Agent(
        role=prompt.render("role"),
        goal=prompt.render("goal"),
        backstory=prompt.render("backstory"),
        llm=get_fast_llm(cfg),
        verbose=False,
        allow_delegation=False,
    )


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
    _agent: Agent = field(init=False)

    def __post_init__(self) -> None:
        self._agent = build_memory_agent(self.cfg)

    @property
    def crewai_agent(self) -> Agent:
        """The underlying ``crewai.Agent`` for use in multi-agent crews."""
        return self._agent

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

            query_vector = await _embed(transcript[:500])
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
        stored_ids: list[str] = []
        for memory in memories:
            key = f"memory_hash:{text_hash(memory.text)}:{user_id}"
            if await self.redis.exists(key):  # type: ignore[union-attr]
                logger.debug("memory_dedup_skip", user_id=user_id, text=memory.text[:40])
                continue

            vector = await embed(memory.text)

            if memory.supersedes_id:
                await apply_supersede(self.store, memory.supersedes_id)

            mem_id = await self.store.upsert_memory(user_id, memory, vector)
            await self.redis.setex(key, 86400 * 30, "1")  # type: ignore[union-attr]
            stored_ids.append(mem_id)

        # Newly-learned facts must not be hidden behind a stale retrieval cache.
        if stored_ids:
            await invalidate_user(self.redis, user_id)
        return stored_ids
