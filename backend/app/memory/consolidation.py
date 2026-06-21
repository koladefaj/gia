"""Memory consolidation — synthesise raw memories into higher-order insights.

The extractor records many small facts ("likes Tems", "likes Burna Boy"). This
periodic *reflection* pass reads the whole set and asks the LLM to generalise it
into a handful of INSIGHT memories ("prefers emotionally expressive Afrobeats").

Insights are what make Gia feel like she *knows* the user rather than just
recalling a list — and they're cheap to retrieve and inject. They're stored as
``type="insight"`` and fully superseded each run: an insight is *derived*, not
authored, so the latest synthesis always replaces the prior one.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import UTC, datetime

from backend.app.config import Settings
from backend.app.memory.embeddings import embed_many
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.observability.logging import get_logger
from backend.app.prompts import PromptRegistry, get_registry
from backend.app.providers.llm import get_fast_llm
from backend.app.schemas.memory import MemoryEntry

logger = get_logger(__name__)

AGENT_KEY = "agents.memory"

# Below this many raw memories there's nothing to generalise from.
MIN_RAW_MEMORIES = 4
MAX_INSIGHTS = 4
# Raw memory types that feed synthesis (mood patterns are their own pipeline).
_SOURCE_TYPES = ("preference", "life_fact")


def _parse_insights(raw: str) -> list[str]:
    """Parse the LLM's JSON array-of-strings output, tolerating fences/prose."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", s, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    return [str(x).strip() for x in data if isinstance(x, str) and x.strip()]


async def consolidate_memories(
    user_id: str,
    store: WeaviateMemoryStore,
    cfg: Settings,
    registry: PromptRegistry | None = None,
) -> list[str]:
    """Synthesise a user's raw memories into ``insight`` memories.

    Args:
        user_id:  UUID string of the user.
        store:    ``WeaviateMemoryStore`` to read raw memories / write insights.
        cfg:      Settings (the fast-tier synthesis model).
        registry: Prompt registry; defaults to the singleton.

    Returns:
        The list of synthesised insight strings actually stored (``[]`` when
        there's too little signal or the model returns nothing).
    """
    reg = registry or get_registry()

    raw: list[MemoryEntry] = []
    for memory_type in _SOURCE_TYPES:
        raw.extend(await store.fetch_by_type(user_id, memory_type, limit=50))

    if len(raw) < MIN_RAW_MEMORIES:
        logger.info("consolidation_low_signal", user_id=user_id, raw=len(raw))
        return []

    prompt = reg.get(AGENT_KEY).render(
        "consolidate", raw_memories="\n".join(f"- {m.text}" for m in raw)
    )
    llm = get_fast_llm(cfg)
    try:
        raw_out = await asyncio.to_thread(llm.call, [{"role": "user", "content": prompt}])
    except Exception as exc:  # noqa: BLE001
        logger.warning("consolidation_llm_error", error=str(exc))
        return []

    insights = _parse_insights(raw_out)[:MAX_INSIGHTS]
    if not insights:
        return []

    # Insights are derived, not authored — replace the whole prior set each run.
    for old in await store.fetch_by_type(user_id, "insight", limit=50):
        await store.delete_by_id(old.id)

    vectors = await embed_many(insights)
    stored: list[str] = []
    for text, vector in zip(insights, vectors, strict=True):
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            type="insight",
            text=text,
            confidence=0.9,
            created_at=datetime.now(UTC),
            source="consolidation",
        )
        stored.append(await store.upsert_memory(user_id, entry, vector))

    logger.info("consolidation_done", user_id=user_id, insights=len(stored))
    return insights
