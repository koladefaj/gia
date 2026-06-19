"""Cold-start taste profiler — bootstrap memories from real Spotify listening.

When a user first connects Spotify, we don't have to start from an empty memory
store: we pull their top artists, top tracks, and recent plays, let the LLM
distil that into durable taste preferences, and persist them through the same
pipeline as conversation memories.  By turn one, Gia already knows them.

Everything goes through the MCP-backed ``SpotifyClientProtocol`` (no direct
Spotify API), so the integration stays in one place.

Note: Spotify's API exposes *ranked* top items by time range, but no listening
minutes or play counts — so the profiler never claims hours/counts.
"""

from __future__ import annotations

import asyncio

from redis.asyncio import Redis as AsyncRedis

from backend.app.agents.memory import MemoryService
from backend.app.config import Settings
from backend.app.interfaces import SpotifyClientProtocol
from backend.app.memory.extractor import _parse_extracted_memories
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.observability.logging import get_logger
from backend.app.prompts import PromptRegistry, get_registry
from backend.app.providers.llm import get_fast_llm

logger = get_logger(__name__)

PROFILER_KEY = "agents.profiler"


def _fmt_artists(artists: list[dict]) -> str:
    """Render top-artist dicts as a ranked text list for the prompt."""
    lines = [f"{i + 1}. {a.get('name', '?')}" for i, a in enumerate(artists)]
    return "\n".join(lines) or "none available"


def _fmt_tracks(tracks: list[dict]) -> str:
    """Render track dicts as a ranked ``name by artist`` text list."""
    lines = [
        f"{i + 1}. {t.get('name', '?')} by {t.get('artist', '?')}"
        for i, t in enumerate(tracks)
    ]
    return "\n".join(lines) or "none available"


def _safe(result: object) -> list[dict]:
    """Coerce a gather result to a list, treating exceptions as empty."""
    return result if isinstance(result, list) else []


async def bootstrap_taste_profile(
    user_id: str,
    *,
    spotify: SpotifyClientProtocol,
    store: WeaviateMemoryStore,
    redis: AsyncRedis,
    cfg: Settings,
    registry: PromptRegistry | None = None,
) -> list[str]:
    """Profile a user from their Spotify listening and store taste memories.

    Pulls top artists / top tracks / recent plays (in parallel, degrading
    gracefully), asks the fast LLM to distil durable preferences, and persists
    the non-duplicate ones via ``MemoryService.persist_memories``.

    Args:
        user_id:  UUID string of the user.
        spotify:  Spotify client (MCP-backed or fake).
        store:    Weaviate memory store.
        redis:    App-level async Redis (dedup + cache invalidation).
        cfg:      Application settings.
        registry: Prompt registry; defaults to the process-wide singleton.

    Returns:
        Weaviate UUID strings of the stored taste memories (empty if there was
        no listening data to profile or nothing durable was extracted).
    """
    reg = registry or get_registry()

    results = await asyncio.gather(
        spotify.get_top_artists(limit=10),
        spotify.get_top_tracks(limit=10),
        spotify.get_recently_played(limit=10),
        return_exceptions=True,
    )
    artists, tracks, recent_tracks = _safe(results[0]), _safe(results[1]), _safe(results[2])

    if not artists and not tracks:
        logger.info("profiler_no_listening_data", user_id=user_id)
        return []

    prompt = reg.get(PROFILER_KEY).render(
        "profile",
        top_artists=_fmt_artists(artists),
        top_tracks=_fmt_tracks(tracks),
        recent_tracks=_fmt_tracks(recent_tracks),
    )

    llm = get_fast_llm(cfg)
    try:
        raw = await asyncio.to_thread(llm.call, [{"role": "user", "content": prompt}])
    except Exception as exc:  # noqa: BLE001
        logger.warning("profiler_llm_error", user_id=user_id, error=str(exc))
        return []

    memories = _parse_extracted_memories(raw)
    # Profiler output is always taste preferences.
    for m in memories:
        m.type = "preference"

    stored = await MemoryService(store=store, redis=redis, cfg=cfg).persist_memories(
        user_id, memories
    )
    logger.info(
        "profiler_done",
        user_id=user_id,
        artists=len(artists),
        tracks=len(tracks),
        extracted=len(memories),
        stored=len(stored),
    )
    return stored
