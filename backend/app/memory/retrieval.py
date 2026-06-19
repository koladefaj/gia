"""Context assembly — the entry point for every agent turn.

``build_user_context`` fetches all seven data sources in parallel and
packages the result into a ``UserContext`` that every CrewAI agent can use
directly via ``context.to_prompt_text()``.

Data sources
------------
1. Postgres  — structured profile facts (timezone, genres, volume)
2. Weaviate  — top-8 semantic preferences matching the query
3. Weaviate  — top-3 mood patterns matching the query
4. Weaviate  — top-3 episodic session summaries matching the query
5. Redis     — current-session running notes (``session:{user_id}``)
6. Spotify   — now-playing track dict
7. Spotify   — last 10 recently played tracks

``asyncio.gather(..., return_exceptions=True)`` is used so that a single
failing service (e.g. Spotify MCP not running in dev) does not block the
whole assembly.  Each result is coerced to a safe default on failure.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass

from redis.asyncio import Redis as AsyncRedis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import Settings, settings as _default_settings
from backend.app.db.models import Profile, User
from backend.app.interfaces import SpotifyClientProtocol
from backend.app.memory.cache import cache_key, get_cached, set_cached
from backend.app.memory.embeddings import embed
from backend.app.memory.reranker import rerank
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.observability.logging import get_logger
from backend.app.schemas.memory import MemoryEntry, UserContext

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievalConfig:
    """Tunable retrieval parameters derived from application ``Settings``.

    Bundling these into one object keeps ``build_user_context`` readable and
    makes the retrieval path easy to exercise in tests with hand-built configs.
    """

    hybrid: bool
    alpha: float
    cache_ttl: int
    rerank: bool
    rerank_model: str
    rerank_multiplier: int

    @classmethod
    def from_settings(cls, cfg: Settings) -> "RetrievalConfig":
        """Build a ``RetrievalConfig`` from the application settings."""
        return cls(
            hybrid=cfg.hybrid_enabled,
            alpha=cfg.retrieval_alpha,
            cache_ttl=cfg.retrieval_cache_ttl,
            rerank=cfg.rerank_enabled,
            rerank_model=cfg.rerank_model,
            rerank_multiplier=cfg.rerank_candidate_multiplier,
        )


async def retrieve_memories(
    *,
    store: WeaviateMemoryStore,
    redis: AsyncRedis,
    user_id: str,
    query_text: str,
    query_vector: list[float],
    memory_type: str,
    k: int,
    rcfg: RetrievalConfig,
) -> list[MemoryEntry]:
    """Run the full retrieval pipeline for one memory type.

    Pipeline: Redis cache → hybrid (BM25 + dense) or dense search → optional
    cross-encoder rerank → cache write.  Every stage degrades safely: a cache
    error falls through to a fresh fetch, and a rerank failure falls back to the
    first-stage ranking.

    Args:
        store:        Weaviate memory store.
        redis:        App-level async Redis client (for the retrieval cache).
        user_id:      UUID string of the user.
        query_text:   Raw query (drives BM25 and the cache key).
        query_vector: Embedding of the query (drives dense search).
        memory_type:  ``"preference"`` / ``"mood_pattern"`` / ``"episode"``.
        k:            Number of results to return.
        rcfg:         Resolved retrieval configuration.

    Returns:
        Up to *k* ``MemoryEntry`` ordered by relevance.
    """
    key = cache_key(user_id, memory_type, query_text)
    if rcfg.cache_ttl > 0:
        cached = await get_cached(redis, key)
        if cached is not None:
            logger.debug("retrieval_cache_hit", user_id=user_id, type=memory_type)
            return cached[:k]

    fetch_k = k * rcfg.rerank_multiplier if rcfg.rerank else k

    if rcfg.hybrid:
        entries = await store.hybrid_search(
            user_id, query_text, query_vector, memory_type, k=fetch_k, alpha=rcfg.alpha
        )
        path = "hybrid"
    else:
        entries = await store.search(user_id, query_vector, memory_type, k=fetch_k)
        path = "dense"

    if rcfg.rerank and entries:
        entries = await rerank(
            query_text, entries, top_k=k, model_name=rcfg.rerank_model
        )
        path = f"{path}+rerank"
    else:
        entries = entries[:k]

    if rcfg.cache_ttl > 0:
        await set_cached(redis, key, entries, rcfg.cache_ttl)

    logger.debug(
        "retrieval_done",
        user_id=user_id,
        type=memory_type,
        path=path,
        count=len(entries),
        top_ids=[e.ref for e in entries[:3]],
    )
    return entries


async def _get_profile(user_id: str, db: AsyncSession) -> dict | None:
    """Fetch structured profile facts from Postgres.

    Args:
        user_id: UUID string of the user.
        db:      Open ``AsyncSession``.

    Returns:
        Dict with ``timezone``, ``preferred_genres``, ``preferred_volume``,
        ``spotify_user_id`` keys, or ``None`` if the profile does not exist.
    """
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        return None

    result = await db.execute(
        select(Profile).join(User, User.id == Profile.user_id).where(User.id == uid)
    )
    profile = result.scalar_one_or_none()
    if profile is None:
        return None
    return {
        "timezone": profile.timezone,
        "preferred_genres": profile.preferred_genres,
        "preferred_volume": profile.preferred_volume,
        "spotify_user_id": profile.spotify_user_id,
    }


def _safe_list(result: object, default: list) -> list:
    """Return *default* when *result* is an ``Exception``."""
    return default if isinstance(result, Exception) else result  # type: ignore[return-value]


def _safe(result: object, default: object) -> object:
    """Return *default* when *result* is an ``Exception``."""
    return default if isinstance(result, Exception) else result


async def build_user_context(
    user_id: str,
    query: str,
    *,
    db: AsyncSession,
    store: WeaviateMemoryStore,
    redis: AsyncRedis,
    spotify: SpotifyClientProtocol,
    cfg: Settings | None = None,
) -> UserContext:
    """Assemble all seven data sources into a single ``UserContext``.

    Fetches are issued in parallel.  Individual failures are logged and
    coerced to safe defaults so a degraded service does not block the
    conversation turn.

    Args:
        user_id: UUID string identifying the user.
        query:   The user's current utterance, used as the semantic search
                 query for Weaviate.
        db:      Open ``AsyncSession`` for Postgres lookups.
        store:   ``WeaviateMemoryStore`` bound to an open Weaviate client.
        redis:   App-level ``AsyncRedis`` instance.
        spotify: Spotify client (live or fake).
        cfg:     Application settings driving retrieval (hybrid / cache / rerank).
                 Defaults to the global settings singleton.

    Returns:
        A fully populated ``UserContext`` ready for ``to_prompt_text()``.
    """
    cfg = cfg or _default_settings
    rcfg = RetrievalConfig.from_settings(cfg)
    query_vector = await embed(query)

    def _retrieve(memory_type: str, k: int) -> Awaitable[list[MemoryEntry]]:
        return retrieve_memories(
            store=store,
            redis=redis,
            user_id=user_id,
            query_text=query,
            query_vector=query_vector,
            memory_type=memory_type,
            k=k,
            rcfg=rcfg,
        )

    (
        profile,
        preferences,
        mood_patterns,
        episodes,
        session_raw,
        now_playing,
        recently_played,
    ) = await asyncio.gather(
        _get_profile(user_id, db),
        _retrieve("preference", cfg.retrieval_k_preferences),
        _retrieve("mood_pattern", cfg.retrieval_k_mood),
        _retrieve("episode", cfg.retrieval_k_episodes),
        redis.get(f"session:{user_id}"),
        spotify.get_currently_playing(),
        spotify.get_recently_played(limit=10),
        return_exceptions=True,
    )

    if isinstance(profile, Exception):
        logger.warning("context_profile_error", error=str(profile))
        profile = None
    if isinstance(session_raw, Exception):
        session_raw = None
    if isinstance(now_playing, Exception):
        logger.warning("context_spotify_error", error=str(now_playing))
        now_playing = None
    if isinstance(recently_played, Exception):
        recently_played = []

    preferences = _safe_list(preferences, [])
    mood_patterns = _safe_list(mood_patterns, [])
    episodes = _safe_list(episodes, [])

    return UserContext(
        user_id=user_id,
        profile=profile,
        preferences=preferences,  # type: ignore[arg-type]
        mood_patterns=mood_patterns,  # type: ignore[arg-type]
        episodes=episodes,  # type: ignore[arg-type]
        session_summary=session_raw,  # type: ignore[arg-type]
        now_playing=now_playing,  # type: ignore[arg-type]
        recently_played=recently_played or [],  # type: ignore[arg-type]
    )
