"""FastAPI application entry point.

Lifespan
--------
``startup``
    1. Initialises the Weaviate schema (idempotent collection creation).
    2. Opens the Redis connection pool.
    3. Instantiates the Spotify HTTP client.
    4. Prewarmsall I/O-bound connections in parallel so the first real request
       does not pay connection-setup latency (critical for conversational AI
       where every millisecond of first-response time matters).

    Database schema is **not** managed here.  Run ``alembic upgrade head``
    before starting the server (the docker-compose ``api`` command does this
    automatically).  This keeps startup idempotent and prevents accidental
    schema drift when multiple instances start concurrently.

``shutdown``
    Closes all connection pools gracefully so no handles are leaked between
    restarts or between test runs.

All long-lived resources are stored on ``app.state`` so they can be injected
by ``dependencies.py`` and replaced in tests via ``dependency_overrides``.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
import weaviate
from fastapi import FastAPI
from sqlalchemy import text

from backend.app.api import artist, auth, chat, dj, health, memory, voice
from backend.app.config import settings
from backend.app.db.session import engine
from backend.app.db.weaviate_init import get_weaviate_client, init_weaviate_schema
from backend.app.observability.langfuse import init_langfuse
from backend.app.observability.logging import get_logger, setup_logging
from backend.app.tools.spotify import SpotifyMCPClient

logger = get_logger(__name__)


async def _prewarm_postgres() -> None:
    """Execute a trivial query to open the Postgres connection pool.

    Without this, the first ORM query after startup pays the full TCP connect
    + TLS + auth handshake cost (~50–150 ms).  A single warm connection puts
    the pool in a ready state for all subsequent requests.
    """
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application-scoped resources for the lifetime of the process.

    Opens connections once at startup and closes them on shutdown.  All
    resources are stored on ``app.state`` so dependency providers can inject
    them without using global variables.
    """
    setup_logging(settings.log_level)
    logger.info("gia_starting", env=settings.app_env, llm=settings.llm_provider)

    if settings.langfuse_enabled:
        init_langfuse(
            settings.langfuse_public_key,
            settings.langfuse_secret_key,
            settings.langfuse_host,
        )

    # Weaviate — ensure vector collections exist (idempotent), then keep a
    # persistent client on app.state for the memory engine.
    try:
        await init_weaviate_schema()
        logger.info("weaviate_schema_ready")
    except Exception as exc:
        logger.warning("weaviate_unavailable_at_startup", error=str(exc))
    app.state.weaviate = await asyncio.to_thread(get_weaviate_client)

    # Redis — single pool shared across all requests
    app.state.redis = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
    )

    # Spotify — single HTTP client shared across all requests
    app.state.spotify = SpotifyMCPClient(cfg=settings)

    # Prewarm all I/O connections in parallel.
    # return_exceptions=True prevents a partial failure from aborting the
    # other warmups — e.g. Spotify MCP may not be running in dev mode.
    results = await asyncio.gather(
        _prewarm_postgres(),
        app.state.redis.ping(),
        app.state.spotify.prewarm(),
        return_exceptions=True,
    )
    for name, result in zip(("postgres", "redis", "spotify"), results):
        if isinstance(result, Exception):
            logger.warning("prewarm_failed", service=name, error=str(result))
        else:
            logger.debug("prewarm_ok", service=name)

    logger.info("gia_ready")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    await app.state.spotify.close()
    await app.state.redis.aclose()
    await asyncio.to_thread(app.state.weaviate.close)
    await engine.dispose()
    logger.info("gia_shutdown")


app = FastAPI(
    title="Gia",
    description="Voice Music Companion — backend API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(memory.router)
app.include_router(dj.router)
app.include_router(artist.router)
app.include_router(chat.router)
app.include_router(voice.router)
