"""FastAPI dependency providers — the central DI wiring layer.

Every injectable resource (database session, Redis connection, Spotify client,
settings) is vended through a function here.  Route handlers and service
functions declare what they need via ``Depends()``.  Nothing is imported and
instantiated at call-site level, which means:

  - Tests swap implementations by overriding ``app.dependency_overrides``.
  - Concrete classes can be replaced without touching callers.
  - Lifetime is explicit: per-request sessions, app-level connection pools.

``get_db`` is defined here (not in ``session.py``) because it is a FastAPI
dependency, not an infrastructure primitive.  ``session.py`` owns the engine
and factory; this module owns the request-scoped lifecycle.

Usage example::

    @router.get("/tracks")
    async def list_tracks(
        spotify: SpotifyClientProtocol = Depends(get_spotify_client),
        db: AsyncSession = Depends(get_db),
    ) -> list[dict]:
        ...
"""

from collections.abc import AsyncGenerator

from fastapi import Depends, Request
from redis.asyncio import Redis as AsyncRedis
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import Settings, settings as _default_settings
from backend.app.db.session import AsyncSessionLocal
from backend.app.interfaces import SpotifyClientProtocol
from backend.app.observability.logging import get_logger

logger = get_logger(__name__)


# ── Settings ──────────────────────────────────────────────────────────────────


def get_settings() -> Settings:
    """Return the global ``Settings`` singleton.

    Overrideable in tests via ``app.dependency_overrides[get_settings]``.
    """
    return _default_settings


# ── Database ──────────────────────────────────────────────────────────────────


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a transactional ``AsyncSession`` for use in FastAPI dependencies.

    Commits on clean exit; rolls back and re-raises on any exception.  The
    session is closed automatically by the ``async with AsyncSessionLocal()``
    context manager — no explicit ``finally: session.close()`` is needed because
    ``AsyncSession.__aexit__`` calls ``close()`` regardless of outcome, returning
    the connection to the pool.

    Yields:
        An ``AsyncSession`` bound to an open database transaction.

    Raises:
        Exception: Any exception raised inside the route handler is re-raised
                   after rollback so FastAPI can return the appropriate HTTP
                   error response.

    Example::

        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)) -> list[Item]:
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            logger.warning("db_session_rollback")
            await session.rollback()
            raise


# ── Redis ─────────────────────────────────────────────────────────────────────


def get_redis(request: Request) -> AsyncRedis:
    """Return the app-level Redis connection pool stored on ``app.state``.

    The pool is created in ``lifespan`` and shared across all requests for
    connection efficiency.  Never create a per-request Redis connection.

    Raises:
        AttributeError: If called before the lifespan ``startup`` phase
            initialises ``app.state.redis``.
    """
    return request.app.state.redis  # type: ignore[return-value]


# ── Spotify client ────────────────────────────────────────────────────────────


def get_spotify_client(request: Request) -> SpotifyClientProtocol:
    """Return the ``SpotifyClientProtocol`` implementation stored at startup.

    In production this is a ``SpotifyMCPClient`` pointing at the MCP server.
    In tests, override with a ``FakeSpotifyClient`` via ``dependency_overrides``.

    Raises:
        AttributeError: If called before ``app.state.spotify`` is set.
    """
    return request.app.state.spotify  # type: ignore[return-value]
