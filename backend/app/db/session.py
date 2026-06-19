"""Async SQLAlchemy engine and session factory.

This module provides two module-level singletons:

``engine``
    The async SQLAlchemy engine connected to the configured database URL.
    Shared for the lifetime of the process; disposed in lifespan shutdown.

``AsyncSessionLocal``
    An ``async_sessionmaker`` bound to ``engine``.  Callers open individual
    sessions via ``async with AsyncSessionLocal() as session:``.

The ``get_db`` FastAPI dependency lives in ``backend.app.dependencies`` — not
here — so that the DI layer has a single home and the session factory stays
a pure infrastructure module.
"""

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
