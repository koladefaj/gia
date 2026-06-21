"""Alembic migration environment — async Postgres edition.

Uses SQLAlchemy's async engine so migrations run through the same asyncpg
driver the application uses at runtime.  The database URL is sourced from
``Settings`` (which reads from the environment / ``.env`` file) rather than
from ``alembic.ini``, so the same ``DATABASE_URL`` env var controls both the
app and the migration runner.

To generate a new migration after changing a model::

    uv run alembic revision --autogenerate -m "describe your change"

To apply all pending migrations::

    uv run alembic upgrade head

To roll back the last migration::

    uv run alembic downgrade -1
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import models so that Base.metadata is populated before autogenerate runs.
# The noqa comment suppresses "imported but unused" warnings — the import
# is the side-effect we need.
import backend.app.db.models  # noqa: F401
from alembic import context
from backend.app.config import settings
from backend.app.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without a live database connection.

    Emits SQL to stdout so you can inspect or pipe it elsewhere.
    """
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection):  # type: ignore[no-untyped-def]
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations through it."""
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = settings.database_url

    connectable = async_engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations with a live async database connection."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
