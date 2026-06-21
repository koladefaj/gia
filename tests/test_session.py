"""Tests for the SQLAlchemy async session factory.

Validates the ``get_db`` dependency generator: that it yields an
``AsyncSession``, commits on success, and rolls back on exception.
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
async def test_get_db_yields_session() -> None:
    """``get_db`` yields an ``AsyncSession``."""
    from backend.app.dependencies import get_db

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("backend.app.dependencies.AsyncSessionLocal", return_value=mock_cm):
        gen = get_db()
        session = await gen.__anext__()
        assert session is mock_session
        # Clean up the generator
        with contextlib.suppress(StopAsyncIteration):
            await gen.aclose()


@pytest.mark.asyncio
async def test_get_db_commits_on_success() -> None:
    """``get_db`` calls ``session.commit()`` when no exception is raised."""
    from backend.app.dependencies import get_db

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("backend.app.dependencies.AsyncSessionLocal", return_value=mock_cm):
        async for _ in get_db():
            pass  # simulate route handler completing successfully

    mock_session.commit.assert_called_once()
    mock_session.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_get_db_rollbacks_on_exception() -> None:
    """``get_db`` calls ``session.rollback()`` when the handler raises.

    Uses ``athrow()`` to inject the exception into the suspended generator,
    which is how FastAPI's dependency injection propagates handler exceptions
    back through a ``yield`` dependency.
    """
    from backend.app.dependencies import get_db

    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("backend.app.dependencies.AsyncSessionLocal", return_value=mock_cm):
        gen = get_db()
        await gen.__anext__()  # advance to the yield point

        with pytest.raises(ValueError, match="handler error"):
            await gen.athrow(ValueError("handler error"))

    mock_session.rollback.assert_called_once()
    mock_session.commit.assert_not_called()
