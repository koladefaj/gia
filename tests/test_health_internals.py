"""Tests for health-check helper functions.

Covers ``_check_postgres``, ``_check_weaviate``, and ``_check_redis`` in
isolation so the unit for each is clear and the integration test (which
exercises them together via ``/health``) does not have to cover error paths.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.api.health import _check_postgres, _check_redis, _check_weaviate

# ── _check_postgres ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_postgres_ok() -> None:
    """Returns ``"ok"`` when the database query succeeds."""
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("backend.app.api.health.AsyncSessionLocal", return_value=mock_cm):
        result = await _check_postgres()

    assert result == "ok"


@pytest.mark.asyncio
async def test_check_postgres_returns_error_string_on_failure() -> None:
    """Returns an error string (not an exception) when Postgres is unreachable."""
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(side_effect=ConnectionRefusedError("connection refused"))
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("backend.app.api.health.AsyncSessionLocal", return_value=mock_cm):
        result = await _check_postgres()

    assert result.startswith("error:")
    assert "connection refused" in result


# ── _check_weaviate ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_weaviate_ok() -> None:
    """Returns ``"ok"`` when the Weaviate ``is_ready()`` call succeeds."""
    mock_client = MagicMock()
    mock_client.is_ready = MagicMock()
    mock_client.close = MagicMock()

    with patch("backend.app.api.health.weaviate.connect_to_custom", return_value=mock_client):
        result = await _check_weaviate("http://weaviate:8080")

    assert result == "ok"
    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_check_weaviate_returns_error_string_on_failure() -> None:
    """Returns an error string when Weaviate is unreachable."""
    with patch(
        "backend.app.api.health.weaviate.connect_to_custom",
        side_effect=ConnectionError("timed out"),
    ):
        result = await _check_weaviate("http://weaviate:8080")

    assert result.startswith("error:")
    assert "timed out" in result


@pytest.mark.asyncio
async def test_check_weaviate_parses_url_correctly() -> None:
    """``_check_weaviate`` extracts host and port from the URL."""
    captured: list[dict] = []

    mock_client = MagicMock()
    mock_client.is_ready = MagicMock()
    mock_client.close = MagicMock()

    def fake_connect(**kwargs: object) -> MagicMock:
        captured.append(dict(kwargs))
        return mock_client

    with patch("backend.app.api.health.weaviate.connect_to_custom", side_effect=fake_connect):
        await _check_weaviate("http://weaviate-host:9090")

    assert captured[0]["http_host"] == "weaviate-host"
    assert captured[0]["http_port"] == 9090


# ── _check_redis ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_redis_ok(fake_redis: AsyncMock) -> None:
    """Returns ``"ok"`` when the Redis ping succeeds."""
    fake_redis.ping.return_value = True
    result = await _check_redis(fake_redis)
    assert result == "ok"


@pytest.mark.asyncio
async def test_check_redis_returns_error_string_on_failure(fake_redis: AsyncMock) -> None:
    """Returns an error string when Redis ping raises."""
    fake_redis.ping.side_effect = ConnectionError("redis unreachable")
    result = await _check_redis(fake_redis)
    assert result.startswith("error:")
    assert "redis unreachable" in result
