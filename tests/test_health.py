"""Tests for GET /health and GET /health/llm.

Coverage targets
----------------
- 200 response shape when all services are healthy
- ``status: degraded`` when a dependency reports an error
- Redis ping delegated to the injected client
- LLM health check returns provider name on success
- LLM health check returns error string, not 500, on failure
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_200_when_all_ok(client: AsyncClient) -> None:
    """``/health`` responds 200 and all service statuses are ``ok``."""
    with (
        patch("backend.app.api.health._check_postgres", return_value="ok"),
        patch("backend.app.api.health._check_weaviate", return_value="ok"),
    ):
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["postgres"] == "ok"
    assert body["weaviate"] == "ok"
    assert body["redis"] == "ok"


@pytest.mark.asyncio
async def test_health_degraded_when_postgres_fails(client: AsyncClient) -> None:
    """``/health`` returns ``status: degraded`` if Postgres is unreachable."""
    with (
        patch("backend.app.api.health._check_postgres", return_value="error: connection refused"),
        patch("backend.app.api.health._check_weaviate", return_value="ok"),
    ):
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert "connection refused" in body["postgres"]


@pytest.mark.asyncio
async def test_health_degraded_when_weaviate_fails(client: AsyncClient) -> None:
    """``/health`` returns ``status: degraded`` if Weaviate is unreachable."""
    with (
        patch("backend.app.api.health._check_postgres", return_value="ok"),
        patch("backend.app.api.health._check_weaviate", return_value="error: timeout"),
    ):
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


@pytest.mark.asyncio
async def test_health_includes_config_flags(client: AsyncClient) -> None:
    """``/health`` response includes ``llm_provider`` and ``spotify_configured``."""
    with (
        patch("backend.app.api.health._check_postgres", return_value="ok"),
        patch("backend.app.api.health._check_weaviate", return_value="ok"),
    ):
        response = await client.get("/health")

    body = response.json()
    assert "llm_provider" in body
    assert "spotify_configured" in body


@pytest.mark.asyncio
async def test_health_redis_uses_injected_client(client: AsyncClient, fake_redis: AsyncMock) -> None:
    """``/health`` pings the injected Redis — never creates its own connection."""
    with (
        patch("backend.app.api.health._check_postgres", return_value="ok"),
        patch("backend.app.api.health._check_weaviate", return_value="ok"),
    ):
        await client.get("/health")

    fake_redis.ping.assert_called_once()


@pytest.mark.asyncio
async def test_health_llm_returns_provider(client: AsyncClient) -> None:
    """``/health/llm`` returns the active LLM provider name."""
    mock_llm = MagicMock()
    mock_llm.call.return_value = "ok"

    with patch("backend.app.providers.llm.get_fast_llm", return_value=mock_llm):
        response = await client.get("/health/llm")

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "anthropic"
    assert body["llm"] in ("ok", "no response")


@pytest.mark.asyncio
async def test_health_llm_returns_error_string_not_500(client: AsyncClient) -> None:
    """``/health/llm`` returns 200 with an error string, never a 500."""
    with patch("backend.app.providers.llm.get_fast_llm", side_effect=RuntimeError("API key invalid")):
        response = await client.get("/health/llm")

    assert response.status_code == 200
    assert "error" in response.json()["llm"]
    assert "API key invalid" in response.json()["llm"]
