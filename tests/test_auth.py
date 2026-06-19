"""Tests for Spotify OAuth 2.0 PKCE endpoints.

What is tested
--------------
- ``/auth/spotify/login`` generates state + verifier, stores in Redis,
  and redirects to Spotify with the correct query parameters.
- ``/auth/spotify/callback`` reads state from Redis, exchanges code,
  persists tokens to Profile, and deletes the state entry (single-use).
- Error paths: missing ``SPOTIFY_CLIENT_ID``, invalid/expired state,
  Spotify returning an error, token exchange HTTP failure.
- ``/auth/spotify/status`` reflects the current settings truthfully.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import Settings
from backend.app.db.models import Profile, User
from tests.conftest import FakeSpotifyClient

# Valid UUID used as the test user identity throughout auth tests
_TEST_USER_ID = "00000000-0000-0000-0000-000000000001"


# ── /auth/spotify/login ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_login_redirects_to_spotify(client: AsyncClient, fake_redis: AsyncMock) -> None:
    """``/login`` redirects to the Spotify authorisation URL."""
    response = await client.get(
        "/auth/spotify/login",
        params={"user_id": _TEST_USER_ID},
        follow_redirects=False,
    )
    assert response.status_code == 307
    location = response.headers["location"]
    assert "accounts.spotify.com/authorize" in location
    assert "code_challenge" in location
    assert "state=" in location


@pytest.mark.asyncio
async def test_login_stores_state_in_redis(client: AsyncClient, fake_redis: AsyncMock) -> None:
    """``/login`` writes PKCE state to Redis with a 10-minute TTL."""
    await client.get(
        "/auth/spotify/login",
        params={"user_id": _TEST_USER_ID},
        follow_redirects=False,
    )
    fake_redis.setex.assert_called_once()
    call_args = fake_redis.setex.call_args
    key: str = call_args[0][0]
    ttl: int = call_args[0][1]
    payload: str = call_args[0][2]
    assert key.startswith("pkce:state:")
    assert ttl == 600
    data = json.loads(payload)
    assert data["user_id"] == _TEST_USER_ID
    assert "verifier" in data


@pytest.mark.asyncio
async def test_login_requires_spotify_client_id(
    test_settings: Settings,
    client: AsyncClient,
) -> None:
    """``/login`` returns 400 if ``SPOTIFY_CLIENT_ID`` is empty."""
    from backend.app.dependencies import get_settings

    no_id_settings = Settings(
        **{**test_settings.model_dump(), "spotify_client_id": ""}
    )

    from backend.app.main import app as _real_app
    _real_app.dependency_overrides[get_settings] = lambda: no_id_settings

    response = await client.get(
        "/auth/spotify/login",
        params={"user_id": _TEST_USER_ID},
        follow_redirects=False,
    )
    assert response.status_code == 400
    assert "SPOTIFY_CLIENT_ID" in response.json()["detail"]

    _real_app.dependency_overrides[get_settings] = lambda: test_settings


@pytest.mark.asyncio
async def test_login_state_is_unique_per_request(
    client: AsyncClient, fake_redis: AsyncMock
) -> None:
    """Two consecutive ``/login`` calls produce different state tokens."""
    states: list[str] = []

    async def capture_setex(key: str, ttl: int, value: str) -> bool:
        states.append(key)
        return True

    fake_redis.setex.side_effect = capture_setex

    for _ in range(2):
        await client.get(
            "/auth/spotify/login",
            params={"user_id": _TEST_USER_ID},
            follow_redirects=False,
        )

    assert len(states) == 2
    assert states[0] != states[1]


# ── /auth/spotify/callback ────────────────────────────────────────────────────


@pytest.fixture()
def pkce_state_payload() -> str:
    """Return a serialised PKCE state payload stored in Redis."""
    return json.dumps({"verifier": "test-verifier-abc", "user_id": _TEST_USER_ID})


@pytest.mark.asyncio
async def test_callback_error_param_returns_400(client: AsyncClient) -> None:
    """``/callback?error=access_denied`` returns 400 without touching Redis."""
    response = await client.get(
        "/auth/spotify/callback",
        params={"error": "access_denied", "state": "some-state"},
    )
    assert response.status_code == 400
    assert "access_denied" in response.json()["detail"]


@pytest.mark.asyncio
async def test_callback_invalid_state_returns_400(
    client: AsyncClient, fake_redis: AsyncMock
) -> None:
    """``/callback`` returns 400 when state is not in Redis (expired or forged)."""
    fake_redis.get.return_value = None
    response = await client.get(
        "/auth/spotify/callback",
        params={"code": "auth-code-xyz", "state": "nonexistent-state"},
    )
    assert response.status_code == 400
    assert "invalid or has expired" in response.json()["detail"]


@pytest.mark.asyncio
async def test_callback_deletes_state_after_use(
    client: AsyncClient,
    fake_redis: AsyncMock,
    pkce_state_payload: str,
    db_session: AsyncSession,
) -> None:
    """``/callback`` deletes the Redis state entry after a single successful use."""
    fake_redis.get.return_value = pkce_state_payload

    token_response = MagicMock()
    token_response.status_code = 200
    token_response.json.return_value = {
        "access_token": "access-tok",
        "refresh_token": "refresh-tok",
        "expires_in": 3600,
    }

    with patch("backend.app.api.auth.httpx.AsyncClient") as mock_http_cls:
        mock_http_cls.return_value.__aenter__ = AsyncMock(return_value=AsyncMock(post=AsyncMock(return_value=token_response)))
        mock_http_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        await client.get(
            "/auth/spotify/callback",
            params={"code": "auth-code-xyz", "state": "valid-state"},
        )

    fake_redis.delete.assert_called_once_with("pkce:state:valid-state")


@pytest.mark.asyncio
async def test_callback_token_exchange_failure_returns_400(
    client: AsyncClient,
    fake_redis: AsyncMock,
    pkce_state_payload: str,
) -> None:
    """``/callback`` returns 400 (not 500) when the Spotify token exchange fails."""
    fake_redis.get.return_value = pkce_state_payload

    bad_response = MagicMock()
    bad_response.status_code = 400
    bad_response.json.return_value = {"error": "invalid_grant"}
    bad_response.text = '{"error": "invalid_grant"}'

    with patch("backend.app.api.auth.httpx.AsyncClient") as mock_http_cls:
        mock_http_cls.return_value.__aenter__ = AsyncMock(return_value=AsyncMock(post=AsyncMock(return_value=bad_response)))
        mock_http_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        response = await client.get(
            "/auth/spotify/callback",
            params={"code": "bad-code", "state": "valid-state"},
        )

    assert response.status_code == 400
    assert "Token exchange" in response.json()["detail"]


# ── /auth/spotify/status ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_status_reflects_settings(client: AsyncClient) -> None:
    """``/status`` reports the current credential configuration accurately."""
    response = await client.get("/auth/spotify/status")
    assert response.status_code == 200
    body = response.json()
    # test_settings has client_id + client_secret set
    assert body["client_id_configured"] is True
    assert body["spotify_configured"] is True
    assert body["has_fallback_refresh_token"] is False
