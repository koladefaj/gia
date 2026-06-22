"""Tests for Spotify OAuth 2.0 PKCE endpoints.

What is tested
--------------
- ``/auth/spotify/login`` generates state + verifier, stores in Redis,
  and redirects to Spotify with the correct query parameters.
- ``/auth/spotify/callback`` reads state from Redis, exchanges code, reads the
  Spotify identity from ``/me``, find-or-creates the ``User`` + ``Profile``,
  persists tokens, deletes the state entry (single-use), and redirects back to
  the frontend with the resolved ``user_id``.
- Error paths: missing ``SPOTIFY_CLIENT_ID``, invalid/expired state,
  Spotify returning an error, token exchange HTTP failure.
- ``/auth/spotify/status`` reflects the current settings truthfully.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import Settings
from backend.app.db.models import Profile, User

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
    """Return a serialised PKCE state payload for an existing-user link flow."""
    return json.dumps({"verifier": "test-verifier-abc", "user_id": _TEST_USER_ID})


@pytest.fixture()
def pkce_state_signup() -> str:
    """Return a serialised PKCE state payload for a fresh sign-in (no user_id)."""
    return json.dumps({"verifier": "test-verifier-abc", "user_id": None})


def _mock_httpx(
    *,
    token_json: dict | None = None,
    me_json: dict | None = None,
    token_status: int = 200,
    me_status: int = 200,
) -> MagicMock:
    """Build a patched ``httpx.AsyncClient`` whose ``post`` returns the token
    response and ``get`` returns the ``/me`` identity response.
    """
    token_resp = MagicMock(status_code=token_status)
    token_resp.json.return_value = token_json or {
        "access_token": "access-tok",
        "refresh_token": "refresh-tok",
        "expires_in": 3600,
    }
    token_resp.text = json.dumps(token_resp.json.return_value)

    me_resp = MagicMock(status_code=me_status)
    me_resp.json.return_value = me_json or {
        "id": "spotify-abc",
        "display_name": "Ada",
        "email": "ada@example.com",
    }
    me_resp.text = json.dumps(me_resp.json.return_value)

    inst = AsyncMock()
    inst.post = AsyncMock(return_value=token_resp)
    inst.get = AsyncMock(return_value=me_resp)

    cls = MagicMock()
    cls.return_value.__aenter__ = AsyncMock(return_value=inst)
    cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return cls


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
) -> None:
    """``/callback`` deletes the Redis state entry after a single successful use."""
    fake_redis.get.return_value = pkce_state_payload

    with patch("backend.app.api.auth.httpx.AsyncClient", _mock_httpx()), patch(
        "backend.app.api.auth._schedule_taste_bootstrap"
    ):
        await client.get(
            "/auth/spotify/callback",
            params={"code": "auth-code-xyz", "state": "valid-state"},
            follow_redirects=False,
        )

    fake_redis.delete.assert_called_once_with("pkce:state:valid-state")


@pytest.mark.asyncio
async def test_callback_redirects_to_frontend(
    client: AsyncClient,
    fake_redis: AsyncMock,
    pkce_state_payload: str,
) -> None:
    """A successful ``/callback`` 302-redirects to the frontend with the user_id."""
    fake_redis.get.return_value = pkce_state_payload

    with patch("backend.app.api.auth.httpx.AsyncClient", _mock_httpx()), patch(
        "backend.app.api.auth._schedule_taste_bootstrap"
    ):
        response = await client.get(
            "/auth/spotify/callback",
            params={"code": "auth-code-xyz", "state": "valid-state"},
            follow_redirects=False,
        )

    assert response.status_code == 302
    location = response.headers["location"]
    assert location.startswith("http://localhost:3000/?")
    assert f"user_id={_TEST_USER_ID}" in location
    assert "connected=1" in location


@pytest.mark.asyncio
async def test_callback_creates_new_user_on_fresh_signin(
    client: AsyncClient,
    fake_redis: AsyncMock,
    pkce_state_signup: str,
    db_session: AsyncSession,
) -> None:
    """With no ``user_id`` in state, ``/callback`` creates User + Profile from /me."""
    fake_redis.get.return_value = pkce_state_signup

    with patch("backend.app.api.auth.httpx.AsyncClient", _mock_httpx()), patch(
        "backend.app.api.auth._schedule_taste_bootstrap"
    ) as bootstrap:
        response = await client.get(
            "/auth/spotify/callback",
            params={"code": "auth-code-xyz", "state": "valid-state"},
            follow_redirects=False,
        )

    assert response.status_code == 302

    profile = (
        await db_session.execute(
            select(Profile).where(Profile.spotify_user_id == "spotify-abc")
        )
    ).scalar_one()
    assert profile.display_name == "Ada"
    assert profile.spotify_access_token == "access-tok"
    assert profile.spotify_refresh_token == "refresh-tok"

    user = (
        await db_session.execute(select(User).where(User.id == profile.user_id))
    ).scalar_one()
    assert user.email == "ada@example.com"

    # A brand-new account triggers the one-time taste bootstrap.
    bootstrap.assert_called_once()


@pytest.mark.asyncio
async def test_callback_returning_user_updates_tokens(
    client: AsyncClient,
    fake_redis: AsyncMock,
    pkce_state_signup: str,
    db_session: AsyncSession,
) -> None:
    """A returning Spotify user is matched by spotify_user_id; no second account."""
    import uuid

    existing_user = User(id=uuid.uuid4(), email="ada@example.com")
    db_session.add(existing_user)
    await db_session.flush()
    db_session.add(
        Profile(
            user_id=existing_user.id,
            spotify_user_id="spotify-abc",
            spotify_access_token="old-tok",
        )
    )
    await db_session.flush()

    fake_redis.get.return_value = pkce_state_signup

    with patch("backend.app.api.auth.httpx.AsyncClient", _mock_httpx()), patch(
        "backend.app.api.auth._schedule_taste_bootstrap"
    ) as bootstrap:
        response = await client.get(
            "/auth/spotify/callback",
            params={"code": "auth-code-xyz", "state": "valid-state"},
            follow_redirects=False,
        )

    assert response.status_code == 302
    assert f"user_id={existing_user.id}" in response.headers["location"]

    profiles = (
        await db_session.execute(
            select(Profile).where(Profile.spotify_user_id == "spotify-abc")
        )
    ).scalars().all()
    assert len(profiles) == 1
    assert profiles[0].spotify_access_token == "access-tok"  # refreshed

    # An existing account is not re-bootstrapped.
    bootstrap.assert_not_called()


@pytest.mark.asyncio
async def test_callback_token_exchange_failure_returns_400(
    client: AsyncClient,
    fake_redis: AsyncMock,
    pkce_state_payload: str,
) -> None:
    """``/callback`` returns 400 (not 500) when the Spotify token exchange fails."""
    fake_redis.get.return_value = pkce_state_payload

    with patch(
        "backend.app.api.auth.httpx.AsyncClient",
        _mock_httpx(token_json={"error": "invalid_grant"}, token_status=400),
    ):
        response = await client.get(
            "/auth/spotify/callback",
            params={"code": "bad-code", "state": "valid-state"},
            follow_redirects=False,
        )

    assert response.status_code == 400
    assert "Token exchange" in response.json()["detail"]


# ── Onboarding helpers (direct) ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_spotify_identity_failure_raises() -> None:
    """``_fetch_spotify_identity`` raises 400 when ``/me`` is not 200."""
    from fastapi import HTTPException

    from backend.app.api import auth

    bad = MagicMock(status_code=401)
    bad.text = "unauthorized"
    inst = AsyncMock()
    inst.get = AsyncMock(return_value=bad)
    cls = MagicMock()
    cls.return_value.__aenter__ = AsyncMock(return_value=inst)
    cls.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("backend.app.api.auth.httpx.AsyncClient", cls):
        with pytest.raises(HTTPException) as exc:
            await auth._fetch_spotify_identity("tok")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_upsert_invalid_requested_user_id_raises(
    db_session: AsyncSession,
) -> None:
    """A non-UUID ``requested_user_id`` is rejected with 400."""
    import datetime as _dt

    from fastapi import HTTPException

    from backend.app.api import auth

    with pytest.raises(HTTPException) as exc:
        await auth._upsert_user_from_spotify(
            db_session,
            requested_user_id="not-a-uuid",
            me={"id": "sp"},
            access_token="a",
            refresh_token="r",
            expires_at=_dt.datetime.now(_dt.UTC),
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_upsert_links_requested_user_without_profile(
    db_session: AsyncSession,
) -> None:
    """A supplied user_id with no existing row creates both User and Profile."""
    import datetime as _dt
    import uuid

    from backend.app.api import auth

    uid = uuid.uuid4()
    user_id, created = await auth._upsert_user_from_spotify(
        db_session,
        requested_user_id=str(uid),
        me={"id": "sp-link", "display_name": "Lex", "email": "lex@x.com"},
        access_token="a",
        refresh_token="r",
        expires_at=_dt.datetime.now(_dt.UTC),
    )
    assert user_id == uid
    assert created is True
    profile = (
        await db_session.execute(select(Profile).where(Profile.user_id == uid))
    ).scalar_one()
    assert profile.spotify_user_id == "sp-link"


@pytest.mark.asyncio
async def test_upsert_matches_returning_profile(db_session: AsyncSession) -> None:
    """With no user_id, an existing Spotify profile is matched, not duplicated."""
    import datetime as _dt
    import uuid

    from backend.app.api import auth

    existing = User(id=uuid.uuid4(), email="r@x.com")
    db_session.add(existing)
    await db_session.flush()
    db_session.add(Profile(user_id=existing.id, spotify_user_id="sp-return"))
    await db_session.flush()

    user_id, created = await auth._upsert_user_from_spotify(
        db_session,
        requested_user_id=None,
        me={"id": "sp-return", "display_name": "R"},
        access_token="new-a",
        refresh_token="new-r",
        expires_at=_dt.datetime.now(_dt.UTC),
    )
    assert user_id == existing.id
    assert created is False


@pytest.mark.asyncio
async def test_schedule_taste_bootstrap_runs(
    fake_spotify, fake_redis: AsyncMock
) -> None:
    """``_schedule_taste_bootstrap`` spawns a task that runs the bootstrap."""
    import asyncio
    import uuid

    from backend.app.api import auth

    called = {}

    async def fake_bootstrap(user_id, **kwargs):
        called["user_id"] = user_id
        return ["mem-1", "mem-2"]

    with patch("backend.app.api.auth.bootstrap_taste_profile", fake_bootstrap), patch(
        "backend.app.api.auth.WeaviateMemoryStore"
    ):
        auth._schedule_taste_bootstrap(
            str(uuid.uuid4()),
            spotify=fake_spotify,
            weaviate=MagicMock(),
            redis=fake_redis,
            cfg=Settings(),
        )
        # Let the scheduled task run to completion.
        await asyncio.gather(*list(auth._BG_TASKS))

    assert "user_id" in called


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
