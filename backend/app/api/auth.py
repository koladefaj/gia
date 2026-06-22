"""Spotify OAuth 2.0 PKCE flow — production-grade, multi-user.

Endpoints
---------
GET /auth/spotify/login
    Redirects the user to Spotify's authorisation page.  Accepts an optional
    ``user_id`` query parameter to associate the resulting token with a
    specific user record.  State and verifier are stored in Redis with a
    10-minute TTL so the server is stateless between the redirect and callback.

GET /auth/spotify/callback
    Exchanges the authorisation code for access and refresh tokens, then
    persists the refresh token to the ``Profile`` table so it survives restarts.

GET /auth/spotify/status
    Non-authenticated diagnostic — reports whether credentials are configured.

Security notes
--------------
- PKCE (S256 challenge) prevents authorisation code interception attacks.
- State is cryptographically random; the Redis TTL limits replay windows.
- Tokens are stored in the database keyed by ``user_id``, never in a cookie
  or in-process memory, so horizontal scaling is safe.
"""

import asyncio
import base64
import hashlib
import json
import os
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from redis.asyncio import Redis as AsyncRedis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from weaviate import WeaviateClient

from backend.app.config import Settings
from backend.app.db.models import Profile, User
from backend.app.dependencies import (
    get_db,
    get_redis,
    get_settings,
    get_spotify_client,
    get_weaviate_client,
)
from backend.app.interfaces import SpotifyClientProtocol
from backend.app.memory.profiler import bootstrap_taste_profile
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.observability.logging import get_logger
from backend.app.schemas.auth import SpotifyStatusResponse

router = APIRouter(prefix="/auth/spotify", tags=["auth"])
logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
PKCE_STATE_TTL_SECONDS = 600  # 10 minutes

SCOPES = " ".join([
    "user-read-email",
    "user-read-private",
    "user-read-currently-playing",
    "user-read-recently-played",
    "user-top-read",
    "user-library-modify",
    "user-library-read",
    "streaming",
    "playlist-modify-public",
    "playlist-modify-private",
])

SPOTIFY_ME_URL = "https://api.spotify.com/v1/me"

_REDIS_KEY_PREFIX = "pkce:state:"


# ==============================================================================
# PKCE helpers
# ==============================================================================


def _generate_code_verifier() -> str:
    """Return a cryptographically random PKCE code verifier (43–128 chars).

    The verifier is base64url-encoded with padding stripped, as required by
    RFC 7636 §4.1.
    """
    return base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode()


def _code_challenge(verifier: str) -> str:
    """Derive the S256 code challenge from *verifier*.

    Computes ``BASE64URL(SHA256(ASCII(verifier)))`` per RFC 7636 §4.2.

    Args:
        verifier: A previously generated PKCE code verifier string.

    Returns:
        The base64url-encoded SHA-256 hash of the verifier, with no padding.
    """
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


# ==============================================================================
# Onboarding helpers
# ==============================================================================

# Strong references to in-flight background tasks so the event loop does not
# garbage-collect them mid-run (asyncio holds only weak refs to bare tasks).
_BG_TASKS: set[asyncio.Task] = set()


async def _fetch_spotify_identity(access_token: str) -> dict:
    """Return the authenticated user's Spotify profile (``GET /v1/me``).

    Args:
        access_token: A freshly minted Spotify access token.

    Returns:
        The decoded ``/me`` JSON (``id``, ``display_name``, ``email``, …).

    Raises:
        HTTPException 400: If the profile request fails.
    """
    async with httpx.AsyncClient() as http:
        resp = await http.get(
            SPOTIFY_ME_URL, headers={"Authorization": f"Bearer {access_token}"}
        )
    if resp.status_code != 200:
        logger.error("spotify_me_failed", status=resp.status_code, body=resp.text)
        raise HTTPException(status_code=400, detail="Could not read your Spotify profile.")
    return resp.json()


async def _upsert_user_from_spotify(
    db: AsyncSession,
    *,
    requested_user_id: str | None,
    me: dict,
    access_token: str,
    refresh_token: str,
    expires_at: datetime,
) -> tuple[uuid.UUID, bool]:
    """Find-or-create the Gia user behind a Spotify identity and store tokens.

    Resolution order:
      1. An explicit ``requested_user_id`` (linking flow) — created if absent.
      2. An existing ``Profile`` matching the Spotify account (returning user).
      3. A brand-new ``User`` + ``Profile`` (first-time sign-in).

    Args:
        db:                Request-scoped database session.
        requested_user_id: Optional existing Gia user ID to link to.
        me:                The Spotify ``/me`` payload.
        access_token:      Short-lived Spotify access token.
        refresh_token:     Long-lived Spotify refresh token (may be empty).
        expires_at:        UTC expiry of the access token.

    Returns:
        ``(user_id, created)`` — the resolved user UUID and whether a new
        account was created this call (drives the one-time taste bootstrap).

    Raises:
        HTTPException 400: If ``requested_user_id`` is not a valid UUID.
    """
    spotify_user_id = me.get("id") or ""
    display_name = me.get("display_name")
    email = me.get("email")

    profile: Profile | None = None
    created = False

    if requested_user_id:
        try:
            user_uuid = uuid.UUID(requested_user_id)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"user_id {requested_user_id!r} is not a valid UUID."
            ) from None
        profile = (
            await db.execute(select(Profile).where(Profile.user_id == user_uuid))
        ).scalar_one_or_none()
        if profile is None:
            db.add(User(id=user_uuid, email=email))
            profile = Profile(user_id=user_uuid)
            db.add(profile)
            created = True

    if profile is None and spotify_user_id:
        profile = (
            await db.execute(
                select(Profile).where(Profile.spotify_user_id == spotify_user_id)
            )
        ).scalar_one_or_none()

    if profile is None:
        # Reuse an existing user with this email (returning user who cleared
        # local state); only match on a real address so null emails never
        # collapse distinct accounts. Otherwise create a fresh user.
        user: User | None = None
        if email:
            user = (
                await db.execute(select(User).where(User.email == email))
            ).scalar_one_or_none()
        if user is None:
            user = User(email=email)
            db.add(user)
            await db.flush()  # assign user.id before the Profile FK references it
        profile = Profile(user_id=user.id)
        db.add(profile)
        created = True

    profile.spotify_user_id = spotify_user_id or profile.spotify_user_id
    profile.display_name = display_name or profile.display_name
    profile.spotify_access_token = access_token
    profile.spotify_refresh_token = refresh_token or profile.spotify_refresh_token
    profile.spotify_token_expires_at = expires_at
    await db.flush()
    return profile.user_id, created


def _schedule_taste_bootstrap(
    user_id: str,
    *,
    spotify: SpotifyClientProtocol,
    weaviate: WeaviateClient,
    redis: AsyncRedis,
    cfg: Settings,
) -> None:
    """Kick off the one-time taste-profile bootstrap without blocking the redirect.

    The bootstrap only touches app-level singletons (Weaviate, Redis, the
    Spotify MCP client), never the request-scoped DB session, so it is safe to
    run after the response has been sent. Failures are logged, never surfaced.
    """

    async def _run() -> None:
        try:
            store = WeaviateMemoryStore(client=weaviate)
            stored = await bootstrap_taste_profile(
                user_id, spotify=spotify, store=store, redis=redis, cfg=cfg
            )
            logger.info("spotify_bootstrap_done", user_id=user_id, stored=len(stored))
        except Exception as exc:  # noqa: BLE001 — best-effort, must not crash
            logger.warning("spotify_bootstrap_error", user_id=user_id, error=str(exc))

    task = asyncio.create_task(_run())
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)


# ==============================================================================
# Route handlers
# ==============================================================================


@router.get("/login", summary="Initiate Spotify OAuth 2.0 PKCE flow", status_code=302)
async def spotify_login(
    redis: Annotated[AsyncRedis, Depends(get_redis)],
    cfg: Annotated[Settings, Depends(get_settings)],
    user_id: str | None = Query(
        default=None,
        description=(
            "Existing Gia user ID to link this Spotify account to. Omit for a "
            "fresh sign-in — the callback creates the user from the Spotify "
            "identity."
        ),
    ),
) -> RedirectResponse:
    """Initiate the Spotify OAuth 2.0 PKCE flow.

    Generates a code verifier and a cryptographic state token, stores both in
    Redis keyed by state (TTL = 10 min), then redirects the browser to
    Spotify's authorisation endpoint.

    Two modes:
      * **Fresh sign-in** (``user_id`` omitted) — the callback reads the
        Spotify identity from ``/me`` and creates a ``User`` + ``Profile`` on
        the fly, so a first-time visitor is onboarded end to end.
      * **Link** (``user_id`` supplied) — the resulting tokens are attached to
        that existing user.

    Args:
        user_id: Optional existing Gia user ID to link the Spotify account to.
        redis:   App-level Redis pool (injected).
        cfg:     Application settings (injected).

    Returns:
        A 302 redirect to the Spotify authorisation page.

    Raises:
        HTTPException 400: If ``SPOTIFY_CLIENT_ID`` is not configured.
    """
    if not cfg.spotify_client_id:
        raise HTTPException(
            status_code=400,
            detail="SPOTIFY_CLIENT_ID is not configured. Set it in your .env file.",
        )

    state = secrets.token_urlsafe(16)
    verifier = _generate_code_verifier()
    payload = json.dumps({"verifier": verifier, "user_id": user_id})
    await redis.setex(f"{_REDIS_KEY_PREFIX}{state}", PKCE_STATE_TTL_SECONDS, payload)

    params = {
        "client_id": cfg.spotify_client_id,
        "response_type": "code",
        "redirect_uri": cfg.spotify_redirect_uri,
        "scope": SCOPES,
        "state": state,
        "code_challenge_method": "S256",
        "code_challenge": _code_challenge(verifier),
    }
    redirect_url = f"{SPOTIFY_AUTH_URL}?{urlencode(params)}"
    logger.info("spotify_oauth_initiated", user_id=user_id, state=state)
    return RedirectResponse(redirect_url)


@router.get(
    "/callback",
    summary="Handle Spotify OAuth 2.0 callback",
    status_code=302,
)
async def spotify_callback(
    redis: Annotated[AsyncRedis, Depends(get_redis)],
    db: Annotated[AsyncSession, Depends(get_db)],
    cfg: Annotated[Settings, Depends(get_settings)],
    spotify: Annotated[SpotifyClientProtocol, Depends(get_spotify_client)],
    weaviate: Annotated[WeaviateClient, Depends(get_weaviate_client)],
    code: str = Query(default=""),
    state: str = Query(default=""),
    error: str = Query(default=""),
) -> RedirectResponse:
    """Complete the PKCE exchange, onboard the user, and return to the frontend.

    Spotify redirects here after the user grants or denies permission. This
    handler:
      1. Validates and consumes the single-use PKCE state from Redis.
      2. Exchanges the authorisation code for access + refresh tokens.
      3. Reads the Spotify identity from ``/me`` and find-or-creates the Gia
         ``User`` + ``Profile`` (so first-time visitors are fully onboarded).
      4. For a brand-new account, schedules a one-time taste-profile bootstrap.
      5. Redirects the browser back to the SPA with the resolved ``user_id`` so
         the frontend can adopt the identity and start talking.

    Args:
        code:  The authorisation code returned by Spotify.
        state: The random state value from the initial redirect, used to look
               up the PKCE verifier and user ID in Redis.
        error: Set by Spotify if the user denied permission or another OAuth
               error occurred.

    Returns:
        A 302 redirect to ``{frontend_url}/?user_id=…&connected=1``.

    Raises:
        HTTPException 400: If Spotify returned an error, the state is invalid
                           or expired, or the token exchange request fails.
    """
    if error:
        raise HTTPException(status_code=400, detail=f"Spotify denied authorisation: {error}")

    logger.info("spotify_callback_received", has_code=bool(code), has_state=bool(state))

    raw = await redis.get(f"{_REDIS_KEY_PREFIX}{state}")
    if not raw:
        raise HTTPException(status_code=400, detail="OAuth state is invalid or has expired (10-minute window).")

    # Consume state — single-use to prevent replay
    await redis.delete(f"{_REDIS_KEY_PREFIX}{state}")
    pkce_data = json.loads(raw)
    verifier: str = pkce_data["verifier"]
    requested_user_id: str | None = pkce_data.get("user_id")

    try:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                SPOTIFY_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": cfg.spotify_redirect_uri,
                    "client_id": cfg.spotify_client_id,
                    "code_verifier": verifier,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code != 200:
                logger.error("spotify_token_exchange_failed", status=resp.status_code, body=resp.text)
                raise HTTPException(status_code=400, detail="Token exchange with Spotify failed.")

        token_data = resp.json()
        access_token: str = token_data["access_token"]
        refresh_token: str = token_data.get("refresh_token", "")
        expires_in: int = token_data.get("expires_in", 3600)
        expires_at = datetime.now(UTC) + timedelta(seconds=expires_in)

        me = await _fetch_spotify_identity(access_token)
        user_uuid, created = await _upsert_user_from_spotify(
            db,
            requested_user_id=requested_user_id,
            me=me,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
        )
    except HTTPException:
        raise
    except Exception as exc:
        # Surface the real cause in the logs (message + type, plus traceback)
        # and return the user to the app with an error flag, not a dead 500.
        logger.exception(
            "spotify_onboard_failed", error=str(exc), kind=type(exc).__name__
        )
        err = urlencode({"error": "onboarding_failed"})
        return RedirectResponse(url=f"{cfg.frontend_url}/?{err}", status_code=302)
    user_id = str(user_uuid)

    # A fresh account starts cold — warm it from their real listening so the
    # very first conversation is already personalised. Best-effort, off the
    # request path so the redirect is instant.
    if created:
        _schedule_taste_bootstrap(
            user_id, spotify=spotify, weaviate=weaviate, redis=redis, cfg=cfg
        )

    logger.info(
        "spotify_oauth_complete",
        user_id=user_id,
        created=created,
        expires_at=expires_at.isoformat(),
    )
    query = urlencode({"user_id": user_id, "connected": "1"})
    return RedirectResponse(url=f"{cfg.frontend_url}/?{query}", status_code=302)


@router.get(
    "/status",
    summary="Check Spotify credential configuration",
    response_model=SpotifyStatusResponse,
    status_code=200,
)
async def spotify_status(cfg: Annotated[Settings, Depends(get_settings)]) -> SpotifyStatusResponse:
    """Return the current Spotify credential configuration state.

    Non-authenticated diagnostic endpoint used during setup to confirm that
    environment variables are in place before attempting a live OAuth flow.

    Args:
        cfg: Application settings (injected).

    Returns:
        Dict with booleans indicating which credentials are configured.
    """
    return SpotifyStatusResponse(
        client_id_configured=bool(cfg.spotify_client_id),
        spotify_configured=cfg.spotify_configured,
        has_fallback_refresh_token=False,
    )
