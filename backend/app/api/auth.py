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

import base64
import hashlib
import json
import os
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

import httpx
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from redis.asyncio import Redis as AsyncRedis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import Settings
from backend.app.db.models import Profile
from backend.app.dependencies import get_db, get_redis, get_settings
from backend.app.observability.logging import get_logger
from backend.app.schemas.auth import SpotifyCallbackResponse, SpotifyStatusResponse

router = APIRouter(prefix="/auth/spotify", tags=["auth"])
logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
PKCE_STATE_TTL_SECONDS = 600  # 10 minutes

SCOPES = " ".join([
    "user-read-currently-playing",
    "user-read-recently-played",
    "user-top-read",
    "user-library-modify",
    "user-library-read",
    "streaming",
    "playlist-modify-public",
    "playlist-modify-private",
])

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
# Route handlers
# ==============================================================================


@router.get("/login", summary="Initiate Spotify OAuth 2.0 PKCE flow", status_code=302)
async def spotify_login(
    redis: Annotated[AsyncRedis, Depends(get_redis)],
    cfg: Annotated[Settings, Depends(get_settings)],
    user_id: str = Query(..., description="Internal user ID to associate the Spotify account with"),
) -> RedirectResponse:
    """Initiate the Spotify OAuth 2.0 PKCE flow for a specific user.

    Generates a code verifier and a cryptographic state token, stores both in
    Redis keyed by state (TTL = 10 min), then redirects the browser to
    Spotify's authorisation endpoint.

    Args:
        user_id: The internal Gia user ID whose ``Profile`` will receive the
                 OAuth tokens after the callback succeeds.
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
    response_model=SpotifyCallbackResponse,
    status_code=200,
)
async def spotify_callback(
    redis: Annotated[AsyncRedis, Depends(get_redis)],
    db: Annotated[AsyncSession, Depends(get_db)],
    cfg: Annotated[Settings, Depends(get_settings)],
    code: str = Query(default=""),
    state: str = Query(default=""),
    error: str = Query(default=""),
) -> SpotifyCallbackResponse:
    """Complete the PKCE exchange and persist tokens.

    Spotify redirects here after the user grants or denies permission.  This
    handler retrieves the PKCE state from Redis, exchanges the authorisation
    code for tokens, and stores the refresh token in the user's ``Profile``
    row.  The Redis state entry is deleted after a single use.

    Args:
        code:  The authorisation code returned by Spotify.
        state: The random state value from the initial redirect, used to look
               up the PKCE verifier and user ID in Redis.
        error: Set by Spotify if the user denied permission or another OAuth
               error occurred.
        redis: App-level Redis pool (injected).
        db:    Database session (injected).
        cfg:   Application settings (injected).

    Returns:
        A JSON body confirming success, including the access token expiry time.
        The refresh token is **not** included in the response body; it is stored
        server-side in the ``Profile`` table.

    Raises:
        HTTPException 400: If Spotify returned an error, the state is invalid
                           or expired, or the token exchange request fails.
    """
    if error:
        raise HTTPException(status_code=400, detail=f"Spotify denied authorisation: {error}")

    raw = await redis.get(f"{_REDIS_KEY_PREFIX}{state}")
    if not raw:
        raise HTTPException(status_code=400, detail="OAuth state is invalid or has expired (10-minute window).")

    # Consume state — single-use to prevent replay
    await redis.delete(f"{_REDIS_KEY_PREFIX}{state}")
    pkce_data = json.loads(raw)
    verifier: str = pkce_data["verifier"]
    user_id: str = pkce_data["user_id"]
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"user_id {user_id!r} is not a valid UUID.")

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

    # Persist tokens to Profile
    result = await db.execute(select(Profile).where(Profile.user_id == user_uuid))
    profile = result.scalar_one_or_none()
    if profile:
        profile.spotify_access_token = access_token
        profile.spotify_refresh_token = refresh_token
        profile.spotify_token_expires_at = expires_at
        await db.flush()

    logger.info("spotify_oauth_complete", user_id=user_id, expires_at=expires_at.isoformat())
    return SpotifyCallbackResponse(
        status="ok",
        user_id=user_id,
        expires_at=expires_at,
        note="Refresh token stored in Profile. Access token valid for 1 hour.",
    )


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
