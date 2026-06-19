"""Pydantic response schemas for Spotify OAuth endpoints.

Keeping response shapes in typed models gives:
  - Automatic OpenAPI docs with real field descriptions.
  - FastAPI validation of what the handler returns (catches regressions).
  - Type-safe callers in tests — no more ``response.json()["field"]`` guesses.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class SpotifyCallbackResponse(BaseModel):
    """Response returned after a successful OAuth PKCE code exchange.

    Attributes:
        status:     Always ``"ok"`` on success.
        user_id:    The internal Gia user ID the tokens were saved for.
        expires_at: ISO-8601 timestamp when the access token expires.
        note:       Human-readable reminder that the refresh token is stored
                    server-side, not in this response.
    """

    status: str = Field("ok", description="Always 'ok' on success")
    user_id: str = Field(..., description="Internal Gia user ID")
    expires_at: datetime = Field(..., description="Access token expiry (UTC)")
    note: str = Field(..., description="Server-side storage note")


class SpotifyStatusResponse(BaseModel):
    """Response for the credential status diagnostic endpoint.

    Attributes:
        client_id_configured:      ``True`` when ``SPOTIFY_CLIENT_ID`` is set.
        spotify_configured:        ``True`` when both client ID and secret are set.
        has_fallback_refresh_token: ``True`` when a refresh token exists for
                                    the fallback single-user flow.
    """

    client_id_configured: bool = Field(..., description="SPOTIFY_CLIENT_ID is set")
    spotify_configured: bool = Field(..., description="Both client ID and secret are set")
    has_fallback_refresh_token: bool = Field(
        ..., description="A fallback refresh token exists in settings"
    )
