"""Pydantic response schemas for health-check endpoints."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response for ``GET /health``.

    Attributes:
        status:            ``"ok"`` when all services are healthy,
                           ``"degraded"`` when any dependency reports an error.
        postgres:          ``"ok"`` or an error string from the Postgres ping.
        weaviate:          ``"ok"`` or an error string from the Weaviate ping.
        redis:             ``"ok"`` or an error string from the Redis ping.
        llm_provider:      Active LLM provider name (e.g. ``"anthropic"``).
        spotify_configured: Whether Spotify OAuth credentials are present.
    """

    status: Literal["ok", "degraded"] = Field(..., description="Overall health status")
    postgres: str = Field(..., description="Postgres ping result")
    weaviate: str = Field(..., description="Weaviate ping result")
    redis: str = Field(..., description="Redis ping result")
    llm_provider: str = Field(..., description="Active LLM provider")
    spotify_configured: bool = Field(..., description="Spotify credentials present")


class LLMHealthResponse(BaseModel):
    """Response for ``GET /health/llm``.

    Attributes:
        llm:      ``"ok"``, ``"no response"``, or an error string.
        provider: The LLM provider that was tested (e.g. ``"anthropic"``).
    """

    llm: str = Field(..., description="LLM completion result")
    provider: str = Field(..., description="Provider that was tested")
