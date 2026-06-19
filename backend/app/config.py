"""Application settings loaded from environment variables / .env file.

All configuration lives here via pydantic-settings.  No config values are
read anywhere else — callers inject ``Settings`` through ``Depends(get_settings)``.

Sensitive fields (API keys, secrets) are plain ``str`` with empty defaults so
the app starts without crashing; missing credentials surface as runtime errors
only when the affected feature is actually called, not at import time.

Field validators enforce that enumerated fields (provider names, log levels,
environments) fail fast with a clear error on startup rather than silently
using an invalid value that causes a cryptic error deep in business logic.
"""

import logging

from pydantic import field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application configuration.

    Populated from environment variables (case-insensitive) and the ``.env``
    file in the project root, in that order.  Environment variables take
    precedence over the file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # =============================================================================
    # Infrastructure
    # =============================================================================
    
    # --- App ---
    app_env: str = Field(default="development")
    log_level: str = Field(default="debug")
    secret_key: str = Field(default="change-me-in-production")

    # --- Database ---
    database_url: str = Field(default="postgresql+asyncpg://gia:gia@localhost:5432/gia")

    # --- Weaviate ---
    weaviate_url: str = Field(default="http://localhost:8080")

    # --- Redis ---
    redis_url: str = Field(default="redis://localhost:6379/0")

    # --- LLM ---
    llm_provider: str = Field(default="openai")  # one of "anthropic", "openai", "ollama"
    anthropic_api_key: str = Field(default="")
    openai_api_key: str  = Field(default="")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="gemma3:4b")
    # Override the default persona / fast models per-provider.  Leave empty to
    # use the built-in provider defaults defined in ``providers/llm.py``.
    llm_persona_model: str = Field(default="")
    llm_fast_model: str = Field(default="")

    # --- Spotify ---
    spotify_client_id: str = Field(default="")
    spotify_client_secret: str = Field(default="")
    spotify_redirect_uri: str = Field(default="http://localhost:8000/auth/spotify/callback")
    spotify_mcp_url: str = Field(default="http://localhost:3001")

    # --- TTS ---
    tts_provider: str = Field(default="kokoro")
    elevenlabs_api_key: str = Field(default="")
    elevenlabs_voice_id: str = Field(default="")

    # --- Brave Search ---
    brave_api_key: str = Field(default="")

    # --- Weather (Open-Meteo — no API key required) ---
    # When enabled, the planner can fetch current weather to make music
    # recommendations context-aware ("31°C — something for a shorter run?").
    weather_enabled: bool = Field(default=True)
    # Default coordinates used when the user has no location on file
    # (Lagos, Nigeria — matches the seeded demo user).
    weather_default_lat: float = Field(default=6.5244)
    weather_default_lon: float = Field(default=3.3792)
    weather_default_label: str = Field(default="Lagos")

    # --- Langfuse ---
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # --- Celery ---
    celery_broker_url: str = Field(default="redis://localhost:6379/1")
    celery_result_backend: str = Field(default="redis://localhost:6379/2")

    # =============================================================================
    # Retrieval (RAG hardening)
    # =============================================================================

    # Hybrid search: combine BM25 keyword + dense vector. Disable to fall back to
    # pure dense (the pre-hardening behaviour).
    hybrid_enabled: bool = Field(default=True)
    # alpha weights dense vs keyword in Weaviate hybrid: 1.0 = pure vector,
    # 0.0 = pure BM25, 0.5 = balanced.
    retrieval_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    # Top-k fetched per memory type during context assembly.
    retrieval_k_preferences: int = Field(default=8, ge=1, le=50)
    retrieval_k_mood: int = Field(default=3, ge=1, le=50)
    retrieval_k_episodes: int = Field(default=3, ge=1, le=50)
    # Redis retrieval cache TTL in seconds (0 disables caching).
    retrieval_cache_ttl: int = Field(default=60, ge=0)
    # Multi-agent synthesis — when ON, a final LLM pass merges several agents'
    # outputs into one coherent reply instead of concatenating them. OFF by
    # default: single-agent turns (the common case) need no merge, and it adds
    # an LLM call to the voice path.
    synthesis_enabled: bool = Field(default=False)

    # Cross-encoder reranking — OFF by default to protect the voice latency
    # budget. Flip on for Celery/eval or to demo the recall gain.
    rerank_enabled: bool = Field(default=False)
    rerank_model: str = Field(default="BAAI/bge-reranker-base")
    # When reranking, fetch this many candidates before trimming to the final k.
    rerank_candidate_multiplier: int = Field(default=3, ge=1, le=10)

    # =============================================================================
    # Tool resilience
    # =============================================================================

    # Per-call timeout (seconds) applied to external tool calls.
    tool_timeout_s: float = Field(default=8.0, gt=0.0)
    # Consecutive failures before a tool's circuit breaker opens.
    tool_circuit_threshold: int = Field(default=5, ge=1)
    # Seconds the breaker stays open before allowing a probe call.
    tool_circuit_cooldown_s: float = Field(default=30.0, gt=0.0)

    # --- Field validators ---
    @field_validator("llm_provider")
    @classmethod
    def _validate_llm_provider(cls, v: str) -> str:
        """Reject unknown LLM providers before they reach the factory.

        Args:
            v: Raw value from the environment.

        Returns:
            The validated provider string.

        Raises:
            ValueError: If *v* is not one of the supported provider names.
        """
        allowed = {"anthropic", "openai", "ollama"}
        if v not in allowed:
            raise ValueError(
                f"LLM_PROVIDER must be one of {sorted(allowed)!r}; got {v!r}"
            )
        return v

    @field_validator("app_env")
    @classmethod
    def _validate_app_env(cls, v: str) -> str:
        """Ensure ``APP_ENV`` is a recognised deployment stage.

        Args:
            v: Raw value from the environment.

        Returns:
            The validated environment string.

        Raises:
            ValueError: If *v* is not one of the allowed environment names.
        """
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(
                f"APP_ENV must be one of {sorted(allowed)!r}; got {v!r}"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        """Ensure ``LOG_LEVEL`` maps to a valid Python logging level.

        Args:
            v: Raw value from the environment (e.g. ``"info"``, ``"DEBUG"``).

        Returns:
            The lower-cased level string.

        Raises:
            ValueError: If *v* does not correspond to a ``logging`` constant.
        """
        if not hasattr(logging, v.upper()):
            raise ValueError(
                f"LOG_LEVEL {v!r} is not a valid Python log level "
                "(use debug, info, warning, error, or critical)"
            )
        return v.lower()

    # --- Derived properties ---

    @property
    def langfuse_enabled(self) -> bool:
        """Return ``True`` when both Langfuse keys are present."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def spotify_configured(self) -> bool:
        """Return ``True`` when Spotify OAuth credentials are in place."""
        return bool(self.spotify_client_id and self.spotify_client_secret)


settings = Settings()
