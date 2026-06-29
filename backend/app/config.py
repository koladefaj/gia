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

from pydantic import Field, field_validator
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
    # Where the browser is sent back to after a successful Spotify OAuth flow.
    # The callback appends ``?user_id=…&connected=1`` so the SPA can adopt the
    # freshly-created identity. Must match the deployed frontend origin.
    frontend_url: str = Field(default="http://localhost:3000")

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

    # --- Per-component models (intent-aware voice pipeline) ---
    # All model names are config so they swap without code changes.  gpt-5.5 is
    # not generally available yet; planner/conversation default to gpt-4o and log
    # a fallback if a 5.5 name is set but unreachable (see providers/openai_stream).
    router_model: str = Field(default="gpt-4o-mini")
    memory_model: str = Field(default="gpt-4o-mini")
    # Memory embeddings via the OpenAI API (no local torch/sentence-transformers).
    # text-embedding-3-small is 1536-dim; change → recreate Weaviate + re-seed.
    embedding_model: str = Field(default="text-embedding-3-small")
    artist_model: str = Field(default="gpt-4o")
    planner_model: str = Field(default="gpt-4o")
    conversation_model: str = Field(default="gpt-4o")
    # Below this router confidence, escalate to the Planner instead of dispatching.
    router_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    # Tier-1 router: when the sub-ms keyword classifier is confident a turn is pure
    # conversation (a greeting/small-talk with zero music/artist/mood/queue signal),
    # skip the ~2s gpt-4o-mini router entirely and use a warm GENERAL_CHAT decision.
    # Conservative — only the unambiguous-chat case short-circuits; anything that
    # might need query resolution still goes to the LLM. Flip off to force the LLM.
    router_fast_path_enabled: bool = Field(default=True)
    # Tier-2: the distilled local classifier (frozen MiniLM + linear heads, see
    # ml/router). Predicts the categorical decision in ~20-40ms on CPU for the
    # confident no-query-resolution turns, skipping the ~1.4s LLM router. OFF by
    # default: needs sentence-transformers (torch) + the trained model present, so
    # it's opt-in for environments that have them. Degrades to the LLM if absent.
    router_local_enabled: bool = Field(default=False)

    # --- Spotify ---
    spotify_client_id: str = Field(default="")
    spotify_client_secret: str = Field(default="")
    spotify_redirect_uri: str = Field(default="http://localhost:8000/auth/spotify/callback")
    spotify_mcp_url: str = Field(default="http://localhost:3001")
    # Path to the built marcelmarais/spotify-mcp-server entrypoint (build/index.js).
    # When set, SpotifyMCPClient spawns it over stdio (MCP protocol). Empty = the
    # client stays inert and Spotify-dependent features degrade gracefully.
    spotify_mcp_server_path: str = Field(default="")
    spotify_mcp_command: str = Field(default="node")
    # Path to the MCP server's spotify-config.json (holds the OAuth tokens the
    # direct Web API client reuses for endpoints the MCP server lacks, e.g.
    # playlist creation via the new /v1/me/playlists). Empty = derive from
    # spotify_mcp_server_path's repo root.
    spotify_config_path: str = Field(default="")

    # --- TTS ---
    # ElevenLabs (HTTP, no local deps) is the default now. Set to "kokoro" only
    # with the local-tts extra installed for zero-cost local synthesis.
    tts_provider: str = Field(default="elevenlabs")
    elevenlabs_api_key: str = Field(default="")
    elevenlabs_voice_id: str = Field(default="")
    # Force eleven_v3 (warm, expressive) for EVERY line instead of the hybrid that
    # drops plain logistics sentences to the faster-but-flatter eleven_flash_v2_5.
    # ON for a consistently warm voice (the voice is the product); flash is ~1.1s
    # faster per turn, so flip OFF to trade warmth for latency.
    tts_force_v3: bool = Field(default=True)
    # Sentence-streaming TTS — synthesise the reply sentence by sentence as the
    # text becomes available (the server splits on punctuation boundaries) instead
    # of waiting for the whole reply, so the first sentence's audio plays while the
    # rest is still being generated/synthesised. Masks latency on both the
    # decomposed pipeline and the realtime path. The tradeoff is per-sentence v3
    # prosody (less surrounding context than a whole-reply call); flip OFF to
    # restore single-pass whole-reply synthesis (warmer prosody, slower first audio).
    tts_stream_sentences: bool = Field(default=True)

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
    retrieval_k_life_facts: int = Field(default=4, ge=1, le=50)
    # Synthesised higher-order insights (memory consolidation) — few, high-signal.
    retrieval_k_insights: int = Field(default=3, ge=1, le=50)
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

    # CrewAI multi-agent curation (Scout → Curator) — a real collaborative crew.
    # OFF by default and OFF the live voice path: inter-agent hand-off adds LLM
    # round-trips, so it belongs in enrichment / "deep pick" flows, not the
    # sub-second reply. See backend/app/agents/curator_crew.py.
    crewai_curator_enabled: bool = Field(default=False)

    # Speech-to-text. Two families behind one switch:
    #   Batch (record → upload → transcribe; serial, before /chat):
    #     "local"  — faster-whisper on the GPU (free, fast, English-only base.en)
    #     "openai" — Whisper API (whisper-1), better on accented English, paid
    #   Streaming (continuous audio over WS, interim + final transcripts):
    #     "deepgram"      — nova-3 streaming, ~300ms finals, cheap, strong accents
    #     "openai_stream" — OpenAI Realtime transcription (gpt-4o-mini-transcribe)
    # The streaming providers feed the WS /voice/stream endpoint and remove the
    # serial transcription wait from the voice path; the batch providers still
    # back the one-shot /voice/transcribe used as a fallback.
    stt_provider: str = Field(default="local")  # local | openai | deepgram | openai_stream
    # Local faster-whisper model. base.en is fast but English-only and weak on
    # heavy accents; "large-v3" (multilingual) is far more accurate and fits the
    # RTX 4060. Changing this re-downloads the model on next start.
    stt_model: str = Field(default="base.en")

    # --- Deepgram (streaming STT) ---
    deepgram_api_key: str = Field(default="")
    # Streaming model. We use Deepgram **Flux** (/v2/listen) — the conversational
    # model built for voice agents: it does end-of-turn detection itself and emits
    # eager/confirmed turn-end events, which is exactly the early-intent signal.
    # "flux-general-en" is English; "flux-general-multi" adds other languages.
    deepgram_model: str = Field(default="flux-general-en")
    # End-of-turn detection knobs (passed straight to Flux). eot_threshold gates
    # the high-confidence EndOfTurn (final); eager_eot_threshold gates the early
    # EagerEndOfTurn used for speculative replies. Higher eot = Flux waits for more
    # confidence you're actually done, so it stops cutting you off mid-sentence
    # (the tradeoff is the turn ends slightly later). 0.8 holds through natural
    # pauses; eot_timeout_ms caps how long it will wait so a real pause still ends
    # the turn. Lower eager = earlier prewarm but more false starts.
    deepgram_eot_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    deepgram_eager_eot_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # Max ms of trailing silence before Flux forces the turn end (so a high
    # eot_threshold never leaves the turn hanging).
    deepgram_eot_timeout_ms: int = Field(default=4000, ge=500, le=15000)

    # =============================================================================
    # Voice mode — decomposed pipeline vs. speech-to-speech (realtime)
    # =============================================================================

    # How a voice turn is run end-to-end. Two families behind one switch, mirroring
    # the stt_provider / tts_provider pattern:
    #   "pipeline" — the decomposed path: streaming STT → router cascade →
    #                specialist agents → streaming TTS. Every stage is observable
    #                and individually optimised (this is the default and the
    #                primary, fully-measured path).
    #   "realtime" — speech-to-speech: the browser streams PCM16 to the backend,
    #                which bridges to OpenAI's Realtime model (gpt-realtime). The
    #                model owns the turn end to end (native barge-in, turn-taking,
    #                prosody) and calls the SAME memory / DJ / artist / weather code
    #                as tools. No STT→text→TTS serialisation on the hot path.
    # The two modes share memory, Spotify, Brave, and Langfuse; they differ only in
    # who orchestrates the turn. See backend/app/providers/realtime.py.
    voice_mode: str = Field(default="pipeline")  # pipeline | realtime

    # OpenAI Realtime model used as the ears + brain (audio in, TEXT out — the
    # voice itself comes from ElevenLabs v3, see tts_provider). Config so it swaps
    # without code changes (the "all model names are config" rule): "gpt-realtime"
    # is the GA voice-agent model; set "gpt-realtime-2" once it's enabled on the
    # account for the reasoning-capable variant. Reuses ``openai_api_key``.
    realtime_model: str = Field(default="gpt-realtime")
    # Where the realtime reply's VOICE comes from:
    #   "elevenlabs" — gpt-realtime emits text, ElevenLabs v3 speaks it (the warm,
    #                  tagged brand voice; needs a working ElevenLabs key).
    #   "model"      — gpt-realtime speaks directly (pure speech-to-speech, lowest
    #                  latency, billed under OpenAI). Graceful fallback when
    #                  ElevenLabs is unavailable, at the cost of the brand voice.
    realtime_voice_source: str = Field(default="elevenlabs")  # elevenlabs | model
    # Output voice when realtime_voice_source == "model" (ignored for elevenlabs).
    # One of OpenAI's realtime voices (marin/cedar are the warmest GA additions).
    realtime_voice: str = Field(default="marin")
    # Turn detection. "semantic_vad" lets the model decide turn-ends from meaning
    # (best for natural barge-in / back-channel); "server_vad" is plain silence
    # detection. null is not exposed — the realtime path needs server-side turns.
    realtime_vad: str = Field(default="semantic_vad")  # semantic_vad | server_vad
    # Transcription model for the *user's* audio. Enabling input transcription
    # keeps the observability + memory pipeline alive in realtime mode: the user
    # text feeds Langfuse traces and the Celery memory extractor exactly as the
    # final transcript does in pipeline mode.
    realtime_transcription_model: str = Field(default="gpt-4o-mini-transcribe")

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

    @field_validator("voice_mode")
    @classmethod
    def _validate_voice_mode(cls, v: str) -> str:
        """Reject unknown voice modes before the endpoint branches on them.

        Args:
            v: Raw value from the environment.

        Returns:
            The lower-cased, validated voice mode.

        Raises:
            ValueError: If *v* is not ``pipeline`` or ``realtime``.
        """
        allowed = {"pipeline", "realtime"}
        v = v.lower()
        if v not in allowed:
            raise ValueError(
                f"VOICE_MODE must be one of {sorted(allowed)!r}; got {v!r}"
            )
        return v

    @field_validator("realtime_voice_source")
    @classmethod
    def _validate_realtime_voice_source(cls, v: str) -> str:
        """Ensure ``REALTIME_VOICE_SOURCE`` is ``elevenlabs`` or ``model``.

        Args:
            v: Raw value from the environment.

        Returns:
            The lower-cased, validated voice source.

        Raises:
            ValueError: If *v* is not one of the allowed sources.
        """
        allowed = {"elevenlabs", "model"}
        v = v.lower()
        if v not in allowed:
            raise ValueError(
                f"REALTIME_VOICE_SOURCE must be one of {sorted(allowed)!r}; got {v!r}"
            )
        return v

    @field_validator("realtime_vad")
    @classmethod
    def _validate_realtime_vad(cls, v: str) -> str:
        """Ensure ``REALTIME_VAD`` names a supported turn-detection strategy.

        Args:
            v: Raw value from the environment.

        Returns:
            The lower-cased, validated VAD strategy.

        Raises:
            ValueError: If *v* is not ``semantic_vad`` or ``server_vad``.
        """
        allowed = {"semantic_vad", "server_vad"}
        v = v.lower()
        if v not in allowed:
            raise ValueError(
                f"REALTIME_VAD must be one of {sorted(allowed)!r}; got {v!r}"
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
