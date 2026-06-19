"""Tests for ``Settings`` and its derived properties.

Validates loading, defaults, derived properties, and the behaviour of
``langfuse_enabled`` / ``spotify_configured`` under different configurations.
"""

from __future__ import annotations

import pytest

from backend.app.config import Settings


def test_settings_defaults_are_safe() -> None:
    """Default settings must not crash at import and must have sensible values."""
    cfg = Settings(
        database_url="postgresql+asyncpg://x:x@localhost/test",
        weaviate_url="http://localhost:8080",
        redis_url="redis://localhost:6379/0",
    )
    assert cfg.app_env == "development"
    assert cfg.log_level == "debug"
    assert cfg.llm_provider == "openai"


def test_langfuse_enabled_requires_both_keys() -> None:
    """``langfuse_enabled`` is ``False`` unless both public and secret keys are set."""
    cfg_no_keys = Settings(
        langfuse_public_key="",
        langfuse_secret_key="",
        database_url="postgresql+asyncpg://x:x@localhost/test",
    )
    assert cfg_no_keys.langfuse_enabled is False

    cfg_one_key = Settings(
        langfuse_public_key="pk-test",
        langfuse_secret_key="",
        database_url="postgresql+asyncpg://x:x@localhost/test",
    )
    assert cfg_one_key.langfuse_enabled is False

    cfg_both_keys = Settings(
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        database_url="postgresql+asyncpg://x:x@localhost/test",
    )
    assert cfg_both_keys.langfuse_enabled is True


def test_spotify_configured_requires_both_credentials() -> None:
    """``spotify_configured`` is ``False`` unless both client ID and secret are set."""
    assert Settings(
        spotify_client_id="",
        spotify_client_secret="",
        database_url="postgresql+asyncpg://x:x@localhost/test",
    ).spotify_configured is False

    assert Settings(
        spotify_client_id="id",
        spotify_client_secret="",
        database_url="postgresql+asyncpg://x:x@localhost/test",
    ).spotify_configured is False

    assert Settings(
        spotify_client_id="id",
        spotify_client_secret="secret",
        database_url="postgresql+asyncpg://x:x@localhost/test",
    ).spotify_configured is True


def test_settings_env_override(test_settings: Settings) -> None:
    """Field values from constructor kwargs override any ``.env`` file."""
    assert test_settings.app_env == "development"
    assert test_settings.anthropic_api_key == "sk-ant-test"


def test_settings_extra_fields_are_ignored() -> None:
    """Unknown env vars do not raise (``extra="ignore"`` in model_config)."""
    cfg = Settings(
        TOTALLY_UNKNOWN_FIELD="ignored",  # type: ignore[call-arg]
        database_url="postgresql+asyncpg://x:x@localhost/test",
    )
    assert not hasattr(cfg, "totally_unknown_field")
