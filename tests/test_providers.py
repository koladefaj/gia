"""Tests for the LLM provider factory.

Validates that ``get_llm`` and ``get_fast_llm`` produce the correct model IDs
for each provider, raise on unknown providers, and never read global settings
directly (they accept ``Settings`` as an argument).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.app.config import Settings
from backend.app.providers.llm import _build_llm, get_fast_llm, get_llm


@pytest.fixture()
def anthropic_cfg(test_settings: Settings) -> Settings:
    """Settings with ``llm_provider=anthropic``."""
    return Settings(**{**test_settings.model_dump(), "llm_provider": "anthropic"})


@pytest.fixture()
def openai_cfg(test_settings: Settings) -> Settings:
    """Settings with ``llm_provider=openai``."""
    return Settings(**{**test_settings.model_dump(), "llm_provider": "openai"})


@pytest.fixture()
def ollama_cfg(test_settings: Settings) -> Settings:
    """Settings with ``llm_provider=ollama``."""
    return Settings(
        **{**test_settings.model_dump(), "llm_provider": "ollama", "ollama_model": "llama3.2"}
    )


# ── get_llm ───────────────────────────────────────────────────────────────────


def test_get_llm_anthropic_uses_sonnet_by_default(anthropic_cfg: Settings) -> None:
    """``get_llm`` with Anthropic defaults to ``claude-sonnet-4-6``."""
    with patch("backend.app.providers.llm.LLM") as mock_llm_cls:
        get_llm(anthropic_cfg)
    mock_llm_cls.assert_called_once()
    assert mock_llm_cls.call_args.kwargs["model"] == "claude-sonnet-4-6"


def test_get_llm_openai_uses_gpt4o_by_default(openai_cfg: Settings) -> None:
    """``get_llm`` with OpenAI defaults to ``gpt-4o``."""
    with patch("backend.app.providers.llm.LLM") as mock_llm_cls:
        get_llm(openai_cfg)
    assert mock_llm_cls.call_args.kwargs["model"] == "gpt-4o"


def test_get_llm_ollama_prefixes_model(ollama_cfg: Settings) -> None:
    """``get_llm`` with Ollama prepends ``ollama/`` to the model name."""
    with patch("backend.app.providers.llm.LLM") as mock_llm_cls:
        get_llm(ollama_cfg)
    model = mock_llm_cls.call_args.kwargs["model"]
    assert model.startswith("ollama/")


def test_get_llm_accepts_model_override(anthropic_cfg: Settings) -> None:
    """``get_llm`` passes through an explicit ``model`` argument."""
    with patch("backend.app.providers.llm.LLM") as mock_llm_cls:
        get_llm(anthropic_cfg, model="claude-opus-4-8")
    assert mock_llm_cls.call_args.kwargs["model"] == "claude-opus-4-8"


def test_get_llm_unknown_provider_raises(test_settings: Settings) -> None:
    """``Settings`` rejects unknown LLM providers at construction time.

    The validator in ``config.py`` now catches invalid providers before they
    can reach the LLM factory, giving a clearer error at startup.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="LLM_PROVIDER"):
        Settings(**{**test_settings.model_dump(), "llm_provider": "grok"})


# ── get_fast_llm ──────────────────────────────────────────────────────────────


def test_get_fast_llm_anthropic_uses_haiku(anthropic_cfg: Settings) -> None:
    """``get_fast_llm`` with Anthropic defaults to Claude Haiku."""
    with patch("backend.app.providers.llm.LLM") as mock_llm_cls:
        get_fast_llm(anthropic_cfg)
    model = mock_llm_cls.call_args.kwargs["model"]
    assert "haiku" in model.lower()


def test_get_fast_llm_openai_uses_mini(openai_cfg: Settings) -> None:
    """``get_fast_llm`` with OpenAI defaults to ``gpt-4o-mini``."""
    with patch("backend.app.providers.llm.LLM") as mock_llm_cls:
        get_fast_llm(openai_cfg)
    assert mock_llm_cls.call_args.kwargs["model"] == "gpt-4o-mini"


def test_get_fast_llm_accepts_model_override(anthropic_cfg: Settings) -> None:
    """``get_fast_llm`` forwards an explicit ``model`` argument."""
    with patch("backend.app.providers.llm.LLM") as mock_llm_cls:
        get_fast_llm(anthropic_cfg, model="claude-sonnet-4-6")
    assert mock_llm_cls.call_args.kwargs["model"] == "claude-sonnet-4-6"


def test_get_fast_llm_uses_injected_settings_not_globals(test_settings: Settings) -> None:
    """``get_fast_llm`` reads ``cfg``, not a global ``settings`` object."""
    local_cfg = Settings(**{**test_settings.model_dump(), "llm_provider": "openai"})
    with patch("backend.app.providers.llm.LLM") as mock_llm_cls:
        get_fast_llm(local_cfg)
    model = mock_llm_cls.call_args.kwargs["model"]
    assert "mini" in model
