"""LLM provider factory.

Returns an ``LLM`` instance for the configured provider.  The factory is called
by agents at construction time, passing in the injected ``Settings`` object so
the factory never reads globals.

Supported providers
-------------------
- ``anthropic`` — Claude models via Anthropic API
- ``openai``    — GPT models via OpenAI API
- ``ollama``    — Local models via Ollama (no API key required)

Per-agent model selection
-------------------------
Agents that need a cheaper model (routing, memory extraction) call
``get_fast_llm(cfg)``; those that need full expressiveness (persona) call
``get_llm(cfg)``::

    router_llm  = get_fast_llm(cfg)   # haiku / gpt-4o-mini / local
    persona_llm = get_llm(cfg)         # sonnet / gpt-4o / local

Usage::

    from backend.app.providers.llm import get_llm
    llm = get_llm(cfg, model="claude-opus-4-8")
"""

from __future__ import annotations

import litellm

from backend.app.config import Settings

# Default model per provider — update here to change all agents at once.
_PERSONA_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
}

_FAST_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
}


class LLM:
    """Thin litellm wrapper that exposes the same ``call()`` interface
    the agents use, while keeping all provider details out of call sites."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._base_url = base_url

    def call(self, messages: list[dict], **kwargs: object) -> str:
        """Send *messages* and return the response text."""
        response = litellm.completion(
            model=self.model,
            messages=messages,
            api_key=self._api_key,
            base_url=self._base_url,
            **kwargs,
        )
        return response.choices[0].message.content or ""


def _persona_default(cfg: Settings) -> str:
    if cfg.llm_provider == "ollama":
        return cfg.ollama_model
    return _PERSONA_MODELS.get(cfg.llm_provider, "")


def _fast_default(cfg: Settings) -> str:
    if cfg.llm_provider == "ollama":
        return cfg.ollama_model
    return _FAST_MODELS.get(cfg.llm_provider, "")


def get_llm(cfg: Settings, model: str | None = None) -> LLM:
    """Return a full-capability LLM for persona and deep-reasoning agents."""
    resolved = model or cfg.llm_persona_model or _persona_default(cfg)
    return _build_llm(cfg, resolved)


def get_fast_llm(cfg: Settings, model: str | None = None) -> LLM:
    """Return a cheap, fast LLM for logistics agents (routing, extraction)."""
    resolved = model or cfg.llm_fast_model or _fast_default(cfg)
    return _build_llm(cfg, resolved)


def _build_llm(cfg: Settings, model: str) -> LLM:
    provider = cfg.llm_provider

    def _qualified(prefix: str) -> str:
        return model if "/" in model else f"{prefix}/{model}"

    if provider == "anthropic":
        return LLM(model=_qualified("anthropic"), api_key=cfg.anthropic_api_key)

    if provider == "openai":
        return LLM(model=_qualified("openai"), api_key=cfg.openai_api_key)

    if provider == "ollama":
        return LLM(model=_qualified("ollama"), base_url=cfg.ollama_base_url)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        "Set LLM_PROVIDER to one of: anthropic | openai | ollama"
    )
