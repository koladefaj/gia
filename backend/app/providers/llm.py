"""LLM provider factory — Phalanx pattern.

Returns a ``crewai.LLM`` instance for the configured provider.  The factory
is called by agents at crew-construction time, passing in the injected
``Settings`` object so the factory never reads globals.

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

from crewai import LLM

from backend.app.config import Settings

# Default model per provider — update here to change all agents at once
_PERSONA_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "ollama": "llama3.2",
}

_FAST_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2",
}


def get_llm(cfg: Settings, model: str | None = None) -> LLM:
    """Return a full-capability LLM for persona and deep-reasoning agents.

    Resolution order for the model string:
    1. ``model`` argument (explicit per-call override).
    2. ``cfg.llm_persona_model`` (env var ``LLM_PERSONA_MODEL`` — deploy-time config).
    3. Built-in provider default from ``_PERSONA_MODELS``.

    Args:
        cfg:   Application settings providing the provider choice and API keys.
        model: Optional per-call model override.

    Returns:
        A configured ``crewai.LLM`` instance ready for use in a ``CrewAI``
        agent or crew.

    Raises:
        ValueError: If ``cfg.llm_provider`` is not one of the supported values.
    """
    resolved = model or cfg.llm_persona_model or _PERSONA_MODELS.get(cfg.llm_provider, "")
    return _build_llm(cfg, resolved)


def get_fast_llm(cfg: Settings, model: str | None = None) -> LLM:
    """Return a cheap, fast LLM for logistics agents (routing, extraction).

    Resolution order for the model string:
    1. ``model`` argument (explicit per-call override).
    2. ``cfg.llm_fast_model`` (env var ``LLM_FAST_MODEL``).
    3. Built-in provider default from ``_FAST_MODELS``.

    Args:
        cfg:   Application settings.
        model: Optional per-call model override.

    Returns:
        A configured ``crewai.LLM`` instance.

    Raises:
        ValueError: If ``cfg.llm_provider`` is not recognised.
    """
    resolved = model or cfg.llm_fast_model or _FAST_MODELS.get(cfg.llm_provider, "")
    return _build_llm(cfg, resolved)


def _build_llm(cfg: Settings, model: str) -> LLM:
    """Construct a ``crewai.LLM`` for the given provider and model.

    Internal factory — callers should use ``get_llm`` or ``get_fast_llm``
    rather than calling this directly.

    Args:
        cfg:   Application settings.
        model: Exact model identifier string to pass to the provider.

    Returns:
        Configured ``crewai.LLM``.

    Raises:
        ValueError: If ``cfg.llm_provider`` is not one of
                    ``anthropic | openai | ollama``.
    """
    provider = cfg.llm_provider

    if provider == "anthropic":
        return LLM(model=model, api_key=cfg.anthropic_api_key)

    if provider == "openai":
        return LLM(model=model, api_key=cfg.openai_api_key)

    if provider == "ollama":
        return LLM(model=f"ollama/{model}", base_url=cfg.ollama_base_url)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        "Set LLM_PROVIDER to one of: anthropic | openai | ollama"
    )
