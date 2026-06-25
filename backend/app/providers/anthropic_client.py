"""Shared async Anthropic client for the voice pipeline's streaming path.

The conversation agent streams tokens so TTS can start on the first sentence
while the rest is still generating. For the OpenAI provider that goes through the
Langfuse drop-in (see ``openai_client``); Anthropic has no Langfuse drop-in, so
this is the plain SDK client, cached once per credential like the OpenAI one.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from backend.app.config import Settings

# Persona-tier default for Anthropic — mirrors ``providers.llm._PERSONA_MODELS``
# so the streamed reply uses the same model as the blocking persona path.
_PERSONA_DEFAULT = "claude-sonnet-4-6"


@lru_cache(maxsize=4)
def _client_for(api_key: str) -> Any:
    """Return a cached ``AsyncAnthropic`` keyed by credentials (one per loop is fine)."""
    from anthropic import AsyncAnthropic  # noqa: PLC0415

    return AsyncAnthropic(api_key=api_key)


def get_async_anthropic(cfg: Settings) -> Any:
    """Return the shared ``AsyncAnthropic`` client for the configured key."""
    return _client_for(cfg.anthropic_api_key)


def persona_model(cfg: Settings) -> str:
    """Resolve the persona-tier model string for the streamed conversational reply."""
    return cfg.llm_persona_model or _PERSONA_DEFAULT
