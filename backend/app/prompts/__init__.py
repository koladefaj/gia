"""Externalised, versioned prompt library and registry.

Public API::

    from backend.app.prompts import PromptRegistry, get_registry, RenderablePrompt

Prompt content lives in ``templates/*.yaml`` — edit those, not Python, to change
Gia's voice or any agent's instructions.
"""

from backend.app.prompts.registry import (
    PromptNotFoundError,
    PromptRegistry,
    RenderablePrompt,
    get_registry,
)
from backend.app.prompts.schema import PromptDoc

__all__ = [
    "PromptDoc",
    "PromptNotFoundError",
    "PromptRegistry",
    "RenderablePrompt",
    "get_registry",
]
