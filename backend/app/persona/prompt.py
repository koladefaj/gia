"""Gia's persona — now sourced from the externalised prompt registry.

The persona text used to be a hardcoded string constant here.  It now lives in
``prompts/templates/persona/gia.yaml`` and is loaded through the
``PromptRegistry`` so it can be versioned and edited without touching code.

``GIA_PERSONA`` is kept as a module-level convenience for the common case and
for backward compatibility.  New code should prefer rendering from the registry
directly (``registry.get("persona.gia").render()``) so a custom/overridden
registry — e.g. in tests — is respected.
"""

from __future__ import annotations

from backend.app.prompts import RenderablePrompt, get_registry

PERSONA_KEY = "persona.gia"


def render_persona(registry=None) -> str:
    """Render Gia's persona body from the prompt registry.

    Args:
        registry: An optional ``PromptRegistry`` to render from.  Defaults to
            the process-wide singleton via ``get_registry()``.

    Returns:
        The rendered persona system-prompt text.
    """
    reg = registry or get_registry()
    prompt: RenderablePrompt = reg.get(PERSONA_KEY)
    return prompt.render()


# Backward-compatible module constant, rendered once at import from the registry.
GIA_PERSONA = render_persona()
