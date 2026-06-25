"""Pydantic schema for externalised prompt documents.

Every ``*.yaml`` file under ``prompts/templates`` is parsed into a ``PromptDoc``.
Validating at load time means a malformed prompt fails loudly on startup (or in
a unit test) rather than producing a silently-empty system prompt at request
time — the kind of bug that is invisible until a demo.

A document carries one or more named **sections**, each a Jinja template string.
Single-body prompts (persona, router) use one section conventionally named
``body``.  Multi-part agent prompts use sections like ``role``, ``goal``,
``backstory`` and ``task`` so a CrewAI ``Agent`` can be assembled entirely from
data.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PromptDoc(BaseModel):
    """A single externalised, versioned prompt definition.

    Attributes:
        id:          Dotted key identifying the prompt (e.g. ``"agents.dj"``).
        version:     Version label, conventionally ``"v1"``, ``"v2"`` ...  The
                     registry resolves the highest version when none is requested.
        description: Human-readable note about what this prompt is for.
        metadata:    Free-form dict (e.g. ``model_tier``, ``tags``,
                     ``audio_tags_allowed``) — never rendered, available to
                     callers for routing/telemetry.
        sections:    Mapping of section name → raw Jinja template string.  At
                     least one section is required.
    """

    id: str
    version: str = "v1"
    description: str = ""
    metadata: dict = Field(default_factory=dict)
    sections: dict[str, str]

    @field_validator("id")
    @classmethod
    def _id_non_empty(cls, v: str) -> str:
        """Reject blank ids so the registry key is always meaningful."""
        if not v or not v.strip():
            raise ValueError("PromptDoc.id must be a non-empty dotted key")
        return v.strip()

    @field_validator("sections")
    @classmethod
    def _sections_non_empty(cls, v: dict[str, str]) -> dict[str, str]:
        """Require at least one section; reject empty template bodies.

        An empty section almost always means a YAML indentation mistake, and a
        blank system prompt degrades the model silently — so we fail fast.
        """
        if not v:
            raise ValueError("PromptDoc.sections must contain at least one section")
        for name, body in v.items():
            if not isinstance(body, str) or not body.strip():
                raise ValueError(f"PromptDoc section {name!r} is empty")
        return v
