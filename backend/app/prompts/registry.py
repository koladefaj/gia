"""File-based prompt registry — the single source of truth for every prompt.

Why this exists
---------------
Prompts were previously Python string constants concatenated inline inside each
agent service.  Changing Gia's voice meant editing code in five places, and the
prompts were impossible to diff, version, or A/B independently of a deploy.

The registry decouples *prompt content* from *application code*:

  - Prompts live as versioned ``*.yaml`` files under ``templates/``.
  - They are rendered with Jinja2 using ``StrictUndefined`` so a missing
    template variable raises immediately instead of producing a malformed
    prompt.
  - Callers reference a stable key (``"agents.dj"``) and never see a raw string.

Lifecycle
---------
In FastAPI the registry is built once in ``lifespan`` and stored on
``app.state.prompts`` (injected via ``dependencies.get_prompt_registry``).
Code without a request context (Celery tasks, module-level defaults) uses the
process-wide lazy singleton ``get_registry()``.
"""

from __future__ import annotations

import threading
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined, Template

from backend.app.observability.logging import get_logger
from backend.app.prompts.schema import PromptDoc

logger = get_logger(__name__)

# Default location of the YAML prompt library, resolved relative to this file so
# it works regardless of the process working directory.
TEMPLATES_DIR = Path(__file__).parent / "templates"

DEFAULT_SECTION = "body"


class PromptNotFoundError(KeyError):
    """Raised when a requested prompt id or version does not exist."""


class RenderablePrompt:
    """A loaded prompt document with its sections pre-compiled to templates.

    Compilation happens once at load time, so ``render`` is just a dictionary
    lookup plus a Jinja render — cheap enough to call on every turn.

    Attributes:
        id:       The prompt's dotted key.
        version:  The prompt's version label.
        metadata: The document's free-form metadata dict.
    """

    def __init__(self, doc: PromptDoc, env: Environment) -> None:
        self._doc = doc
        self._compiled: dict[str, Template] = {
            name: env.from_string(body) for name, body in doc.sections.items()
        }

    @property
    def id(self) -> str:
        return self._doc.id

    @property
    def version(self) -> str:
        return self._doc.version

    @property
    def metadata(self) -> dict:
        return self._doc.metadata

    @property
    def sections(self) -> list[str]:
        """Names of the sections this prompt defines."""
        return list(self._compiled)

    def render(self, section: str = DEFAULT_SECTION, /, **context: object) -> str:
        """Render *section* with *context* and return the stripped result.

        Args:
            section: Section name (defaults to ``"body"`` for single-body prompts).
            **context: Template variables.  Any variable referenced by the
                template but missing here raises ``jinja2.UndefinedError`` —
                this is intentional (fail loud, not silent).

        Returns:
            The rendered prompt text, with surrounding whitespace stripped.

        Raises:
            PromptNotFoundError: If *section* is not defined on this prompt.
        """
        try:
            template = self._compiled[section]
        except KeyError as exc:
            raise PromptNotFoundError(
                f"Prompt {self._doc.id!r} has no section {section!r}; "
                f"available: {self.sections}"
            ) from exc
        return template.render(**context).strip()

    def raw(self, section: str = DEFAULT_SECTION) -> str:
        """Return the un-rendered template string for *section* (for tests/inspection)."""
        return self._doc.sections[section]


def _version_sort_key(version: str) -> tuple[int, str]:
    """Order versions so ``v10`` sorts after ``v2``.

    Parses the leading integer in a ``vN`` label.  Non-numeric labels fall back
    to ``(-1, label)`` so they never out-rank a numbered version.

    Args:
        version: A version label such as ``"v1"`` or ``"v12"``.

    Returns:
        A sort key tuple; higher compares greater.
    """
    digits = "".join(c for c in version if c.isdigit())
    return (int(digits), version) if digits else (-1, version)


class PromptRegistry:
    """In-memory index of every ``PromptDoc`` discovered under a templates dir.

    The registry is keyed by ``(id, version)`` and also tracks the latest
    version per id so callers can omit the version for the common case.

    Thread-safety: ``reload`` swaps the internal dicts under a lock, so reads on
    other threads always see a consistent snapshot.
    """

    def __init__(self, templates_dir: Path | str = TEMPLATES_DIR) -> None:
        self._dir = Path(templates_dir)
        self._env = Environment(
            undefined=StrictUndefined,
            autoescape=False,  # prompts are plain text, never HTML
            trim_blocks=False,
            lstrip_blocks=False,
            keep_trailing_newline=False,
        )
        self._lock = threading.Lock()
        self._prompts: dict[tuple[str, str], RenderablePrompt] = {}
        self._latest: dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        """Load (or re-load) every ``*.yaml`` under the templates directory.

        Each file is parsed, validated into a ``PromptDoc``, compiled, and
        indexed.  Safe to call at runtime for dev hot-editing.

        Raises:
            FileNotFoundError: If the templates directory does not exist.
            ValueError:        If two files declare the same ``(id, version)``.
        """
        if not self._dir.is_dir():
            raise FileNotFoundError(f"Prompt templates dir not found: {self._dir}")

        prompts: dict[tuple[str, str], RenderablePrompt] = {}
        latest: dict[str, str] = {}

        for path in sorted(self._dir.rglob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if raw is None:
                continue
            doc = PromptDoc.model_validate(raw)
            key = (doc.id, doc.version)
            if key in prompts:
                raise ValueError(
                    f"Duplicate prompt {doc.id!r} version {doc.version!r} "
                    f"(second definition in {path})"
                )
            prompts[key] = RenderablePrompt(doc, self._env)

            current = latest.get(doc.id)
            if current is None or _version_sort_key(doc.version) > _version_sort_key(current):
                latest[doc.id] = doc.version

        with self._lock:
            self._prompts = prompts
            self._latest = latest

        logger.info("prompt_registry_loaded", count=len(prompts), dir=str(self._dir))

    def get(self, key: str, version: str | None = None) -> RenderablePrompt:
        """Return the prompt for *key*, latest version unless *version* is given.

        Args:
            key:     Dotted prompt id (e.g. ``"persona.gia"``).
            version: Explicit version label, or ``None`` for the latest.

        Returns:
            The matching ``RenderablePrompt``.

        Raises:
            PromptNotFoundError: If the id (or specific version) is unknown.
        """
        with self._lock:
            resolved = version or self._latest.get(key)
            if resolved is None:
                raise PromptNotFoundError(f"No prompt registered for id {key!r}")
            prompt = self._prompts.get((key, resolved))
        if prompt is None:
            raise PromptNotFoundError(
                f"Prompt {key!r} has no version {resolved!r}"
            )
        return prompt

    def ids(self) -> list[str]:
        """Return all registered prompt ids (for diagnostics / a /prompts route)."""
        with self._lock:
            return sorted(self._latest)


# ── Process-wide lazy singleton (for non-request contexts) ────────────────────

_singleton: PromptRegistry | None = None
_singleton_lock = threading.Lock()


def get_registry() -> PromptRegistry:
    """Return the shared ``PromptRegistry``, building it on first use.

    FastAPI request handlers should prefer ``Depends(get_prompt_registry)`` so
    the instance on ``app.state`` is used (and overridable in tests).  This
    accessor exists for Celery tasks and module-level defaults that have no
    request to draw from.
    """
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = PromptRegistry()
    return _singleton
