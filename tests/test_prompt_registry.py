"""Tests for the externalised prompt registry.

Covers loading/validation, version resolution, Jinja rendering with
``StrictUndefined``, and the real shipped template library so a malformed YAML
prompt fails here rather than in production.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from jinja2 import UndefinedError

from backend.app.prompts.registry import PromptNotFoundError, PromptRegistry
from backend.app.prompts.schema import PromptDoc


def _write(dir_: Path, name: str, content: str) -> None:
    path = dir_ / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


# ── Schema validation ─────────────────────────────────────────────────────────


def test_prompt_doc_rejects_empty_sections() -> None:
    with pytest.raises(ValueError):
        PromptDoc(id="x", sections={})


def test_prompt_doc_rejects_blank_section_body() -> None:
    with pytest.raises(ValueError):
        PromptDoc(id="x", sections={"body": "   "})


def test_prompt_doc_rejects_blank_id() -> None:
    with pytest.raises(ValueError):
        PromptDoc(id="  ", sections={"body": "hi"})


# ── Loading & rendering ───────────────────────────────────────────────────────


def test_registry_loads_and_renders(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "greet.yaml",
        """
        id: test.greet
        version: v1
        sections:
          body: "Hello {{ name }}"
        """,
    )
    reg = PromptRegistry(tmp_path)
    assert reg.get("test.greet").render(name="Gia") == "Hello Gia"


def test_missing_variable_raises(tmp_path: Path) -> None:
    """StrictUndefined makes a missing variable a loud error, not silent blank."""
    _write(
        tmp_path,
        "greet.yaml",
        """
        id: test.greet
        sections:
          body: "Hello {{ name }}"
        """,
    )
    reg = PromptRegistry(tmp_path)
    with pytest.raises(UndefinedError):
        reg.get("test.greet").render()


def test_latest_version_is_resolved(tmp_path: Path) -> None:
    _write(tmp_path, "v1.yaml", """
        id: test.p
        version: v1
        sections: {body: "one"}
    """)
    _write(tmp_path, "v2.yaml", """
        id: test.p
        version: v2
        sections: {body: "two"}
    """)
    _write(tmp_path, "v10.yaml", """
        id: test.p
        version: v10
        sections: {body: "ten"}
    """)
    reg = PromptRegistry(tmp_path)
    # v10 must out-rank v2 (numeric, not lexical, ordering)
    assert reg.get("test.p").render() == "ten"
    assert reg.get("test.p", version="v1").render() == "one"


def test_unknown_id_and_version_raise(tmp_path: Path) -> None:
    _write(tmp_path, "p.yaml", """
        id: test.p
        version: v1
        sections: {body: "hi"}
    """)
    reg = PromptRegistry(tmp_path)
    with pytest.raises(PromptNotFoundError):
        reg.get("does.not.exist")
    with pytest.raises(PromptNotFoundError):
        reg.get("test.p", version="v9")


def test_unknown_section_raises(tmp_path: Path) -> None:
    _write(tmp_path, "p.yaml", """
        id: test.p
        sections: {body: "hi"}
    """)
    reg = PromptRegistry(tmp_path)
    with pytest.raises(PromptNotFoundError):
        reg.get("test.p").render("nope")


def test_duplicate_id_version_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "a.yaml", """
        id: dup
        version: v1
        sections: {body: "a"}
    """)
    _write(tmp_path, "b.yaml", """
        id: dup
        version: v1
        sections: {body: "b"}
    """)
    with pytest.raises(ValueError, match="Duplicate"):
        PromptRegistry(tmp_path)


def test_reload_picks_up_edits(tmp_path: Path) -> None:
    _write(tmp_path, "p.yaml", """
        id: test.p
        sections: {body: "before"}
    """)
    reg = PromptRegistry(tmp_path)
    assert reg.get("test.p").render() == "before"
    _write(tmp_path, "p.yaml", """
        id: test.p
        sections: {body: "after"}
    """)
    reg.reload()
    assert reg.get("test.p").render() == "after"


def test_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PromptRegistry(tmp_path / "nope")


# ── The real shipped template library ─────────────────────────────────────────


def test_shipped_templates_are_valid() -> None:
    """Every YAML under the real templates dir loads and the key prompts exist."""
    reg = PromptRegistry()
    ids = reg.ids()
    for expected in (
        "persona.gia",
        "agents.dj",
        "agents.artist",
        "agents.mood",
        "agents.memory",
        "agents.router",
    ):
        assert expected in ids

    # Persona renders with no variables.
    assert reg.get("persona.gia").render()
    # Agent identities expose role/goal/backstory.
    for agent in ("agents.dj", "agents.artist", "agents.mood", "agents.router", "agents.memory"):
        prompt = reg.get(agent)
        assert prompt.render("role")
        assert prompt.render("goal")
        assert prompt.render("backstory")


def test_dj_task_renders_with_context() -> None:
    reg = PromptRegistry()
    out = reg.get("agents.dj").render(
        "task",
        persona="PERSONA",
        user_context="CTX",
        query="chill afrobeats",
        seed_name="Free Mind",
        seed_artist="Tems",
        queued_names="a, b",
        start_playback=False,
        requested_title=None,
        missing_titles=None,
    )
    assert "Free Mind by Tems" in out
    # The template never references raw audio-feature numbers.
    assert "chill afrobeats" in out
