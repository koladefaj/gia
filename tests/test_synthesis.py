"""Tests for multi-agent reply synthesis."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.app.agents.synthesis import synthesize_reply


def _cfg():
    from backend.app.config import Settings

    return Settings(openai_api_key="sk-test")


@pytest.mark.asyncio
async def test_single_part_skips_llm() -> None:
    # No merge needed → no LLM call should happen.
    with patch("backend.app.agents.synthesis.get_fast_llm", side_effect=AssertionError):
        out = await synthesize_reply(["just one"], "q", _cfg())
    assert out == "just one"


@pytest.mark.asyncio
async def test_empty_parts_return_empty() -> None:
    out = await synthesize_reply(["", "   "], "q", _cfg())
    assert out == ""


@pytest.mark.asyncio
async def test_multiple_parts_merged_by_llm() -> None:
    fake_llm = MagicMock()
    fake_llm.call = MagicMock(return_value="A warm woven reply.")
    with patch("backend.app.agents.synthesis.get_fast_llm", return_value=fake_llm), \
         patch("backend.app.agents.synthesis.render_persona", return_value="PERSONA"):
        out = await synthesize_reply(["DJ pick.", "Artist aside."], "play Tems", _cfg())
    assert out == "A warm woven reply."
    fake_llm.call.assert_called_once()


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_join() -> None:
    fake_llm = MagicMock()
    fake_llm.call = MagicMock(side_effect=RuntimeError("boom"))
    with patch("backend.app.agents.synthesis.get_fast_llm", return_value=fake_llm), \
         patch("backend.app.agents.synthesis.render_persona", return_value="PERSONA"):
        out = await synthesize_reply(["one", "two"], "q", _cfg())
    assert out == "one two"
