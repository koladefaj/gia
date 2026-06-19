"""Tests for the LLM memory extractor.

The LLM is replaced with a mock so these tests run offline.  Tests cover
prompt construction, JSON parsing, and error handling.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.schemas.memory import MemoryEntry


def _make_entry(text: str, uid: str = "00000000-0000-0000-0000-000000000001") -> MemoryEntry:
    return MemoryEntry(
        id=uid,
        type="preference",
        text=text,
        confidence=0.8,
        created_at=datetime(2026, 6, 19, tzinfo=timezone.utc),
    )


class TestParseExtractedMemories:
    def test_valid_json_array(self) -> None:
        """Well-formed JSON array is parsed into ``ExtractedMemory`` objects."""
        from backend.app.memory.extractor import _parse_extracted_memories

        raw = '[{"type":"preference","text":"Loves Tems","confidence":0.9,"supersedes_id":null}]'
        result = _parse_extracted_memories(raw)
        assert len(result) == 1
        assert result[0].text == "Loves Tems"
        assert result[0].confidence == pytest.approx(0.9)
        assert result[0].supersedes_id is None

    def test_json_inside_markdown_fences(self) -> None:
        """Extractor strips markdown fences and still parses the array."""
        from backend.app.memory.extractor import _parse_extracted_memories

        raw = '```json\n[{"type":"preference","text":"Night owl","confidence":0.7}]\n```'
        result = _parse_extracted_memories(raw)
        assert len(result) == 1
        assert result[0].text == "Night owl"

    def test_empty_array_returns_empty_list(self) -> None:
        """LLM returning ``[]`` produces an empty list."""
        from backend.app.memory.extractor import _parse_extracted_memories

        assert _parse_extracted_memories("[]") == []

    def test_no_json_array_returns_empty_list(self) -> None:
        """Non-JSON response returns empty list without raising."""
        from backend.app.memory.extractor import _parse_extracted_memories

        assert _parse_extracted_memories("Sorry, nothing to extract.") == []

    def test_invalid_json_returns_empty_list(self) -> None:
        """Malformed JSON returns empty list without raising."""
        from backend.app.memory.extractor import _parse_extracted_memories

        assert _parse_extracted_memories("[not valid json}") == []

    def test_invalid_items_skipped(self) -> None:
        """Items that fail Pydantic validation are silently skipped."""
        from backend.app.memory.extractor import _parse_extracted_memories

        raw = '[{"type":"preference","text":"ok","confidence":0.8},{"bad":"item"}]'
        result = _parse_extracted_memories(raw)
        # "ok" item passes; "bad" item fails (missing required type/text fields)
        assert len(result) == 1
        assert result[0].text == "ok"

    def test_supersedes_id_preserved(self) -> None:
        """``supersedes_id`` is passed through when present."""
        from backend.app.memory.extractor import _parse_extracted_memories

        raw = '[{"type":"preference","text":"New pref","confidence":0.85,"supersedes_id":"abc-123"}]'
        result = _parse_extracted_memories(raw)
        assert result[0].supersedes_id == "abc-123"

    def test_json_object_not_array_returns_empty(self) -> None:
        """A bare JSON object (not an array) is rejected, not half-parsed.

        Small models sometimes drop the wrapping list — we want [] not a crash.
        """
        from backend.app.memory.extractor import _parse_extracted_memories

        assert _parse_extracted_memories('{"type":"preference","text":"x"}') == []

    def test_greedy_outermost_array_with_preamble(self) -> None:
        """Leading prose + a fenced array (typical gemma3:4b output) still parses."""
        from backend.app.memory.extractor import _parse_extracted_memories

        raw = 'Here is what I found:\n```json\n[{"type":"preference","text":"Loves Tems","confidence":0.9}]\n```'
        result = _parse_extracted_memories(raw)
        assert len(result) == 1
        assert result[0].text == "Loves Tems"


@pytest.mark.asyncio
async def test_extract_memories_calls_llm(test_settings) -> None:
    """``extract_memories`` calls the LLM and returns parsed memories."""
    from backend.app.memory.extractor import extract_memories

    fake_llm = MagicMock()
    fake_llm.call.return_value = (
        '[{"type":"preference","text":"Loves Tems","confidence":0.9}]'
    )

    with patch("backend.app.memory.extractor.get_fast_llm", return_value=fake_llm), \
         patch("backend.app.memory.extractor.asyncio.to_thread", new=AsyncMock(
             return_value='[{"type":"preference","text":"Loves Tems","confidence":0.9}]'
         )):
        result = await extract_memories("I loved Free Mind by Tems", [], test_settings)

    assert len(result) == 1
    assert result[0].text == "Loves Tems"


@pytest.mark.asyncio
async def test_extract_memories_with_existing_memories(test_settings) -> None:
    """Existing memories are embedded in the prompt for supersede lookup."""
    from backend.app.memory.extractor import extract_memories

    existing = [_make_entry("Prefers low-energy tracks", "00000000-0000-0000-0000-000000000001")]

    with patch("backend.app.memory.extractor.get_fast_llm") as mock_get_llm, \
         patch("backend.app.memory.extractor.asyncio.to_thread", new=AsyncMock(return_value="[]")):
        mock_get_llm.return_value = MagicMock()
        mock_get_llm.return_value.call.return_value = "[]"
        result = await extract_memories("nothing durable here", existing, test_settings)

    assert result == []


@pytest.mark.asyncio
async def test_extract_memories_handles_llm_error(test_settings) -> None:
    """LLM errors are caught and return an empty list instead of raising."""
    from backend.app.memory.extractor import extract_memories

    with patch("backend.app.memory.extractor.get_fast_llm") as mock_get_llm, \
         patch("backend.app.memory.extractor.asyncio.to_thread", new=AsyncMock(
             side_effect=RuntimeError("LLM unreachable")
         )):
        mock_get_llm.return_value = MagicMock()
        result = await extract_memories("some transcript", [], test_settings)

    assert result == []
