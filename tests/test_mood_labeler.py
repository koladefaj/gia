"""Tests for the LLM mood labeler."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from backend.app.mood.classifier import MOOD_LABELS
from backend.app.mood.labeler import label_mood


@pytest.mark.asyncio
async def test_label_mood_returns_vocabulary_label(test_settings) -> None:
    tracks = [{"name": "Free Mind", "artist": "Tems"}, {"name": "Essence", "artist": "Wizkid"}]
    with patch("backend.app.mood.labeler.get_fast_llm") as mock_llm:
        mock_llm.return_value.call = lambda *a, **k: "chill"
        label = await label_mood(tracks, test_settings)
    assert label == "chill"


@pytest.mark.asyncio
async def test_label_mood_coerces_synonym(test_settings) -> None:
    with patch("backend.app.mood.labeler.get_fast_llm") as mock_llm:
        mock_llm.return_value.call = lambda *a, **k: "an energetic, party set"
        label = await label_mood([{"name": "Last Last", "artist": "Burna Boy"}], test_settings)
    assert label == "hype"


@pytest.mark.asyncio
async def test_label_mood_empty_is_neutral(test_settings) -> None:
    assert await label_mood([], test_settings) == "neutral"


@pytest.mark.asyncio
async def test_label_mood_llm_error_is_neutral(test_settings) -> None:
    with patch("backend.app.mood.labeler.asyncio.to_thread",
               new=AsyncMock(side_effect=RuntimeError("down"))):
        label = await label_mood([{"name": "x", "artist": "y"}], test_settings)
    assert label == "neutral"


@pytest.mark.asyncio
async def test_label_mood_result_always_in_vocab(test_settings) -> None:
    with patch("backend.app.mood.labeler.get_fast_llm") as mock_llm:
        mock_llm.return_value.call = lambda *a, **k: "totally unknown vibe"
        label = await label_mood([{"name": "x", "artist": "y"}], test_settings)
    assert label in MOOD_LABELS
