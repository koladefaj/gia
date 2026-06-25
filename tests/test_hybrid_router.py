"""Tests for the structured hybrid router."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from backend.app.agents.hybrid_router import classify_turn
from backend.app.providers.openai_client import extract_json_object
from backend.app.schemas.chat import IntentType
from backend.app.schemas.router import EngagementMode, Tone

_VALID = (
    '{"intent":"NEWS_QUERY","tone":"surprised","confidence":0.9,'
    '"engagement_mode":"react_then_execute","needs_search":true,'
    '"needs_memory":false,"needs_music":false,"needs_artist_lookup":false}'
)


@pytest.mark.asyncio
async def test_classify_turn_parses_decision(test_settings) -> None:
    with patch("backend.app.agents.hybrid_router._complete",
               new=AsyncMock(return_value=_VALID)):
        decision = await classify_turn("did you see what Drake said?", test_settings)
    assert decision.intent == IntentType.NEWS_QUERY
    assert decision.tone == Tone.SURPRISED
    assert decision.engagement_mode == EngagementMode.REACT_THEN_EXECUTE
    assert decision.needs_search is True
    assert decision.reacts is True


@pytest.mark.asyncio
async def test_classify_turn_tolerates_fences(test_settings) -> None:
    fenced = f"```json\n{_VALID}\n```"
    with patch("backend.app.agents.hybrid_router._complete",
               new=AsyncMock(return_value=fenced)):
        decision = await classify_turn("hi", test_settings)
    assert decision.intent == IntentType.NEWS_QUERY


@pytest.mark.asyncio
async def test_classify_turn_bad_json_falls_back(test_settings) -> None:
    with patch("backend.app.agents.hybrid_router._complete",
               new=AsyncMock(return_value="not json at all")):
        decision = await classify_turn("hi", test_settings)
    assert decision.intent == IntentType.GENERAL_CHAT
    assert decision.confidence == 0.0


@pytest.mark.asyncio
async def test_classify_turn_call_error_falls_back(test_settings) -> None:
    with patch("backend.app.agents.hybrid_router._complete",
               new=AsyncMock(side_effect=RuntimeError("openai down"))):
        decision = await classify_turn("hi", test_settings)
    assert decision.intent == IntentType.GENERAL_CHAT


def test_extract_json_object_from_prose() -> None:
    obj = extract_json_object('Sure! {"intent": "GENERAL_CHAT"} hope that helps')
    assert obj["intent"] == "GENERAL_CHAT"


def test_extract_json_object_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        extract_json_object("[1, 2, 3]")
