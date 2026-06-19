"""Tests for the planner (intent → execution plan + signals)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from backend.app.agents.planner import build_plan
from backend.app.schemas.chat import IntentType


def _cfg():
    from backend.app.config import Settings

    return Settings(openai_api_key="sk-test")


@pytest.mark.asyncio
async def test_plan_maps_music_find_to_dj() -> None:
    with patch(
        "backend.app.agents.planner.classify_intent",
        new=AsyncMock(return_value=(IntentType.MUSIC_FIND, 1.0)),
    ):
        plan = await build_plan("find me something chill", _cfg())
    assert plan.intent is IntentType.MUSIC_FIND
    assert plan.steps == ["dj"]
    assert plan.signals == []


@pytest.mark.asyncio
async def test_plan_mixed_fans_out() -> None:
    with patch(
        "backend.app.agents.planner.classify_intent",
        new=AsyncMock(return_value=(IntentType.MIXED, 0.8)),
    ):
        plan = await build_plan("play some Tems and tell me about her", _cfg())
    assert plan.steps == ["dj", "artist"]


@pytest.mark.asyncio
async def test_plan_adds_weather_signal_for_workout() -> None:
    with patch(
        "backend.app.agents.planner.classify_intent",
        new=AsyncMock(return_value=(IntentType.MUSIC_FIND, 1.0)),
    ):
        plan = await build_plan("I'm going for a run, play something", _cfg())
    assert "weather" in plan.signals
    assert plan.steps == ["dj"]


@pytest.mark.asyncio
async def test_plan_no_weather_without_dj_step() -> None:
    # An artist question that happens to mention "outside" should NOT pull
    # weather, because there's no music recommendation to make it relevant.
    with patch(
        "backend.app.agents.planner.classify_intent",
        new=AsyncMock(return_value=(IntentType.ARTIST_INFO, 1.0)),
    ):
        plan = await build_plan("is Burna Boy playing outside this weekend?", _cfg())
    assert plan.signals == []


@pytest.mark.asyncio
async def test_plan_no_weather_for_plain_music_request() -> None:
    with patch(
        "backend.app.agents.planner.classify_intent",
        new=AsyncMock(return_value=(IntentType.MUSIC_FIND, 1.0)),
    ):
        plan = await build_plan("find me some afrobeats", _cfg())
    assert plan.signals == []
