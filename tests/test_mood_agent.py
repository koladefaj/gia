"""Tests for ``MoodService`` and ``build_mood_agent``."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture()
def fake_spotify_mood() -> MagicMock:
    sp = MagicMock()
    sp.get_currently_playing = AsyncMock(return_value={
        "uri": "spotify:track:001",
        "name": "Last Last",
        "artist": "Burna Boy",
        "energy": 0.78,
        "valence": 0.68,
        "is_playing": True,
    })
    return sp


@pytest.fixture()
def fake_store_with_pattern() -> MagicMock:
    store = MagicMock()
    pattern_mem = MagicMock()
    pattern_mem.id = "pattern-1"
    pattern_mem.text = (
        "Mood pattern for sunday_evening: wind-down. "
        "avg energy=0.31, avg valence=0.72, avg tempo=90 BPM. "
        "Based on 8 sessions (consistency σ=0.045)."
    )
    store.search = AsyncMock(return_value=[pattern_mem])
    return store


@pytest.mark.asyncio
async def test_mood_service_detects_deviation(
    fake_spotify_mood, fake_store_with_pattern, test_settings
) -> None:
    """MoodService detects high-energy track deviating from wind-down pattern."""
    from backend.app.agents.mood import MoodService

    with patch("backend.app.agents.mood.get_pattern_for_now", new=AsyncMock(
        return_value=MagicMock(text=fake_store_with_pattern.search.return_value[0].text)
    )), \
    patch("backend.app.agents.mood.asyncio.to_thread", new=AsyncMock(
        return_value="[thoughtful] You're usually on wind-down stuff around Sunday evening."
    )), \
    patch("backend.app.agents.mood.datetime") as mock_dt:
        from datetime import datetime, timezone
        mock_dt.now.return_value = datetime(2026, 6, 21, 20, 0, tzinfo=timezone.utc)

        service = MoodService(
            spotify=fake_spotify_mood,
            store=fake_store_with_pattern,
            cfg=test_settings,
        )
        result = await service.analyze("user-1")

    assert result.deviation is True
    assert result.current_label == "hype"
    assert result.pattern_label == "wind-down"
    assert result.proactive_draft is not None
    assert "wind-down" in result.proactive_draft.lower()


@pytest.mark.asyncio
async def test_mood_service_no_deviation_when_matching(
    fake_store_with_pattern, test_settings
) -> None:
    """No deviation when current features match the known pattern."""
    from backend.app.agents.mood import MoodService

    sp = MagicMock()
    sp.get_currently_playing = AsyncMock(return_value={
        "energy": 0.30,  # Close to pattern 0.31
        "valence": 0.71,
        "name": "Free Mind",
    })

    with patch("backend.app.agents.mood.get_pattern_for_now", new=AsyncMock(
        return_value=MagicMock(text=fake_store_with_pattern.search.return_value[0].text)
    )), \
    patch("backend.app.agents.mood.datetime") as mock_dt:
        from datetime import datetime, timezone
        mock_dt.now.return_value = datetime(2026, 6, 21, 20, 0, tzinfo=timezone.utc)

        service = MoodService(spotify=sp, store=fake_store_with_pattern, cfg=test_settings)
        result = await service.analyze("user-1")

    assert result.deviation is False
    assert result.proactive_draft is None


@pytest.mark.asyncio
async def test_mood_service_no_pattern_returns_neutral(
    fake_spotify_mood, test_settings
) -> None:
    """No known pattern → returns current label, no proactive draft."""
    from backend.app.agents.mood import MoodService

    store = MagicMock()

    with patch("backend.app.agents.mood.get_pattern_for_now", new=AsyncMock(return_value=None)):
        service = MoodService(spotify=fake_spotify_mood, store=store, cfg=test_settings)
        result = await service.analyze("user-1")

    assert result.pattern_label is None
    assert result.proactive_draft is None
    assert result.current_label == "hype"


@pytest.mark.asyncio
async def test_mood_service_handles_spotify_error(test_settings) -> None:
    """Spotify errors produce a neutral result without crashing."""
    from backend.app.agents.mood import MoodService

    sp = MagicMock()
    sp.get_currently_playing = AsyncMock(side_effect=RuntimeError("Spotify down"))
    store = MagicMock()

    service = MoodService(spotify=sp, store=store, cfg=test_settings)
    result = await service.analyze("user-1")

    assert result.current_label == "neutral"


def test_build_mood_agent_returns_crewai_agent(test_settings) -> None:
    """``build_mood_agent`` returns a configured CrewAI Agent."""
    from crewai import Agent

    from backend.app.agents.mood import build_mood_agent

    with patch("backend.app.agents.mood.get_fast_llm", return_value="gpt-4o-mini"):
        agent = build_mood_agent(test_settings)

    assert isinstance(agent, Agent)
    assert "Mood" in agent.role
