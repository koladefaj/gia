"""Tests for ``MoodService`` and ``build_mood_agent``."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# A stored pattern: bucket + label, in the format inference writes / proactive parses.
_PATTERN_TEXT = "Mood pattern for sunday_evening: chill. Often plays Tems, Wizkid. Based on 8 plays."


@pytest.fixture()
def fake_spotify_mood() -> MagicMock:
    sp = MagicMock()
    sp.get_recently_played = AsyncMock(return_value=[
        {"name": "Last Last", "artist": "Burna Boy"},
        {"name": "Ye", "artist": "Burna Boy"},
    ])
    return sp


@pytest.mark.asyncio
async def test_mood_service_detects_deviation(fake_spotify_mood, test_settings) -> None:
    """Current listening reads as a different mood than the bucket's pattern."""
    from backend.app.agents.mood import MoodService

    with patch("backend.app.agents.mood.get_pattern_for_now",
               new=AsyncMock(return_value=MagicMock(text=_PATTERN_TEXT))), \
         patch("backend.app.agents.mood.label_mood", new=AsyncMock(return_value="hype")), \
         patch("backend.app.agents.mood.asyncio.to_thread",
               new=AsyncMock(return_value="[thoughtful] You're usually on chill stuff around Sunday evening.")):
        service = MoodService(spotify=fake_spotify_mood, store=MagicMock(), cfg=test_settings)
        result = await service.analyze("user-1")

    assert result.deviation is True
    assert result.current_label == "hype"
    assert result.pattern_label == "chill"
    assert result.proactive_draft is not None


@pytest.mark.asyncio
async def test_mood_service_no_deviation_when_matching(fake_spotify_mood, test_settings) -> None:
    """No deviation when the current mood matches the known pattern."""
    from backend.app.agents.mood import MoodService

    with patch("backend.app.agents.mood.get_pattern_for_now",
               new=AsyncMock(return_value=MagicMock(text=_PATTERN_TEXT))), \
         patch("backend.app.agents.mood.label_mood", new=AsyncMock(return_value="chill")):
        service = MoodService(spotify=fake_spotify_mood, store=MagicMock(), cfg=test_settings)
        result = await service.analyze("user-1")

    assert result.deviation is False
    assert result.current_label == "chill"
    assert result.pattern_label == "chill"
    assert result.proactive_draft is None


@pytest.mark.asyncio
async def test_mood_service_no_pattern_returns_current(fake_spotify_mood, test_settings) -> None:
    """No known pattern → returns the current label, no proactive draft."""
    from backend.app.agents.mood import MoodService

    with patch("backend.app.agents.mood.get_pattern_for_now", new=AsyncMock(return_value=None)), \
         patch("backend.app.agents.mood.label_mood", new=AsyncMock(return_value="hype")):
        service = MoodService(spotify=fake_spotify_mood, store=MagicMock(), cfg=test_settings)
        result = await service.analyze("user-1")

    assert result.pattern_label is None
    assert result.proactive_draft is None
    assert result.current_label == "hype"


@pytest.mark.asyncio
async def test_mood_service_handles_spotify_error(test_settings) -> None:
    """A recently-played error degrades to a neutral result without crashing."""
    from backend.app.agents.mood import MoodService

    sp = MagicMock()
    sp.get_recently_played = AsyncMock(side_effect=RuntimeError("Spotify down"))

    service = MoodService(spotify=sp, store=MagicMock(), cfg=test_settings)
    result = await service.analyze("user-1")

    assert result.current_label == "neutral"


