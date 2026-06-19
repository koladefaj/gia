"""Tests for ``DJService`` and ``build_dj_agent``."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture()
def fake_spotify_dj() -> MagicMock:
    sp = MagicMock()
    sp.search_tracks = AsyncMock(return_value=[
        {"uri": "spotify:track:001", "name": "Free Mind", "artist": "Tems",
         "energy": 0.38, "valence": 0.71, "tempo": 92.0, "key": 5, "mode": 0, "danceability": 0.62},
        {"uri": "spotify:track:002", "name": "Last Last", "artist": "Burna Boy",
         "energy": 0.78, "valence": 0.68, "tempo": 118.0, "key": 7, "mode": 1, "danceability": 0.80},
        {"uri": "spotify:track:003", "name": "Calm Down", "artist": "Rema",
         "energy": 0.55, "valence": 0.74, "tempo": 107.0, "key": 9, "mode": 0, "danceability": 0.75},
        {"uri": "spotify:track:004", "name": "Essence", "artist": "Wizkid",
         "energy": 0.42, "valence": 0.80, "tempo": 96.0, "key": 2, "mode": 1, "danceability": 0.70},
        {"uri": "spotify:track:005", "name": "Ye", "artist": "Burna Boy",
         "energy": 0.60, "valence": 0.65, "tempo": 110.0, "key": 4, "mode": 0, "danceability": 0.72},
    ])
    sp.get_audio_features = AsyncMock(side_effect=lambda uris: [
        {"uri": u, "name": f"Track {u[-3:]}", "artist": "Artist",
         "energy": 0.5, "valence": 0.5, "tempo": 100.0, "key": 0, "mode": 1, "danceability": 0.5}
        for u in uris
    ])
    sp.start_playback = AsyncMock(return_value={"status": "playing"})
    return sp


@pytest.mark.asyncio
async def test_dj_service_returns_response(fake_spotify_dj, test_settings) -> None:
    """``DJService.recommend`` returns a ``DJResponse`` with seed + queue."""
    from backend.app.agents.dj import DJService

    # Use search_tracks return directly as audio features too
    fake_spotify_dj.get_audio_features = AsyncMock(side_effect=lambda uris: [
        {"uri": u, "name": "Track", "artist": "Artist",
         "energy": 0.5, "valence": 0.5, "tempo": 100.0,
         "key": 0, "mode": 1, "danceability": 0.5}
        for u in uris
    ])

    with patch("backend.app.agents.dj.asyncio.to_thread", new=AsyncMock(return_value="Here's Free Mind. Fits perfectly.")):
        service = DJService(spotify=fake_spotify_dj, cfg=test_settings)
        result = await service.recommend("chill Afrobeats", n=3)

    assert result.recommendation != ""
    assert result.primary_track.uri == "spotify:track:001"
    assert len(result.queue.tracks) <= 3
    assert result.playback_started is False


@pytest.mark.asyncio
async def test_dj_service_starts_playback_when_requested(fake_spotify_dj, test_settings) -> None:
    """``start_playback=True`` triggers ``spotify.start_playback``."""
    from backend.app.agents.dj import DJService

    fake_spotify_dj.get_audio_features = AsyncMock(side_effect=lambda uris: [
        {"uri": u, "name": "T", "artist": "A", "energy": 0.5, "valence": 0.5,
         "tempo": 100.0, "key": 0, "mode": 1, "danceability": 0.5}
        for u in uris
    ])

    with patch("backend.app.agents.dj.asyncio.to_thread", new=AsyncMock(return_value="Playing now.")):
        service = DJService(spotify=fake_spotify_dj, cfg=test_settings)
        result = await service.recommend("hype", start_playback=True, n=2)

    assert result.playback_started is True
    fake_spotify_dj.start_playback.assert_called_once()


@pytest.mark.asyncio
async def test_dj_service_raises_when_no_tracks(test_settings) -> None:
    """``recommend`` raises ``ValueError`` when Spotify returns no tracks."""
    from backend.app.agents.dj import DJService

    sp = MagicMock()
    sp.search_tracks = AsyncMock(return_value=[])

    service = DJService(spotify=sp, cfg=test_settings)
    with pytest.raises(ValueError, match="No tracks found"):
        await service.recommend("totally obscure query")


@pytest.mark.asyncio
async def test_dj_service_llm_error_falls_back(fake_spotify_dj, test_settings) -> None:
    """LLM errors produce a safe fallback string, not an exception."""
    from backend.app.agents.dj import DJService

    fake_spotify_dj.get_audio_features = AsyncMock(side_effect=lambda uris: [
        {"uri": u, "name": "T", "artist": "A", "energy": 0.5, "valence": 0.5,
         "tempo": 100.0, "key": 0, "mode": 1, "danceability": 0.5}
        for u in uris
    ])

    with patch("backend.app.agents.dj.asyncio.to_thread", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
        service = DJService(spotify=fake_spotify_dj, cfg=test_settings)
        result = await service.recommend("chill vibes", n=2)

    assert "Track" in result.recommendation or "vibe" in result.recommendation


def test_build_dj_agent_returns_crewai_agent(test_settings) -> None:
    """``build_dj_agent`` returns a properly configured CrewAI ``Agent``."""
    from crewai import Agent

    from backend.app.agents.dj import build_dj_agent

    with patch("backend.app.agents.dj.get_llm", return_value="gpt-4o-mini"):
        agent = build_dj_agent(test_settings)

    assert isinstance(agent, Agent)
    assert "DJ" in agent.role
