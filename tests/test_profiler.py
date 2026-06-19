"""Tests for the cold-start Spotify taste profiler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.agents.memory import MemoryService
from backend.app.config import Settings
from backend.app.memory.profiler import bootstrap_taste_profile


def _cfg() -> Settings:
    return Settings(openai_api_key="sk-test")


def _spotify(artists: list[dict], tracks: list[dict], recent: list[dict]) -> MagicMock:
    sp = MagicMock()
    sp.get_top_artists = AsyncMock(return_value=artists)
    sp.get_top_tracks = AsyncMock(return_value=tracks)
    sp.get_recently_played = AsyncMock(return_value=recent)
    return sp


@pytest.mark.asyncio
async def test_bootstrap_profiles_and_persists() -> None:
    spotify = _spotify(
        artists=[{"name": "Burna Boy"}, {"name": "Tems"}],
        tracks=[{"name": "Last Last", "artist": "Burna Boy"}],
        recent=[],
    )
    fake_llm = MagicMock()
    # LLM returns an "episode" type — the profiler must force it to "preference".
    fake_llm.call = MagicMock(
        return_value='[{"type":"episode","text":"Core Afrobeats listener","confidence":0.9}]'
    )

    with patch("backend.app.memory.profiler.get_fast_llm", return_value=fake_llm), \
         patch.object(MemoryService, "persist_memories",
                      new=AsyncMock(return_value=["mem-1"])) as persist:
        ids = await bootstrap_taste_profile(
            "uid", spotify=spotify, store=MagicMock(), redis=MagicMock(), cfg=_cfg()
        )

    assert ids == ["mem-1"]
    spotify.get_top_artists.assert_awaited_once()
    spotify.get_top_tracks.assert_awaited_once()
    persisted = persist.call_args[0][1]
    assert persisted[0].type == "preference"
    assert "Afrobeats" in persisted[0].text


@pytest.mark.asyncio
async def test_bootstrap_no_listening_data_returns_empty() -> None:
    spotify = _spotify(artists=[], tracks=[], recent=[])
    with patch("backend.app.memory.profiler.get_fast_llm", side_effect=AssertionError):
        ids = await bootstrap_taste_profile(
            "uid", spotify=spotify, store=MagicMock(), redis=MagicMock(), cfg=_cfg()
        )
    assert ids == []


@pytest.mark.asyncio
async def test_bootstrap_degrades_on_spotify_error() -> None:
    spotify = MagicMock()
    spotify.get_top_artists = AsyncMock(side_effect=RuntimeError("mcp down"))
    spotify.get_top_tracks = AsyncMock(return_value=[{"name": "X", "artist": "Y"}])
    spotify.get_recently_played = AsyncMock(return_value=[])
    fake_llm = MagicMock()
    fake_llm.call = MagicMock(return_value="[]")

    with patch("backend.app.memory.profiler.get_fast_llm", return_value=fake_llm), \
         patch.object(MemoryService, "persist_memories", new=AsyncMock(return_value=[])):
        ids = await bootstrap_taste_profile(
            "uid", spotify=spotify, store=MagicMock(), redis=MagicMock(), cfg=_cfg()
        )
    # top_artists failed but top_tracks succeeded → still proceeds, returns []
    assert ids == []
