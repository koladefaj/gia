"""Tests for ``DJService`` and ``build_dj_agent``."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture()
def fake_spotify_dj() -> MagicMock:
    # Real Spotify search results carry only metadata — no audio features.
    sp = MagicMock()
    sp.search_tracks = AsyncMock(return_value=[
        {"uri": "spotify:track:001", "name": "Free Mind", "artist": "Tems"},
        {"uri": "spotify:track:002", "name": "Last Last", "artist": "Burna Boy"},
        {"uri": "spotify:track:003", "name": "Calm Down", "artist": "Rema"},
        {"uri": "spotify:track:004", "name": "Essence", "artist": "Wizkid"},
        {"uri": "spotify:track:005", "name": "Ye", "artist": "Burna Boy"},
    ])
    sp.start_playback = AsyncMock(return_value={"status": "playing"})
    return sp


@pytest.mark.asyncio
async def test_dj_search_only_returns_seed_and_queue(fake_spotify_dj, test_settings) -> None:
    """``search_only`` returns (seed, queue) without LLM or playback."""
    from backend.app.agents.dj import DJService

    service = DJService(spotify=fake_spotify_dj, cfg=test_settings)
    seed, queue = await service.search_only("chill Afrobeats", n=2)

    assert seed.uri == "spotify:track:001"
    assert len(queue) == 2
    fake_spotify_dj.start_playback.assert_not_called()


@pytest.mark.asyncio
async def test_dj_recommend_uses_prefetched_without_searching(test_settings) -> None:
    """``prefetched`` is used verbatim — recommend does not search again."""
    from backend.app.agents.dj import DJService
    from backend.app.schemas.dj import TrackItem

    sp = MagicMock()
    sp.search_tracks = AsyncMock(return_value=[])  # would raise if called
    seed = TrackItem(uri="spotify:track:zz", name="SICKO MODE", artist="Travis Scott")
    prefetched = (seed, [TrackItem(uri="spotify:track:yy", name="FE!N", artist="Travis Scott")])

    with patch("backend.app.agents.dj.asyncio.to_thread", new=AsyncMock(return_value="On it.")):
        service = DJService(spotify=sp, cfg=test_settings)
        result = await service.recommend("Travis Scott", prefetched=prefetched)

    assert result.primary_track.uri == "spotify:track:zz"
    assert len(result.queue.tracks) == 1
    sp.search_tracks.assert_not_called()


@pytest.mark.asyncio
async def test_dj_service_returns_response(fake_spotify_dj, test_settings) -> None:
    """``DJService.recommend`` returns a ``DJResponse`` with seed + queue."""
    from backend.app.agents.dj import DJService

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

    with patch("backend.app.agents.dj.asyncio.to_thread", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
        service = DJService(spotify=fake_spotify_dj, cfg=test_settings)
        result = await service.recommend("chill vibes", n=2)

    assert "Track" in result.recommendation or "vibe" in result.recommendation


@pytest.mark.asyncio
async def test_dj_prompt_flags_named_track_mismatch(fake_spotify_dj, test_settings) -> None:
    """When the user named a specific track, the prompt instructs a 'did you mean' check."""
    from backend.app.agents.dj import DJService

    to_thread = AsyncMock(return_value="ok")
    with patch("backend.app.agents.dj.asyncio.to_thread", to_thread):
        service = DJService(spotify=fake_spotify_dj, cfg=test_settings)
        await service.recommend(
            "Virginia Island Drake", requested_titles=["Virginia Island"], n=2,
        )

    prompt = to_thread.call_args.args[1][0]["content"]
    assert "Virginia Island" in prompt
    assert "couldn't find" in prompt  # the mismatch disclaimer is wired in


@pytest.mark.asyncio
async def test_dj_per_title_queues_named_tracks_in_order(test_settings) -> None:
    """≥2 named titles → each searched separately, queued in the user's order."""
    from backend.app.agents.dj import DJService

    async def fake_search(query, limit=10):
        # Each title resolves to its own track (top hit named after the query).
        return [{"uri": f"spotify:track:{query}", "name": query, "artist": "Hillsong"}]

    sp = MagicMock()
    sp.search_tracks = AsyncMock(side_effect=fake_search)

    with patch("backend.app.agents.dj.asyncio.to_thread", new=AsyncMock(return_value="ok")):
        service = DJService(spotify=sp, cfg=test_settings)
        result = await service.recommend(
            "So Will I Hillsong Promises",
            requested_titles=["So Will I", "Promises"],
        )

    assert result.primary_track.name == "So Will I"
    assert [t.name for t in result.queue.tracks] == ["Promises"]
    # Each title was searched on its own, not as one combined query.
    searched = [c.args[0] for c in sp.search_tracks.call_args_list]
    assert searched == ["So Will I", "Promises"]
    # Audio features are never fetched (Spotify no longer exposes them).
    assert not hasattr(sp.get_audio_features, "assert_called") or not sp.get_audio_features.called


@pytest.mark.asyncio
async def test_dj_per_title_reports_missing_titles(test_settings) -> None:
    """A named title Spotify can't find is surfaced to the prompt as missing."""
    from backend.app.agents.dj import DJService

    async def fake_search(query, limit=10):
        if query == "Ghost Song":
            return []
        return [{"uri": "spotify:track:1", "name": query, "artist": "A"}]

    sp = MagicMock()
    sp.search_tracks = AsyncMock(side_effect=fake_search)

    to_thread = AsyncMock(return_value="ok")
    with patch("backend.app.agents.dj.asyncio.to_thread", to_thread):
        service = DJService(spotify=sp, cfg=test_settings)
        await service.recommend("q", requested_titles=["Real Song", "Ghost Song"])

    prompt = to_thread.call_args.args[1][0]["content"]
    assert "Ghost Song" in prompt and "could NOT find" in prompt


@pytest.mark.asyncio
async def test_dj_prompt_has_no_mismatch_block_for_vibe(fake_spotify_dj, test_settings) -> None:
    """A vibe request (no named track) gets no 'did you mean' instruction."""
    from backend.app.agents.dj import DJService

    fake_spotify_dj.get_audio_features = AsyncMock(side_effect=lambda uris: [
        {"uri": u, "name": "T", "artist": "A", "energy": 0.5, "valence": 0.5,
         "tempo": 100.0, "key": 0, "mode": 1, "danceability": 0.5}
        for u in uris
    ])

    to_thread = AsyncMock(return_value="ok")
    with patch("backend.app.agents.dj.asyncio.to_thread", to_thread):
        service = DJService(spotify=fake_spotify_dj, cfg=test_settings)
        await service.recommend("something chill", n=2)

    prompt = to_thread.call_args.args[1][0]["content"]
    assert "couldn't find" not in prompt


