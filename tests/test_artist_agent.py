"""Tests for ``ArtistService`` and ``build_artist_agent``."""

from __future__ import annotations

from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.agents.artist import extract_artist_name


class TestExtractArtistName:
    """The gate that stops small talk being looked up as an artist."""

    @pytest.mark.parametrize("message,expected", [
        ("tell me about Odumodublvck", "Odumodublvck"),
        ("tell me about Tems lately", "Tems"),
        ("who is Burna Boy?", "Burna Boy"),
        ("who's Asake", "Asake"),
        ("what about Wizkid", "Wizkid"),
        ("what has Drake been up to", "Drake"),
        ("anything new from Rema", "Rema"),
        ("Tems", "Tems"),
        ("Burna Boy", "Burna Boy"),
    ])
    def test_extracts_name(self, message: str, expected: str) -> None:
        assert extract_artist_name(message) == expected

    @pytest.mark.parametrize("message", [
        "whats the weather like",
        "how are you doing",
        "hey gia",
        "find me something chill",
        "what's my mood",
        "thanks so much",
        "",
        "i'm a little tired",
        # Conversational affirmations — replies to "want me to play him?", never
        # an artist to look up (this is what produced the "crossword" hallucination).
        "yeah sure",
        "yeah",
        "nah",
        "yep",
        "cool nice",
        "alright sure",
        "maybe",
    ])
    def test_rejects_non_artist(self, message: str) -> None:
        assert extract_artist_name(message) == ""


@pytest.fixture()
def fake_spotify_artist() -> MagicMock:
    sp = MagicMock()
    sp.search_tracks = AsyncMock(return_value=[
        {"uri": "spotify:track:a01", "name": "Declan", "artist": "Odumodublvck"},
        {"uri": "spotify:track:a02", "name": "Greek God", "artist": "Odumodublvck"},
    ])
    return sp


@pytest.fixture()
def fake_brave_results() -> list[dict]:
    return [
        {
            "title": "Odumodublvck wins award at Headies 2026",
            "url": "https://example.com/headies",
            "description": "The Abuja-born rapper took home Rap Album of the Year.",
        }
    ]


@pytest.mark.asyncio
async def test_artist_service_returns_response(
    fake_spotify_artist, fake_brave_results, test_settings
) -> None:
    """``ArtistService.get_info`` returns a populated ``ArtistInfoResponse``."""
    from backend.app.agents.artist import ArtistService
    from backend.app.tools.brave import BraveSearchClient

    fake_brave = MagicMock(spec=BraveSearchClient)
    fake_brave.search = AsyncMock(return_value=fake_brave_results)

    with patch("backend.app.agents.artist.asyncio.to_thread", new=AsyncMock(
        return_value="[curious] Odumodublvck has been on a tear lately."
    )):
        service = ArtistService(spotify=fake_spotify_artist, brave=fake_brave, cfg=test_settings)
        result = await service.get_info("Odumodublvck")

    assert result.artist_name == "Odumodublvck"
    assert "Odumodublvck" in result.response
    assert len(result.top_tracks) == 2
    assert result.recent_news[0].title == "Odumodublvck wins award at Headies 2026"


@pytest.mark.asyncio
async def test_artist_service_handles_spotify_error(fake_brave_results, test_settings) -> None:
    """Spotify errors are caught — response still generated with empty tracks."""
    from backend.app.agents.artist import ArtistService
    from backend.app.tools.brave import BraveSearchClient

    sp = MagicMock()
    sp.search_tracks = AsyncMock(side_effect=RuntimeError("Spotify down"))

    fake_brave = MagicMock(spec=BraveSearchClient)
    fake_brave.search = AsyncMock(return_value=fake_brave_results)

    with patch("backend.app.agents.artist.asyncio.to_thread", new=AsyncMock(return_value="Still got info.")):
        service = ArtistService(spotify=sp, brave=fake_brave, cfg=test_settings)
        result = await service.get_info("Odumodublvck")

    assert result.top_tracks == []
    assert result.artist_name == "Odumodublvck"


@pytest.mark.asyncio
async def test_artist_service_handles_brave_error(fake_spotify_artist, test_settings) -> None:
    """Brave errors are caught — response generated with empty news."""
    from backend.app.agents.artist import ArtistService
    from backend.app.tools.brave import BraveSearchClient

    fake_brave = MagicMock(spec=BraveSearchClient)
    fake_brave.search = AsyncMock(side_effect=RuntimeError("Brave down"))

    with patch("backend.app.agents.artist.asyncio.to_thread", new=AsyncMock(return_value="No recent news.")):
        service = ArtistService(spotify=fake_spotify_artist, brave=fake_brave, cfg=test_settings)
        result = await service.get_info("Odumodublvck")

    assert result.recent_news == []


@pytest.mark.asyncio
async def test_artist_service_llm_error_falls_back(
    fake_spotify_artist, fake_brave_results, test_settings
) -> None:
    """LLM errors produce a safe fallback string."""
    from backend.app.agents.artist import ArtistService
    from backend.app.tools.brave import BraveSearchClient

    fake_brave = MagicMock(spec=BraveSearchClient)
    fake_brave.search = AsyncMock(return_value=fake_brave_results)

    with patch("backend.app.agents.artist.asyncio.to_thread", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
        service = ArtistService(spotify=fake_spotify_artist, brave=fake_brave, cfg=test_settings)
        result = await service.get_info("Odumodublvck")

    assert "Odumodublvck" in result.response
    assert "moment" in result.response


@pytest.mark.asyncio
async def test_artist_service_includes_user_memory(
    fake_spotify_artist, fake_brave_results, test_settings
) -> None:
    """User history from Weaviate is fetched when user_id is provided."""
    from datetime import datetime

    from backend.app.agents.artist import ArtistService
    from backend.app.schemas.memory import MemoryEntry
    from backend.app.tools.brave import BraveSearchClient

    fake_brave = MagicMock(spec=BraveSearchClient)
    fake_brave.search = AsyncMock(return_value=fake_brave_results)

    mem = MemoryEntry(
        id="abc",
        type="preference",
        text="User loves Odumodublvck's aggressive flow",
        confidence=0.9,
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
    )

    fake_store = MagicMock()
    fake_store.search = AsyncMock(return_value=[mem])

    captured_prompt: list[str] = []

    async def fake_to_thread(fn, *args, **kwargs):
        prompt = args[0][0]["content"] if args else ""
        captured_prompt.append(prompt)
        return "Great response."

    with patch("backend.app.agents.artist.asyncio.to_thread", new=fake_to_thread), \
         patch("backend.app.agents.artist.embed", new=AsyncMock(return_value=[0.0] * 768)):
        service = ArtistService(
            spotify=fake_spotify_artist, brave=fake_brave, cfg=test_settings, store=fake_store
        )
        await service.get_info("Odumodublvck", user_id="00000000-0000-0000-0000-000000000001")

    assert any("aggressive flow" in p for p in captured_prompt)


