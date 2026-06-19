"""Tests for ``build_user_context`` — the context assembly function."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.schemas.memory import MemoryEntry, UserContext

_USER_ID = "00000000-0000-0000-0000-000000000001"
_NOW = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)


def _entry(text: str, uid: str = "00000000-0000-0000-0000-000000000011") -> MemoryEntry:
    return MemoryEntry(
        id=uid,
        type="preference",
        text=text,
        confidence=0.85,
        created_at=_NOW,
    )


@pytest.fixture()
def fake_store() -> MagicMock:
    store = MagicMock()
    store.search = AsyncMock(return_value=[])
    return store


@pytest.fixture()
def fake_redis() -> MagicMock:
    r = MagicMock()
    r.get = AsyncMock(return_value=None)
    return r


@pytest.fixture()
def fake_spotify() -> MagicMock:
    sp = MagicMock()
    sp.get_currently_playing = AsyncMock(return_value=None)
    sp.get_recently_played = AsyncMock(return_value=[])
    return sp


@pytest.fixture()
def fake_db() -> MagicMock:
    db = MagicMock()
    # scalar_one_or_none returns None → no profile
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = None
    db.execute = AsyncMock(return_value=execute_result)
    return db


@pytest.mark.asyncio
async def test_build_user_context_returns_user_context(
    fake_store, fake_redis, fake_spotify, fake_db
) -> None:
    """``build_user_context`` returns a ``UserContext`` with the correct user_id."""
    from backend.app.memory.retrieval import build_user_context

    with patch(
        "backend.app.memory.retrieval.embed",
        new=AsyncMock(return_value=[0.0] * 768),
    ):
        ctx = await build_user_context(
            _USER_ID,
            "wind-down music",
            db=fake_db,
            store=fake_store,
            redis=fake_redis,
            spotify=fake_spotify,
        )

    assert isinstance(ctx, UserContext)
    assert ctx.user_id == _USER_ID


@pytest.mark.asyncio
async def test_build_user_context_includes_preferences(
    fake_store, fake_redis, fake_spotify, fake_db
) -> None:
    """Preferences returned by Weaviate appear in the context."""
    fake_store.search = AsyncMock(side_effect=[
        [_entry("Loves Tems")],  # preferences
        [],                       # mood_patterns
        [],                       # episodes
    ])

    from backend.app.memory.retrieval import build_user_context

    with patch(
        "backend.app.memory.retrieval.embed",
        new=AsyncMock(return_value=[0.0] * 768),
    ):
        ctx = await build_user_context(
            _USER_ID,
            "music",
            db=fake_db,
            store=fake_store,
            redis=fake_redis,
            spotify=fake_spotify,
        )

    assert len(ctx.preferences) == 1
    assert ctx.preferences[0].text == "Loves Tems"


@pytest.mark.asyncio
async def test_build_user_context_handles_spotify_error(
    fake_store, fake_redis, fake_db
) -> None:
    """Spotify errors are caught; now_playing is ``None`` and recently_played is ``[]``."""
    bad_spotify = MagicMock()
    bad_spotify.get_currently_playing = AsyncMock(side_effect=RuntimeError("MCP down"))
    bad_spotify.get_recently_played = AsyncMock(side_effect=RuntimeError("MCP down"))

    from backend.app.memory.retrieval import build_user_context

    with patch(
        "backend.app.memory.retrieval.embed",
        new=AsyncMock(return_value=[0.0] * 768),
    ):
        ctx = await build_user_context(
            _USER_ID,
            "music",
            db=fake_db,
            store=fake_store,
            redis=fake_redis,
            spotify=bad_spotify,
        )

    assert ctx.now_playing is None
    assert ctx.recently_played == []


@pytest.mark.asyncio
async def test_build_user_context_includes_session_summary(
    fake_store, fake_redis, fake_spotify, fake_db
) -> None:
    """Redis session notes are surfaced in ``session_summary``."""
    fake_redis.get = AsyncMock(return_value="Asked for Afrobeats wind-down")

    from backend.app.memory.retrieval import build_user_context

    with patch(
        "backend.app.memory.retrieval.embed",
        new=AsyncMock(return_value=[0.0] * 768),
    ):
        ctx = await build_user_context(
            _USER_ID,
            "music",
            db=fake_db,
            store=fake_store,
            redis=fake_redis,
            spotify=fake_spotify,
        )

    assert ctx.session_summary == "Asked for Afrobeats wind-down"


@pytest.mark.asyncio
async def test_build_user_context_invalid_uuid_returns_no_profile(
    fake_store, fake_redis, fake_spotify, fake_db
) -> None:
    """Invalid UUID string produces ``profile=None`` rather than raising."""
    from backend.app.memory.retrieval import build_user_context

    with patch(
        "backend.app.memory.retrieval.embed",
        new=AsyncMock(return_value=[0.0] * 768),
    ):
        ctx = await build_user_context(
            "not-a-uuid",
            "music",
            db=fake_db,
            store=fake_store,
            redis=fake_redis,
            spotify=fake_spotify,
        )

    assert ctx.profile is None


class TestUserContextToPromptText:
    def _context(self, **kwargs) -> UserContext:
        return UserContext(user_id=_USER_ID, **kwargs)

    def test_empty_context_renders_header(self) -> None:
        ctx = self._context()
        text = ctx.to_prompt_text()
        assert "What Gia knows" in text

    def test_profile_rendered(self) -> None:
        ctx = self._context(
            profile={"timezone": "Africa/Lagos", "preferred_genres": ["afrobeats"], "preferred_volume": 0.6}
        )
        text = ctx.to_prompt_text()
        assert "Africa/Lagos" in text
        assert "afrobeats" in text

    def test_preferences_rendered(self) -> None:
        ctx = self._context(preferences=[_entry("Loves Tems")])
        text = ctx.to_prompt_text()
        assert "Loves Tems" in text
        assert "Preferences" in text

    def test_now_playing_rendered(self) -> None:
        ctx = self._context(now_playing={"name": "Free Mind", "artist": "Tems", "energy": 0.38})
        text = ctx.to_prompt_text()
        assert "Free Mind" in text
        assert "Tems" in text

    def test_session_summary_rendered(self) -> None:
        ctx = self._context(session_summary="User asked for wind-down playlist")
        text = ctx.to_prompt_text()
        assert "wind-down playlist" in text

    def test_no_profile_genres_shows_none_set(self) -> None:
        ctx = self._context(profile={"timezone": "UTC", "preferred_genres": [], "preferred_volume": 0.7})
        text = ctx.to_prompt_text()
        assert "none set" in text
