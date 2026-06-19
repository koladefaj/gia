"""Shared pytest fixtures and application test wiring.

Design principles
-----------------
- No network calls in unit tests.  Every external service is replaced by a
  ``Fake*`` implementation that satisfies the same ``Protocol`` interface.
- ``app.dependency_overrides`` is used for injection — no monkey-patching of
  modules, so overrides are scoped to the test and don't bleed across tests.
- Database tests use a real async SQLite engine (``aiosqlite``).  This keeps
  schema validation honest without needing a running Postgres instance.
- ``AsyncClient`` from ``httpx`` drives the FastAPI app via ASGI transport so
  routes are exercised end-to-end including middleware and lifespan events.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.config import Settings
from backend.app.db.base import Base
from backend.app.dependencies import get_brave_client, get_db, get_redis, get_settings, get_spotify_client, get_weather_client, get_weaviate_client
from backend.app.interfaces import SpotifyClientProtocol
from backend.app.main import app as _real_app


# ── Test settings ─────────────────────────────────────────────────────────────


@pytest.fixture()
def test_settings() -> Settings:
    """Return a ``Settings`` instance with safe test values.

    Uses an in-memory SQLite URL so no real Postgres is required for unit tests.
    All API keys are replaced with recognisable dummy values to make assertion
    errors easier to diagnose.
    """
    return Settings(
        app_env="development",
        log_level="debug",
        secret_key="test-secret",
        database_url="sqlite+aiosqlite:///:memory:",
        weaviate_url="http://weaviate-test:8080",
        redis_url="redis://redis-test:6379/0",
        llm_provider="anthropic",
        anthropic_api_key="sk-ant-test",
        openai_api_key="sk-openai-test",
        spotify_client_id="spotify-client-id-test",
        spotify_client_secret="spotify-client-secret-test",
        spotify_redirect_uri="http://localhost:8000/auth/spotify/callback",
        spotify_mcp_url="http://spotify-mcp-test:3001",
        langfuse_public_key="",
        langfuse_secret_key="",
        celery_broker_url="redis://redis-test:6379/1",
        celery_result_backend="redis://redis-test:6379/2",
    )


# ── In-memory SQLite database ─────────────────────────────────────────────────


@pytest_asyncio.fixture()
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async SQLAlchemy session backed by an in-memory SQLite database.

    Tables are created fresh for each test and dropped afterwards, giving every
    test a clean schema without needing a running Postgres instance.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ── Fake Spotify client ───────────────────────────────────────────────────────

_FAKE_TRACKS = [
    {
        "uri": "spotify:track:001",
        "name": "Free Mind",
        "artist": "Tems",
        "energy": 0.38,
        "valence": 0.71,
        "tempo": 92.0,
        "key": 5,
        "mode": 0,
        "danceability": 0.62,
    },
    {
        "uri": "spotify:track:002",
        "name": "Last Last",
        "artist": "Burna Boy",
        "energy": 0.78,
        "valence": 0.68,
        "tempo": 118.0,
        "key": 7,
        "mode": 1,
        "danceability": 0.80,
    },
]

_FAKE_ARTISTS = [
    {"uri": "spotify:artist:a01", "name": "Tems", "genres": ["afropop", "r&b"]},
    {"uri": "spotify:artist:a02", "name": "Burna Boy", "genres": ["afrobeats"]},
]


class FakeSpotifyClient:
    """Deterministic in-memory implementation of ``SpotifyClientProtocol``.

    All methods return the minimal data needed for unit tests.  Callers can
    mutate ``self.tracks`` and ``self.artists`` in a test to control responses
    without subclassing.

    This class is intentionally **not** a ``MagicMock`` so that tests break
    loudly if the ``SpotifyClientProtocol`` interface changes and the fake
    diverges from it.
    """

    def __init__(self) -> None:
        self.tracks: list[dict] = list(_FAKE_TRACKS)
        self.artists: list[dict] = list(_FAKE_ARTISTS)
        self.playback_started: list[str] = []
        self.saved_tracks: list[str] = []
        self.queued_tracks: list[str] = []
        self.created_playlists: list[dict] = []

    async def get_currently_playing(self) -> dict | None:
        """Return the first fake track as if it were currently playing."""
        return {**self.tracks[0], "is_playing": True, "progress_ms": 45000}

    async def get_recently_played(self, limit: int = 10) -> list[dict]:
        """Return up to *limit* fake tracks."""
        return self.tracks[:limit]

    async def get_top_artists(self, time_range: str = "medium_term", limit: int = 10) -> list[dict]:
        """Return up to *limit* fake artists."""
        return self.artists[:limit]

    async def get_audio_features(self, uris: list[str]) -> list[dict]:
        """Return fake audio features for each URI in *uris*."""
        by_uri = {t["uri"]: t for t in self.tracks}
        return [by_uri.get(u, self.tracks[0]) for u in uris]

    async def search_tracks(self, query: str, limit: int = 10) -> list[dict]:
        """Return all fake tracks regardless of *query*."""
        return self.tracks[:limit]

    async def start_playback(self, uri: str, device_id: str | None = None) -> dict:
        """Record that playback was started and return confirmation."""
        self.playback_started.append(uri)
        return {"status": "playing", "uri": uri}

    async def save_track(self, uri: str) -> dict:
        """Record that the track was saved and return confirmation."""
        self.saved_tracks.append(uri)
        return {"status": "saved", "uri": uri}

    async def add_to_queue(self, uri: str) -> dict:
        """Record that the track was queued and return confirmation."""
        self.queued_tracks.append(uri)
        return {"status": "queued", "uri": uri}

    async def create_playlist(self, name: str, description: str = "") -> dict:
        """Record the new playlist and return metadata."""
        playlist = {"id": f"fake-{name}", "name": name, "uri": f"spotify:playlist:fake-{name}"}
        self.created_playlists.append(playlist)
        return playlist

    async def add_tracks_to_playlist(self, playlist_id: str, uris: list[str]) -> dict:
        """Return confirmation without mutating state."""
        return {"status": "ok", "added": len(uris)}

    async def get_artist_info(self, artist_id: str) -> dict:
        """Return the first fake artist."""
        return self.artists[0]

    async def get_artist_top_tracks(self, artist_id: str) -> list[dict]:
        """Return all fake tracks as the artist's top tracks."""
        return self.tracks


assert isinstance(FakeSpotifyClient(), SpotifyClientProtocol), (
    "FakeSpotifyClient does not satisfy SpotifyClientProtocol — update the fake."
)


@pytest.fixture()
def fake_spotify() -> FakeSpotifyClient:
    """Return a fresh ``FakeSpotifyClient`` for each test."""
    return FakeSpotifyClient()


# ── Fake Redis ────────────────────────────────────────────────────────────────


@pytest.fixture()
def fake_redis() -> AsyncMock:
    """Return an ``AsyncMock`` that mimics an async Redis client.

    Pre-configured methods:
      - ``ping()`` → ``True``
      - ``get(key)`` → ``None`` (override per-test with ``side_effect``)
      - ``setex()`` → ``True``
      - ``delete()`` → 1
    """
    mock = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    mock.get = AsyncMock(return_value=None)
    mock.setex = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.aclose = AsyncMock()
    return mock


# ── FastAPI test client ───────────────────────────────────────────────────────


@pytest_asyncio.fixture()
async def client(
    test_settings: Settings,
    fake_spotify: FakeSpotifyClient,
    fake_redis: AsyncMock,
    db_session: AsyncSession,
) -> AsyncGenerator[AsyncClient, None]:
    """Yield an ``httpx.AsyncClient`` wired to the real FastAPI app.

    Dependency overrides replace every external service with a fake so tests
    run without any live infrastructure:

    - ``get_settings``      → ``test_settings``
    - ``get_db``            → ``db_session`` (SQLite in-memory)
    - ``get_redis``         → ``fake_redis`` (AsyncMock)
    - ``get_spotify_client``→ ``fake_spotify`` (FakeSpotifyClient)

    The app lifespan is bypassed by setting ``app.state`` directly, which
    avoids needing real Postgres/Weaviate/Redis at startup.
    """
    fake_weaviate = MagicMock()
    fake_brave = AsyncMock()
    fake_brave.search = AsyncMock(return_value=[])
    from backend.app.tools.weather import MockWeatherClient
    fake_weather = MockWeatherClient()

    _real_app.dependency_overrides[get_settings] = lambda: test_settings
    _real_app.dependency_overrides[get_db] = lambda: db_session
    _real_app.dependency_overrides[get_redis] = lambda: fake_redis
    _real_app.dependency_overrides[get_spotify_client] = lambda: fake_spotify
    _real_app.dependency_overrides[get_weaviate_client] = lambda: fake_weaviate
    _real_app.dependency_overrides[get_brave_client] = lambda: fake_brave
    _real_app.dependency_overrides[get_weather_client] = lambda: fake_weather

    # Set app state directly so lifespan doesn't try to connect to real services
    _real_app.state.redis = fake_redis
    _real_app.state.spotify = fake_spotify
    _real_app.state.weaviate = fake_weaviate

    async with AsyncClient(
        transport=ASGITransport(app=_real_app), base_url="http://test"
    ) as c:
        yield c

    _real_app.dependency_overrides.clear()
