"""Spotify client backed by the ``marcelmarais/spotify-mcp-server`` over MCP stdio.

The server is a stdio Model Context Protocol server: we spawn it once
(``cfg.spotify_mcp_server_path``) and hold a single ``ClientSession`` for the
app's lifetime, created in ``prewarm`` and closed in ``close``.  Calls are
serialised through an ``asyncio.Lock`` because one stdio session multiplexes a
single pipe.

Two impedance mismatches are handled here so the rest of Gia sees a normal
``SpotifyClientProtocol``:

1. **Text, not JSON.** The MCP tools return formatted markdown; the parsers in
   ``spotify_parse`` turn that back into dicts.
2. **No audio features / artist endpoints.** The server exposes no
   ``audio-features`` (Spotify deprecated it) or artist-info tool, so
   ``get_audio_features`` returns neutral placeholders (the DJ still builds a
   queue, just without real key/energy data) and the artist helpers degrade
   gracefully.  ``ArtistService`` already gets an artist's tracks via
   ``search_tracks``, so nothing depends on the missing tools.

For unit tests, inject a ``FakeSpotifyClient`` via ``dependency_overrides`` — the
real client is never constructed in tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import AsyncExitStack, contextmanager
from typing import Any

from backend.app.config import Settings
from backend.app.observability.logging import get_logger
from backend.app.tools.spotify_parse import (
    parse_artists,
    parse_now_playing,
    parse_tracks,
)

log = get_logger(__name__)


# ── Langfuse tracing helper ───────────────────────────────────────────────────


class _NoopSpan:
    """Silent stand-in when Langfuse is not configured."""

    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *_: object) -> None:
        pass


@contextmanager
def _span(cfg: Settings, name: str) -> Generator[Any, None, None]:
    """Yield a Langfuse span or a no-op context when tracing is disabled."""
    if cfg.langfuse_enabled:
        try:
            from langfuse import Langfuse  # noqa: PLC0415

            lf = Langfuse(
                public_key=cfg.langfuse_public_key,
                secret_key=cfg.langfuse_secret_key,
                host=cfg.langfuse_host,
            )
            with lf.start_span(name=name) as span:
                yield span
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("langfuse_span_failed", name=name, error=str(exc))
    yield _NoopSpan()


# ── MCP stdio bridge ──────────────────────────────────────────────────────────


class _McpBridge:
    """Owns a single persistent MCP ``ClientSession`` over stdio.

    The session is opened in :meth:`start` (FastAPI ``lifespan`` startup) and
    closed in :meth:`stop` (shutdown).  ``call`` serialises tool invocations.
    """

    def __init__(self, command: str, server_path: str) -> None:
        self._command = command
        self._server_path = server_path
        self._stack: AsyncExitStack | None = None
        self._session: Any = None
        self._lock = asyncio.Lock()

    @property
    def started(self) -> bool:
        return self._session is not None

    async def start(self) -> None:
        """Spawn the MCP server and initialise the session."""
        from mcp import ClientSession, StdioServerParameters  # noqa: PLC0415
        from mcp.client.stdio import stdio_client  # noqa: PLC0415

        params = StdioServerParameters(command=self._command, args=[self._server_path])
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        log.info("spotify_mcp_session_started", server=self._server_path)

    async def stop(self) -> None:
        """Tear down the session and child process."""
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception as exc:  # noqa: BLE001
                log.warning("spotify_mcp_session_close_error", error=str(exc))
        self._stack = None
        self._session = None

    async def call(self, tool: str, arguments: dict) -> str:
        """Invoke *tool* and return its concatenated text content.

        Raises:
            RuntimeError: If the session has not been started.
        """
        if self._session is None:
            raise RuntimeError("Spotify MCP session is not started")
        async with self._lock:
            result = await self._session.call_tool(tool, arguments)
        # Concatenate any text-bearing content blocks. We don't filter on a
        # ``type`` field: across mcp SDK versions the attribute/value varies, and
        # filtering on it can silently drop every block (→ empty result).
        return "\n".join(
            text for c in result.content if (text := getattr(c, "text", "")) is not None and text
        )


# Neutral audio features used when real ones are unavailable (Spotify deprecated
# the endpoint and the MCP server exposes none). Keeps the DJ queue builder
# functional with degraded — but non-crashing — sequencing.
_NEUTRAL_FEATURES = {
    "energy": 0.5, "valence": 0.5, "tempo": 120.0,
    "danceability": 0.6, "key": 0, "mode": 1,
}


class SpotifyMCPClient:
    """``SpotifyClientProtocol`` implementation over the MCP stdio server.

    A single client is created at startup (``main.lifespan``) and shared via
    ``Depends(get_spotify_client)``.  When ``cfg.spotify_mcp_server_path`` is
    empty the client stays inert: ``prewarm`` is a no-op and every method raises,
    which callers already handle as a degraded-Spotify path.
    """

    def __init__(self, cfg: Settings) -> None:
        self._cfg = cfg
        self._bridge = _McpBridge(cfg.spotify_mcp_command, cfg.spotify_mcp_server_path)

    async def prewarm(self) -> None:
        """Start the MCP session at app startup (no-op if unconfigured)."""
        if not self._cfg.spotify_mcp_server_path:
            log.info("spotify_mcp_disabled", reason="spotify_mcp_server_path not set")
            return
        await self._bridge.start()

    async def close(self) -> None:
        """Stop the MCP session at app shutdown."""
        await self._bridge.stop()

    async def _call(self, tool: str, **arguments: Any) -> str:
        with _span(self._cfg, f"spotify.{tool}"):
            return await self._bridge.call(tool, arguments)

    # ── Reads ────────────────────────────────────────────────────────────────

    async def get_currently_playing(self) -> dict | None:
        return parse_now_playing(await self._call("getNowPlaying"))

    async def get_recently_played(self, limit: int = 10) -> list[dict]:
        return parse_tracks(await self._call("getRecentlyPlayed", limit=limit))

    async def get_top_artists(self, time_range: str = "medium_term", limit: int = 10) -> list[dict]:
        return parse_artists(
            await self._call("getTopArtists", timeRange=time_range, limit=limit)
        )

    async def get_audio_features(self, uris: list[str]) -> list[dict]:
        """Return neutral placeholder features (no real source available).

        Spotify deprecated ``/audio-features`` and the MCP server exposes no
        equivalent, so per-track energy/valence/key/mode cannot be fetched.  We
        return neutral values keyed by uri so the DJ's queue builder keeps
        working (Camelot sequencing is degraded but never crashes).
        """
        return [{"uri": uri, **_NEUTRAL_FEATURES} for uri in uris]

    # Spotify caps search ``limit`` at 10 for Development-Mode apps (a >10 value
    # returns HTTP 400 "Invalid limit"), so we clamp it here defensively.
    _SEARCH_LIMIT_MAX = 10

    async def search_tracks(self, query: str, limit: int = 10) -> list[dict]:
        capped = min(limit, self._SEARCH_LIMIT_MAX)
        return parse_tracks(
            await self._call("searchSpotify", query=query, type="track", limit=capped)
        )

    # ── Playback / writes ────────────────────────────────────────────────────

    async def start_playback(self, uri: str, device_id: str | None = None) -> dict:
        args: dict[str, Any] = {"uri": uri}
        if device_id:
            args["deviceId"] = device_id
        text = await self._call("playMusic", **args)
        return {"status": "playing", "uri": uri, "detail": text}

    async def save_track(self, uri: str) -> dict:
        """Saving a single track is not exposed by the MCP server (album-only)."""
        log.info("spotify_save_track_unsupported", uri=uri)
        return {"status": "unsupported", "uri": uri}

    async def add_to_queue(self, uri: str) -> dict:
        text = await self._call("addToQueue", uri=uri)
        return {"status": "queued", "uri": uri, "detail": text}

    async def create_playlist(self, name: str, description: str = "") -> dict:
        text = await self._call("createPlaylist", name=name, description=description)
        from backend.app.tools.spotify_parse import _ID  # noqa: PLC0415

        match = _ID.search(text)
        playlist_id = match.group("id") if match else ""
        return {"id": playlist_id, "name": name, "detail": text}

    async def add_tracks_to_playlist(self, playlist_id: str, uris: list[str]) -> dict:
        text = await self._call(
            "addTracksToPlaylist", playlistId=playlist_id, trackUris=uris
        )
        return {"status": "ok", "added": len(uris), "detail": text}

    # ── Artist helpers (no MCP tool — graceful fallbacks) ────────────────────

    async def get_artist_info(self, artist_id: str) -> dict:
        """No artist-info tool exists; return a minimal placeholder."""
        return {"id": artist_id, "name": "", "genres": []}

    async def get_artist_top_tracks(self, artist_id: str) -> list[dict]:
        """No artist-top-tracks tool exists; ArtistService uses search instead."""
        return []
