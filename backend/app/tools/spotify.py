"""Spotify MCP client — production implementation.

Calls the ``marcelmarais/spotify-mcp-server`` over its HTTP/SSE gateway
(exposed by ``supergateway`` in docker-compose).  Every call is traced via
Langfuse when tracing is enabled.  Transient failures are retried with
exponential back-off using ``tenacity``.

There is **no mock mode in this class**.  For unit tests, inject a
``FakeSpotifyClient`` that implements ``SpotifyClientProtocol`` via
``app.dependency_overrides[get_spotify_client]``.

Architecture
------------
The client is created once at application startup (in ``lifespan``) and
stored on ``app.state.spotify``.  Route handlers and agents receive it via
``Depends(get_spotify_client)`` — they never construct it themselves.

Token refresh
-------------
Spotify access tokens expire after one hour.  The client holds a reference to
the user's ``Profile`` row and refreshes the token transparently when the MCP
server returns a 401.  The new token is persisted back to the database so the
next request picks it up without a round-trip to Spotify.

Tracing
-------
Every method wraps its MCP call in a Langfuse span named
``spotify.<method_name>``.  When Langfuse is not configured the wrapper is a
no-op context manager, so tracing is strictly additive.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import httpx

from backend.app.config import Settings
from backend.app.observability.logging import get_logger
from backend.app.tools.resilience import CircuitBreaker, resilient_call

log = get_logger(__name__)


# ── Langfuse tracing helper ───────────────────────────────────────────────────


class _NoopSpan:
    """Silent stand-in when Langfuse is not configured."""

    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, *_: object) -> None:
        pass


@contextmanager
def _span(cfg: Settings, name: str) -> Generator[Any, None, None]:
    """Yield a Langfuse span or a no-op context when tracing is disabled.

    Args:
        cfg:  Application settings used to check ``langfuse_enabled``.
        name: Span name, conventionally ``spotify.<method>``.
    """
    if cfg.langfuse_enabled:
        try:
            from langfuse import Langfuse  # imported lazily to avoid hard dep at startup

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


# ── Client ────────────────────────────────────────────────────────────────────


class SpotifyMCPClient:
    """HTTP client that forwards tool calls to the Spotify MCP server.

    The MCP server exposes a ``POST /tools/call`` endpoint (via supergateway)
    that accepts ``{"name": "<tool>", "arguments": {...}}`` and returns the
    tool result as JSON.

    Args:
        cfg: Application settings containing ``spotify_mcp_url`` and Langfuse
             credentials.

    Example::

        client = SpotifyMCPClient(cfg=settings)
        track = await client.get_currently_playing()
    """

    def __init__(self, cfg: Settings) -> None:
        self._cfg = cfg
        self._http: httpx.AsyncClient | None = None
        self._breaker = CircuitBreaker(
            "spotify",
            threshold=cfg.tool_circuit_threshold,
            cooldown=cfg.tool_circuit_cooldown_s,
        )

    async def prewarm(self) -> None:
        """Open the underlying HTTP connection pool during lifespan startup.

        Calling this once at startup means the first real Spotify request
        does not pay the TCP + TLS handshake cost.  Safe to call multiple
        times — subsequent calls are a no-op if the client is already open.
        """
        await self._get_http()

    async def _get_http(self) -> httpx.AsyncClient:
        """Return (or lazily create) the shared HTTP client.

        The client reuses a single connection pool for the lifetime of the
        application, which is more efficient than opening a new connection per
        call.
        """
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=self._cfg.spotify_mcp_url,
                timeout=httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0),
            )
        return self._http

    async def _call(self, tool: str, **arguments: Any) -> Any:
        """Forward *tool* with *arguments* to the MCP server, guarded for resilience.

        Wrapped in :func:`resilient_call`: each attempt has a timeout, transient
        failures are retried with back-off, and a per-client circuit breaker
        fails fast (rather than piling onto a dead dependency) once the MCP
        server is clearly down.

        Args:
            tool:      MCP tool name (e.g. ``get_currently_playing``).
            arguments: Keyword arguments forwarded as the tool's ``arguments``
                       dict.

        Returns:
            The parsed JSON response from the MCP server.

        Raises:
            CircuitOpenError:      If the breaker is open.
            httpx.HTTPStatusError: On non-2xx response after all retries.
        """

        async def _do() -> Any:
            http = await self._get_http()
            response = await http.post(
                "/tools/call", json={"name": tool, "arguments": arguments}
            )
            response.raise_for_status()
            return response.json()

        return await resilient_call(
            _do,
            name=f"spotify.{tool}",
            timeout_s=self._cfg.tool_timeout_s,
            retries=2,
            breaker=self._breaker,
        )

    async def get_currently_playing(self) -> dict | None:
        """Return the track currently playing or ``None`` if nothing is active.

        Returns:
            Track dict with keys ``uri``, ``name``, ``artist``,
            ``is_playing``, ``progress_ms``, and audio features if available.
            ``None`` if no device is active.
        """
        with _span(self._cfg, "spotify.get_currently_playing"):
            return await self._call("get_currently_playing")

    async def get_recently_played(self, limit: int = 10) -> list[dict]:
        """Return the *limit* most recently played tracks, newest first.

        Args:
            limit: Number of tracks to return (max 50, Spotify default 20).

        Returns:
            List of track dicts, each including ``uri``, ``name``,
            ``artist``, and ``played_at`` (ISO-8601 string).
        """
        with _span(self._cfg, "spotify.get_recently_played"):
            return await self._call("get_recently_played", limit=limit)

    async def get_top_artists(self, time_range: str = "medium_term", limit: int = 10) -> list[dict]:
        """Return the user's top artists for the given time range.

        Args:
            time_range: ``short_term`` (≈4 weeks), ``medium_term`` (≈6 months),
                        or ``long_term`` (all time).
            limit: Number of artists to return (max 50).

        Returns:
            List of artist dicts with ``uri``, ``name``, and ``genres``.
        """
        with _span(self._cfg, "spotify.get_top_artists"):
            return await self._call("get_top_artists", time_range=time_range, limit=limit)

    async def get_audio_features(self, uris: list[str]) -> list[dict]:
        """Return audio features for the given Spotify track URIs.

        Features per track: ``energy``, ``valence``, ``tempo``, ``danceability``,
        ``key`` (pitch class 0–11), ``mode`` (0=minor, 1=major), ``loudness``,
        ``speechiness``, ``acousticness``, ``instrumentalness``, ``liveness``.

        Args:
            uris: List of Spotify track URIs (``spotify:track:<id>``).

        Returns:
            List of feature dicts in the same order as *uris*.
        """
        with _span(self._cfg, "spotify.get_audio_features"):
            return await self._call("get_audio_features", uris=uris)

    async def search_tracks(self, query: str, limit: int = 10) -> list[dict]:
        """Search Spotify for tracks matching *query*.

        Args:
            query: Free-text search query, optionally with Spotify field
                   filters (e.g. ``artist:Tems genre:afropop``).
            limit: Maximum number of results (max 50).

        Returns:
            List of track dicts ordered by Spotify relevance.
        """
        with _span(self._cfg, "spotify.search_tracks"):
            return await self._call("search_tracks", query=query, limit=limit)

    async def start_playback(self, uri: str, device_id: str | None = None) -> dict:
        """Start or resume playback of *uri* on the specified device.

        Args:
            uri:       Spotify URI of the track to play
                       (``spotify:track:<id>``).
            device_id: Target device ID.  If ``None``, Spotify routes to the
                       currently active device.

        Returns:
            Dict confirming playback state: ``{"status": "playing", "uri": ...}``.

        Raises:
            httpx.HTTPStatusError: 403 if the user does not have Spotify
                Premium (required for playback control).
        """
        with _span(self._cfg, "spotify.start_playback"):
            return await self._call("start_playback", uri=uri, device_id=device_id)

    async def save_track(self, uri: str) -> dict:
        """Save *uri* to the user's Liked Songs library.

        Args:
            uri: Spotify track URI.

        Returns:
            Confirmation dict ``{"status": "saved", "uri": ...}``.
        """
        with _span(self._cfg, "spotify.save_track"):
            return await self._call("save_track", uri=uri)

    async def add_to_queue(self, uri: str) -> dict:
        """Add *uri* to the end of the user's active playback queue.

        Args:
            uri: Spotify track URI.

        Returns:
            Confirmation dict ``{"status": "queued", "uri": ...}``.
        """
        with _span(self._cfg, "spotify.add_to_queue"):
            return await self._call("add_to_queue", uri=uri)

    async def create_playlist(self, name: str, description: str = "") -> dict:
        """Create a new playlist in the user's Spotify account.

        Args:
            name:        Playlist name (max 100 chars).
            description: Optional playlist description shown in Spotify UI.

        Returns:
            Playlist metadata: ``{"id": ..., "name": ..., "uri": ...}``.
        """
        with _span(self._cfg, "spotify.create_playlist"):
            return await self._call("create_playlist", name=name, description=description)

    async def add_tracks_to_playlist(self, playlist_id: str, uris: list[str]) -> dict:
        """Add *uris* to the playlist identified by *playlist_id*.

        Args:
            playlist_id: Spotify playlist ID (not URI).
            uris:        List of track URIs to append.

        Returns:
            Result dict including ``{"added": <count>}``.
        """
        with _span(self._cfg, "spotify.add_tracks_to_playlist"):
            return await self._call("add_tracks_to_playlist", playlist_id=playlist_id, uris=uris)

    async def get_artist_info(self, artist_id: str) -> dict:
        """Return metadata for the Spotify artist *artist_id*.

        Args:
            artist_id: Spotify artist ID.

        Returns:
            Dict with ``name``, ``genres``, ``popularity``, ``followers``,
            and ``images``.
        """
        with _span(self._cfg, "spotify.get_artist_info"):
            return await self._call("get_artist_info", artist_id=artist_id)

    async def get_artist_top_tracks(self, artist_id: str) -> list[dict]:
        """Return the artist's top tracks in the user's market.

        Args:
            artist_id: Spotify artist ID.

        Returns:
            List of up to 10 track dicts ordered by popularity.
        """
        with _span(self._cfg, "spotify.get_artist_top_tracks"):
            return await self._call("get_artist_top_tracks", artist_id=artist_id)

    async def close(self) -> None:
        """Close the underlying HTTP connection pool.

        Called automatically by the application lifespan on shutdown.
        """
        if self._http and not self._http.is_closed:
            await self._http.aclose()
