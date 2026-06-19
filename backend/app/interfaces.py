"""Protocol interfaces for all external service clients.

Defining explicit protocols instead of inheriting concrete classes keeps
every dependency on an abstraction, not an implementation. Swap mock ↔ live
by injecting a different concrete class — callers never change.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class SpotifyClientProtocol(Protocol):
    """Interface for all Spotify operations used by Gia's agents.

    Concrete implementations:
      - ``SpotifyMCPClient`` — calls the MCP server / Spotify Web API
      - ``MockSpotifyClient`` — deterministic fixture data for testing
    """

    async def get_currently_playing(self) -> dict | None:
        """Return the currently playing track or ``None`` if nothing is playing."""
        ...

    async def get_recently_played(self, limit: int = 10) -> list[dict]:
        """Return the ``limit`` most recently played tracks, newest first."""
        ...

    async def get_top_artists(self, time_range: str = "medium_term", limit: int = 10) -> list[dict]:
        """Return the user's top artists for the given time range.

        Args:
            time_range: One of ``short_term`` (4 wks), ``medium_term`` (6 mo),
                        ``long_term`` (all time).
            limit: Number of artists to return (max 50).
        """
        ...

    async def get_audio_features(self, uris: list[str]) -> list[dict]:
        """Return audio feature objects for the given Spotify track URIs.

        Features include: energy, valence, tempo, danceability, key, mode.
        """
        ...

    async def search_tracks(self, query: str, limit: int = 10) -> list[dict]:
        """Search Spotify for tracks matching *query*."""
        ...

    async def start_playback(self, uri: str, device_id: str | None = None) -> dict:
        """Start playback of track *uri* on the given device (or active device)."""
        ...

    async def save_track(self, uri: str) -> dict:
        """Save track *uri* to the user's Liked Songs library."""
        ...

    async def add_to_queue(self, uri: str) -> dict:
        """Add track *uri* to the end of the user's playback queue."""
        ...

    async def create_playlist(self, name: str, description: str = "") -> dict:
        """Create a new playlist and return its metadata (id, uri, name)."""
        ...

    async def add_tracks_to_playlist(self, playlist_id: str, uris: list[str]) -> dict:
        """Add *uris* to the playlist identified by *playlist_id*."""
        ...

    async def get_artist_info(self, artist_id: str) -> dict:
        """Return artist metadata (name, genres, popularity) for *artist_id*."""
        ...

    async def get_artist_top_tracks(self, artist_id: str) -> list[dict]:
        """Return the top tracks for *artist_id* in the user's market."""
        ...


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Minimal interface Gia's agents expect from any LLM implementation."""

    def call(self, messages: list[dict], **kwargs: object) -> str:
        """Send *messages* to the model and return the text response."""
        ...
