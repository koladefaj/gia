"""Pydantic schemas for the DJ agent."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrackItem(BaseModel):
    """A single track with its Spotify and audio-feature metadata.

    Attributes:
        uri:          Spotify URI (``spotify:track:...``).
        name:         Track name.
        artist:       Primary artist name.
        energy:       Spotify audio feature 0–1 (loud/fast = high).
        valence:      Spotify audio feature 0–1 (happy = high).
        tempo:        BPM.
        key:          Spotify key integer 0–11 (C=0, C♯=1 …).
        mode:         0 = minor, 1 = major.
        danceability: Spotify audio feature 0–1.
        camelot_key:  Camelot wheel label (e.g. ``"8B"``), or ``None``.
    """

    uri: str
    name: str
    artist: str
    energy: float = 0.5
    valence: float = 0.5
    tempo: float = 120.0
    key: int = 0
    mode: int = 1
    danceability: float = 0.5
    camelot_key: str | None = None


class CrossfadeQueue(BaseModel):
    """An ordered list of tracks for sequential playback with crossfade metadata.

    Attributes:
        seed_uri:      The track that anchors the queue's energy and key.
        tracks:        Ordered tracks to play after the seed.
        crossfade_ms:  Overlap in milliseconds between consecutive tracks.
    """

    seed_uri: str
    tracks: list[TrackItem]
    crossfade_ms: int = 3000


class DJRequest(BaseModel):
    """Request body for ``POST /dj/recommend``.

    Attributes:
        query:           Natural-language request ("something chill, Afrobeats").
        user_id:         Optional UUID — enables context-aware recommendations.
        start_playback:  If ``True``, immediately starts the seed track on Spotify.
        n:               Number of tracks to sequence after the seed (default 4).
    """

    query: str
    user_id: str | None = None
    start_playback: bool = False
    n: int = Field(default=4, ge=1, le=10)


class DJResponse(BaseModel):
    """Response from ``POST /dj/recommend``.

    Attributes:
        recommendation:  Gia's natural-language reasoning for the pick.
        primary_track:   The recommended seed track.
        queue:           The ordered crossfade queue.
        playback_started: Whether Spotify playback was started in this call.
    """

    recommendation: str
    primary_track: TrackItem
    queue: CrossfadeQueue
    playback_started: bool
