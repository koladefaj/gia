"""Pydantic schemas for the Artist agent."""

from __future__ import annotations

from pydantic import BaseModel


class ArtistInfoRequest(BaseModel):
    """Request body for ``POST /artist/info``.

    Attributes:
        artist_name: The artist to research (e.g. ``"Odumodublvck"``).
        user_id:     Optional UUID — enables personalised response referencing
                     the user's history with this artist.
    """

    artist_name: str
    user_id: str | None = None


class BraveResult(BaseModel):
    """A single Brave Search result.

    Attributes:
        title:       Page title.
        url:         Page URL.
        description: Short description or snippet.
    """

    title: str
    url: str
    description: str


class ArtistInfoResponse(BaseModel):
    """Response from ``POST /artist/info``.

    Attributes:
        artist_name:  The queried artist name.
        response:     Gia's warm, personalised narrative about the artist.
        top_tracks:   Spotify top tracks for the artist.
        recent_news:  Brave Search results for the artist (recent activity).
    """

    artist_name: str
    response: str
    top_tracks: list[dict]
    recent_news: list[BraveResult]
