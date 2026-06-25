"""Parsers for the Spotify MCP server's text responses.

The ``marcelmarais/spotify-mcp-server`` tools are built for an LLM to *read*:
they return formatted markdown, not JSON.  These pure functions turn that text
back into the structured dicts the rest of Gia expects, so the parsing lives in
one tested place instead of being smeared through the client.

Observed line formats (from a live server)::

    1. "Free Mind" by Tems (4:08) - ID: 2mzM4Y0Rnx2BDZqRnhQ5Q6
    1. "Forgiveness" by Asake (2:39) - ID: 5u4... - Played at: 19/06/2026, 14:43:10
    1. Burna Boy - ID: 3wcj11K77LjEY1PkEazffa   (artist search)
"""

from __future__ import annotations

import re

# A numbered track line: `N. "NAME" by ARTIST (DURATION[, popularity: P]) - ID: ID [...]`
_TRACK_LINE = re.compile(
    r'^\s*\d+\.\s+"(?P<name>.+?)"\s+by\s+(?P<artist>.+?)\s+'
    r"\((?P<dur>[^)]*?)\)(?:,\s*popularity:\s*\d+)?\s+-\s+ID:\s+(?P<id>[A-Za-z0-9]+)",
    re.MULTILINE,
)

# A numbered artist line: `N. ARTIST NAME - ID: ID`
_ARTIST_LINE = re.compile(
    r"^\s*\d+\.\s+(?P<name>.+?)\s+-\s+ID:\s+(?P<id>[A-Za-z0-9]+)\s*$",
    re.MULTILINE,
)

# A standalone Spotify track id (22 base62 chars).
_ID = re.compile(r"\b(?P<id>[A-Za-z0-9]{22})\b")


def parse_tracks(text: str) -> list[dict]:
    """Parse a track-listing response into ``{uri, id, name, artist}`` dicts.

    Args:
        text: The MCP tool's text output (search / recently-played / queue).

    Returns:
        Ordered list of track dicts (empty if none matched).
    """
    out: list[dict] = []
    for m in _TRACK_LINE.finditer(text):
        track_id = m.group("id")
        out.append(
            {
                "uri": f"spotify:track:{track_id}",
                "id": track_id,
                "name": m.group("name").strip(),
                "artist": m.group("artist").strip(),
            }
        )
    return out


def parse_artists(text: str) -> list[dict]:
    """Parse a ``getTopArtists`` / artist-search response into artist dicts.

    Track lines (which also end in ``- ID:``) are excluded so a mixed response
    doesn't leak tracks into the artist list.

    Args:
        text: The MCP tool's text output.

    Returns:
        Ordered list of ``{uri, id, name, genres}`` dicts.
    """
    track_ids = {t["id"] for t in parse_tracks(text)}
    out: list[dict] = []
    for m in _ARTIST_LINE.finditer(text):
        artist_id = m.group("id")
        if artist_id in track_ids:
            continue
        out.append(
            {
                "uri": f"spotify:artist:{artist_id}",
                "id": artist_id,
                "name": m.group("name").strip(),
                "genres": [],
            }
        )
    return out


def parse_now_playing(text: str) -> dict | None:
    """Parse a ``getNowPlaying`` response into a track dict, or ``None``.

    The server returns a sentence like *"No track is currently playing"* when
    idle; any response without a recoverable track id yields ``None``.

    Args:
        text: The MCP tool's text output.

    Returns:
        ``{uri, id, name, artist, is_playing}`` or ``None``.
    """
    lowered = text.lower()
    if "no track" in lowered or "not playing" in lowered or "nothing" in lowered:
        return None

    tracks = parse_tracks(text)
    if tracks:
        t = tracks[0]
        return {**t, "is_playing": True}

    # Fallback: best-effort name/artist + a bare id.
    id_match = _ID.search(text)
    name_match = re.search(r'"(?P<name>.+?)"\s+by\s+(?P<artist>.+?)(?:\s+\(|\n|$)', text)
    if id_match and name_match:
        track_id = id_match.group("id")
        return {
            "uri": f"spotify:track:{track_id}",
            "id": track_id,
            "name": name_match.group("name").strip(),
            "artist": name_match.group("artist").strip(),
            "is_playing": True,
        }
    return None
