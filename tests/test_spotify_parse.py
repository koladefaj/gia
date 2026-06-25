"""Tests for parsing the Spotify MCP server's text responses."""

from __future__ import annotations

from backend.app.tools.spotify_parse import (
    parse_artists,
    parse_now_playing,
    parse_tracks,
)

_SEARCH = (
    '# Search results for "Tems Free Mind" (type: track)\n\n'
    '1. "Free Mind" by Tems (4:08) - ID: 2mzM4Y0Rnx2BDZqRnhQ5Q6\n'
    '2. "Damages" by Tems (2:49), popularity: 71 - ID: 3Xfwu3xtPqmJ4nM4jpBm8O\n'
    '3. "Mrs Sativa" by Halfco Baby, Zan, Benty (4:23) - ID: 5ARd8VUjVywyr4eSh6cBLA\n'
)

_RECENT = (
    "# Recently Played Tracks\n\n"
    '1. "Forgiveness" by Asake (2:39) - ID: 5u4rozuOBse9MgrAzGspQy - Played at: 19/06/2026, 14:43:10\n'
)


def test_parse_tracks_basic() -> None:
    tracks = parse_tracks(_SEARCH)
    assert len(tracks) == 3
    assert tracks[0] == {
        "uri": "spotify:track:2mzM4Y0Rnx2BDZqRnhQ5Q6",
        "id": "2mzM4Y0Rnx2BDZqRnhQ5Q6",
        "name": "Free Mind",
        "artist": "Tems",
    }


def test_parse_tracks_handles_popularity_and_multiartist() -> None:
    tracks = parse_tracks(_SEARCH)
    assert tracks[1]["name"] == "Damages"  # popularity suffix tolerated
    assert tracks[2]["artist"] == "Halfco Baby, Zan, Benty"  # multi-artist preserved


def test_parse_tracks_recently_played_with_timestamp() -> None:
    tracks = parse_tracks(_RECENT)
    assert tracks[0]["name"] == "Forgiveness"
    assert tracks[0]["artist"] == "Asake"
    assert tracks[0]["id"] == "5u4rozuOBse9MgrAzGspQy"


def test_parse_tracks_empty() -> None:
    assert parse_tracks("No track results found for \"zzz\"") == []


def test_parse_artists_excludes_track_lines() -> None:
    text = (
        "# Top Artists\n\n"
        "1. Burna Boy - ID: 3wcj11K77LjEY1PkEazffaaa\n"
        "2. Tems - ID: 0Xy5mNQQQQ5tFbCv2tmuuu\n"
    )
    artists = parse_artists(text)
    assert [a["name"] for a in artists] == ["Burna Boy", "Tems"]
    assert artists[0]["uri"].startswith("spotify:artist:")


def test_parse_now_playing_none_when_idle() -> None:
    assert parse_now_playing("No track is currently playing.") is None


def test_parse_now_playing_extracts_track() -> None:
    text = '1. "Free Mind" by Tems (4:08) - ID: 2mzM4Y0Rnx2BDZqRnhQ5Q6'
    np = parse_now_playing(text)
    assert np is not None
    assert np["name"] == "Free Mind"
    assert np["is_playing"] is True
