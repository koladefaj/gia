"""Tests for recently-played → ListeningEvent ingestion."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.app.mood.ingest import ingest_recently_played

_USER = "00000000-0000-0000-0000-000000000001"


def _db(existing_uris: list[str]) -> MagicMock:
    db = MagicMock()
    db.execute = AsyncMock(return_value=MagicMock(all=lambda: [(u,) for u in existing_uris]))
    db.add = MagicMock()
    db.commit = AsyncMock()
    return db


def _spotify(tracks: list[dict]) -> MagicMock:
    sp = MagicMock()
    sp.get_recently_played = AsyncMock(return_value=tracks)
    return sp


@pytest.mark.asyncio
async def test_ingest_writes_new_rows() -> None:
    sp = _spotify([
        {"uri": "spotify:track:1", "name": "A", "artist": "X"},
        {"uri": "spotify:track:2", "name": "B", "artist": "Y"},
    ])
    db = _db([])
    added = await ingest_recently_played(_USER, sp, db)
    assert added == 2
    assert db.add.call_count == 2
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_dedups_already_recorded() -> None:
    sp = _spotify([
        {"uri": "spotify:track:1", "name": "A", "artist": "X"},  # already recorded
        {"uri": "spotify:track:2", "name": "B", "artist": "Y"},  # new
    ])
    db = _db(["spotify:track:1"])
    added = await ingest_recently_played(_USER, sp, db)
    assert added == 1
    assert db.add.call_count == 1


@pytest.mark.asyncio
async def test_ingest_empty_returns_zero_no_commit() -> None:
    sp = _spotify([])
    db = _db([])
    added = await ingest_recently_played(_USER, sp, db)
    assert added == 0
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_spotify_error_returns_zero() -> None:
    sp = MagicMock()
    sp.get_recently_played = AsyncMock(side_effect=RuntimeError("down"))
    db = _db([])
    added = await ingest_recently_played(_USER, sp, db)
    assert added == 0
