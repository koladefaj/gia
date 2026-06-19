"""Tests for SQLAlchemy database models.

Uses the ``db_session`` fixture (SQLite in-memory) to verify that models
can be created, persisted, queried, and that relationships resolve correctly.
No Postgres instance is required.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import ConversationSession, ListeningEvent, Profile, User


def _u(n: int) -> uuid.UUID:
    """Return a deterministic UUID from a small integer, for readable test IDs."""
    return uuid.UUID(f"00000000-0000-0000-0000-{n:012d}")


# ── User ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_user_can_be_created(db_session: AsyncSession) -> None:
    """A ``User`` row can be inserted and retrieved by primary key."""
    uid = _u(1)
    user = User(id=uid, email="test@gia.local")
    db_session.add(user)
    await db_session.flush()

    result = await db_session.execute(select(User).where(User.id == uid))
    fetched = result.scalar_one_or_none()
    assert fetched is not None
    assert fetched.email == "test@gia.local"


@pytest.mark.asyncio
async def test_user_email_must_be_unique(db_session: AsyncSession) -> None:
    """Two users with the same email raise an integrity error."""
    from sqlalchemy.exc import IntegrityError

    db_session.add(User(id=_u(2), email="dup@gia.local"))
    db_session.add(User(id=_u(3), email="dup@gia.local"))
    with pytest.raises(IntegrityError):
        await db_session.flush()


# ── Profile ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_profile_linked_to_user(db_session: AsyncSession) -> None:
    """A ``Profile`` is retrievable via its ``User`` relationship."""
    uid, pid = _u(4), _u(104)
    user = User(id=uid, email="kolade@gia.local")
    profile = Profile(
        id=pid,
        user_id=uid,
        spotify_user_id="kolade_spotify",
        timezone="Europe/London",
        preferred_genres=["afrobeats", "r&b"],
    )
    db_session.add(user)
    db_session.add(profile)
    await db_session.flush()

    result = await db_session.execute(select(User).where(User.id == uid))
    fetched_user = result.scalar_one()
    await db_session.refresh(fetched_user, ["profile"])
    assert fetched_user.profile is not None
    assert fetched_user.profile.timezone == "Europe/London"
    assert "afrobeats" in fetched_user.profile.preferred_genres


@pytest.mark.asyncio
async def test_profile_preferred_genres_defaults_to_empty_list(db_session: AsyncSession) -> None:
    """``Profile.preferred_genres`` defaults to ``[]`` when not provided."""
    uid, pid = _u(5), _u(105)
    db_session.add(User(id=uid))
    profile = Profile(id=pid, user_id=uid)
    db_session.add(profile)
    await db_session.flush()
    await db_session.refresh(profile)
    assert profile.preferred_genres == []


# ── ListeningEvent ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_listening_event_stores_audio_features(db_session: AsyncSession) -> None:
    """``ListeningEvent`` correctly stores and retrieves audio feature floats."""
    uid = _u(6)
    db_session.add(User(id=uid))
    await db_session.flush()

    event = ListeningEvent(
        id=uuid.uuid4(),
        user_id=uid,
        track_uri="spotify:track:001",
        track_name="Free Mind",
        artist_name="Tems",
        energy=0.38,
        valence=0.71,
        tempo=92.0,
        danceability=0.62,
        key=5,
        mode=0,
        played_at=datetime.now(UTC),
    )
    db_session.add(event)
    await db_session.flush()

    result = await db_session.execute(
        select(ListeningEvent).where(ListeningEvent.user_id == uid)
    )
    fetched = result.scalar_one()
    assert abs(fetched.energy - 0.38) < 0.001
    assert fetched.track_name == "Free Mind"
    assert fetched.key == 5


@pytest.mark.asyncio
async def test_listening_event_audio_features_nullable(db_session: AsyncSession) -> None:
    """Audio feature columns are nullable — events without features are valid."""
    uid = _u(7)
    db_session.add(User(id=uid))
    await db_session.flush()

    event = ListeningEvent(
        id=uuid.uuid4(),
        user_id=uid,
        track_uri="spotify:track:002",
        played_at=datetime.now(UTC),
    )
    db_session.add(event)
    await db_session.flush()

    result = await db_session.execute(
        select(ListeningEvent).where(ListeningEvent.user_id == uid)
    )
    fetched = result.scalar_one()
    assert fetched.energy is None
    assert fetched.valence is None


# ── ConversationSession ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_conversation_session_intent_log_defaults_to_empty(db_session: AsyncSession) -> None:
    """``ConversationSession.intent_log`` defaults to an empty list."""
    uid = _u(8)
    db_session.add(User(id=uid))
    await db_session.flush()

    session_row = ConversationSession(
        id=uuid.uuid4(),
        user_id=uid,
        started_at=datetime.now(UTC),
    )
    db_session.add(session_row)
    await db_session.flush()
    await db_session.refresh(session_row)
    assert session_row.intent_log == []


@pytest.mark.asyncio
async def test_conversation_session_summary_is_nullable(db_session: AsyncSession) -> None:
    """``ConversationSession.summary`` can be ``None`` for in-progress sessions."""
    uid = _u(9)
    db_session.add(User(id=uid))
    await db_session.flush()

    session_row = ConversationSession(
        id=uuid.uuid4(),
        user_id=uid,
        started_at=datetime.now(UTC),
    )
    db_session.add(session_row)
    await db_session.flush()
    assert session_row.summary is None
