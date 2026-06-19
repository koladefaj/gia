"""Day 2 seed script — creates a rich returning user so the demo feels alive from turn 1.

Run:
    python scripts/seed_user.py

What it creates:
  Postgres  — User, Profile, 30 ListeningEvents across time buckets
  Weaviate  — 8 UserMemory objects (preferences + mood pattern + episodes)

Vectors: random 768-dim placeholders for now. Real BGE embeddings wired on Day 3.
"""

import asyncio
import random
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import weaviate
import weaviate.classes as wvc
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.config import settings
from backend.app.db.base import Base
from backend.app.db.models import ConversationSession, ListeningEvent, Profile, User
from backend.app.db.weaviate_init import init_weaviate_schema_sync
from backend.app.observability.logging import get_logger, setup_logging

setup_logging("debug")
log = get_logger(__name__)

USER_ID = "kolade-demo"
PROFILE_ID = "kolade-demo-profile"

# ── Postgres seed ─────────────────────────────────────────────────────────────

_TRACK_POOL = [
    ("spotify:track:001", "Free Mind", "Tems", 0.38, 0.71, 92.0, 0.62, 5, 0),
    ("spotify:track:002", "Last Last", "Burna Boy", 0.78, 0.68, 118.0, 0.80, 7, 1),
    ("spotify:track:003", "Essence", "Wizkid", 0.61, 0.76, 108.0, 0.75, 2, 1),
    ("spotify:track:004", "Infinity", "Odumodublvck", 0.85, 0.55, 142.0, 0.72, 9, 0),
    ("spotify:track:005", "Bounce", "Omah Lay", 0.45, 0.73, 98.0, 0.68, 4, 1),
    ("spotify:track:006", "Calm Down", "Rema", 0.52, 0.82, 105.0, 0.78, 6, 1),
    ("spotify:track:007", "Peru", "Fireboy DML", 0.41, 0.69, 95.0, 0.65, 3, 0),
    ("spotify:track:008", "Ye", "Burna Boy", 0.33, 0.77, 88.0, 0.60, 1, 1),
    ("spotify:track:009", "Ku Lo Sa", "Oxlade", 0.44, 0.79, 102.0, 0.70, 8, 1),
    ("spotify:track:010", "Soweto", "Victony", 0.58, 0.74, 112.0, 0.76, 0, 1),
]


def _make_listening_events(user_id: str) -> list[ListeningEvent]:
    """30 events spread across time buckets so mood inference has signal."""
    now = datetime.now(UTC)
    events = []

    def _event(track: tuple, played_at: datetime) -> ListeningEvent:
        uri, name, artist, energy, valence, tempo, dance, key, mode = track
        return ListeningEvent(
            id=str(uuid.uuid4()),
            user_id=user_id,
            track_uri=uri,
            track_name=name,
            artist_name=artist,
            energy=energy + random.uniform(-0.05, 0.05),
            valence=valence + random.uniform(-0.05, 0.05),
            tempo=tempo + random.uniform(-3, 3),
            danceability=dance,
            key=key,
            mode=mode,
            played_at=played_at,
        )

    # Sunday evenings (8–10 PM) — low energy, wind-down → 10 events
    low_energy_tracks = [t for t in _TRACK_POOL if t[3] < 0.50]
    for i in range(10):
        days_ago = 7 * (i // 2 + 1)
        hour = random.randint(20, 22)
        ts = (now - timedelta(days=days_ago)).replace(hour=hour, minute=random.randint(0, 59))
        events.append(_event(random.choice(low_energy_tracks), ts))

    # Monday mornings (7–9 AM) — high energy, focus → 10 events
    high_energy_tracks = [t for t in _TRACK_POOL if t[3] > 0.70]
    for i in range(10):
        days_ago = 7 * (i // 2 + 1) - 1  # day before Sunday
        hour = random.randint(7, 9)
        ts = (now - timedelta(days=days_ago)).replace(hour=hour, minute=random.randint(0, 59))
        events.append(_event(random.choice(high_energy_tracks), ts))

    # Miscellaneous weekday afternoons → 10 events
    for i in range(10):
        days_ago = random.randint(1, 30)
        hour = random.randint(13, 18)
        ts = (now - timedelta(days=days_ago)).replace(hour=hour, minute=random.randint(0, 59))
        events.append(_event(random.choice(_TRACK_POOL), ts))

    return events


async def seed_postgres() -> None:
    engine = create_async_engine(settings.database_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        session: AsyncSession

        # Idempotent — skip if already seeded
        from sqlalchemy import select
        existing = await session.execute(select(User).where(User.id == USER_ID))
        if existing.scalar_one_or_none():
            log.info("seed_postgres_skipped", reason="user already exists")
            await engine.dispose()
            return

        user = User(id=USER_ID, email="kolade@demo.local")
        profile = Profile(
            id=PROFILE_ID,
            user_id=USER_ID,
            spotify_user_id="kolade_spotify",
            timezone="Europe/London",
            preferred_genres=["afrobeats", "afropop", "r&b"],
            preferred_volume=0.75,
        )
        session.add(user)
        session.add(profile)

        events = _make_listening_events(USER_ID)
        session.add_all(events)

        # One past session so memory extractor has something to reference
        past_session = ConversationSession(
            id=str(uuid.uuid4()),
            user_id=USER_ID,
            started_at=datetime.now(UTC) - timedelta(days=4),
            ended_at=datetime.now(UTC) - timedelta(days=4, minutes=-35),
            summary="User asked for Afrobeats wind-down. Confirmed Tems, saved Free Mind. Created playlist 'Afro Vibes'.",
        )
        session.add(past_session)

        await session.commit()
        log.info("seed_postgres_done", user_id=USER_ID, events=len(events))

    await engine.dispose()


# ── Weaviate seed ─────────────────────────────────────────────────────────────

def _rand_vector(dim: int = 768) -> list[float]:
    """Placeholder vector — replaced by real BGE embeddings on Day 3."""
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


_MEMORIES = [
    {
        "type": "preference",
        "text": "Loves Tems for wind-down sessions — low-energy, high-valence tracks like Free Mind.",
        "confidence": 0.92,
    },
    {
        "type": "preference",
        "text": "Core Afrobeats listener. Burna Boy, Wizkid, and Davido are staples in regular rotation.",
        "confidence": 0.95,
    },
    {
        "type": "preference",
        "text": "Currently going through an Odumodublvck phase — drawn to his Afro-fusion style.",
        "confidence": 0.85,
    },
    {
        "type": "preference",
        "text": "Prefers high-energy tracks during weekday morning sessions (7–9 AM).",
        "confidence": 0.88,
    },
    {
        "type": "preference",
        "text": "Rejects club-energy tracks before 10 PM unless explicitly requested.",
        "confidence": 0.80,
    },
    {
        "type": "preference",
        "text": "Has confirmed Tems tracks across 3+ sessions — reliable positive signal.",
        "confidence": 0.90,
    },
    {
        "type": "mood_pattern",
        "text": "Sunday 8–10 PM: consistently low energy (avg 0.35) + high valence (avg 0.73). Unwinding pattern — 8 sessions of evidence.",
        "confidence": 0.87,
    },
    {
        "type": "episode",
        "text": "Session 2026-06-15: asked for Afrobeats wind-down, Gia recommended Tems, user confirmed and saved Free Mind. Created playlist 'Afro Vibes' with 5 tracks.",
        "confidence": 1.0,
    },
]


def seed_weaviate() -> None:
    init_weaviate_schema_sync()

    client = weaviate.connect_to_custom(
        http_host=settings.weaviate_url.replace("http://", "").split(":")[0],
        http_port=int(settings.weaviate_url.split(":")[-1]),
        http_secure=False,
        grpc_host=settings.weaviate_url.replace("http://", "").split(":")[0],
        grpc_port=50051,
        grpc_secure=False,
    )

    try:
        collection = client.collections.get("UserMemory")

        # Idempotent — check if already seeded
        existing = collection.query.fetch_objects(
            filters=wvc.query.Filter.by_property("user_id").equal(USER_ID),
            limit=1,
        )
        if existing.objects:
            log.info("seed_weaviate_skipped", reason="memories already exist")
            return

        now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        with collection.batch.dynamic() as batch:
            for mem in _MEMORIES:
                batch.add_object(
                    properties={
                        "user_id": USER_ID,
                        "type": mem["type"],
                        "text": mem["text"],
                        "confidence": mem["confidence"],
                        "created_at": now_iso,
                        "supersedes_id": "",
                    },
                    vector=_rand_vector(),
                )

        log.info("seed_weaviate_done", user_id=USER_ID, memories=len(_MEMORIES))
    finally:
        client.close()


async def main() -> None:
    log.info("seed_starting")
    await seed_postgres()
    await asyncio.to_thread(seed_weaviate)
    log.info("seed_complete", user_id=USER_ID)
    print(f"\n✓ Seed complete. Demo user: {USER_ID}")
    print("  Run: docker compose up → GET /health → GET /auth/spotify/status")


if __name__ == "__main__":
    asyncio.run(main())
