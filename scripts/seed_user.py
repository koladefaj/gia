"""Seed (or re-seed) the demo user so Gia feels like she already knows *you*.

Run inside the api container (so the in-cluster hostnames resolve):

    docker compose exec api python scripts/seed_user.py --reset

What it creates for the demo user (`USER_ID`):
  Postgres  — User, Profile (name/timezone/genres), listening history across
              time buckets (morning runs / late nights / vibing)
  Weaviate  — UserMemory objects: insights + preferences + life facts +
              mood patterns + a recent episode, each with a real embedding

`--reset` first deletes everything tied to this one demo user (Postgres rows,
Weaviate objects, Redis keys) so a re-seed is a clean replace, not a merge. It
only touches `USER_ID` — it is not a full-database wipe.

To use the seeded identity in the browser, open once:
    http://localhost:3000/?user_id=<USER_ID>
(the frontend adopts it from the URL and persists it in localStorage).
"""

import asyncio
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import weaviate
import weaviate.classes as wvc
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.config import settings
from backend.app.db.base import Base
from backend.app.db.models import ConversationSession, ListeningEvent, Profile, User
from backend.app.db.weaviate_init import init_weaviate_schema_sync
from backend.app.memory.embeddings import embed
from backend.app.observability.logging import get_logger, setup_logging

setup_logging("info")


def _alembic_head() -> str:
    """Head revision id, read from the migration files (no DB, no cwd dependency)."""
    from alembic.config import Config  # noqa: PLC0415
    from alembic.script import ScriptDirectory  # noqa: PLC0415

    cfg = Config()
    cfg.set_main_option(
        "script_location", str(Path(__file__).resolve().parent.parent / "alembic")
    )
    return ScriptDirectory.from_config(cfg).get_current_head() or ""


async def _stamp_alembic(conn) -> None:
    """Stamp alembic to head after ``create_all``.

    The seed builds the schema with ``Base.metadata.create_all``; without this,
    the app container's startup ``alembic upgrade head`` then tries to re-create
    the tables and crash-loops with "relation users already exists" on a fresh
    DB. Stamping makes that upgrade a no-op. Only writes when ``alembic_version``
    is empty, so it never clobbers a real migration state.
    """
    head = _alembic_head()
    if not head:
        return
    await conn.execute(
        text(
            "CREATE TABLE IF NOT EXISTS alembic_version ("
            "version_num VARCHAR(32) NOT NULL "
            "CONSTRAINT alembic_version_pkc PRIMARY KEY)"
        )
    )
    existing = (
        await conn.execute(text("SELECT version_num FROM alembic_version LIMIT 1"))
    ).first()
    if existing is None:
        await conn.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:v)"), {"v": head}
        )


log = get_logger(__name__)

# Stable, deterministic UUID so the demo user is the same across re-seeds and the
# id is a real UUID (the User/Profile columns are native ``Uuid``).
USER_UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "gia.kolade-demo")
PROFILE_UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "gia.kolade-demo.profile")
USER_ID = str(USER_UUID)
PROFILE_ID = str(PROFILE_UUID)

DISPLAY_NAME = "Kolade"  # change here if the demo user's name differs

# ── Listening history ─────────────────────────────────────────────────────────
# (uri, track, artist). URIs are placeholders — listening history is mood/taste
# signal; live recommendations come from Spotify search, not these ids.

# Morning runs — Central Cee, "Can't Rush Greatness" (UK rap, high energy)
_MORNING = [
    ("spotify:track:cc01", "Sprinter", "Central Cee"),
    ("spotify:track:cc02", "Limitless", "Central Cee"),
    ("spotify:track:cc03", "GenZ Luv", "Central Cee"),
    ("spotify:track:cc04", "Commitment Issues", "Central Cee"),
    ("spotify:track:cc05", "CRG", "Central Cee"),
    ("spotify:track:cc06", "Truth in the Lies", "Central Cee"),
]

# Late nights — Drake (R&B leaning)
_NIGHT = [
    ("spotify:track:dk01", "Marvins Room", "Drake"),
    ("spotify:track:dk02", "Jaded", "Drake"),
    ("spotify:track:dk03", "Teenage Fever", "Drake"),
    ("spotify:track:dk04", "Whisper My Name", "Drake"),
    ("spotify:track:dk05", "Dust", "Drake"),
    ("spotify:track:dk06", "Shabang", "Drake"),
    ("spotify:track:dk07", "Somebody Loves Me", "Drake, PARTYNEXTDOOR"),
    ("spotify:track:dk08", "Fortworth", "Drake, PARTYNEXTDOOR"),
]

# Just vibing — Afrobeats / new-wave "burti" + upcoming artists
_VIBE = [
    ("spotify:track:mv01", "Mofe", "Mavo"),
    ("spotify:track:mv02", "Aura Salad", "Mavo"),
    ("spotify:track:mv03", "Guapanese", "Mavo"),
    ("spotify:track:mv04", "Money Constant", "Mavo"),
    ("spotify:track:ss01", "Super Power", "Suono Sai"),
    ("spotify:track:ss02", "Igbo Boy", "Suono Sai"),
    ("spotify:track:zl01", "Chose Me", "Zaylevelten"),
    ("spotify:track:zl02", "Go Again", "Zaylevelten"),
    ("spotify:track:mc01", "We 2 Fly", "Monochrome"),
    ("spotify:track:mc02", "LV 444", "Monochrome"),
]


def _make_listening_events(user_id: uuid.UUID) -> list[ListeningEvent]:
    """Listening history across time buckets so mood inference has real signal."""
    import random

    now = datetime.now(UTC)
    events: list[ListeningEvent] = []

    def _event(track: tuple, played_at: datetime) -> ListeningEvent:
        uri, name, artist = track
        return ListeningEvent(
            id=uuid.uuid4(),
            user_id=user_id,
            track_uri=uri,
            track_name=name,
            artist_name=artist,
            played_at=played_at,
        )

    # Weekday mornings (7–9 AM) — runs, Central Cee → 10 events
    for i in range(10):
        ts = (now - timedelta(days=i + 1)).replace(hour=random.randint(7, 9), minute=random.randint(0, 59))
        events.append(_event(random.choice(_MORNING), ts))

    # Late nights (10 PM–1 AM) — Drake → 12 events
    for i in range(12):
        ts = (now - timedelta(days=i + 1)).replace(hour=random.choice([22, 23, 0, 1]), minute=random.randint(0, 59))
        events.append(_event(random.choice(_NIGHT), ts))

    # Afternoons / weekends (1–6 PM) — vibing, Afrobeats → 12 events
    for _ in range(12):
        ts = (now - timedelta(days=random.randint(1, 21))).replace(hour=random.randint(13, 18), minute=random.randint(0, 59))
        events.append(_event(random.choice(_VIBE), ts))

    return events


# ── Memories (Weaviate UserMemory) ────────────────────────────────────────────
# Types must match what retrieval reads: insight | preference | life_fact |
# mood_pattern | episode. Insights are normally generated by the consolidation
# worker; seeding them directly makes the "who they are" section alive from turn 1.

_MEMORIES = [
    # — Insights: the synthesised big picture —
    {"type": "insight", "confidence": 0.95,
     "text": "Maps music to the moment: UK rap (Central Cee) to move on morning runs, "
             "Drake's R&B to wind down late at night, and Afrobeats when just vibing. "
             "Genre-fluid, but intentional about which mood gets which sound."},
    {"type": "insight", "confidence": 0.92,
     "text": "Has an ear for the new wave — actively champions upcoming artists "
             "(Suono Sai, Monochrome, Zaylevelten) and the emerging 'burti' Afrobeats "
             "sound Mavo is shaping. Likes being early on talent."},

    # — Preferences: the taste, with the WHY —
    {"type": "preference", "confidence": 0.95,
     "text": "Loves Drake, especially his R&B side — Drake is late-night listening "
             "(Marvins Room, Jaded, Whisper My Name, Teenage Fever)."},
    {"type": "preference", "confidence": 0.93,
     "text": "Into Central Cee for his UK rap; runs the album 'Can't Rush Greatness' "
             "on morning jogs (Sprinter, Limitless, GenZ Luv, CRG)."},
    {"type": "preference", "confidence": 0.90,
     "text": "Big on Mavo for the way they reshaped the Afrobeats sound into the new "
             "'burti' wave (Mofe, Aura Salad, Guapanese, Money Constant)."},
    {"type": "preference", "confidence": 0.88,
     "text": "Champions upcoming artists and likes discovering them early — Suono Sai "
             "(Super Power, Igbo Boy), Zaylevelten (Chose Me, Go Again), Monochrome "
             "(We 2 Fly, LV 444)."},
    {"type": "preference", "confidence": 0.90,
     "text": "Loves the Drake x PARTYNEXTDOOR collabs (Somebody Loves Me, Fortworth)."},
    {"type": "preference", "confidence": 0.85,
     "text": "Afrobeats is the default when just vibing; switches to Drake's R&B at "
             "night and Central Cee's UK rap to get moving."},

    # — Life facts: what makes her a companion, not a jukebox —
    {"type": "life_fact", "confidence": 1.0,
     "text": "21 years old; birthday is in August."},
    {"type": "life_fact", "confidence": 1.0,
     "text": "Lives in Abuja, Nigeria."},
    {"type": "life_fact", "confidence": 0.9,
     "text": "Used to play football but hasn't in months — thinking about coming out of "
             "'retirement' and getting back to it."},
    {"type": "life_fact", "confidence": 1.0,
     "text": "Has a dog named Rex."},

    # — Mood patterns: time-of-day tendencies —
    {"type": "mood_pattern", "confidence": 0.9,
     "text": "Morning runs: high-energy UK rap — Central Cee's 'Can't Rush Greatness' on repeat."},
    {"type": "mood_pattern", "confidence": 0.9,
     "text": "Late nights: winds down with Drake, R&B-leaning and lower energy."},
    {"type": "mood_pattern", "confidence": 0.88,
     "text": "When just vibing (afternoons/weekends): Afrobeats — Mavo, Suono Sai, and the new-wave upcoming artists."},

    # — Episode: a recent session for callbacks —
    {"type": "episode", "confidence": 1.0,
     "text": "Recent session: went full Drake for the night — played 'Fortworth' (with "
             "PARTYNEXTDOOR) and queued a few more of his tracks."},
]


# ── Reset (scoped to this one demo user) ──────────────────────────────────────

async def reset_user() -> None:
    """Delete everything tied to USER_ID — Postgres rows, Weaviate objects, Redis
    keys — so a re-seed is a clean replace. Scoped to the demo user only."""
    log.info("reset_starting", user_id=USER_ID)

    # Postgres — children first (FK order), then profile + user.
    engine = create_async_engine(settings.database_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _stamp_alembic(conn)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        session: AsyncSession
        await session.execute(delete(ListeningEvent).where(ListeningEvent.user_id == USER_UUID))
        await session.execute(delete(ConversationSession).where(ConversationSession.user_id == USER_UUID))
        await session.execute(delete(Profile).where(Profile.user_id == USER_UUID))
        await session.execute(delete(User).where(User.id == USER_UUID))
        await session.commit()
    await engine.dispose()
    log.info("reset_postgres_done")

    # Weaviate — delete this user's objects in both collections.
    def _wipe_weaviate() -> None:
        host = settings.weaviate_url.replace("http://", "").replace("https://", "")
        h, _, p = host.partition(":")
        client = weaviate.connect_to_custom(
            http_host=h, http_port=int(p or 8080), http_secure=False,
            grpc_host=h, grpc_port=50051, grpc_secure=False,
        )
        try:
            for name in ("UserMemory", "MoodPattern"):
                if client.collections.exists(name):
                    client.collections.get(name).data.delete_many(
                        where=wvc.query.Filter.by_property("user_id").equal(USER_ID)
                    )
        finally:
            client.close()

    await asyncio.to_thread(_wipe_weaviate)
    log.info("reset_weaviate_done")

    # Redis — drop any keys mentioning this user (session notes, retrieval cache).
    # SCAN keeps it scoped; no FLUSHDB. Pending-flush members are set entries
    # (not keys), so they're removed separately.
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        async for key in r.scan_iter(match=f"*{USER_ID}*", count=200):
            await r.delete(key)
        members = await r.zrange("gia:pending_flush", 0, -1)
        mine = [m for m in members if m.startswith(f"{USER_ID}:")]
        if mine:
            await r.zrem("gia:pending_flush", *mine)
        await r.aclose()
    except Exception as exc:  # noqa: BLE001 — redis is best-effort here
        log.warning("reset_redis_skipped", error=str(exc))
    log.info("reset_redis_done")


# ── Seed ──────────────────────────────────────────────────────────────────────

async def seed_postgres() -> None:
    engine = create_async_engine(settings.database_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _stamp_alembic(conn)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        session: AsyncSession
        existing = await session.execute(select(User).where(User.id == USER_UUID))
        if existing.scalar_one_or_none():
            log.info("seed_postgres_skipped", reason="user exists (pass --reset to replace)")
            await engine.dispose()
            return

        user = User(id=USER_UUID, email="kolade@demo.local")
        profile = Profile(
            id=PROFILE_UUID,
            user_id=USER_UUID,
            spotify_user_id="kolade_spotify",
            display_name=DISPLAY_NAME,
            timezone="Africa/Lagos",
            preferred_genres=["afrobeats", "uk rap", "r&b", "hip-hop"],
            preferred_volume=0.75,
        )
        session.add_all([user, profile])

        events = _make_listening_events(USER_UUID)
        session.add_all(events)

        past_session = ConversationSession(
            id=uuid.uuid4(),
            user_id=USER_UUID,
            started_at=datetime.now(UTC) - timedelta(days=1),
            ended_at=datetime.now(UTC) - timedelta(days=1, minutes=-25),
            summary="Late-night Drake session — played Fortworth (with PARTYNEXTDOOR) and queued more.",
        )
        session.add(past_session)

        await session.commit()
        log.info("seed_postgres_done", user_id=USER_ID, events=len(events))

    await engine.dispose()


def seed_weaviate(vectors: list[list[float]]) -> None:
    """Insert the seed memories with real embeddings (same order as ``_MEMORIES``)."""
    init_weaviate_schema_sync()

    host = settings.weaviate_url.replace("http://", "").replace("https://", "")
    h, _, p = host.partition(":")
    client = weaviate.connect_to_custom(
        http_host=h, http_port=int(p or 8080), http_secure=False,
        grpc_host=h, grpc_port=50051, grpc_secure=False,
    )
    try:
        collection = client.collections.get("UserMemory")
        existing = collection.query.fetch_objects(
            filters=wvc.query.Filter.by_property("user_id").equal(USER_ID), limit=1,
        )
        if existing.objects:
            log.info("seed_weaviate_skipped", reason="memories exist (pass --reset to replace)")
            return

        now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        with collection.batch.dynamic() as batch:
            for mem, vector in zip(_MEMORIES, vectors, strict=True):
                batch.add_object(
                    properties={
                        "user_id": USER_ID,
                        "type": mem["type"],
                        "text": mem["text"],
                        "confidence": mem["confidence"],
                        "created_at": now_iso,
                        "supersedes_id": "",
                    },
                    vector=vector,
                )
        log.info("seed_weaviate_done", user_id=USER_ID, memories=len(_MEMORIES))
    finally:
        client.close()


async def main() -> None:
    do_reset = "--reset" in sys.argv
    log.info("seed_starting", reset=do_reset)
    if do_reset:
        await reset_user()

    await seed_postgres()
    log.info("seed_embedding_memories", count=len(_MEMORIES))
    vectors = [await embed(mem["text"]) for mem in _MEMORIES]
    await asyncio.to_thread(seed_weaviate, vectors)

    log.info("seed_complete", user_id=USER_ID)
    print("\n[OK] Demo user seeded.")
    print(f"  user_id: {USER_ID}")
    print(f"  Open the browser with this identity:  http://localhost:3000/?user_id={USER_ID}")
    print("  (the frontend adopts it from the URL and remembers it)")


if __name__ == "__main__":
    asyncio.run(main())
