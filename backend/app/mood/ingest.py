"""Listening-history ingestion — recently-played → ``ListeningEvent`` rows.

This is the missing piece that made mood inference dead: nothing recorded what
the user played. The API has the Spotify client (the worker doesn't), so it
polls recently-played and appends rows here, throttled.

Spotify's MCP recently-played carries no per-track timestamp, so ``played_at``
is stamped at ingestion time (staggered by index to keep ordering). Polled
frequently, the newest tracks were played within the poll interval, so the time
bucket stays accurate enough for patterning — an approximation, documented here.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import ListeningEvent
from backend.app.interfaces import SpotifyClientProtocol
from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

# recently-played overlaps poll-to-poll, so without a dedup window every poll
# would re-insert the same tracks. Skip URIs already recorded this recently.
_DEDUP_WINDOW = timedelta(hours=6)


async def ingest_recently_played(
    user_id: str,
    spotify: SpotifyClientProtocol,
    db: AsyncSession,
    limit: int = 50,
) -> int:
    """Append newly-played tracks to ``listening_events`` for *user_id*.

    Args:
        user_id: UUID string of the user.
        spotify: Spotify client (recently-played source).
        db:      Async SQLAlchemy session.
        limit:   Max recently-played tracks to pull.

    Returns:
        The number of new rows inserted (0 on a Spotify error or no new plays).
    """
    try:
        recent = await spotify.get_recently_played(limit=limit)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ingest_spotify_error", error=str(exc))
        return 0
    if not recent:
        return 0

    uid = uuid.UUID(user_id)
    now = datetime.now(UTC)

    rows = await db.execute(
        select(ListeningEvent.track_uri)
        .where(ListeningEvent.user_id == uid)
        .where(ListeningEvent.played_at >= now - _DEDUP_WINDOW)
    )
    seen = {uri for (uri,) in rows.all()}

    added = 0
    for i, track in enumerate(recent):
        uri = track.get("uri")
        if not uri or uri in seen:
            continue
        db.add(ListeningEvent(
            user_id=uid,
            track_uri=uri,
            track_name=track.get("name"),
            artist_name=track.get("artist"),
            played_at=now - timedelta(minutes=i),
        ))
        seen.add(uri)
        added += 1

    if added:
        await db.commit()
        logger.info("ingest_recently_played", user_id=user_id, added=added)
    return added
