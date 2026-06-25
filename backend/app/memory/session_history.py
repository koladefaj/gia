"""Short-term conversation memory — the last few turns of a chat session.

The Weaviate memory engine remembers *durable* facts about a user; this is the
short-term working memory that makes a single conversation cohere.  Without it,
every ``/chat`` turn is an island: "play it now", "queue that one", "what did you
just say" have no referent, so Gia answers as if each message were the first.

Turns live in a capped Redis list keyed by session id (oldest → newest) and
expire after a couple of idle hours.  Everything degrades quietly: a Redis hiccup
yields an empty history rather than a failed turn.
"""

from __future__ import annotations

import json
from typing import Any

from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

_MAX_TURNS = 12       # keep the last ~6 exchanges
_HISTORY_TTL = 7200   # 2 hours of inactivity


def _key(session_id: str) -> str:
    return f"chat:hist:{session_id}"


async def append_turn(redis: Any, session_id: str, role: str, text: str) -> None:
    """Append one message to the session's history ring.

    Args:
        redis:      Async Redis client.
        session_id: Conversation session id.
        role:       ``"user"`` or ``"gia"``.
        text:       The message text (audio tags and all — they're stripped on read).
    """
    if not session_id or not text.strip():
        return
    try:
        key = _key(session_id)
        await redis.rpush(key, json.dumps({"role": role, "text": text.strip()}))
        await redis.ltrim(key, -_MAX_TURNS, -1)
        await redis.expire(key, _HISTORY_TTL)
    except Exception as exc:  # noqa: BLE001
        logger.warning("history_append_error", error=str(exc))


async def get_history(redis: Any, session_id: str, limit: int = _MAX_TURNS) -> list[dict]:
    """Return recent turns (oldest → newest) for *session_id*.

    Returns an empty list on any error so a degraded Redis never blocks a turn.
    """
    if not session_id:
        return []
    try:
        raw = await redis.lrange(_key(session_id), -limit, -1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("history_get_error", error=str(exc))
        return []
    out: list[dict] = []
    for item in raw:
        try:
            out.append(json.loads(item))
        except (json.JSONDecodeError, TypeError):
            continue
    return out


def format_history(turns: list[dict]) -> str:
    """Render turns as a compact ``User:`` / ``Gia:`` transcript for prompts."""
    lines = []
    for t in turns:
        who = "Gia" if t.get("role") == "gia" else "User"
        text = (t.get("text") or "").strip()
        if text:
            lines.append(f"{who}: {text}")
    return "\n".join(lines)
