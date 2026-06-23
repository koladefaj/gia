"""Speculative router prewarm — overlap the router with the user's last words.

The ``gpt-4o-mini`` router (~2s) sits on the critical path for music/specialist
turns: nothing can search or play until it resolves intent + ``search_query``.
Streaming STT (Deepgram Flux) emits an ``EagerEndOfTurn`` a beat *before* the
user actually finishes, with a transcript guaranteed to match the eventual
``EndOfTurn``. So we start the router on that eager transcript and stash the
in-flight task here; when ``/chat`` arrives with the final transcript it simply
awaits the stashed task instead of starting the router cold — the router latency
now overlaps the tail of the utterance.

Correctness is unaffected: classification is read-only (no playback/search side
effects), and the prewarm key is the *exact* normalised transcript, so ``/chat``
only reuses a decision computed from identical input. A superseded eager (the
user kept talking → ``TurnResumed``) just leaves an unused task that expires.

Process-local by design: prewarm and ``/chat`` are two requests, but in a single
worker they share one event loop, so an ``asyncio.Task`` is the cheapest hand-off.
Across workers (no shared loop) ``take`` simply misses and ``/chat`` runs its own
router — a graceful, correct fallback, never a stall.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable

from backend.app.observability.logging import get_logger
from backend.app.schemas.router import RouterDecision

logger = get_logger(__name__)

# Entries live briefly — long enough to bridge an eager→final gap (sub-second)
# plus client/network slack, short enough that abandoned turns self-clean.
_TTL_S = 30.0

# key -> (created_monotonic, task)
_PENDING: dict[str, tuple[float, asyncio.Task[RouterDecision]]] = {}


def _key(session_id: str, message: str) -> str:
    """Stable key from session + normalised transcript.

    Flux guarantees the final transcript matches the eager one, so a normalised
    (trimmed, lower-cased) hash collides exactly when ``/chat``'s message is the
    one we prewarmed — no fuzzy matching needed.
    """
    norm = " ".join(message.strip().lower().split())
    digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()  # noqa: S324 — not security
    return f"{session_id}:{digest}"


def _prune(now: float) -> None:
    """Drop and cancel entries older than the TTL."""
    stale = [k for k, (ts, _) in _PENDING.items() if now - ts > _TTL_S]
    for k in stale:
        _, task = _PENDING.pop(k)
        if not task.done():
            task.cancel()


def start(
    session_id: str,
    message: str,
    factory: Callable[[], Awaitable[RouterDecision]],
) -> None:
    """Kick off a router classification for *message* and stash the task.

    No-op when an identical (session, message) prewarm is already in flight, so
    repeated eager events for the same text don't double-spend. *factory* builds
    the ``classify_turn`` coroutine (closing over the same history ``/chat`` will
    use), called only when we actually start one.
    """
    now = time.monotonic()
    _prune(now)
    key = _key(session_id, message)
    if key in _PENDING:
        return
    task = asyncio.create_task(factory())
    # Swallow the result if the turn is superseded and the task is never taken,
    # so a stray exception/result doesn't warn "never retrieved".
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    _PENDING[key] = (now, task)
    logger.debug("router_prewarm_start", key=key)


async def take(session_id: str, message: str) -> RouterDecision | None:
    """Return the prewarmed decision for *message*, or ``None`` to run cold.

    Awaits the in-flight task (so a router still finishing overlaps the await),
    and removes it from the cache. ``None`` means no matching prewarm — the
    caller should classify normally.
    """
    entry = _PENDING.pop(_key(session_id, message), None)
    if entry is None:
        return None
    _, task = entry
    try:
        return await task
    except Exception as exc:  # noqa: BLE001 — classify_turn shouldn't raise, but be safe
        logger.warning("router_prewarm_take_failed", error=str(exc))
        return None
