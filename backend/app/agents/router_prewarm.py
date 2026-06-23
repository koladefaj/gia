"""Speculative router prewarm — overlap the router with the user's last words.

The ``gpt-4o-mini`` router (~2s) sits on the critical path for music/specialist
turns: nothing can search or play until it resolves intent + ``search_query``.
Streaming STT (Deepgram Flux) emits an ``EagerEndOfTurn`` a beat *before* the
user actually finishes, with a transcript guaranteed to match the eventual
``EndOfTurn``. So we start the router on that eager transcript; when ``/chat``
arrives with the final transcript it reuses that result instead of starting the
router cold — the router latency now overlaps the tail of the utterance.

Two-tier hand-off so it works whether or not the two requests land on the same
process:

  * **Same worker (fast path):** the in-flight ``asyncio.Task`` is kept in a
    process-local map; ``take`` awaits it directly — zero serialisation, and it
    works even if the router is still running.
  * **Different worker:** the completed ``RouterDecision`` is also written to
    Redis (keyed by the exact normalised transcript), with an in-flight marker.
    A ``take`` on another worker finds the result, or briefly waits on the marker
    for the router to finish — so the eager head start survives across workers.

Correctness is unaffected: classification is read-only (no playback/search side
effects), the key is the *exact* normalised transcript (so ``/chat`` only reuses
a decision from identical input), and the result is consumed on take. Any miss
(typed turn, batch STT, dead prewarm) falls back to a cold router — graceful,
never a stall.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from collections.abc import Awaitable, Callable

from redis.asyncio import Redis as AsyncRedis

from backend.app.observability.logging import get_logger
from backend.app.schemas.router import RouterDecision, safe_default_decision

logger = get_logger(__name__)

# Entries live briefly — long enough to bridge an eager→final gap (sub-second)
# plus client/network slack, short enough that abandoned turns self-clean.
_TTL_S = 30
# Max time a cross-worker ``take`` waits on an in-flight prewarm before giving up
# and running cold. Covers a full router round-trip; a dead prewarm wastes at
# most this long (rare).
_WAIT_S = 2.5
_POLL_S = 0.04

_PENDING_KEY = "gia:prewarm:pending:"
_RESULT_KEY = "gia:prewarm:result:"

# Process-local in-flight tasks: key -> (created_monotonic, task)
_LOCAL: dict[str, tuple[float, asyncio.Task[RouterDecision]]] = {}


def _key(session_id: str, message: str) -> str:
    """Stable key from session + normalised transcript.

    Flux guarantees the final transcript matches the eager one, so a normalised
    (trimmed, collapsed-whitespace, lower-cased) hash collides exactly when
    ``/chat``'s message is the one we prewarmed — no fuzzy matching needed.
    """
    norm = " ".join(message.strip().lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()  # noqa: S324 — not security


def _prune(now: float) -> None:
    """Drop and cancel local entries older than the TTL."""
    stale = [k for k, (ts, _) in _LOCAL.items() if now - ts > _TTL_S]
    for k in stale:
        _, task = _LOCAL.pop(k)
        if not task.done():
            task.cancel()


async def _run_and_cache(
    redis: AsyncRedis, key: str, factory: Callable[[], Awaitable[RouterDecision]]
) -> RouterDecision:
    """Run the classification, publish the result to Redis, clear the marker."""
    try:
        decision = await factory()
    except Exception as exc:  # noqa: BLE001 — classify_turn shouldn't raise, but be safe
        logger.warning("router_prewarm_classify_failed", error=str(exc))
        decision = safe_default_decision()
    with contextlib.suppress(Exception):
        await redis.set(_RESULT_KEY + key, decision.model_dump_json(), ex=_TTL_S)
        await redis.delete(_PENDING_KEY + key)
    return decision


async def start(
    redis: AsyncRedis,
    session_id: str,
    message: str,
    factory: Callable[[], Awaitable[RouterDecision]],
) -> None:
    """Kick off a router classification for *message* and stash it.

    No-op when an identical (session, message) prewarm is already in flight on
    this worker, so repeated eager events for the same text don't double-spend.
    Marks the turn in-flight in Redis first so a ``take`` on another worker knows
    to wait for it. *factory* builds the ``classify_turn`` coroutine (closing over
    the same history ``/chat`` will use).
    """
    now = time.monotonic()
    _prune(now)
    key = _key(session_id, message)
    if key in _LOCAL:
        return
    with contextlib.suppress(Exception):
        await redis.set(_PENDING_KEY + key, "1", ex=_TTL_S)
    task = asyncio.create_task(_run_and_cache(redis, key, factory))
    # Swallow the result if the turn is superseded and never taken, so a stray
    # exception/result doesn't warn "never retrieved".
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    _LOCAL[key] = (now, task)
    logger.debug("router_prewarm_start", key=key)


async def take(
    redis: AsyncRedis, session_id: str, message: str
) -> RouterDecision | None:
    """Return the prewarmed decision for *message*, or ``None`` to run cold.

    Fast path: if this worker holds the in-flight task, await it (overlapping a
    still-running router). Otherwise look in Redis — return a finished result, or
    wait up to ``_WAIT_S`` on an in-flight marker from another worker. The result
    is consumed (deleted) so a later identical phrase classifies fresh.
    """
    key = _key(session_id, message)

    entry = _LOCAL.pop(key, None)
    if entry is not None:
        try:
            decision = await entry[1]
        except Exception as exc:  # noqa: BLE001
            logger.warning("router_prewarm_take_failed", error=str(exc))
            return None
        with contextlib.suppress(Exception):
            await redis.delete(_RESULT_KEY + key, _PENDING_KEY + key)
        return decision

    return await _take_redis(redis, key)


async def _take_redis(redis: AsyncRedis, key: str) -> RouterDecision | None:
    """Cross-worker take: read the cached result, waiting briefly if it's still
    being computed elsewhere. Returns ``None`` on miss / error (caller runs cold)."""
    try:
        raw = await redis.get(_RESULT_KEY + key)
        if raw is None and await redis.exists(_PENDING_KEY + key):
            deadline = time.monotonic() + _WAIT_S
            while raw is None and time.monotonic() < deadline:
                await asyncio.sleep(_POLL_S)
                raw = await redis.get(_RESULT_KEY + key)
        if raw is None:
            return None
        with contextlib.suppress(Exception):
            await redis.delete(_RESULT_KEY + key)
        # RouterDecision.model_validate_json accepts str or bytes.
        return RouterDecision.model_validate_json(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("router_prewarm_redis_take_failed", error=str(exc))
        return None
