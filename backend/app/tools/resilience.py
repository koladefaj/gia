"""Reusable resilience primitives for external tool calls.

In production the thing that takes a conversational AI down is rarely a logic
bug — it is a dependency that hangs, flaps, or fails in a storm.  This module
provides the three guards every outbound call should have, composed into one
helper:

  * **timeout** — a hung dependency must not hang the turn (``asyncio.wait_for``).
  * **retry** — transient blips are retried with exponential back-off.
  * **circuit breaker** — once a dependency is clearly down, stop hammering it:
    fail fast for a cooldown window so the event loop and the dependency both
    get room to recover, then probe with a single half-open call.

The breaker is intentionally tiny and in-process (per client instance).  For a
single-process app that is exactly the right scope; a distributed breaker would
be over-engineering here.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import Enum

from backend.app.observability.logging import get_logger

logger = get_logger(__name__)


class CircuitState(str, Enum):
    """The three states of a circuit breaker."""

    CLOSED = "closed"  # healthy — calls flow through
    OPEN = "open"  # tripped — calls fail fast
    HALF_OPEN = "half_open"  # cooldown elapsed — allow one probe


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the breaker is open."""


class CircuitBreaker:
    """A minimal in-process circuit breaker.

    Opens after *threshold* consecutive failures, stays open for *cooldown*
    seconds, then allows a single half-open probe.  A success closes it; a
    failure re-opens it for another cooldown.

    Attributes:
        name:      Identifier used in logs (e.g. ``"spotify"``).
        threshold: Consecutive failures that trip the breaker.
        cooldown:  Seconds to stay open before probing.
    """

    def __init__(self, name: str, *, threshold: int = 5, cooldown: float = 30.0) -> None:
        self.name = name
        self.threshold = threshold
        self.cooldown = cooldown
        self._failures = 0
        self._opened_at = 0.0
        self._state = CircuitState.CLOSED

    @property
    def state(self) -> CircuitState:
        """Current state, transitioning OPEN → HALF_OPEN once cooldown elapses."""
        if self._state is CircuitState.OPEN and (
            time.monotonic() - self._opened_at >= self.cooldown
        ):
            self._state = CircuitState.HALF_OPEN
            logger.info("circuit_half_open", breaker=self.name)
        return self._state

    def allow(self) -> bool:
        """Return ``True`` if a call may proceed under the current state."""
        return self.state is not CircuitState.OPEN

    def record_success(self) -> None:
        """Reset failure tracking and close the breaker."""
        self._failures = 0
        if self._state is not CircuitState.CLOSED:
            logger.info("circuit_closed", breaker=self.name)
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Count a failure and open the breaker if the threshold is reached."""
        self._failures += 1
        if self._failures >= self.threshold or self._state is CircuitState.HALF_OPEN:
            self._opened_at = time.monotonic()
            if self._state is not CircuitState.OPEN:
                logger.warning(
                    "circuit_open", breaker=self.name, failures=self._failures
                )
            self._state = CircuitState.OPEN


async def resilient_call[T](
    fn: Callable[[], Awaitable[T]],
    *,
    name: str,
    timeout_s: float,
    retries: int = 2,
    backoff_base: float = 0.5,
    breaker: CircuitBreaker | None = None,
) -> T:
    """Invoke *fn* with timeout, retry, and (optional) circuit-breaker guards.

    Args:
        fn:           Zero-arg coroutine factory to call (wrap your real call in
                      a ``lambda``/closure capturing its arguments).
        name:         Label for logs and breaker correlation.
        timeout_s:    Per-attempt timeout in seconds.
        retries:      Additional attempts after the first (so ``retries=2`` =
                      up to 3 calls total).
        backoff_base: Base seconds for exponential back-off between attempts.
        breaker:      Optional breaker shared across calls to the same tool.

    Returns:
        Whatever *fn* returns on success.

    Raises:
        CircuitOpenError: If *breaker* is open when the call is attempted.
        Exception:        The last error if every attempt fails (after the
                          breaker has been updated).
    """
    if breaker is not None and not breaker.allow():
        raise CircuitOpenError(f"{name} circuit is open")

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            result = await asyncio.wait_for(fn(), timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "resilient_call_attempt_failed",
                tool=name,
                attempt=attempt + 1,
                error=str(exc),
            )
            if attempt < retries:
                await asyncio.sleep(backoff_base * (2**attempt))
            continue
        else:
            if breaker is not None:
                breaker.record_success()
            return result

    if breaker is not None:
        breaker.record_failure()
    assert last_exc is not None  # loop ran at least once
    raise last_exc
