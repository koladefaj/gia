"""Tests for the tool-resilience primitives."""

from __future__ import annotations

import asyncio

import pytest

from backend.app.tools.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    resilient_call,
)

# ── CircuitBreaker ────────────────────────────────────────────────────────────


def test_breaker_opens_after_threshold() -> None:
    cb = CircuitBreaker("t", threshold=3, cooldown=10.0)
    assert cb.state is CircuitState.CLOSED
    for _ in range(3):
        cb.record_failure()
    assert cb.state is CircuitState.OPEN
    assert cb.allow() is False


def test_breaker_success_resets() -> None:
    cb = CircuitBreaker("t", threshold=2, cooldown=10.0)
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    # One failure after a reset is below threshold → still closed.
    assert cb.state is CircuitState.CLOSED


def test_breaker_half_opens_after_cooldown() -> None:
    cb = CircuitBreaker("t", threshold=1, cooldown=0.0)
    cb.record_failure()
    # cooldown 0 → immediately eligible for a half-open probe
    assert cb.state is CircuitState.HALF_OPEN
    assert cb.allow() is True
    # A failure while half-open re-opens.
    cb.record_failure()
    assert cb._state is CircuitState.OPEN  # noqa: SLF001


# ── resilient_call ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resilient_call_returns_on_success() -> None:
    async def ok() -> int:
        return 42

    assert await resilient_call(ok, name="t", timeout_s=1.0) == 42


@pytest.mark.asyncio
async def test_resilient_call_retries_then_succeeds() -> None:
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "ok"

    out = await resilient_call(
        flaky, name="t", timeout_s=1.0, retries=2, backoff_base=0.0
    )
    assert out == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_resilient_call_raises_after_exhaustion_and_opens_breaker() -> None:
    cb = CircuitBreaker("t", threshold=1, cooldown=10.0)

    async def always_fail() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await resilient_call(
            always_fail, name="t", timeout_s=1.0, retries=1, backoff_base=0.0, breaker=cb
        )
    assert cb.state is CircuitState.OPEN


@pytest.mark.asyncio
async def test_resilient_call_rejects_when_breaker_open() -> None:
    cb = CircuitBreaker("t", threshold=1, cooldown=10.0)
    cb.record_failure()  # open it

    async def ok() -> int:
        return 1

    with pytest.raises(CircuitOpenError):
        await resilient_call(ok, name="t", timeout_s=1.0, breaker=cb)


@pytest.mark.asyncio
async def test_resilient_call_times_out() -> None:
    async def slow() -> None:
        await asyncio.sleep(1.0)

    with pytest.raises(asyncio.TimeoutError):
        await resilient_call(
            slow, name="t", timeout_s=0.01, retries=0, backoff_base=0.0
        )
