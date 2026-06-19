"""Tests for the MCP bridge's owner-task + auto-reconnect logic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.app.tools.spotify import _McpBridge


def _result(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(text=text, type="text")])


class _FakeSession:
    def __init__(self, *, fail: bool) -> None:
        self._fail = fail

    async def call_tool(self, tool: str, args: dict):  # noqa: ANN201
        if self._fail:
            raise RuntimeError("stream corrupted (simulated token-refresh)")
        return _result(f"ok:{tool}")


@pytest.mark.asyncio
async def test_call_before_start_raises() -> None:
    bridge = _McpBridge("node", "x")
    with pytest.raises(RuntimeError, match="not started"):
        await bridge.call("searchSpotify", {})


@pytest.mark.asyncio
async def test_happy_path_returns_text() -> None:
    async def factory():  # noqa: ANN202
        return _FakeSession(fail=False)

    bridge = _McpBridge("node", "x", session_factory=factory)
    await bridge.start()
    try:
        assert await bridge.call("searchSpotify", {}) == "ok:searchSpotify"
    finally:
        await bridge.stop()


@pytest.mark.asyncio
async def test_reconnects_after_session_failure() -> None:
    """A failed call tears down the session and retries on a fresh one."""
    connects = {"n": 0}

    async def factory():  # noqa: ANN202
        connects["n"] += 1
        # First connection yields a session that fails; the reconnect succeeds.
        return _FakeSession(fail=(connects["n"] == 1))

    bridge = _McpBridge("node", "x", session_factory=factory)
    await bridge.start()
    try:
        result = await bridge.call("getNowPlaying", {})
    finally:
        await bridge.stop()

    assert result == "ok:getNowPlaying"
    assert connects["n"] == 2  # reconnected exactly once


@pytest.mark.asyncio
async def test_raises_after_exhausting_retries() -> None:
    async def factory():  # noqa: ANN202
        return _FakeSession(fail=True)  # always fails

    bridge = _McpBridge("node", "x", session_factory=factory)
    await bridge.start()
    try:
        with pytest.raises(RuntimeError, match="stream corrupted"):
            await bridge.call("searchSpotify", {})
    finally:
        await bridge.stop()
