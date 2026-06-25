"""Tests for short-term conversation history (Redis-backed)."""

from __future__ import annotations

import pytest

from backend.app.memory.session_history import (
    _MAX_TURNS,
    append_turn,
    format_history,
    get_history,
)


class FakeListRedis:
    """Minimal async Redis supporting the list ops session_history uses."""

    def __init__(self) -> None:
        self.lists: dict[str, list] = {}

    @staticmethod
    def _slice(lst: list, start: int, end: int) -> list:
        n = len(lst)
        s = start if start >= 0 else n + start
        e = end if end >= 0 else n + end
        return lst[max(s, 0):e + 1]

    async def rpush(self, key: str, val: str) -> None:
        self.lists.setdefault(key, []).append(val)

    async def ltrim(self, key: str, start: int, end: int) -> None:
        self.lists[key] = self._slice(self.lists.get(key, []), start, end)

    async def lrange(self, key: str, start: int, end: int) -> list:
        return self._slice(self.lists.get(key, []), start, end)

    async def expire(self, key: str, ttl: int) -> None:
        pass


@pytest.mark.asyncio
async def test_append_and_get_roundtrip() -> None:
    r = FakeListRedis()
    await append_turn(r, "s1", "user", "play drake")
    await append_turn(r, "s1", "gia", "on it")
    turns = await get_history(r, "s1")
    assert [t["role"] for t in turns] == ["user", "gia"]
    assert turns[0]["text"] == "play drake"


@pytest.mark.asyncio
async def test_history_is_capped() -> None:
    r = FakeListRedis()
    for i in range(_MAX_TURNS + 8):
        await append_turn(r, "s1", "user", f"msg{i}")
    turns = await get_history(r, "s1")
    assert len(turns) == _MAX_TURNS
    assert turns[-1]["text"] == f"msg{_MAX_TURNS + 7}"  # newest kept


@pytest.mark.asyncio
async def test_blank_inputs_are_noops() -> None:
    r = FakeListRedis()
    await append_turn(r, "", "user", "x")      # no session id
    await append_turn(r, "s1", "user", "   ")  # empty text
    assert await get_history(r, "s1") == []
    assert await get_history(r, "") == []


def test_format_history() -> None:
    txt = format_history([
        {"role": "user", "text": "play drake"},
        {"role": "gia", "text": "on it"},
    ])
    assert txt == "User: play drake\nGia: on it"


def test_format_history_empty() -> None:
    assert format_history([]) == ""
