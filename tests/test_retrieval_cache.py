"""Tests for the Redis-backed retrieval cache."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone

import pytest

from backend.app.memory.cache import (
    cache_key,
    get_cached,
    invalidate_user,
    set_cached,
)
from backend.app.schemas.memory import MemoryEntry

_NOW = datetime(2026, 6, 19, tzinfo=timezone.utc)


class FakeRedis:
    """Tiny in-memory stand-in supporting the cache's Redis surface."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value

    async def delete(self, *keys: str) -> int:
        n = 0
        for k in keys:
            n += 1 if self.store.pop(k, None) is not None else 0
        return n

    async def scan_iter(self, match: str) -> AsyncIterator[str]:
        prefix = match.rstrip("*")
        for k in list(self.store):
            if k.startswith(prefix):
                yield k


def _entry(uid: str, text: str) -> MemoryEntry:
    return MemoryEntry(
        id=uid, type="preference", text=text, confidence=0.8, created_at=_NOW
    )


def test_cache_key_is_stable_and_normalised() -> None:
    a = cache_key("u1", "preference", "Wind Down")
    b = cache_key("u1", "preference", "  wind down  ")
    assert a == b  # whitespace + case normalised
    assert a != cache_key("u1", "preference", "hype")
    assert a.startswith("retr:u1:preference:")


@pytest.mark.asyncio
async def test_set_then_get_round_trip() -> None:
    r = FakeRedis()
    key = cache_key("u1", "preference", "q")
    entries = [_entry("11111111-0000-0000-0000-000000000001", "loves Tems")]
    await set_cached(r, key, entries, ttl=60)

    got = await get_cached(r, key)
    assert got is not None
    assert len(got) == 1
    assert got[0].text == "loves Tems"
    assert got[0].id == "11111111-0000-0000-0000-000000000001"


@pytest.mark.asyncio
async def test_get_miss_returns_none() -> None:
    r = FakeRedis()
    assert await get_cached(r, "retr:nope") is None


@pytest.mark.asyncio
async def test_ttl_zero_disables_write() -> None:
    r = FakeRedis()
    key = cache_key("u1", "preference", "q")
    await set_cached(r, key, [_entry("a" * 8, "x")], ttl=0)
    assert await get_cached(r, key) is None


@pytest.mark.asyncio
async def test_corrupt_value_is_a_miss() -> None:
    r = FakeRedis()
    key = cache_key("u1", "preference", "q")
    r.store[key] = "{not json"
    assert await get_cached(r, key) is None


@pytest.mark.asyncio
async def test_invalidate_user_clears_only_that_user() -> None:
    r = FakeRedis()
    await set_cached(r, cache_key("u1", "preference", "a"), [_entry("a" * 8, "x")], 60)
    await set_cached(r, cache_key("u1", "episode", "b"), [_entry("b" * 8, "y")], 60)
    await set_cached(r, cache_key("u2", "preference", "c"), [_entry("c" * 8, "z")], 60)

    deleted = await invalidate_user(r, "u1")
    assert deleted == 2
    assert await get_cached(r, cache_key("u2", "preference", "c")) is not None
