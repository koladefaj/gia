"""Tests for the OpenAI embedding service.

The OpenAI client is mocked so these tests run offline with no API calls.
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _fake_client(vector: list[float]) -> MagicMock:
    """Build a fake AsyncOpenAI whose embeddings.create returns *vector*."""
    client = MagicMock()
    resp = MagicMock(data=[MagicMock(embedding=vector, index=0)])
    client.embeddings.create = AsyncMock(return_value=resp)
    return client


def _fake_batch_client(vectors: list[list[float]]) -> MagicMock:
    """Build a fake AsyncOpenAI whose embeddings.create returns *vectors* (with .index)."""
    client = MagicMock()
    data = [MagicMock(embedding=v, index=i) for i, v in enumerate(vectors)]
    client.embeddings.create = AsyncMock(return_value=MagicMock(data=data))
    return client


@pytest.mark.asyncio
async def test_embed_returns_list_of_floats() -> None:
    """``embed`` returns the embedding vector from the OpenAI response."""
    client = _fake_client([0.1, 0.2, 0.3])
    with patch("backend.app.memory.embeddings.get_async_openai", return_value=client):
        from backend.app.memory.embeddings import embed

        result = await embed("test text")

    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_calls_openai_with_text_and_model() -> None:
    """``embed`` sends the text + configured model to the embeddings endpoint."""
    client = _fake_client([0.0] * 8)
    with patch("backend.app.memory.embeddings.get_async_openai", return_value=client):
        from backend.app.memory.embeddings import embed

        await embed("hello world")

    kwargs = client.embeddings.create.call_args.kwargs
    # Single embeds now go through the batch path, so input is a 1-element list.
    assert kwargs["input"] == ["hello world"]
    assert "model" in kwargs


@pytest.mark.asyncio
async def test_embed_many_batches_into_one_call() -> None:
    """``embed_many`` sends all texts in a single request and returns them in order."""
    client = _fake_batch_client([[1.0], [2.0], [3.0]])
    with patch("backend.app.memory.embeddings.get_async_openai", return_value=client):
        from backend.app.memory.embeddings import embed_many

        out = await embed_many(["a", "b", "c"])

    assert out == [[1.0], [2.0], [3.0]]
    client.embeddings.create.assert_called_once()
    assert client.embeddings.create.call_args.kwargs["input"] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_embed_many_realigns_by_index() -> None:
    """Out-of-order API results are realigned to the input order via ``.index``."""
    client = MagicMock()
    # Returned shuffled; .index points back to the original position.
    data = [MagicMock(embedding=[2.0], index=1), MagicMock(embedding=[0.0], index=0)]
    client.embeddings.create = AsyncMock(return_value=MagicMock(data=data))
    with patch("backend.app.memory.embeddings.get_async_openai", return_value=client):
        from backend.app.memory.embeddings import embed_many

        out = await embed_many(["x", "y"])

    assert out == [[0.0], [2.0]]


@pytest.mark.asyncio
async def test_embed_many_empty_returns_empty_without_calling_api() -> None:
    client = _fake_batch_client([])
    with patch("backend.app.memory.embeddings.get_async_openai", return_value=client):
        from backend.app.memory.embeddings import embed_many

        out = await embed_many([])

    assert out == []
    client.embeddings.create.assert_not_called()


@pytest.mark.asyncio
async def test_embed_many_serves_hits_from_cache_and_only_embeds_misses() -> None:
    """Cached texts skip the API; only the misses are sent, vectors stay aligned."""
    import json

    from backend.app.memory.embeddings import text_hash

    async def fake_get(key: str):
        return json.dumps([9.9]) if key == f"embed_cache:{text_hash('a')}" else None

    redis = MagicMock()
    redis.get = AsyncMock(side_effect=fake_get)
    redis.set = AsyncMock()

    client = _fake_batch_client([[2.0]])  # only the miss ("b") is embedded
    with patch("backend.app.memory.embeddings.get_async_openai", return_value=client):
        from backend.app.memory.embeddings import embed_many

        out = await embed_many(["a", "b"], redis=redis)

    assert out == [[9.9], [2.0]]
    assert client.embeddings.create.call_args.kwargs["input"] == ["b"]


def test_text_hash_returns_sha256_hex() -> None:
    """``text_hash`` is deterministic and matches hashlib output."""
    from backend.app.memory.embeddings import text_hash

    text = "User loves Tems during wind-down sessions"
    expected = hashlib.sha256(text.encode()).hexdigest()
    assert text_hash(text) == expected
    assert len(text_hash(text)) == 64


def test_text_hash_different_inputs_give_different_hashes() -> None:
    """Different texts produce different SHA-256 hashes."""
    from backend.app.memory.embeddings import text_hash

    assert text_hash("text A") != text_hash("text B")


def test_text_hash_same_input_is_stable() -> None:
    """Same text always produces the same hash (deterministic)."""
    from backend.app.memory.embeddings import text_hash

    h = text_hash("stable")
    assert text_hash("stable") == h
    assert text_hash("stable") == h
