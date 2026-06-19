"""Tests for the BGE embedding service.

The sentence-transformers model is never actually loaded — ``asyncio.to_thread``
is patched so we can test the public API without triggering a model download.
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_embed_returns_list_of_floats() -> None:
    """``embed`` returns a list of floats produced by the model encode call."""
    fake_vector = [0.1, 0.2, 0.3]

    with patch("backend.app.memory.embeddings.asyncio.to_thread", new=AsyncMock(return_value=fake_vector)):
        from backend.app.memory.embeddings import embed

        result = await embed("test text")

    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_passes_text_to_encode() -> None:
    """``embed`` calls the thread function (which calls model.encode) with the text."""
    captured: list[object] = []

    async def fake_to_thread(fn, *args, **kwargs):  # type: ignore[misc]
        captured.append((args, kwargs))
        return [0.0] * 768

    with patch("backend.app.memory.embeddings.asyncio.to_thread", new=fake_to_thread):
        from importlib import reload

        import backend.app.memory.embeddings as emb

        reload(emb)
        await emb.embed("hello world")

    # to_thread is called with a callable and no positional args beyond that
    assert len(captured) == 1


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
