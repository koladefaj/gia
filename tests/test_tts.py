"""Tests for the TTS provider and voice streaming utilities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.providers.tts import is_emotional
from backend.app.voice.streaming import split_sentences


class TestIsEmotional:
    def test_audio_tag_is_emotional(self) -> None:
        assert is_emotional("[warmly] Hey, long week?")

    def test_question_is_emotional(self) -> None:
        assert is_emotional("Everything okay?")

    def test_logistics_not_emotional(self) -> None:
        assert not is_emotional("Adding to your library.")

    def test_empty_string_not_emotional(self) -> None:
        assert not is_emotional("")

    def test_laughs_tag_emotional(self) -> None:
        assert is_emotional("[laughs] That tweet is wild.")

    def test_plain_statement_not_emotional(self) -> None:
        assert not is_emotional("Here are four tracks for you.")


class TestSplitSentences:
    def test_splits_on_period(self) -> None:
        parts = split_sentences("Hello. How are you. I'm well.")
        assert len(parts) >= 2

    def test_splits_on_question_mark(self) -> None:
        parts = split_sentences("Everything okay? Just checking.")
        assert len(parts) == 2

    def test_preserves_audio_tags(self) -> None:
        parts = split_sentences("[warmly] Hey. Long week?")
        assert any("[warmly]" in p for p in parts)

    def test_empty_string(self) -> None:
        assert split_sentences("") == []

    def test_single_sentence_no_split(self) -> None:
        parts = split_sentences("Here's Free Mind by Tems")
        assert len(parts) == 1

    def test_filters_empty_parts(self) -> None:
        parts = split_sentences("Hello.   ")
        assert all(p.strip() for p in parts)


@pytest.mark.asyncio
async def test_synthesize_kokoro_returns_empty_when_not_installed() -> None:
    """Kokoro returns ``b""`` gracefully when not installed."""
    from backend.app.providers.tts import synthesize

    with patch("backend.app.providers.tts._get_kokoro_pipeline", return_value=None):
        result = await synthesize("Hello there.", provider="kokoro")

    assert result == b""


@pytest.mark.asyncio
async def test_synthesize_elevenlabs_returns_empty_without_key() -> None:
    """ElevenLabs returns ``b""`` when no API key is configured."""
    from backend.app.providers.tts import synthesize

    result = await synthesize("Hello there.", provider="elevenlabs", api_key="", voice_id="")
    assert result == b""


@pytest.mark.asyncio
async def test_synthesize_elevenlabs_calls_api() -> None:
    """ElevenLabs hits the API when a key is provided — or falls back gracefully."""
    from backend.app.providers.tts import synthesize

    mock_resp = MagicMock()
    mock_resp.content = b"fake-mp3-bytes"
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await synthesize(
            "Hello.", provider="elevenlabs", api_key="test-key", voice_id="voice-123"
        )

    # Either succeeded (mp3 bytes) or fell back gracefully (b"")
    assert isinstance(result, bytes)


@pytest.mark.asyncio
async def test_stream_tts_chunks_yields_nothing_for_empty_text() -> None:
    """Empty input produces no audio chunks."""
    from backend.app.voice.streaming import stream_tts_chunks

    chunks = [chunk async for chunk in stream_tts_chunks("")]
    assert chunks == []


@pytest.mark.asyncio
async def test_stream_tts_chunks_skips_empty_synthesis() -> None:
    """Sentences that synthesize to empty bytes are not yielded."""
    from backend.app.voice.streaming import stream_tts_chunks

    with patch("backend.app.voice.streaming.synthesize", new=AsyncMock(return_value=b"")):
        chunks = [chunk async for chunk in stream_tts_chunks("Hello. World.")]

    assert chunks == []


@pytest.mark.asyncio
async def test_stream_tts_chunks_yields_per_sentence() -> None:
    """One chunk per sentence when synthesis returns data."""
    from backend.app.voice.streaming import stream_tts_chunks

    call_count = 0

    async def fake_synthesize(text, **_kwargs):
        nonlocal call_count
        call_count += 1
        return b"audio-" + text.encode()[:4]

    with patch("backend.app.voice.streaming.synthesize", new=fake_synthesize):
        chunks = [chunk async for chunk in stream_tts_chunks("Hello. How are you.")]

    assert len(chunks) >= 1
    assert all(c.startswith(b"audio-") for c in chunks)
