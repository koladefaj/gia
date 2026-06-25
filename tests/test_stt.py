"""Tests for the STT provider."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_transcribe_returns_empty_when_no_provider() -> None:
    """``transcribe`` returns ``""`` when neither faster-whisper nor OpenAI is available."""
    from backend.app.providers.stt import transcribe

    with (
        patch("backend.app.providers.stt._get_local_model", return_value=None),
        patch.dict(os.environ, {}, clear=True),
    ):
        result = await transcribe(b"fake-audio-bytes")

    assert result == ""


@pytest.mark.asyncio
async def test_transcribe_returns_text_from_local_whisper() -> None:
    """``transcribe`` joins segment texts from the local faster-whisper model."""
    from backend.app.providers.stt import transcribe

    seg1 = MagicMock()
    seg1.text = " Hello there"
    seg2 = MagicMock()
    seg2.text = " how are you"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([seg1, seg2], MagicMock())

    with patch("backend.app.providers.stt._get_local_model", return_value=mock_model):
        result = await transcribe(b"audio-bytes", language="en")

    assert result == "Hello there how are you"


@pytest.mark.asyncio
async def test_transcribe_falls_back_to_openai_when_whisper_absent() -> None:
    """``transcribe`` uses OpenAI Whisper API when faster-whisper is not installed."""
    from backend.app.providers.stt import transcribe

    with (
        patch("backend.app.providers.stt._get_local_model", return_value=None),
        patch(
            "backend.app.providers.stt._transcribe_openai",
            new=AsyncMock(return_value="play me some afrobeats"),
        ),
    ):
        result = await transcribe(b"audio-bytes", cfg=MagicMock(openai_api_key="sk-test"))

    assert result == "play me some afrobeats"


@pytest.mark.asyncio
async def test_transcribe_openai_body() -> None:
    """``_transcribe_openai`` calls the Whisper API and returns stripped text."""
    from backend.app.providers.stt import _transcribe_openai

    mock_result = MagicMock()
    mock_result.text = "  what's playing  "

    mock_transcriptions = MagicMock()
    mock_transcriptions.create = AsyncMock(return_value=mock_result)

    mock_client = MagicMock()
    mock_client.audio.transcriptions = mock_transcriptions

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        result = await _transcribe_openai(b"audio", "en", "sk-test")

    assert result == "what's playing"


@pytest.mark.asyncio
async def test_transcribe_openai_error_returns_empty() -> None:
    """``_transcribe_openai`` returns ``""`` and does not raise on API errors."""
    from backend.app.providers.stt import _transcribe_openai

    with patch("openai.AsyncOpenAI", side_effect=RuntimeError("api error")):
        result = await _transcribe_openai(b"audio", "en", "sk-bad")

    assert result == ""


@pytest.mark.asyncio
async def test_transcribe_handles_local_model_error() -> None:
    """Transcription errors from the local model return ``""`` without raising."""
    from backend.app.providers.stt import transcribe

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = RuntimeError("model crashed")

    with patch("backend.app.providers.stt._get_local_model", return_value=mock_model):
        result = await transcribe(b"audio-bytes")

    assert result == ""


@pytest.mark.asyncio
async def test_transcribe_empty_segments() -> None:
    """Empty segment list from local model returns ``""``."""
    from backend.app.providers.stt import transcribe

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], MagicMock())

    with patch("backend.app.providers.stt._get_local_model", return_value=mock_model):
        result = await transcribe(b"silence")

    assert result == ""
