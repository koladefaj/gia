"""Tests for the STT provider."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_transcribe_returns_empty_when_whisper_not_installed() -> None:
    """``transcribe`` returns ``""`` gracefully when faster-whisper is absent."""
    from backend.app.providers.stt import transcribe

    with patch("backend.app.providers.stt._get_model", return_value=None):
        result = await transcribe(b"fake-audio-bytes")

    assert result == ""


@pytest.mark.asyncio
async def test_transcribe_returns_text_from_whisper() -> None:
    """``transcribe`` joins segment texts from the model."""
    from unittest.mock import MagicMock

    from backend.app.providers.stt import transcribe

    seg1 = MagicMock()
    seg1.text = " Hello there"
    seg2 = MagicMock()
    seg2.text = " how are you"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([seg1, seg2], MagicMock())

    with patch("backend.app.providers.stt._get_model", return_value=mock_model):
        result = await transcribe(b"audio-bytes", language="en")

    assert result == "Hello there how are you"


@pytest.mark.asyncio
async def test_transcribe_handles_model_error() -> None:
    """Transcription errors return ``""`` without raising."""
    from unittest.mock import MagicMock

    from backend.app.providers.stt import transcribe

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = RuntimeError("model crashed")

    with patch("backend.app.providers.stt._get_model", return_value=mock_model):
        result = await transcribe(b"audio-bytes")

    assert result == ""


@pytest.mark.asyncio
async def test_transcribe_empty_segments() -> None:
    """Empty segment list returns ``""``."""
    from unittest.mock import MagicMock

    from backend.app.providers.stt import transcribe

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], MagicMock())

    with patch("backend.app.providers.stt._get_model", return_value=mock_model):
        result = await transcribe(b"silence")

    assert result == ""
