"""Tests for ``POST /voice/transcribe`` and ``POST /voice/speak``."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_transcribe_returns_empty_when_whisper_unavailable(client: AsyncClient) -> None:
    """``POST /voice/transcribe`` returns transcript='' when STT unavailable."""
    with patch("backend.app.api.voice.transcribe", new=AsyncMock(return_value="")):
        response = await client.post(
            "/voice/transcribe",
            files={"audio": ("test.wav", b"fake-wav-bytes", "audio/wav")},
            data={"language": "en"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["transcript"] == ""
    assert data["language"] == "en"


@pytest.mark.asyncio
async def test_transcribe_returns_text(client: AsyncClient) -> None:
    """``POST /voice/transcribe`` returns the transcript from STT."""
    with patch("backend.app.api.voice.transcribe", new=AsyncMock(return_value="find me something chill")):
        response = await client.post(
            "/voice/transcribe",
            files={"audio": ("test.wav", b"wav-audio-data", "audio/wav")},
            data={"language": "en"},
        )

    assert response.status_code == 200
    assert response.json()["transcript"] == "find me something chill"


@pytest.mark.asyncio
async def test_transcribe_empty_audio_returns_400(client: AsyncClient) -> None:
    """Empty audio upload returns 400."""
    response = await client.post(
        "/voice/transcribe",
        files={"audio": ("test.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_speak_returns_audio_bytes(client: AsyncClient) -> None:
    """``POST /voice/speak`` returns audio bytes when TTS is available."""
    fake_audio = b"RIFF" + b"\x00" * 44  # minimal WAV header

    with patch("backend.app.api.voice.synthesize", new=AsyncMock(return_value=fake_audio)):
        response = await client.post(
            "/voice/speak",
            json={"text": "Hey, long week?"},
        )

    assert response.status_code == 200
    assert response.content == fake_audio
    assert "audio" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_speak_returns_empty_body_when_tts_unavailable(client: AsyncClient) -> None:
    """``POST /voice/speak`` returns 200 with empty body when TTS unavailable."""
    with patch("backend.app.api.voice.synthesize", new=AsyncMock(return_value=b"")):
        response = await client.post(
            "/voice/speak",
            json={"text": "Hello there."},
        )

    assert response.status_code == 200
    assert response.content == b""


@pytest.mark.asyncio
async def test_speak_provider_override(client: AsyncClient) -> None:
    """Provider override in the request body is respected."""
    with patch("backend.app.api.voice.synthesize", new=AsyncMock(return_value=b"mp3-data")) as mock_synth:
        await client.post(
            "/voice/speak",
            json={"text": "Hello.", "provider": "elevenlabs"},
        )

    call_kwargs = mock_synth.call_args.kwargs
    assert call_kwargs["provider"] == "elevenlabs"
