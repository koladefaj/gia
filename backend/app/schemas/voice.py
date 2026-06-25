from __future__ import annotations

from pydantic import BaseModel


class TranscribeResponse(BaseModel):
    """Response from ``POST /voice/transcribe``.

    Attributes:
        transcript: The recognised text, or ``""`` if STT is unavailable.
        language:   Detected or provided language code.
    """

    transcript: str
    language: str = "en"


class SpeakRequest(BaseModel):
    """Request body for ``POST /voice/speak``.

    Attributes:
        text:     Text to synthesise (audio tags supported for ElevenLabs v3).
        provider: Override TTS provider (``"kokoro"`` or ``"elevenlabs"``).
                  Defaults to the app-level ``TTS_PROVIDER`` setting.
    """

    text: str
    provider: str | None = None
