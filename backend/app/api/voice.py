"""Voice API — STT transcription and TTS synthesis endpoints.

``POST /voice/transcribe`` accepts a raw audio upload and returns a
transcript string.  ``POST /voice/speak`` accepts text and returns an
audio blob.

These endpoints are used by the Next.js frontend's ``MediaRecorder`` loop
(Day 10).  They are deliberately thin — all intelligence is in ``/chat``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from backend.app.config import Settings
from backend.app.dependencies import get_settings
from backend.app.observability.logging import get_logger
from backend.app.providers.stt import transcribe
from backend.app.providers.tts import synthesize

logger = get_logger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])


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


@router.post("/transcribe", summary="Transcribe audio to text", status_code=200, response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, WebM, OGG)"),
    language: str = Form(default="en"),
    cfg: Settings = Depends(get_settings),
) -> TranscribeResponse:
    """Transcribe an uploaded audio file to text.

    Uses faster-whisper locally (RTX 4060).  Returns ``transcript=""``
    gracefully when faster-whisper is not installed.

    Args:
        audio:    Uploaded audio file.
        language: BCP-47 language code hint (default ``"en"``).

    Returns:
        ``TranscribeResponse`` with ``transcript`` and ``language``.
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    logger.info("voice_transcribe_request", bytes=len(audio_bytes), language=language)
    text = await transcribe(audio_bytes, language=language)
    logger.info("voice_transcribe_done", transcript_len=len(text))

    return TranscribeResponse(transcript=text, language=language)


@router.post("/speak", summary="Synthesise text to speech", status_code=200)
async def speak(
    body: SpeakRequest,
    cfg: Settings = Depends(get_settings),
) -> Response:
    """Synthesise *body.text* to audio and return the raw bytes.

    Returns WAV (Kokoro) or MP3 (ElevenLabs) depending on the provider.
    Returns a 200 with empty body when no TTS provider is available in dev.

    Args:
        body: ``SpeakRequest`` with ``text`` and optional ``provider`` override.

    Returns:
        ``Response`` with ``Content-Type: audio/wav`` or ``audio/mpeg``.
    """
    provider = body.provider or cfg.tts_provider
    api_key = cfg.elevenlabs_api_key if provider == "elevenlabs" else ""
    voice_id = cfg.elevenlabs_voice_id if provider == "elevenlabs" else ""

    logger.info("voice_speak_request", provider=provider, text_len=len(body.text))
    chunk = await synthesize(
        body.text,
        provider=provider,
        api_key=api_key,
        voice_id=voice_id,
    )

    if not chunk:
        return Response(content=b"", media_type="audio/wav")

    media_type = "audio/mpeg" if provider == "elevenlabs" else "audio/wav"
    return Response(content=chunk, media_type=media_type)
