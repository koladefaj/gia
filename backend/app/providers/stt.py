"""Speech-to-text provider — faster-whisper (local, RTX 4060).

Transcribes audio blobs to text.  Uses ``faster-whisper`` when installed;
degrades to returning an empty string with a warning log when the library
is absent (dev without a GPU, CI environment).

Usage::

    from backend.app.providers.stt import transcribe

    text = await transcribe(audio_bytes, language="en")

faster-whisper installation::

    pip install faster-whisper

Model used: ``base.en`` (small, fast, good accuracy for English music conversations).
Override with the ``STT_MODEL`` environment variable or ``cfg.stt_model``.
"""

from __future__ import annotations

import asyncio
import io
from functools import lru_cache

from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL = "base.en"
_DEFAULT_DEVICE = "auto"
_DEFAULT_COMPUTE_TYPE = "int8"


@lru_cache(maxsize=1)
def _get_model(model_name: str = _DEFAULT_MODEL):
    """Load and cache the faster-whisper model.

    Returns ``None`` if faster-whisper is not installed.

    Args:
        model_name: Whisper model name (e.g. ``"base.en"``, ``"small"``).

    Returns:
        A ``WhisperModel`` instance, or ``None``.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        model = WhisperModel(
            model_name,
            device=_DEFAULT_DEVICE,
            compute_type=_DEFAULT_COMPUTE_TYPE,
        )
        logger.info("stt_model_loaded", model=model_name)
        return model
    except ImportError:
        logger.warning(
            "faster_whisper_not_installed",
            hint="pip install faster-whisper  # requires libcudart for GPU",
        )
        return None


def _transcribe_sync(audio_bytes: bytes, model_name: str, language: str | None) -> str:
    """Run faster-whisper transcription synchronously (CPU-bound, thread-pool safe).

    Args:
        audio_bytes: Raw audio data (WAV, MP3, WebM, etc.).
        model_name:  Whisper model identifier.
        language:    Optional two-letter BCP-47 code (``"en"``, ``"fr"``).

    Returns:
        Transcribed text, or ``""`` if the model is unavailable.
    """
    model = _get_model(model_name)
    if model is None:
        return ""

    try:
        audio_io = io.BytesIO(audio_bytes)
        segments, _info = model.transcribe(
            audio_io,
            language=language,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.debug("stt_transcribed", length=len(text))
        return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("stt_transcribe_error", error=str(exc))
        return ""


async def transcribe(
    audio_bytes: bytes,
    *,
    model: str = _DEFAULT_MODEL,
    language: str | None = "en",
) -> str:
    """Transcribe *audio_bytes* to text using faster-whisper.

    Runs the CPU-bound model in a thread pool so the event loop is not blocked.

    Args:
        audio_bytes: Raw audio data (WAV, MP3, WebM, OGG, etc.).
        model:       Whisper model name.  Defaults to ``"base.en"``.
        language:    Optional language hint to skip detection.

    Returns:
        Transcribed text string.  Returns ``""`` when whisper is unavailable.
    """
    return await asyncio.to_thread(_transcribe_sync, audio_bytes, model, language)
