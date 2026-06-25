"""Speech-to-text provider — faster-whisper (local) with OpenAI Whisper fallback.

Priority:
  1. faster-whisper (local, GPU-accelerated, free) — used when installed.
  2. OpenAI Whisper API (``whisper-1``) — used when faster-whisper is absent
     and ``OPENAI_API_KEY`` is set.
  3. Empty string — graceful no-op for dev environments without either.

Usage::

    from backend.app.providers.stt import transcribe

    text = await transcribe(audio_bytes, language="en", cfg=cfg)
"""

from __future__ import annotations

import asyncio
import io
import os
from functools import lru_cache

from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL = "base.en"


def _pick_device() -> tuple[str, str]:
    """Choose the fastest available ctranslate2 backend.

    A visible CUDA GPU (the RTX 4060 here) runs ``int8_float16`` — several times
    faster than CPU and comfortably under the round-trip of a spoken turn, so the
    GPU *lowers* transcription latency rather than adding any. Falls back to CPU
    ``int8`` when no GPU is passed through, so the same image still runs on a
    laptop or in CI.
    """
    try:
        import ctranslate2  # type: ignore[import-untyped]

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "int8_float16"
    except Exception:  # noqa: BLE001 — any probe failure → safe CPU path
        pass
    return "cpu", "int8"


@lru_cache(maxsize=1)
def _get_local_model(model_name: str = _DEFAULT_MODEL):
    """Load and cache the faster-whisper model, or return None if not installed."""
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
    except ImportError:
        logger.info("faster_whisper_not_installed", hint="falling back to OpenAI Whisper API")
        return None

    device, compute_type = _pick_device()
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        logger.info("stt_model_loaded", model=model_name, device=device, compute_type=compute_type)
        return model
    except Exception as exc:  # noqa: BLE001 — GPU libs missing/mismatched → CPU
        if device == "cpu":
            raise
        logger.warning("stt_gpu_init_failed", error=str(exc), hint="falling back to CPU int8")
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        logger.info("stt_model_loaded", model=model_name, device="cpu", compute_type="int8")
        return model


def warmup(model: str = _DEFAULT_MODEL) -> None:
    """Eagerly load the model (into VRAM on GPU) so the first spoken turn doesn't
    pay the one-time model-load cost. Safe no-op when faster-whisper is absent."""
    _get_local_model(model)


def _transcribe_sync(audio_bytes: bytes, model_name: str, language: str | None) -> str:
    """Run faster-whisper transcription synchronously."""
    model = _get_local_model(model_name)
    if model is None:
        return ""
    try:
        segments, _info = model.transcribe(
            io.BytesIO(audio_bytes),
            language=language,
            beam_size=1,  # greedy — ~2x faster than beam search, fine for short turns
            # No server-side VAD: the frontend already segments on voice activity,
            # so a second VAD pass here is redundant — and on short, browser-decoded
            # (webm/opus) clips it was misclassifying the whole utterance as silence
            # and dropping it, producing empty transcripts. Trust the client's turn.
            vad_filter=False,
            condition_on_previous_text=False,  # no context carry → faster, fewer loops
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.debug("stt_local_transcribed", length=len(text))
        return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("stt_local_error", error=str(exc))
        return ""


async def _transcribe_openai(audio_bytes: bytes, language: str | None, api_key: str) -> str:
    """Transcribe via the OpenAI Whisper API (``whisper-1``)."""
    try:
        from openai import AsyncOpenAI  # noqa: PLC0415

        client = AsyncOpenAI(api_key=api_key)
        audio_io = io.BytesIO(audio_bytes)
        audio_io.name = "recording.webm"
        result = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_io,
            language=language or "en",
        )
        text = result.text.strip()
        logger.debug("stt_openai_transcribed", length=len(text))
        return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("stt_openai_error", error=str(exc))
        return ""


async def transcribe(
    audio_bytes: bytes,
    *,
    model: str = _DEFAULT_MODEL,
    language: str | None = "en",
    cfg=None,  # Settings | None — avoid circular import at module level
) -> str:
    """Transcribe *audio_bytes* to text.

    Tries faster-whisper first; falls back to the OpenAI Whisper API when
    faster-whisper is not installed and the app is configured with an OpenAI key.

    Args:
        audio_bytes: Raw audio data (WAV, MP3, WebM, OGG, etc.).
        model:       faster-whisper model name.  Defaults to ``"base.en"``.
        language:    Optional BCP-47 language code hint.
        cfg:         App settings (used to resolve the OpenAI API key).

    Returns:
        Transcribed text string.  Returns ``""`` when no STT provider is available.
    """
    # Provider + model from settings (env STT_PROVIDER / STT_MODEL), with the
    # function args as fallbacks. STT_PROVIDER=openai forces the Whisper API —
    # a larger multilingual model that handles accented English (Nigerian, etc.)
    # noticeably better than the local base.en.
    provider = (getattr(cfg, "stt_provider", "") or "local") if cfg else "local"
    model_name = (getattr(cfg, "stt_model", "") if cfg else "") or model

    def _openai_key() -> str:
        key = getattr(cfg, "openai_api_key", "") if cfg else ""
        return key or os.getenv("OPENAI_API_KEY", "")

    # Path 1: explicit OpenAI Whisper API
    if provider == "openai":
        api_key = _openai_key()
        if api_key:
            return await _transcribe_openai(audio_bytes, language, api_key)
        logger.warning("stt_openai_selected_no_key", hint="set OPENAI_API_KEY")
        return ""

    # Path 2: local faster-whisper (GPU, free, low-latency)
    local_model = _get_local_model(model_name)
    if local_model is not None:
        return await asyncio.to_thread(_transcribe_sync, audio_bytes, model_name, language)

    # Path 3: fall back to the OpenAI API when faster-whisper isn't installed
    api_key = _openai_key()
    if api_key:
        return await _transcribe_openai(audio_bytes, language, api_key)

    logger.warning(
        "stt_unavailable", hint="install faster-whisper, or set STT_PROVIDER=openai + OPENAI_API_KEY"
    )
    return ""
