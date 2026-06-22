"""TTS provider — Kokoro (local) + ElevenLabs v3 (production).

Two-tier strategy:

1. **Development / iteration**: Kokoro runs locally, zero cost, fast turnaround.
   Use for testing agent logic, sentence streaming, and audio queue mechanics.

2. **Production / tuning days**: ElevenLabs eleven_v3 for emotional warmth,
   audio tags, and natural pacing.  Switch by setting ``TTS_PROVIDER=elevenlabs``.

Hybrid for latency (from Section 7):
  - Emotional sentences (containing audio tags or ending with ``?``) → ``eleven_v3``
  - Logistics sentences → ``eleven_flash_v2_5``

Usage::

    from backend.app.providers.tts import synthesize, is_emotional

    chunk = await synthesize("Here's Free Mind by Tems.", provider="kokoro")
    async for chunk in stream_sentence(sentence):
        yield chunk
"""

from __future__ import annotations

import asyncio
import io
import re
from functools import lru_cache

from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

# Matches any ``[audio tag]`` (case-insensitive) — used both to strip tags for
# non-ElevenLabs TTS and to decide which ElevenLabs model a sentence needs.
_AUDIO_TAG_RE = re.compile(r"\[[a-z][a-z ]*\]", re.IGNORECASE)


def strip_audio_tags(text: str) -> str:
    """Remove ElevenLabs-style ``[audio tags]`` and tidy the leftover spacing.

    ElevenLabs v3 interprets tags like ``[warmly]`` as delivery cues, but a
    plain TTS engine (Kokoro) would read the literal word "warmly" aloud. We
    strip them so local audio stays clean; the production ElevenLabs path keeps
    the tags untouched.
    """
    return re.sub(r"\s{2,}", " ", _AUDIO_TAG_RE.sub("", text)).strip()


def is_emotional(sentence: str) -> bool:
    """Return ``True`` when a sentence warrants expressive TTS (ElevenLabs v3).

    Used by the hybrid model-picker to decide whether to route to the full
    ``eleven_v3`` model or the faster ``eleven_flash_v2_5``.

    Args:
        sentence: One sentence of Gia's reply.

    Returns:
        ``True`` if the sentence contains an audio tag or is a question.
    """
    # ANY audio tag must route to eleven_v3 — the faster eleven_flash_v2_5 can't
    # render tags and would read them aloud (e.g. saying "laughs softly").
    return bool(_AUDIO_TAG_RE.search(sentence)) or sentence.strip().endswith("?")


# ── Kokoro (local) ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_kokoro_pipeline():
    """Lazy-load the Kokoro TTS pipeline.

    Returns ``None`` when kokoro is not installed (graceful degradation).
    """
    try:
        from kokoro import KPipeline  # type: ignore[import-untyped]

        pipeline = KPipeline(lang_code="a")
        logger.info("kokoro_pipeline_loaded")
        return pipeline
    except ImportError:
        logger.warning("kokoro_not_installed", hint="pip install kokoro soundfile")
        return None


def _kokoro_synthesize_sync(text: str, voice: str = "af_heart") -> bytes:
    """Synthesize *text* with Kokoro synchronously.

    Args:
        text:  Text to synthesise.
        voice: Kokoro voice code (default ``"af_heart"`` — warm female).

    Returns:
        WAV audio bytes, or ``b""`` if Kokoro is unavailable.
    """
    pipeline = _get_kokoro_pipeline()
    if pipeline is None:
        return b""

    # Kokoro has no notion of delivery tags — strip them so it doesn't read
    # "[warmly]" aloud. Empty after stripping → nothing to synthesise.
    text = strip_audio_tags(text)
    if not text:
        return b""

    try:
        import numpy as np
        import soundfile as sf  # type: ignore[import-untyped]

        generator = pipeline(text, voice=voice, speed=1.0)
        chunks = [audio for _, _, audio in generator]
        if not chunks:
            return b""

        combined = np.concatenate(chunks)
        buf = io.BytesIO()
        sf.write(buf, combined, samplerate=24000, format="WAV")
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        logger.warning("kokoro_synthesis_error", error=str(exc))
        return b""


# ── ElevenLabs v3 ─────────────────────────────────────────────────────────────


async def _elevenlabs_synthesize(
    text: str,
    api_key: str,
    voice_id: str,
    emotional: bool = False,
) -> bytes:
    """Synthesize *text* with ElevenLabs v3 (async HTTP call).

    Args:
        text:      Text to synthesize (audio tags are passed through).
        api_key:   ElevenLabs API key.
        voice_id:  ElevenLabs voice ID.
        emotional: If ``True``, use ``eleven_v3``; else ``eleven_flash_v2_5``.

    Returns:
        MP3 audio bytes, or ``b""`` on error / missing key.
    """
    if not api_key:
        logger.debug("elevenlabs_no_key")
        return b""

    model_id = "eleven_v3" if emotional else "eleven_flash_v2_5"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    try:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                json={
                    "text": text,
                    "model_id": model_id,
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                },
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
            )
            resp.raise_for_status()
            logger.debug("elevenlabs_ok", model=model_id, chars=len(text))
            return resp.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("elevenlabs_error", error=str(exc))
        return b""


# ── Public API ─────────────────────────────────────────────────────────────────


async def synthesize(
    text: str,
    *,
    provider: str = "kokoro",
    api_key: str = "",
    voice_id: str = "",
    voice: str = "af_heart",
) -> bytes:
    """Synthesize *text* to audio bytes using the configured TTS provider.

    Args:
        text:     Text to synthesise (audio tags supported for ElevenLabs v3).
        provider: ``"kokoro"`` or ``"elevenlabs"``.
        api_key:  ElevenLabs API key (ignored for Kokoro).
        voice_id: ElevenLabs voice ID (ignored for Kokoro).
        voice:    Kokoro voice code (ignored for ElevenLabs).

    Returns:
        Raw audio bytes (WAV for Kokoro, MP3 for ElevenLabs), or ``b""``
        when the provider is unavailable.
    """
    if provider == "elevenlabs":
        emotional = is_emotional(text)
        return await _elevenlabs_synthesize(text, api_key, voice_id, emotional)

    return await asyncio.to_thread(_kokoro_synthesize_sync, text, voice)
