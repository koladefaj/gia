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
from collections.abc import AsyncIterator
from functools import lru_cache

from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

# Matches any ``[audio tag]`` (case-insensitive) — used both to strip tags for
# non-ElevenLabs TTS and to decide which ElevenLabs model a sentence needs.
_AUDIO_TAG_RE = re.compile(r"\[[a-z][a-z ]*\]", re.IGNORECASE)


def strip_audio_tags(text: str) -> str:
    """Remove ElevenLabs-style ``[audio tags]`` and tidy the leftover spacing.

    ElevenLabs v3 interprets tags like ``[warm]`` as delivery cues, but a
    plain TTS engine (Kokoro) would read the literal word "warm" aloud. We
    strip them so local audio stays clean; the production ElevenLabs path keeps
    the tags untouched.
    """
    return re.sub(r"\s{2,}", " ", _AUDIO_TAG_RE.sub("", text)).strip()


def has_audio_tag(text: str) -> bool:
    """Return ``True`` when *text* contains an ElevenLabs ``[audio tag]``.

    Used to decide whether a line already carries a v3 delivery cue (so callers
    can add one when it doesn't, routing the line to the expressive model).
    """
    return bool(_AUDIO_TAG_RE.search(text))


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
    # "[warm]" aloud. Empty after stripping → nothing to synthesise.
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

_ELEVEN_BASE = "https://api.elevenlabs.io/v1/text-to-speech"
# Voice settings shared by the blocking and streaming paths — one place to tune.
_VOICE_SETTINGS = {"stability": 0.5, "similarity_boost": 0.75}

# Pooled async client: a fresh AsyncClient per call paid a TLS handshake every
# turn (a real chunk of TTS latency). One keep-alive pool reuses the connection.
_http_client = None  # type: ignore[var-annotated]


def _get_http_client():
    """Return the process-wide pooled httpx client (built on first use)."""
    global _http_client
    if _http_client is None:
        import httpx

        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=60.0),
        )
    return _http_client


async def aclose_http_client() -> None:
    """Close the pooled client on app shutdown (idempotent)."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def _eleven_model(emotional: bool) -> str:
    """Pick the ElevenLabs model: expressive ``eleven_v3`` vs faster flash."""
    # eleven_v3 carries the emotional warmth + audio-tag rendering; it needs the
    # full reply (~250+ chars) for good prosody, which is why we synthesize the
    # whole reply at once rather than per sentence.
    return "eleven_v3" if emotional else "eleven_flash_v2_5"


def _eleven_payload(text: str, emotional: bool) -> dict:
    """Build the JSON body for an ElevenLabs synthesis request."""
    return {
        "text": text,
        "model_id": _eleven_model(emotional),
        "voice_settings": _VOICE_SETTINGS,
    }


def _eleven_headers(api_key: str) -> dict:
    return {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }


async def _elevenlabs_synthesize(
    text: str,
    api_key: str,
    voice_id: str,
    emotional: bool = False,
) -> bytes:
    """Synthesize *text* with ElevenLabs v3 (async HTTP call, whole file).

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

    try:
        client = _get_http_client()
        resp = await client.post(
            f"{_ELEVEN_BASE}/{voice_id}",
            json=_eleven_payload(text, emotional),
            headers=_eleven_headers(api_key),
        )
        resp.raise_for_status()
        logger.debug("elevenlabs_ok", model=_eleven_model(emotional), chars=len(text))
        return resp.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("elevenlabs_error", error=str(exc))
        return b""


async def _elevenlabs_stream(
    text: str,
    api_key: str,
    voice_id: str,
    emotional: bool = False,
) -> AsyncIterator[bytes]:
    """Stream MP3 audio for *text* from ElevenLabs' ``/stream`` endpoint.

    The whole reply text is sent up-front (so v3 keeps full-context prosody and
    tag rendering), but audio bytes are forwarded as they are generated — first
    bytes arrive well before the complete file, cutting time-to-first-audio.

    Yields raw MP3 byte chunks; yields nothing on error / missing key.
    """
    if not api_key:
        logger.debug("elevenlabs_no_key")
        return

    try:
        client = _get_http_client()
        async with client.stream(
            "POST",
            f"{_ELEVEN_BASE}/{voice_id}/stream",
            json=_eleven_payload(text, emotional),
            headers=_eleven_headers(api_key),
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk
        logger.debug("elevenlabs_stream_ok", model=_eleven_model(emotional), chars=len(text))
    except Exception as exc:  # noqa: BLE001
        logger.warning("elevenlabs_stream_error", error=str(exc))


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


async def synthesize_stream(
    text: str,
    *,
    provider: str = "kokoro",
    api_key: str = "",
    voice_id: str = "",
    voice: str = "af_heart",
) -> AsyncIterator[bytes]:
    """Stream audio bytes for *text* as they are produced by the provider.

    For ElevenLabs this hits the ``/stream`` endpoint so the first audio bytes
    reach the caller before the whole file is rendered — the latency win. Kokoro
    has no streaming API, so it falls back to a single synthesized blob (still
    correct, just not progressive). Yields nothing when the provider is
    unavailable.

    Args:
        text:     Text to synthesise (audio tags supported for ElevenLabs v3).
        provider: ``"kokoro"`` or ``"elevenlabs"``.
        api_key:  ElevenLabs API key (ignored for Kokoro).
        voice_id: ElevenLabs voice ID (ignored for Kokoro).
        voice:    Kokoro voice code (ignored for ElevenLabs).

    Yields:
        Raw audio byte chunks (MP3 for ElevenLabs, one WAV blob for Kokoro).
    """
    if provider == "elevenlabs":
        async for chunk in _elevenlabs_stream(text, api_key, voice_id, is_emotional(text)):
            yield chunk
        return

    blob = await asyncio.to_thread(_kokoro_synthesize_sync, text, voice)
    if blob:
        yield blob
