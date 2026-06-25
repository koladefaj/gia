"""Voice streaming utilities — sentence splitting, chunk queuing, SSE helpers.

Sentence-level streaming achieves sub-second first-word latency: the first
sentence is synthesised and starts playing before the full reply is ready.

Design (from Section 7):
  1. Split Gia's reply into sentences at ``. ? !`` and natural pause points.
  2. For each sentence: choose TTS model (v3 for emotional, Flash otherwise).
  3. Synthesise and enqueue the audio chunk.
  4. The frontend dequeues and plays in order, gapless.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from backend.app.observability.logging import get_logger
from backend.app.providers.tts import synthesize

logger = get_logger(__name__)

# Sentence boundary: end of `. ? !` followed by whitespace or end-of-string.
# Audio tags like [pause] count as boundaries too.
_SENTENCE_RE = re.compile(
    r"(?<=[.?!])\s+|(?=\[pause\])|(?<=\[pause\])\s*",
    flags=re.IGNORECASE,
)


# Incremental boundary: sentence-ending punctuation (optionally followed by a
# closing quote/bracket) THEN whitespace — the trailing whitespace is what tells
# us the sentence is actually finished, so "3.5" or "Dr." mid-token don't flush
# early. A ``[pause]`` tag is a hard boundary on its own.
_STREAM_BOUNDARY_RE = re.compile(
    r"[.?!]+[\"')\]]?(?=\s)|\[pause\]",
    flags=re.IGNORECASE,
)


async def stream_sentences(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """Reassemble a stream of text deltas into complete sentences.

    Buffers incoming token deltas and emits each sentence the moment its closing
    boundary arrives, so TTS can start on sentence one while the model is still
    generating sentence two.  Any trailing text with no terminal punctuation is
    flushed when the source stream ends.

    Args:
        chunks: Async iterator of text fragments (e.g. from ``stream_general``).

    Yields:
        Complete, non-empty sentence strings in order.
    """
    buffer = ""
    async for delta in chunks:
        if not delta:
            continue
        buffer += delta
        while True:
            match = _STREAM_BOUNDARY_RE.search(buffer)
            if not match:
                break
            end = match.end()
            sentence = buffer[:end].strip()
            buffer = buffer[end:].lstrip()
            if sentence:
                yield sentence
    tail = buffer.strip()
    if tail:
        yield tail


def split_sentences(text: str) -> list[str]:
    """Split *text* into a list of synthesisable sentence fragments.

    Preserves audio tags within their sentence.  Empty strings are filtered out.

    Args:
        text: Full reply text (may include ``[warmly]``, ``[pause]``, etc.).

    Returns:
        Non-empty sentence fragments in order.
    """
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


async def stream_tts_chunks(
    text: str,
    *,
    provider: str = "kokoro",
    api_key: str = "",
    voice_id: str = "",
    voice: str = "af_heart",
) -> AsyncIterator[bytes]:
    """Yield audio chunks sentence by sentence.

    Each sentence is synthesised independently so the first chunk is available
    as soon as the first sentence is ready, rather than waiting for the full
    reply.

    Args:
        text:     Complete reply text.
        provider: TTS provider (``"kokoro"`` or ``"elevenlabs"``).
        api_key:  ElevenLabs key (ignored for Kokoro).
        voice_id: ElevenLabs voice ID (ignored for Kokoro).
        voice:    Kokoro voice code.

    Yields:
        Raw audio bytes for each non-empty sentence chunk.
    """
    sentences = split_sentences(text)
    if not sentences:
        return

    for sentence in sentences:
        try:
            chunk = await synthesize(
                sentence,
                provider=provider,
                api_key=api_key,
                voice_id=voice_id,
                voice=voice,
            )
            if chunk:
                logger.debug("tts_chunk_ready", sentence_len=len(sentence), chunk_bytes=len(chunk))
                yield chunk
        except Exception as exc:  # noqa: BLE001
            logger.warning("tts_chunk_error", sentence=sentence[:50], error=str(exc))
