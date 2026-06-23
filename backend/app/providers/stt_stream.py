"""Streaming speech-to-text — provider-agnostic, real-time transcription.

Unlike the batch path in ``stt.py`` (record → upload → transcribe, serial and
*before* ``/chat`` starts), this module keeps a live socket to a streaming ASR
provider and emits **interim** and **final** transcripts as the user speaks. The
serial transcription wait disappears, and the interim results are what let the
router fire mid-utterance (early-intent, Phase 2).

Two adapters sit behind one switch (``STT_PROVIDER``), mirroring the
``tts_provider`` pattern:

  * ``deepgram``      — Deepgram nova streaming (``wss://api.deepgram.com``)
  * ``openai_stream`` — OpenAI Realtime transcription (``gpt-4o-mini-transcribe``)

Both accept **mono PCM16** audio, so the browser streams one wire format (24 kHz
mono Int16) and the provider choice is invisible to everything upstream. Each
adapter is an async context manager that you feed audio with ``send_audio`` and
drain transcripts from via ``events()``; ``send_audio`` and ``events()`` run
concurrently on the same socket (the ``websockets`` client allows it).
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from urllib.parse import urlencode

import websockets
from websockets.asyncio.client import ClientConnection

from backend.app.config import Settings
from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

# One wire format for every provider: 24 kHz mono PCM16. Deepgram accepts any
# sample rate (we just declare it); OpenAI Realtime *requires* 24 kHz pcm16, so
# 24 kHz is the common denominator the browser resamples to.
SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class TranscriptEvent:
    """One transcript update from the provider.

    The fields are normalised across providers so the endpoint and frontend
    never branch on which ASR is behind the switch.

    Attributes:
        text:         The transcript so far (interim) or the settled text (final).
        is_final:     Provider has committed this segment — it won't be revised.
                      (Deepgram Flux ``EndOfTurn`` / OpenAI ``...completed``.)
        speech_final: Provider detected end-of-utterance (the speaker paused).
                      Drives turn segmentation without a client-side silence timer.
        eager:        Deepgram Flux ``EagerEndOfTurn`` — a medium-confidence turn
                      end. The hook for early-intent (Phase 2): draft the reply on
                      this, commit on the matching ``is_final``. Always ``False``
                      for providers without eager turn detection.
        resumed:      Deepgram Flux ``TurnResumed`` — a prior ``eager`` guess was
                      wrong and the user kept talking; cancel any speculative work.
    """

    text: str
    is_final: bool
    speech_final: bool = False
    eager: bool = False
    resumed: bool = False


@runtime_checkable
class StreamingTranscriber(Protocol):
    """A live transcription session over a provider WebSocket."""

    async def __aenter__(self) -> StreamingTranscriber: ...
    async def __aexit__(self, *exc: object) -> None: ...

    async def send_audio(self, pcm: bytes) -> None:
        """Forward a chunk of raw mono PCM16 audio to the provider."""

    async def finish(self) -> None:
        """Signal end-of-audio so the provider flushes any pending transcript."""

    def events(self) -> AsyncIterator[TranscriptEvent]:
        """Yield transcript updates until the provider closes the stream."""


# ── Deepgram ──────────────────────────────────────────────────────────────────


class DeepgramTranscriber:
    """Deepgram **Flux** streaming adapter (``/v2/listen``, ``flux-general-en``).

    Flux is Deepgram's conversational model: it does end-of-turn detection itself
    and emits ``TurnInfo`` events, so the client needs no silence timer. Raw PCM16
    is streamed in (~80 ms chunks recommended); ``TurnInfo`` events come back with
    an ``event`` field:

      * ``StartOfTurn``    — turn began (guaranteed non-empty transcript; barge-in)
      * ``Update``         — interim transcript (~every 0.25 s)
      * ``EagerEndOfTurn`` — medium-confidence turn end → ``eager`` (Phase 2)
      * ``TurnResumed``    — the eager guess was wrong → ``resumed`` (Phase 2)
      * ``EndOfTurn``      — high-confidence turn end → ``is_final`` / ``speech_final``

    ``eager_eot_threshold`` must be set for the eager/resumed events to fire; we
    set it so Phase 2 can light up without a protocol change.
    """

    _URL = "wss://api.deepgram.com/v2/listen"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "flux-general-en",
        sample_rate: int = SAMPLE_RATE,
        eot_threshold: float = 0.7,
        eager_eot_threshold: float = 0.5,
    ) -> None:
        self._api_key = api_key
        # Flux takes a smaller param set than nova: no smart_format/punctuate/
        # language for raw audio — just model, encoding, sample_rate, and the EOT
        # knobs. encoding+sample_rate are required for non-containerized PCM.
        self._params = {
            "model": model,
            "encoding": "linear16",
            "sample_rate": str(sample_rate),
            "eot_threshold": str(eot_threshold),
            "eager_eot_threshold": str(eager_eot_threshold),
        }
        self._ws: ClientConnection | None = None

    async def __aenter__(self) -> DeepgramTranscriber:
        url = f"{self._URL}?{urlencode(self._params)}"
        self._ws = await websockets.connect(
            url,
            additional_headers={"Authorization": f"Token {self._api_key}"},
            max_size=None,
        )
        logger.info("deepgram_flux_open", model=self._params["model"])
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

    async def send_audio(self, pcm: bytes) -> None:
        if self._ws is not None and pcm:
            await self._ws.send(pcm)

    async def finish(self) -> None:
        # Tell Flux no more audio is coming so it flushes the final turn rather
        # than waiting on its timeout.
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.send(json.dumps({"type": "CloseStream"}))

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        if self._ws is None:
            return
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            # Flux sends TurnInfo (carrying `event`) plus Connected/Metadata
            # frames we ignore. Some frames flatten `event` to the top level.
            event = msg.get("event") or msg.get("type")
            if event not in (
                "StartOfTurn", "Update", "EagerEndOfTurn", "TurnResumed", "EndOfTurn",
            ):
                continue
            text = (msg.get("transcript") or "").strip()
            # TurnResumed carries no new words — it's a control signal to cancel
            # speculative work, so forward it even with an empty transcript.
            if not text and event != "TurnResumed":
                continue
            yield TranscriptEvent(
                text=text,
                is_final=event == "EndOfTurn",
                speech_final=event == "EndOfTurn",
                eager=event == "EagerEndOfTurn",
                resumed=event == "TurnResumed",
            )


# ── OpenAI Realtime ───────────────────────────────────────────────────────────


class OpenAIRealtimeTranscriber:
    """OpenAI Realtime transcription adapter (``gpt-4o-mini-transcribe``).

    Opens a transcription-intent Realtime session, streams base64 PCM16 via
    ``input_audio_buffer.append``, and reads ``...input_audio_transcription``
    delta/completed events. Server VAD handles turn detection. Deltas are
    accumulated into the running interim text; ``completed`` settles the segment.

    Best-effort: shipped behind the same switch as Deepgram but exercised less,
    since Deepgram is the primary tested path.
    """

    _URL = "wss://api.openai.com/v1/realtime?intent=transcription"

    def __init__(self, *, api_key: str, language: str = "en", model: str = "gpt-4o-mini-transcribe") -> None:
        self._api_key = api_key
        self._language = language
        self._model = model
        self._ws: ClientConnection | None = None
        self._buffer = ""  # running interim text for the in-flight segment

    async def __aenter__(self) -> OpenAIRealtimeTranscriber:
        self._ws = await websockets.connect(
            self._URL,
            additional_headers={
                "Authorization": f"Bearer {self._api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
            max_size=None,
        )
        await self._ws.send(json.dumps({
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {"model": self._model, "language": self._language},
                "turn_detection": {"type": "server_vad", "silence_duration_ms": 500},
            },
        }))
        logger.info("openai_realtime_stream_open", model=self._model)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

    async def send_audio(self, pcm: bytes) -> None:
        if self._ws is not None and pcm:
            await self._ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode(),
            }))

    async def finish(self) -> None:
        # Commit whatever audio is buffered; server VAD usually does this on the
        # pause, but an explicit commit flushes a trailing tail.
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        if self._ws is None:
            return
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            etype = msg.get("type", "")
            if etype.endswith("input_audio_transcription.delta"):
                delta = msg.get("delta", "")
                if delta:
                    self._buffer += delta
                    yield TranscriptEvent(text=self._buffer.strip(), is_final=False)
            elif etype.endswith("input_audio_transcription.completed"):
                text = (msg.get("transcript", "") or self._buffer).strip()
                self._buffer = ""
                if text:
                    yield TranscriptEvent(text=text, is_final=True, speech_final=True)
            elif etype == "error":
                logger.warning("openai_realtime_error", error=msg.get("error"))


# ── Factory ───────────────────────────────────────────────────────────────────


def streaming_provider(cfg: Settings, override: str | None = None) -> str:
    """Resolve the effective STT provider (query override beats config)."""
    return (override or getattr(cfg, "stt_provider", "") or "local").lower()


def is_streaming(provider: str) -> bool:
    """Whether *provider* names a streaming (not batch) STT backend."""
    return provider in ("deepgram", "openai_stream")


def get_streaming_transcriber(
    cfg: Settings,
    *,
    provider: str | None = None,
    language: str = "en",
) -> StreamingTranscriber | None:
    """Build a streaming transcriber for the configured provider.

    Returns ``None`` when the resolved provider is a batch backend (``local`` /
    ``openai``) or when its credentials are missing — the caller then falls back
    to the one-shot ``/voice/transcribe`` path.

    Args:
        cfg:      Application settings (provider + credentials).
        provider: Optional explicit provider, overriding ``cfg.stt_provider``.
        language: BCP-47 language hint passed to the provider.
    """
    name = streaming_provider(cfg, provider)

    if name == "deepgram":
        key = getattr(cfg, "deepgram_api_key", "") or os.getenv("DEEPGRAM_API_KEY", "")
        if not key:
            logger.warning("stt_stream_no_key", provider="deepgram")
            return None
        return DeepgramTranscriber(
            api_key=key,
            model=getattr(cfg, "deepgram_model", "flux-general-en") or "flux-general-en",
            eot_threshold=getattr(cfg, "deepgram_eot_threshold", 0.7),
            eager_eot_threshold=getattr(cfg, "deepgram_eager_eot_threshold", 0.5),
        )

    if name == "openai_stream":
        key = getattr(cfg, "openai_api_key", "") or os.getenv("OPENAI_API_KEY", "")
        if not key:
            logger.warning("stt_stream_no_key", provider="openai_stream")
            return None
        return OpenAIRealtimeTranscriber(api_key=key, language=language)

    return None  # batch provider — no streaming session
