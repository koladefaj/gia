"""Streaming voice API — ``WS /voice/stream``.

A thin WebSocket proxy between the browser and the configured streaming ASR
provider (Deepgram Flux or OpenAI Realtime). The browser streams raw mono PCM16
frames up; the backend forwards them to the provider and streams normalised
transcript frames back down:

  ``{"type": "ready"}``                 — provider socket is open, start talking
  ``{"type": "partial", "text": ...}``  — interim transcript (revisable)
  ``{"type": "eager",   "text": ...}``  — Flux EagerEndOfTurn (Phase 2 early-intent)
  ``{"type": "resumed"}``               — Flux TurnResumed (cancel speculative work)
  ``{"type": "final",   "text": ...}``  — committed transcript for this turn
  ``{"type": "error",   "error": ...}`` — provider unavailable; client falls back

Keeping the provider key server-side (and the wire format identical across
providers) means the frontend never learns which ASR is behind the switch — the
same contract as ``stt_provider`` everywhere else.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from backend.app.config import Settings
from backend.app.dependencies import get_settings
from backend.app.observability.logging import get_logger
from backend.app.providers.stt_stream import (
    get_streaming_transcriber,
    streaming_provider,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])


def _frame(ev) -> dict:  # ev: TranscriptEvent
    """Map a provider ``TranscriptEvent`` to a browser frame."""
    if ev.resumed:
        return {"type": "resumed"}
    if ev.eager:
        return {"type": "eager", "text": ev.text}
    if ev.is_final:
        return {"type": "final", "text": ev.text}
    return {"type": "partial", "text": ev.text}


@router.websocket("/stream")
async def stream(
    ws: WebSocket,
    cfg: Annotated[Settings, Depends(get_settings)],
    provider: str | None = None,
    language: str = "en",
) -> None:
    """Proxy a live transcription session for one turn.

    Query params:
        provider: Optional override of ``stt_provider`` (``deepgram`` /
                  ``openai_stream``); defaults to the configured value.
        language: BCP-47 hint forwarded to the provider.

    The socket carries binary audio up and JSON frames down. It closes when the
    client sends ``{"type": "stop"}``, disconnects, or the provider stream ends.
    """
    await ws.accept()

    transcriber = get_streaming_transcriber(cfg, provider=provider, language=language)
    if transcriber is None:
        # Misconfigured / batch provider: tell the client to use the one-shot
        # /voice/transcribe path instead of leaving the socket hanging.
        name = streaming_provider(cfg, provider)
        await ws.send_json({"type": "error", "error": f"streaming STT unavailable for provider '{name}'"})
        await ws.close()
        return

    async def pump_up() -> bool:
        """Browser audio frames → provider.

        Returns ``True`` on a clean ``stop`` (the provider should be allowed to
        flush its final transcript), ``False`` on client disconnect (tear down
        immediately — nothing more is coming).
        """
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return False
            data = msg.get("bytes")
            if data:
                await transcriber.send_audio(data)
                continue
            text = msg.get("text")
            if text and '"stop"' in text:
                await transcriber.finish()
                return True

    async def pump_down() -> None:
        """Provider transcript events → browser, until the provider closes."""
        async for ev in transcriber.events():
            if ws.application_state != WebSocketState.CONNECTED:
                break
            await ws.send_json(_frame(ev))

    try:
        async with transcriber:
            await ws.send_json({"type": "ready"})
            down = asyncio.create_task(pump_down())
            clean_stop = await pump_up()
            if clean_stop:
                # finish() was sent; let the provider flush remaining finals and
                # close the stream (which ends pump_down). Bounded so a wedged
                # provider can't hang the socket.
                try:
                    await asyncio.wait_for(down, timeout=5.0)
                except TimeoutError:
                    down.cancel()
            else:
                down.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await down
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("voice_stream_error", error=str(exc))
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_json({"type": "error", "error": str(exc)})
    finally:
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.close()
