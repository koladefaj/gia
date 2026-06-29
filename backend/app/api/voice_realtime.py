"""Speech-to-speech voice API — ``WS /voice/realtime``.

The realtime counterpart to ``voice_stream.py``, and a **hybrid**: ``gpt-realtime``
is the ears + brain (audio in, reasoning, tool calls, reply as **text**), and
**ElevenLabs v3 is the voice**. This endpoint owns that second half — it takes the
model's text and streams it through the existing ElevenLabs path, so the warm,
audio-tagged brand voice is preserved (the voice is the product).

The model reaches the same memory / Spotify / Brave / weather code the pipeline
uses, via :class:`~backend.app.providers.realtime.RealtimeTools` — built here
from the request-scoped clients so tool execution keeps every dependency intact.

Wire format (browser ⇄ backend) — audio frames match ``/chat`` so the frontend
reuses the same streaming player:

  ``{"type": "ready"}``                       — session armed, start talking
  ``{"type": "user_transcript", "text": ...}``      — finalised user words
  ``{"type": "reply_chunk", "text": ...}``    — a chunk of the reply (captions)
  ``{"type": "audio_start"}``                 — ElevenLabs synthesis beginning
  ``{"type": "audio_chunk", "data": <b64 mp3>, "streaming": true, "seq": n}``
  ``{"type": "audio_end", "chunks": n}``      — reply audio complete
  ``{"type": "flush"}``                       — barge-in: drop buffered playback
  ``{"type": "speech_started"}``              — user is talking
  ``{"type": "tool", "name": ...}``           — a tool is running (status)
  ``{"type": "response_done"}``               — turn complete
  ``{"type": "error", "error": ...}``         — session unavailable / failed

Finalised user and assistant text are appended to the session history and the
durable-memory extractor is enqueued on the same throttle the pipeline uses.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import time
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from backend.app.config import Settings
from backend.app.dependencies import (
    get_brave_client,
    get_db,
    get_settings,
    get_weather_client,
)
from backend.app.memory.retrieval import build_user_context
from backend.app.memory.session_history import append_turn
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.observability.langfuse import crew_trace
from backend.app.observability.logging import get_logger
from backend.app.prompts import get_registry
from backend.app.providers.realtime import (
    RealtimeTools,
    build_session,
    realtime_enabled,
)
from backend.app.providers.tts import should_use_v3, synthesize_stream
from backend.app.tools.spotify_web import SpotifyWebClient
from backend.app.voice.streaming import stream_sentences, synthesize_sentence_stream

logger = get_logger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])

# Acknowledge-then-search: a music or web lookup takes ~1–2s, and the model
# otherwise emits the tool call silently and only speaks once the result is back —
# ~3s of dead air on a "play X" turn. The Realtime API lets one response carry
# BOTH spoken output and a function call, so we tell the model to speak a short
# ack first and call the tool in the same turn; the user hears Gia at ~first-token
# latency while the search runs behind her voice. (The decomposed pipeline keeps
# the same scoped filler for music commands.)
_REALTIME_GUIDANCE = (
    "## Staying responsive (important)\n"
    "Some tools take a moment — a Spotify search or a web lookup can take a "
    "second or two. NEVER go silent while one runs. The instant you decide to "
    "call such a tool, FIRST say a short, warm acknowledgement out loud — e.g. "
    "'[warm] Say less — one sec…' / 'On it.' / 'Let me find that.' — and THEN "
    "make the tool call, in the SAME turn, so the user hears you immediately and "
    "the lookup happens behind your voice. When the result comes back, continue "
    "naturally (e.g. '…here's <track> by <artist>'). Don't acknowledge for "
    "instant, no-tool replies — only when a tool will actually run."
)


async def _build_instructions(
    user_id: str | None,
    *,
    store: WeaviateMemoryStore | None,
    db,
    redis,
    spotify,
    cfg: Settings,
) -> str:
    """Assemble the session prompt: the Gia persona plus the user's memory context.

    The realtime model has no per-turn context-assembly step (it owns the turn),
    so the persona and what we already know about the listener are injected once,
    up front, as the session instructions — alongside the ``recall_memory`` tool
    for deeper, topic-specific lookups mid-conversation. The persona already
    prescribes ``[warm]`` / ``[laughs]`` audio tags, which the model writes inline
    and ElevenLabs v3 renders. A flaky memory backend degrades to the bare persona.
    """
    persona = get_registry().get("persona.gia").render()
    base = f"{persona}\n\n{_REALTIME_GUIDANCE}"
    context_text = ""
    if user_id and store:
        try:
            ctx = await build_user_context(
                user_id, "hello",
                db=db, store=store, redis=redis, spotify=spotify, cfg=cfg,
            )
            context_text = ctx.to_prompt_text()
        except Exception as exc:  # noqa: BLE001
            logger.warning("realtime_context_error", error=str(exc))
    if not context_text:
        return base
    return (
        f"{base}\n\n"
        "## What you know about this listener\n"
        f"{context_text}\n\n"
        "Use this naturally — don't recite it. Call recall_memory for anything "
        "more specific you need mid-conversation."
    )


async def _persist_and_extract(
    user_id: str | None,
    session_id: str,
    user_text: str,
    gia_text: str,
    redis,
) -> None:
    """Append the completed exchange to history and enqueue memory extraction.

    Keeps realtime turns flowing into the same continuity + durable-memory
    pipeline as ``chat.py``: history for "play it now"/"that one" reference
    resolution next turn, and the throttled Celery extractor for preferences and
    life facts. Best-effort — a persistence hiccup never interrupts the call.
    """
    try:
        if user_text:
            await append_turn(redis, session_id, "user", user_text)
        if gia_text:
            await append_turn(redis, session_id, "gia", gia_text)
        if not user_id:
            return
        await redis.zadd("gia:pending_flush", {f"{user_id}:{session_id}": time.time()})
        if await redis.set(f"extract_throttle:{session_id}", "1", ex=2700, nx=True):
            from backend.worker.celery_app import celery_app  # noqa: PLC0415

            celery_app.send_task(
                "backend.worker.tasks.memory_extraction.extract_session_memories",
                args=[user_id, session_id],
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("realtime_persist_error", error=str(exc))


@router.websocket("/realtime")
async def realtime(
    ws: WebSocket,
    brave: Annotated[object, Depends(get_brave_client)],
    weather: Annotated[object, Depends(get_weather_client)],
    db: Annotated[object, Depends(get_db)],
    cfg: Annotated[Settings, Depends(get_settings)],
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Proxy a speech-to-speech session: gpt-realtime understands, ElevenLabs speaks.

    Query params:
        user_id:    Optional user UUID — enables memory injection + the
                    ``recall_memory`` tool and persists the conversation.
        session_id: Conversation id for history continuity (defaults to a
                    transient one when omitted).

    The socket carries binary audio up and JSON frames down; it closes on a
    client ``{"type": "stop"}``, a disconnect, or when the model ends the stream.
    """
    await ws.accept()

    if not realtime_enabled(cfg):
        # Wrong mode or no OpenAI key: tell the client to use the decomposed
        # /voice/stream + /chat path instead of hanging the socket.
        await ws.send_json({
            "type": "error",
            "error": "realtime voice mode unavailable (set VOICE_MODE=realtime and OPENAI_API_KEY)",
        })
        await ws.close()
        return

    # App-level clients live on app.state (set in lifespan). The HTTP
    # get_*_client deps require a Request, which a WebSocket route can't provide,
    # so read these three directly off app.state instead of via Depends.
    spotify = ws.app.state.spotify
    redis = ws.app.state.redis
    weaviate = ws.app.state.weaviate

    sid = session_id or f"realtime-{int(time.time())}"
    store = WeaviateMemoryStore(client=weaviate) if user_id else None
    instructions = await _build_instructions(
        user_id, store=store, db=db, redis=redis, spotify=spotify, cfg=cfg
    )
    tools = RealtimeTools(
        cfg=cfg, spotify=spotify, brave=brave, store=store,
        db=db, redis=redis, weather=weather, user_id=user_id,
        spotify_web=SpotifyWebClient(cfg),  # fast direct search path
    )
    session = build_session(cfg, instructions=instructions, tools=tools)

    # ElevenLabs is the voice (resolved once, like chat.py).
    tts_provider = cfg.tts_provider
    tts_api_key = cfg.elevenlabs_api_key if tts_provider == "elevenlabs" else ""
    tts_voice_id = cfg.elevenlabs_voice_id if tts_provider == "elevenlabs" else ""

    # Serialise sends — the synthesis task and the event loop both write the
    # socket, and Starlette doesn't guarantee concurrent send safety.
    send_lock = asyncio.Lock()
    # Latest finalised user turn, paired with the next assistant text so the
    # exchange lands in history + the extractor together.
    last_user_text = ""
    # Per-turn Langfuse accumulators: start time (set on the user transcript) and
    # the tools that fired this turn, emitted as one trace when the reply lands.
    turn_t0: float | None = None
    turn_tools: list[tuple[str, float]] = []

    async def emit_turn_trace(user_text: str, assistant_text: str) -> None:
        """Emit one Langfuse trace for a completed realtime turn (no-op if off).

        Mirrors the pipeline's per-turn trace: input = user words, output = reply,
        a span per tool that fired, and self-eval scores (tools used, end-to-end
        latency). Built one-shot at turn end so it never has to stay "current"
        across the event loop's awaits (which would break OTel nesting).
        """
        if not (user_text or assistant_text):
            return
        try:
            async with crew_trace(
                sid, user_id, user_input=user_text,
                trace_name="gia-realtime-turn", tags=["realtime", "voice"],
            ) as trace:
                for name, at_ms in turn_tools:
                    with trace.span(name, f"+{round(at_ms)}ms"):
                        pass
                trace.set_output(assistant_text)
                trace.score("tools_used", len(turn_tools), data_type="NUMERIC")
                if turn_t0 is not None:
                    trace.score(
                        "turn_latency_ms",
                        round((time.monotonic() - turn_t0) * 1000, 1),
                        data_type="NUMERIC",
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("realtime_trace_error", error=str(exc))
    # Voice source: "elevenlabs" → model emits text, we synthesise it through
    # ElevenLabs; "model" → the model speaks directly and we just relay its audio.
    voice_source = cfg.realtime_voice_source
    stream_mode = cfg.tts_stream_sentences
    # Sentence-streaming: a per-turn queue of text deltas drives a consumer that
    # synthesises sentence by sentence. ``synth_tasks`` tracks every in-flight
    # synthesis (sentence consumers + whole-text fallback) so a barge-in cancels
    # all of them.
    delta_q: asyncio.Queue[str | None] | None = None
    synth_tasks: set[asyncio.Task] = set()

    async def send(frame: dict) -> None:
        async with send_lock:
            if ws.application_state == WebSocketState.CONNECTED:
                await ws.send_json(frame)

    def _track(task: asyncio.Task) -> None:
        synth_tasks.add(task)
        task.add_done_callback(synth_tasks.discard)

    async def run_sentence_synth(q: asyncio.Queue[str | None]) -> None:
        """Consume one turn's text deltas and synthesise sentence by sentence.

        ``stream_sentences`` reassembles the deltas into complete sentences (a
        ``None`` on the queue marks end-of-turn); ``synthesize_sentence_stream``
        synthesises each as it lands, so the first sentence's audio plays while
        gpt-realtime is still generating the next — the latency mask. Frames carry
        a monotonic ``seq`` and one ``audio_end`` per turn, so the frontend's
        streaming player is unchanged.
        """
        async def deltas() -> AsyncIterator[str]:
            while True:
                d = await q.get()
                if d is None:
                    return
                yield d

        async for kind, payload in synthesize_sentence_stream(
            stream_sentences(deltas()),
            provider=tts_provider, api_key=tts_api_key, voice_id=tts_voice_id,
        ):
            await send({"type": kind, **payload})

    async def synthesize_whole(text: str) -> None:
        """Whole-reply fallback (``tts_stream_sentences=False``): one v3 pass."""
        emotional = should_use_v3(text)
        started = False
        seq = 0
        try:
            async for chunk in synthesize_stream(
                text, provider=tts_provider, api_key=tts_api_key, voice_id=tts_voice_id
            ):
                if not chunk:
                    continue
                if not started:
                    started = True
                    await send({"type": "audio_start", "model": tts_provider, "emotional": emotional})
                await send({
                    "type": "audio_chunk",
                    "data": base64.b64encode(chunk).decode(),
                    "model": tts_provider,
                    "emotional": emotional,
                    "seq": seq,
                    "streaming": True,
                })
                seq += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("realtime_tts_error", error=str(exc))
        if started:
            await send({"type": "audio_end", "chunks": seq})

    async def pump_up() -> bool:
        """Browser audio frames → model. ``True`` on a clean stop, ``False`` on disconnect."""
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return False
            data = msg.get("bytes")
            if data:
                await session.send_audio(data)
                continue
            text = msg.get("text")
            if text and '"stop"' in text:
                await session.finish()
                return True

    async def cancel_synth() -> None:
        """Cancel all in-flight synthesis (barge-in / teardown) and end the turn queue."""
        nonlocal delta_q
        delta_q = None
        for task in list(synth_tasks):
            if not task.done():
                task.cancel()
        for task in list(synth_tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def pump_down() -> None:
        """Model events → browser: captions, ElevenLabs audio, barge-in, persistence."""
        nonlocal last_user_text, delta_q, turn_t0, turn_tools
        async for ev in session.events():
            if ws.application_state != WebSocketState.CONNECTED:
                break
            if ev.kind == "user_transcript":
                last_user_text = ev.text
                turn_t0 = time.monotonic()  # turn clock starts at the user's words
                turn_tools = []
                await send({"type": "user_transcript", "text": ev.text})
            elif ev.kind == "audio":
                # Model-voice mode: relay the model's own audio (base64 PCM16).
                await send({"type": "audio", "data": ev.audio_b64})
            elif ev.kind == "assistant_delta":
                await send({"type": "reply_chunk", "text": ev.text})
                if voice_source == "elevenlabs" and stream_mode:
                    # Feed the delta to the current turn's sentence synthesiser,
                    # starting one on the first delta of the turn.
                    if delta_q is None:
                        delta_q = asyncio.Queue()
                        _track(asyncio.create_task(run_sentence_synth(delta_q)))
                    delta_q.put_nowait(ev.text)
            elif ev.kind == "assistant_text":
                # Reply text finalised. ElevenLabs mode: close the sentence queue
                # (consumer flushes tail + audio_end) or synthesise the whole text.
                # Model mode: audio was already relayed, nothing to synthesise.
                if voice_source == "elevenlabs":
                    if stream_mode:
                        if delta_q is not None:
                            delta_q.put_nowait(None)
                            delta_q = None
                    else:
                        _track(asyncio.create_task(synthesize_whole(ev.text)))
                await _persist_and_extract(user_id, sid, last_user_text, ev.text, redis)
                await emit_turn_trace(last_user_text, ev.text)
                last_user_text = ""
            elif ev.kind == "speech_started":
                # Barge-in: stop synthesising and tell the client to drop playback.
                await cancel_synth()
                await send({"type": "flush"})
                await send({"type": "speech_started"})
            elif ev.kind == "tool":
                at = (time.monotonic() - turn_t0) * 1000 if turn_t0 else 0.0
                turn_tools.append((ev.tool_name, at))
                await send({"type": "tool", "name": ev.tool_name})
            elif ev.kind == "response_done":
                await send({"type": "response_done"})
            elif ev.kind == "error":
                await send({"type": "error", "error": ev.text})

    try:
        async with session:
            await ws.send_json({"type": "ready"})
            down = asyncio.create_task(pump_down())
            clean_stop = await pump_up()
            if clean_stop:
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
        logger.warning("voice_realtime_error", error=str(exc))
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_json({"type": "error", "error": str(exc)})
    finally:
        await cancel_synth()
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.close()
