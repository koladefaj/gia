"""Chat API — the primary conversation endpoint.

``POST /chat`` accepts a user message and streams back Server-Sent Events
(SSE) that describe crew execution in real-time:

  ``agent_start``   — an agent has begun processing
  ``tool_call``     — an agent invoked a tool (Spotify, Brave, Weaviate, etc.)
  ``agent_done``    — an agent finished; includes latency_ms
  ``reply_chunk``   — a sentence of Gia's reply
  ``audio_chunk``   — a base64-encoded TTS audio chunk for that sentence
  ``done``          — stream end (intent, proactive, session summary)
  ``error``         — something went wrong (non-fatal; stream continues if possible)

The event shape is designed for the Day 10 frontend.  Keep it stable.

Crew execution order:
  1. Build user context (Memory agent)
  2. Pop pending proactive draft (if any)
  3. Router classifies intent
  4. Execute relevant agents (DJ / Artist / Mood) in parallel where possible
  5. Stream reply sentences + TTS chunks
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from weaviate import WeaviateClient

from backend.app.agents.artist import ArtistService
from backend.app.agents.dj import DJService
from backend.app.agents.mood import MoodService
from backend.app.agents.planner import build_plan
from backend.app.agents.synthesis import synthesize_reply
from backend.app.config import Settings
from backend.app.dependencies import (
    get_brave_client,
    get_db,
    get_redis,
    get_settings,
    get_spotify_client,
    get_weather_client,
    get_weaviate_client,
)
from backend.app.interfaces import SpotifyClientProtocol, WeatherClientProtocol
from backend.app.memory.retrieval import build_user_context
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.mood.proactive import pop_proactive_draft
from backend.app.observability.langfuse import crew_trace
from backend.app.observability.logging import get_logger
from backend.app.providers.tts import synthesize, is_emotional
from backend.app.schemas.chat import ChatRequest, IntentType
from backend.app.tools.brave import BraveSearchClient
from backend.app.voice.streaming import split_sentences, stream_tts_chunks

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def _sse(event: str, data: dict) -> str:
    """Format a single SSE message frame.

    Args:
        event: Event type string.
        data:  Payload dict (must be JSON-serialisable).

    Returns:
        SSE-formatted string including trailing double-newline.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _weather_note(weather: WeatherClientProtocol, cfg: Settings) -> str | None:
    """Fetch current weather and render a one-line context note for the LLM.

    Uses the configured default coordinates (the demo user's city).  Returns
    ``None`` when the lookup fails so the turn proceeds without weather rather
    than blocking on a degraded service.

    Args:
        weather: Weather client.
        cfg:     Settings (default location).

    Returns:
        A note like ``**Weather:** It's 31°C and clear in Lagos right now.`` or
        ``None``.
    """
    try:
        current = await weather.get_current(
            cfg.weather_default_lat, cfg.weather_default_lon
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("chat_weather_error", error=str(exc))
        return None
    if not current:
        return None
    return (
        f"**Weather:** It's {current['temperature_c']:.0f}°C and "
        f"{current['condition']} in {cfg.weather_default_label} right now."
    )


async def _run_crew(
    request: ChatRequest,
    spotify: SpotifyClientProtocol,
    brave: BraveSearchClient,
    weather: WeatherClientProtocol,
    weaviate: WeaviateClient,
    db: AsyncSession,
    redis,
    cfg: Settings,
) -> AsyncIterator[str]:
    """Core crew execution generator — yields SSE frames.

    Args:
        request: Parsed ``ChatRequest``.
        spotify: Spotify client.
        brave:   Brave Search client.
        weather: Weather client (for context-aware recommendations).
        weaviate: Weaviate client.
        db:      SQLAlchemy async session.
        redis:   Async Redis client.
        cfg:     Application settings.

    Yields:
        SSE-formatted strings.
    """
    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_id
    store = WeaviateMemoryStore(client=weaviate) if user_id else None

    async with crew_trace(session_id, user_id) as trace:

        # ── 1. Build user context ─────────────────────────────────────────────
        user_context_text = ""
        user_context_used = False
        if user_id and store:
            yield _sse("agent_start", {"agent": "memory", "input": request.message})
            with trace.span("memory", request.message) as span:
                try:
                    t0 = time.monotonic()
                    ctx = await build_user_context(
                        user_id, request.message,
                        db=db, store=store, redis=redis, spotify=spotify, cfg=cfg,
                    )
                    user_context_text = ctx.to_prompt_text()
                    user_context_used = bool(user_context_text)
                    span.set_output(f"context_length={len(user_context_text)}")
                    yield _sse("agent_done", {
                        "agent": "memory",
                        "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                        "context_chars": len(user_context_text),
                    })
                except Exception as exc:  # noqa: BLE001
                    logger.warning("chat_context_error", error=str(exc))
                    yield _sse("error", {"agent": "memory", "error": str(exc)})

        # ── 2. Pop pending proactive draft ────────────────────────────────────
        proactive: str | None = None
        if user_id:
            proactive = await pop_proactive_draft(user_id, redis)

        # ── 3. Plan the turn ──────────────────────────────────────────────────
        yield _sse("agent_start", {"agent": "router", "input": request.message})
        with trace.span("router", request.message) as span:
            t0 = time.monotonic()
            plan = await build_plan(request.message, cfg)
            intent, confidence = plan.intent, plan.confidence
            span.set_output(intent.value)
            yield _sse("agent_done", {
                "agent": "router",
                "intent": intent.value,
                "confidence": round(confidence, 2),
                "latency_ms": round((time.monotonic() - t0) * 1000, 1),
            })
        # Richer, additive plan event (steps + signals) for multi-tool turns.
        yield _sse("plan", {
            "intent": intent.value,
            "steps": plan.steps,
            "signals": plan.signals,
        })

        # ── 3b. Gather requested real-world signals ───────────────────────────
        if "weather" in plan.signals:
            yield _sse("tool_call", {"agent": "planner", "tool": "weather", "input": request.message})
            weather_note = await _weather_note(weather, cfg)
            if weather_note:
                user_context_text = f"{user_context_text}\n{weather_note}".strip()
                yield _sse("signal", {"name": "weather", "value": weather_note})

        # ── 4. Execute the planned agents ─────────────────────────────────────
        reply_parts: list[str] = []

        # Proactive observation surfaced first
        if proactive:
            reply_parts.append(proactive)

        if "dj" in plan.steps:
            yield _sse("agent_start", {"agent": "dj", "input": request.message})
            with trace.span("dj", request.message) as span:
                t0 = time.monotonic()
                try:
                    yield _sse("tool_call", {"agent": "dj", "tool": "search_tracks", "input": request.message})
                    dj_svc = DJService(spotify=spotify, cfg=cfg)
                    dj_result = await dj_svc.recommend(
                        query=request.message,
                        user_context_text=user_context_text,
                        start_playback=(intent == IntentType.MUSIC_QUEUE),
                        n=4,
                    )
                    span.set_output(dj_result.recommendation)
                    reply_parts.append(dj_result.recommendation)
                    yield _sse("agent_done", {
                        "agent": "dj",
                        "track": dj_result.primary_track.name,
                        "queue_depth": len(dj_result.queue.tracks),
                        "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                    })
                except Exception as exc:  # noqa: BLE001
                    logger.warning("chat_dj_error", error=str(exc))
                    yield _sse("error", {"agent": "dj", "error": str(exc)})
                    reply_parts.append("I had trouble finding tracks right now — try again in a moment.")

        if "artist" in plan.steps:
            yield _sse("agent_start", {"agent": "artist", "input": request.message})
            with trace.span("artist", request.message) as span:
                t0 = time.monotonic()
                try:
                    yield _sse("tool_call", {"agent": "artist", "tool": "brave_search", "input": request.message})
                    artist_svc = ArtistService(
                        spotify=spotify, brave=brave, cfg=cfg, store=store
                    )
                    artist_result = await artist_svc.get_info(
                        artist_name=request.message,
                        user_id=user_id,
                    )
                    span.set_output(artist_result.response)
                    reply_parts.append(artist_result.response)
                    yield _sse("agent_done", {
                        "agent": "artist",
                        "artist": artist_result.artist_name,
                        "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                    })
                except Exception as exc:  # noqa: BLE001
                    logger.warning("chat_artist_error", error=str(exc))
                    yield _sse("error", {"agent": "artist", "error": str(exc)})

        if "mood" in plan.steps and user_id and store:
            yield _sse("agent_start", {"agent": "mood", "input": request.message})
            with trace.span("mood", request.message) as span:
                t0 = time.monotonic()
                try:
                    mood_svc = MoodService(spotify=spotify, store=store, cfg=cfg)
                    mood_result = await mood_svc.analyze(user_id)
                    span.set_output(mood_result.current_label)
                    if mood_result.proactive_draft:
                        reply_parts.append(mood_result.proactive_draft)
                    else:
                        reply_parts.append(
                            f"[thoughtful] You're currently in a {mood_result.current_label} zone. "
                            + (
                                f"That's pretty typical for {mood_result.bucket.replace('_', ' ')} for you."
                                if mood_result.pattern_label else
                                "I'm still building up your pattern data — ask me again after a few more sessions."
                            )
                        )
                    yield _sse("agent_done", {
                        "agent": "mood",
                        "label": mood_result.current_label,
                        "deviation": mood_result.deviation,
                        "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                    })
                except Exception as exc:  # noqa: BLE001
                    logger.warning("chat_mood_error", error=str(exc))
                    yield _sse("error", {"agent": "mood", "error": str(exc)})

        # Fallback reply when no agent produced output
        if not reply_parts:
            reply_parts.append(
                "I'm not sure what you're after — could you tell me more? "
                "I can find music, talk about an artist, or check your mood patterns."
            )

        # ── 5. Synthesise + stream reply + TTS ────────────────────────────────
        # When several agents contributed (and synthesis is enabled), merge
        # them into one coherent reply; otherwise join. A pending proactive note
        # is kept verbatim at the front so its phrasing is never reworded.
        if cfg.synthesis_enabled and len([p for p in reply_parts if p.strip()]) > 1:
            with trace.span("synthesis", request.message) as span:
                synth_input = reply_parts[1:] if proactive else reply_parts
                merged = await synthesize_reply(synth_input, request.message, cfg)
                full_reply = f"{proactive} {merged}".strip() if proactive else merged
                span.set_output(full_reply)
        else:
            full_reply = " ".join(reply_parts)

        tts_provider = cfg.tts_provider
        tts_api_key = cfg.elevenlabs_api_key if tts_provider == "elevenlabs" else ""
        tts_voice_id = cfg.elevenlabs_voice_id if tts_provider == "elevenlabs" else ""

        for sentence in split_sentences(full_reply):
            yield _sse("reply_chunk", {"text": sentence})
            try:
                chunk = await synthesize(
                    sentence,
                    provider=tts_provider,
                    api_key=tts_api_key,
                    voice_id=tts_voice_id,
                )
                if chunk:
                    yield _sse("audio_chunk", {
                        "data": base64.b64encode(chunk).decode(),
                        "model": tts_provider,
                        "emotional": is_emotional(sentence),
                    })
            except Exception as exc:  # noqa: BLE001
                logger.warning("chat_tts_error", error=str(exc))

        # ── 6. Done ───────────────────────────────────────────────────────────
        yield _sse("done", {
            "intent": intent.value,
            "session_id": session_id,
            "user_context_used": user_context_used,
            "proactive": proactive,
            "agent_traces": [
                {
                    "agent": s.agent,
                    "input": s.input[:200],
                    "output": s.output[:200],
                    "latency_ms": round(s.latency_ms, 1),
                }
                for s in trace.spans
            ],
        })


@router.post("")
async def chat(
    request: ChatRequest,
    spotify: SpotifyClientProtocol = Depends(get_spotify_client),
    brave: BraveSearchClient = Depends(get_brave_client),
    weather: WeatherClientProtocol = Depends(get_weather_client),
    weaviate: WeaviateClient = Depends(get_weaviate_client),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
    cfg: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Run the Gia crew and stream SSE events back to the client.

    The client should consume events with ``EventSource`` (or the ``useSSE``
    hook in the Next.js frontend).

    Args:
        request: Parsed ``ChatRequest`` with ``message``, optional ``user_id``
                 and ``session_id``.

    Returns:
        ``StreamingResponse`` with ``Content-Type: text/event-stream``.
    """
    return StreamingResponse(
        _run_crew(request, spotify, brave, weather, weaviate, db, redis, cfg),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
