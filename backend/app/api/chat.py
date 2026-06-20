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

import base64
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from weaviate import WeaviateClient

from backend.app.agents.acknowledgment import get_selector, should_acknowledge
from backend.app.agents.artist import ArtistService, extract_artist_name
from backend.app.agents.dj import DJService
from backend.app.agents.general import opening_line, respond_general
from backend.app.agents.hybrid_router import classify_turn
from backend.app.agents.mood import MoodService
from backend.app.agents.planner import _wants_weather
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
from backend.app.memory.session_history import append_turn, format_history, get_history
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.mood.proactive import pop_proactive_draft
from backend.app.observability.langfuse import crew_trace
from backend.app.observability.logging import get_logger
from backend.app.providers.tts import is_emotional, synthesize
from backend.app.schemas.chat import ChatRequest, IntentType
from backend.app.schemas.router import RouterDecision
from backend.app.tools.brave import BraveSearchClient
from backend.app.voice.adapter import VoiceAdapter
from backend.app.voice.streaming import split_sentences

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


_voice_adapter = VoiceAdapter()

# Intents the conversational responder handles directly (no specialist agent).
_CONVERSATIONAL_INTENTS = {
    IntentType.GENERAL,
    IntentType.GENERAL_CHAT,
    IntentType.NEWS_QUERY,
    IntentType.MEMORY_QUERY,
}


def _steps_for_decision(decision: RouterDecision) -> list[str]:
    """Map a router decision to the specialist agents to run this turn.

    Slice A drives the existing DJ/Artist/Mood agents from the router's
    ``needs_*`` flags + intent. NEWS/MEMORY/GENERAL_CHAT fall through to the
    conversational responder (the streaming conversation agent lands in Slice B).
    """
    steps: list[str] = []
    if decision.needs_music or decision.intent in (
        IntentType.MUSIC_FIND, IntentType.MUSIC_QUEUE,
    ):
        steps.append("dj")
    if decision.needs_artist_lookup or decision.intent == IntentType.ARTIST_INFO:
        steps.append("artist")
    if decision.intent == IntentType.MOOD_CHECK:
        steps.append("mood")
    return steps


_NOW_PLAYING_RE = re.compile(
    r"\b(what('?s| is| am i)?\s+(currently\s+)?(playing|on right now|on now)"
    r"|what song is this|what'?s this song|now playing|current track)\b",
    re.IGNORECASE,
)


def _is_now_playing_query(message: str) -> bool:
    """Return ``True`` when the user is asking what's playing right now."""
    return bool(_NOW_PLAYING_RE.search(message))


async def _tts_frame(text: str, provider: str, api_key: str, voice_id: str) -> str | None:
    """Synthesise *text* and return an ``audio_chunk`` SSE frame (or ``None``)."""
    chunk = await synthesize(text, provider=provider, api_key=api_key, voice_id=voice_id)
    if not chunk:
        return None
    return _sse("audio_chunk", {
        "data": base64.b64encode(chunk).decode(),
        "model": provider,
        "emotional": is_emotional(text),
    })


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

    # TTS settings resolved once — used by both the acknowledgment (spoken first)
    # and the streamed reply.
    tts_provider = cfg.tts_provider
    tts_api_key = cfg.elevenlabs_api_key if tts_provider == "elevenlabs" else ""
    tts_voice_id = cfg.elevenlabs_voice_id if tts_provider == "elevenlabs" else ""

    async with crew_trace(session_id, user_id, user_input=request.message) as trace:

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

        # ── 2. Pop pending proactive draft + load short-term history ──────────
        proactive: str | None = None
        if user_id:
            proactive = await pop_proactive_draft(user_id, redis)

        # Recent turns of THIS conversation — lets the router resolve "play it
        # now"/"that one" and lets Gia pick up where she left off instead of
        # treating every message as the first.
        history_text = format_history(await get_history(redis, session_id))

        # ── 3. Route the turn (structured: intent + tone + engagement) ─────────
        # One small-model call classifies everything the turn needs. It never
        # raises — a failed call degrades to a warm GENERAL_CHAT default.
        yield _sse("agent_start", {"agent": "router", "input": request.message})
        with trace.span("router", request.message) as span:
            t0 = time.monotonic()
            decision = await classify_turn(request.message, cfg, history=history_text)
            intent, confidence = decision.intent, decision.confidence
            # "what's playing?" is a status query — answer from Spotify, don't
            # search/recommend (which is how Gia used to invent an answer).
            now_playing_query = _is_now_playing_query(request.message)
            steps = [] if now_playing_query else _steps_for_decision(decision)
            signals = ["weather"] if _wants_weather(request.message, steps) else []
            span.set_output(
                f"{intent.value}/{decision.tone.value}/{decision.engagement_mode.value}"
            )
            yield _sse("agent_done", {
                "agent": "router",
                "intent": intent.value,
                "tone": decision.tone.value,
                "engagement_mode": decision.engagement_mode.value,
                "confidence": round(confidence, 2),
                "latency_ms": round((time.monotonic() - t0) * 1000, 1),
            })
        yield _sse("plan", {
            "intent": intent.value,
            "steps": steps,
            "signals": signals,
            "tone": decision.tone.value,
            "engagement_mode": decision.engagement_mode.value,
            "needs": {
                "search": decision.needs_search,
                "memory": decision.needs_memory,
                "music": decision.needs_music,
                "artist_lookup": decision.needs_artist_lookup,
            },
        })

        # ── 3a. Immediate acknowledgment — Gia reacts before the work runs ─────
        # No LLM: a local template chosen by intent+tone, spoken right away so the
        # user hears a reaction in ~1s while retrieval / the reply are still cooking.
        if should_acknowledge(decision):
            with trace.span("acknowledgment", request.message) as span:
                t0 = time.monotonic()
                ack_line = get_selector().select(decision.intent, decision.tone, session_id)
                spoken = _voice_adapter.apply(decision.tone.value, ack_line)
                span.set_output(spoken)
                yield _sse("reply_chunk", {"text": spoken})
                yield _sse("acknowledgment", {
                    "text": spoken,
                    "tone": decision.tone.value,
                    "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                })
                frame = await _tts_frame(spoken, tts_provider, tts_api_key, tts_voice_id)
                if frame:
                    yield frame

        # ── 3b. Gather requested real-world signals ───────────────────────────
        if "weather" in signals:
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

        # Status query: report what's actually playing instead of guessing.
        if now_playing_query:
            yield _sse("agent_start", {"agent": "now_playing", "input": request.message})
            try:
                np = await spotify.get_currently_playing()
            except Exception as exc:  # noqa: BLE001
                logger.warning("chat_now_playing_error", error=str(exc))
                np = None
            if np and np.get("name"):
                reply_parts.append(
                    f"Right now you're on {np['name']} by {np.get('artist', '?')}."
                )
            else:
                reply_parts.append(
                    "Nothing's playing at the moment — want me to put something on?"
                )
            yield _sse("agent_done", {"agent": "now_playing"})

        if "dj" in steps:
            # Search the router's resolved query ("Fortworth Drake PARTYNEXTDOOR"),
            # not the raw message ("just play it now"), and let the router decide
            # play-vs-queue via start_playback.
            dj_query = decision.search_query or request.message
            yield _sse("agent_start", {"agent": "dj", "input": dj_query})
            with trace.span("dj", dj_query) as span:
                t0 = time.monotonic()
                try:
                    yield _sse("tool_call", {"agent": "dj", "tool": "search_tracks", "input": dj_query})
                    dj_svc = DJService(spotify=spotify, cfg=cfg)
                    dj_result = await dj_svc.recommend(
                        query=dj_query,
                        user_context_text=user_context_text,
                        start_playback=decision.start_playback,
                        n=4,
                    )
                    # "queue X" → add seed + all crossfade queue tracks to Spotify.
                    if intent == IntentType.MUSIC_QUEUE and not decision.start_playback:
                        tracks_to_queue = [dj_result.primary_track, *dj_result.queue.tracks]
                        for _track in tracks_to_queue:
                            try:
                                await spotify.add_to_queue(_track.uri)
                            except Exception as exc:  # noqa: BLE001
                                logger.warning("chat_queue_error", track=_track.name, error=str(exc))
                                break
                    span.set_output(dj_result.recommendation)
                    reply_parts.append(dj_result.recommendation)
                    yield _sse("agent_done", {
                        "agent": "dj",
                        "track": dj_result.primary_track.name,
                        "queue_depth": len(dj_result.queue.tracks),
                        "playback_started": dj_result.playback_started,
                        "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                    })
                except Exception as exc:  # noqa: BLE001
                    logger.warning("chat_dj_error", error=str(exc))
                    yield _sse("error", {"agent": "dj", "error": str(exc)})
                    reply_parts.append(
                        "[thoughtful] Hmm — I couldn't reach the music just now. "
                        "Give me a moment and ask again?"
                    )

        # Only run the artist agent when we can actually name an artist. This
        # stops generic/small-talk messages (e.g. "whats the weather like") from
        # being looked up as if they were an artist called that.
        artist_name = extract_artist_name(request.message) if "artist" in steps else ""
        if "artist" in steps and not artist_name:
            logger.debug("artist_skipped_no_name", message=request.message)
        if artist_name:
            yield _sse("agent_start", {"agent": "artist", "input": artist_name})
            with trace.span("artist", artist_name) as span:
                t0 = time.monotonic()
                try:
                    yield _sse("tool_call", {"agent": "artist", "tool": "brave_search", "input": artist_name})
                    artist_svc = ArtistService(
                        spotify=spotify, brave=brave, cfg=cfg, store=store
                    )
                    artist_result = await artist_svc.get_info(
                        artist_name=artist_name,
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

        if "mood" in steps and user_id and store:
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

        # Conversational intents (chit-chat, news, memory questions) — and any
        # turn where no specialist produced output — get a persona-grounded reply
        # in Gia's own voice instead of a canned string.
        if not now_playing_query and (
            intent in _CONVERSATIONAL_INTENTS or not [p for p in reply_parts if p.strip()]
        ):
            yield _sse("agent_start", {"agent": "gia", "input": request.message})
            with trace.span("general", request.message) as span:
                t0 = time.monotonic()
                reply = await respond_general(
                    request.message, user_context_text, cfg=cfg, history=history_text
                )
                span.set_output(reply)
                reply_parts.append(reply)
                yield _sse("agent_done", {
                    "agent": "gia",
                    "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                })

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

        # Trace-level output for the turn (visible at the top of the Langfuse trace).
        trace.set_output(full_reply)

        # Persist this exchange so the next turn has continuity ("play it now",
        # "that one", "what did you just say"). The ack filler is intentionally
        # not stored — only the substantive reply.
        await append_turn(redis, session_id, "user", request.message)
        await append_turn(redis, session_id, "gia", full_reply)

        # Fire-and-forget: distil durable memories (preferences + life facts) from
        # the conversation in the background worker. Throttled to ~45 min per
        # session so the extractor sees a meaningful chunk of conversation and
        # embedding API calls are batched rather than per-turn.
        # The flush task (session_flush.beat) handles the final extraction pass
        # for any tail turns that happen after the last throttle window.
        if user_id:
            try:
                # Keep the flush sorted set up-to-date so idle sessions are found.
                await redis.zadd("gia:pending_flush", {f"{user_id}:{session_id}": time.time()})

                if await redis.set(f"extract_throttle:{session_id}", "1", ex=2700, nx=True):
                    from backend.worker.celery_app import celery_app  # noqa: PLC0415
                    celery_app.send_task(
                        "backend.worker.tasks.memory_extraction.extract_session_memories",
                        args=[user_id, session_id],
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("chat_extract_enqueue_error", error=str(exc))

        for sentence in split_sentences(full_reply):
            yield _sse("reply_chunk", {"text": sentence})
            try:
                frame = await _tts_frame(sentence, tts_provider, tts_api_key, tts_voice_id)
                if frame:
                    yield frame
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


@router.get("/opening", summary="Gia's opening line — she speaks first")
async def opening(
    weaviate: Annotated[WeaviateClient, Depends(get_weaviate_client)],
    db: Annotated[AsyncSession, Depends(get_db)],
    redis: Annotated[Redis, Depends(get_redis)],
    spotify: Annotated[SpotifyClientProtocol, Depends(get_spotify_client)],
    cfg: Annotated[Settings, Depends(get_settings)],
    user_id: str | None = None,
) -> dict:
    """Return a warm, varied opening line for a fresh conversation.

    The frontend fetches this on load so Gia greets the user first (by name when
    a ``user_id`` is supplied and we know it), instead of waiting silently. The
    user-context lookup degrades gracefully — a flaky memory backend yields a
    generic-but-warm hello rather than an error.

    Args:
        user_id: Optional user UUID for a personalised greeting.

    Returns:
        ``{"greeting": "<Gia's opening line>"}``.
    """
    user_context_text = ""
    if user_id:
        try:
            store = WeaviateMemoryStore(client=weaviate)
            ctx = await build_user_context(
                user_id, "hello",
                db=db, store=store, redis=redis, spotify=spotify, cfg=cfg,
            )
            user_context_text = ctx.to_prompt_text()
        except Exception as exc:  # noqa: BLE001
            logger.warning("opening_context_error", error=str(exc))
    greeting = await opening_line(user_context_text, cfg=cfg)
    return {"greeting": greeting}


@router.post("", summary="Chat with Gia — returns a stream of SSE events", status_code=200)
async def chat(
    request: ChatRequest,
    spotify: Annotated[SpotifyClientProtocol, Depends(get_spotify_client)],
    brave: Annotated[BraveSearchClient, Depends(get_brave_client)],
    weather: Annotated[WeatherClientProtocol, Depends(get_weather_client)],
    weaviate: Annotated[WeaviateClient, Depends(get_weaviate_client)],
    db: Annotated[AsyncSession, Depends(get_db)],
    redis: Annotated[Redis, Depends(get_redis)],
    cfg: Annotated[Settings, Depends(get_settings)],
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
