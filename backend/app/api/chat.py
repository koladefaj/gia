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
import contextlib
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from weaviate import WeaviateClient

from backend.app.agents.artist import ArtistService, extract_artist_name
from backend.app.agents.dj import DJService
from backend.app.agents.general import opening_line, stream_general
from backend.app.agents.hybrid_router import classify_turn
from backend.app.agents.mood import MoodService
from backend.app.agents.planner import _wants_weather
from backend.app.agents.router import _keyword_classify
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
from backend.app.mood.ingest import ingest_recently_played
from backend.app.mood.proactive import pop_proactive_draft
from backend.app.observability.langfuse import crew_trace
from backend.app.observability.logging import get_logger
from backend.app.providers.tts import is_emotional, synthesize_stream
from backend.app.schemas.chat import ChatRequest, IntentType
from backend.app.schemas.router import RouterDecision
from backend.app.tools.brave import BraveSearchClient
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


# Recent turns of the session transcript handed to the router (≈3 exchanges).
# The reply path keeps the full window (session_history._MAX_TURNS); the router
# only needs enough to resolve references and derive a clean search_query.
_ROUTER_HISTORY_TURNS = 6

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


async def _aiter(items: list[str]) -> AsyncIterator[str]:
    """Adapt a ready list of sentences to the async-iterator interface."""
    for item in items:
        yield item


@asynccontextmanager
async def _task_scope() -> AsyncIterator[list[asyncio.Task]]:
    """Track background tasks and cancel any still pending on exit.

    The memory-context task holds the request's DB session; if the client
    disconnects mid-stream we must cancel it before FastAPI tears the session
    down, otherwise a stray query lands on a closed session. Tasks that already
    finished are left alone.
    """
    tasks: list[asyncio.Task] = []
    try:
        yield tasks
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            with contextlib.suppress(Exception):
                await task


async def _stream_reply_frames(
    sentences: AsyncIterator[str],
    provider: str,
    api_key: str,
    voice_id: str,
    collected: list[str],
) -> AsyncIterator[str]:
    """Stream the reply text sentence-by-sentence, then STREAM the WHOLE reply's
    audio progressively as ElevenLabs renders it.

    The whole reply is synthesised in ONE pass (the full text is sent up-front),
    so prosody and v3 tag rendering stay consistent and there's no mid-reply
    flash↔v3 hop — v3 needs ~250+ chars of context to sound right, which a
    per-sentence call can't give it. But the audio is pulled from ElevenLabs'
    ``/stream`` endpoint and forwarded chunk-by-chunk, so the first audio bytes
    reach the client well before the complete file is rendered — that's the
    latency win that replaces the old fast acknowledgment. Text still streams
    live so captions appear as Gia "types"; every sentence is appended to
    *collected* so the caller can persist the full reply.

    Frames: ``audio_start`` (once, before the first byte) → N ``audio_chunk``
    (each ``streaming: true`` with a monotonic ``seq``) → ``audio_end``. The
    frontend appends the chunks to a MediaSource buffer and plays progressively.
    """
    async for sentence in sentences:
        collected.append(sentence)
        yield _sse("reply_chunk", {"text": sentence})

    full = " ".join(s.strip() for s in collected).strip()
    if not full:
        return

    # is_emotional(full) picks ONE model for the whole reply (no mid-reply
    # flash↔v3 hop): any tag or a question → eleven_v3, else eleven_flash.
    emotional = is_emotional(full)
    started = False
    seq = 0
    try:
        async for chunk in synthesize_stream(
            full, provider=provider, api_key=api_key, voice_id=voice_id
        ):
            if not chunk:
                continue
            if not started:
                started = True
                yield _sse("audio_start", {"model": provider, "emotional": emotional})
            yield _sse("audio_chunk", {
                "data": base64.b64encode(chunk).decode(),
                "model": provider,
                "emotional": emotional,
                "seq": seq,
                "streaming": True,
            })
            seq += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("chat_tts_error", error=str(exc))
    if started:
        yield _sse("audio_end", {"chunks": seq})


# Keyword-classified intents that clearly want a specialist (retrieval/playback),
# so the conversational reply should NOT be pre-generated for them. Everything
# else (greetings, questions, ambiguous MIXED) is treated as probably-chat and is
# worth speculating on while the router LLM runs. MIXED is intentionally absent:
# the keyword path and the router disagree most there, so speculating is the safe
# bet (a wasted call costs cents; a serial router wait costs ~2s of silence).
_SPECULATE_SKIP_INTENTS = {
    IntentType.MUSIC_FIND,
    IntentType.MUSIC_QUEUE,
    IntentType.ARTIST_INFO,
    IntentType.MOOD_CHECK,
}


def _should_speculate(message: str) -> bool:
    """Whether to pre-generate the conversational reply concurrently with the router.

    The router LLM is ~2s and fully blocks the reply. Most turns are chit-chat,
    so we kick off the conversational reply in parallel and use it only if the
    router confirms a conversational intent — correctness is unchanged (nothing
    is emitted before the router lands), we just overlap the generation. We skip
    speculation when the sub-ms keyword classifier already says a specialist will
    clearly run, to avoid burning an LLM call the turn won't use.
    """
    return _keyword_classify(message) not in _SPECULATE_SKIP_INTENTS


async def _collect_general(
    message: str,
    user_context_text: str,
    history_text: str,
    cfg: Settings,
) -> str:
    """Run the conversational reply to completion and return the full text.

    Drains :func:`stream_general` into a single string. Used both for the
    speculative pre-generation and the in-band fallback; the assembled reply is
    streamed to the client (text + progressive audio) in section 5.
    """
    parts: list[str] = []
    async for delta in stream_general(message, user_context_text, cfg=cfg, history=history_text):
        if delta:
            parts.append(delta)
    return "".join(parts).strip()


async def _speculative_general(
    message: str,
    memory_task: asyncio.Task | None,
    history_text: str,
    cfg: Settings,
) -> str:
    """Pre-generate the conversational reply, awaiting the in-flight memory task
    for personalisation context first. Never raises — a failed memory lookup just
    yields an un-personalised (but still warm) reply."""
    user_context_text = ""
    if memory_task is not None:
        try:
            ctx = await memory_task
            user_context_text = ctx.to_prompt_text()
        except Exception as exc:  # noqa: BLE001
            logger.warning("speculative_memory_error", error=str(exc))
    return await _collect_general(message, user_context_text, history_text, cfg)


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
    turn_t0 = time.monotonic()
    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_id
    store = WeaviateMemoryStore(client=weaviate) if user_id else None

    # TTS settings resolved once — used by both the acknowledgment (spoken first)
    # and the streamed reply.
    tts_provider = cfg.tts_provider
    tts_api_key = cfg.elevenlabs_api_key if tts_provider == "elevenlabs" else ""
    tts_voice_id = cfg.elevenlabs_voice_id if tts_provider == "elevenlabs" else ""

    async with crew_trace(session_id, user_id, user_input=request.message) as trace, \
            _task_scope() as bg_tasks:

        # ── 1. Kick off the slowest pre-reply steps concurrently ──────────────
        # Memory-context building and the router LLM are independent, so we start
        # both as background tasks. We also speculatively pre-generate the
        # conversational reply (below) concurrently with the router, so the ~2s
        # router round-trip overlaps the reply generation instead of preceding it.
        #
        # Recent turns of THIS conversation let the router resolve "play it now"/
        # "that one" and let Gia pick up where she left off — fetched first because
        # the router needs it. The reply gets the full window for continuity; the
        # router gets a tighter recent slice (≈3 exchanges) — enough to resolve
        # references and derive a clean search_query, without the extra tokens or
        # the risk of over-anchoring on a stale earlier intent.
        turns = await get_history(redis, session_id)
        history_text = format_history(turns)
        router_history = format_history(turns[-_ROUTER_HISTORY_TURNS:])

        memory_task: asyncio.Task | None = None
        memory_t0 = 0.0
        if user_id and store:
            yield _sse("agent_start", {"agent": "memory", "input": request.message})
            memory_t0 = time.monotonic()
            memory_task = asyncio.create_task(
                build_user_context(
                    user_id, request.message,
                    db=db, store=store, redis=redis, spotify=spotify, cfg=cfg,
                )
            )
            bg_tasks.append(memory_task)

        proactive: str | None = None
        if user_id:
            proactive = await pop_proactive_draft(user_id, redis)

        # "what's playing?" is a status query — answer from Spotify, don't
        # search/recommend (which is how Gia used to invent an answer).
        now_playing_query = _is_now_playing_query(request.message)

        # ── 1b. Speculatively pre-generate the conversational reply ───────────
        # The conversational reply is the turn's content whenever no specialist
        # runs. Most turns are chit-chat, so we start generating it NOW — in
        # parallel with the router — and use it only if the router confirms a
        # conversational intent. Nothing is emitted before the router lands, so
        # correctness is identical; we just stop paying the router latency and the
        # reply latency back-to-back. Skipped when a specialist is clearly coming
        # (keyword classifier), when a proactive note will lead the reply, for
        # now-playing status queries, and for providers without token streaming.
        general_task: asyncio.Task | None = None
        if (
            not now_playing_query
            and not proactive
            and len(request.message.strip()) >= 3
            and cfg.llm_provider in ("openai", "anthropic")
            and _should_speculate(request.message)
        ):
            general_task = asyncio.create_task(
                _speculative_general(request.message, memory_task, history_text, cfg)
            )
            bg_tasks.append(general_task)

        # ── 2. Route the turn (structured: intent + tone + engagement) ─────────
        # One small-model call classifies everything the turn needs. It never
        # raises — a failed call degrades to a warm GENERAL_CHAT default. Started
        # as a task so the speculative reply above overlaps the round-trip.
        yield _sse("agent_start", {"agent": "router", "input": request.message})
        with trace.span("router", request.message) as span:
            router_t0 = time.monotonic()
            # Created inside the span so the OpenAI drop-in generation nests under
            # "router" (asyncio tasks inherit the current OTel context).
            router_task = asyncio.create_task(
                classify_turn(request.message, cfg, history=router_history)
            )
            bg_tasks.append(router_task)

            decision = await router_task
            intent, confidence = decision.intent, decision.confidence
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
                "latency_ms": round((time.monotonic() - router_t0) * 1000, 1),
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

        # ── 3. Resolve the memory context (needed by agents + the reply) ──────
        user_context_text = ""
        user_context_used = False
        if memory_task is not None:
            with trace.span("memory", request.message) as span:
                try:
                    ctx = await memory_task
                    user_context_text = ctx.to_prompt_text()
                    user_context_used = bool(user_context_text)
                    span.set_output(f"context_length={len(user_context_text)}")
                    yield _sse("agent_done", {
                        "agent": "memory",
                        "latency_ms": round((time.monotonic() - memory_t0) * 1000, 1),
                        "context_chars": len(user_context_text),
                    })
                except Exception as exc:  # noqa: BLE001
                    logger.warning("chat_context_error", error=str(exc))
                    yield _sse("error", {"agent": "memory", "error": str(exc)})

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
                        requested_titles=decision.track_titles,
                    )
                    # Push tracks onto Spotify's queue when the user wants them
                    # queued — an explicit MUSIC_QUEUE, or a per-title request that
                    # named several specific tracks ("play X and queue Y next"). If
                    # the seed is already playing we queue only the rest; otherwise
                    # the seed leads. (Vibe MUSIC_FIND keeps its client-side
                    # crossfade queue and isn't pushed to Spotify.)
                    named_multi = len(decision.track_titles) >= 2
                    if intent == IntentType.MUSIC_QUEUE or named_multi:
                        tracks_to_queue = (
                            list(dj_result.queue.tracks)
                            if decision.start_playback
                            else [dj_result.primary_track, *dj_result.queue.tracks]
                        )
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
        substantive_parts = [p for p in reply_parts if p.strip()]
        full_reply = ""
        if not now_playing_query and (
            intent in _CONVERSATIONAL_INTENTS or not substantive_parts
        ):
            # Use the speculative reply generated alongside the router when we have
            # one (the common chit-chat path — it's already done or nearly so, so
            # the router latency was absorbed). Otherwise generate it now: this
            # covers turns we declined to speculate on (a specialist looked likely
            # but produced nothing, or the keyword path flagged a command). The
            # assembled reply is streamed (text + progressive audio) in section 5.
            yield _sse("agent_start", {"agent": "gia", "input": request.message})
            with trace.span("general", request.message) as span:
                t0 = time.monotonic()
                if general_task is not None:
                    reply = await general_task
                else:
                    reply = await _collect_general(
                        request.message, user_context_text, history_text, cfg
                    )
                reply_parts.append(reply)
                span.set_output(reply)
                yield _sse("agent_done", {
                    "agent": "gia",
                    "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                })

        # ── 5. Synthesise + stream reply + TTS ────────────────────────────────
        # Merge the contributing parts into one reply. When several agents
        # contributed (and synthesis is enabled), synthesise them into one
        # coherent reply; otherwise join. A pending proactive note is kept verbatim
        # at the front so its phrasing is never reworded.
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
        # "that one", "what did you just say").
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

        # Stream the assembled reply: captions sentence-by-sentence, then the
        # whole-reply audio forwarded progressively as ElevenLabs renders it.
        async for frame in _stream_reply_frames(
            _aiter(split_sentences(full_reply)),
            tts_provider, tts_api_key, tts_voice_id, [],
        ):
            yield frame

        # Record recent plays for mood patterning — throttled, and only after the
        # reply has streamed so it never adds perceived latency (the db session is
        # free at this point). On a fresh ingest, kick the worker to re-infer this
        # user's per-time-bucket mood patterns.
        if user_id:
            try:
                if await redis.set(f"ingest_throttle:{user_id}", "1", ex=1800, nx=True):
                    added = await ingest_recently_played(user_id, spotify, db)
                    if added:
                        from backend.worker.celery_app import (
                            celery_app,  # noqa: PLC0415
                        )
                        celery_app.send_task(
                            "backend.worker.tasks.mood_inference.run_mood_inference",
                            args=[user_id],
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning("chat_ingest_error", error=str(exc))

        # ── Self-evaluation: log per-turn quality/cost signals to Langfuse ────
        # Deterministic and free — the feedback loop that lets routing and
        # retrieval be tuned from data later instead of by guesswork.
        trace.score("context_used", 1 if user_context_used else 0, data_type="BOOLEAN")
        trace.score("retrieval_used", 1 if steps else 0, data_type="BOOLEAN")
        trace.score("router_confidence", round(confidence, 2), data_type="NUMERIC")
        trace.score(
            "turn_latency_ms",
            round((time.monotonic() - turn_t0) * 1000, 1),
            data_type="NUMERIC",
        )

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
