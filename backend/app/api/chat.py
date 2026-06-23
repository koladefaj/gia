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
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from weaviate import WeaviateClient

from backend.app.agents import router_prewarm
from backend.app.agents.acknowledgment import get_selector
from backend.app.agents.artist import ArtistService, extract_artist_name
from backend.app.agents.dj import DJService
from backend.app.agents.general import opening_line, stream_general
from backend.app.agents.hybrid_router import classify_turn, fast_keyword_decision
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
from backend.app.providers.tts import is_emotional, synthesize, synthesize_stream
from backend.app.schemas.chat import ChatRequest, IntentType
from backend.app.schemas.dj import TrackItem
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


# Markers that a turn is asking about *current* facts — so the model must search
# rather than answer from its (months-old) training data. Catches the cases the
# router's needs_search can miss: "Drake's latest album", "who won", "score".
_FRESH_SEARCH_RE = re.compile(
    r"\b(latest|newest|recent(?:ly)?|current(?:ly)?|today|tonight|right now|"
    r"this (?:week|month|year|season)|so far this year|update[ds]?|news|headlines?|"
    r"happening|breaking|just (?:dropped|released|came out|out)|new (?:album|single|song|release)|"
    r"who won|who'?s winning|the score|standings|fixtures?|results?)\b"
    r"|\b20[2-9]\d\b",
    re.IGNORECASE,
)

# A tighter subset that wants the news feed specifically (vs general web).
_BREAKING_RE = re.compile(
    r"\b(news|breaking|headlines?|what'?s (?:going on|happening)|latest on)\b",
    re.IGNORECASE,
)


def _wants_fresh_search(message: str) -> bool:
    """Whether *message* asks about current facts that need a live search."""
    return bool(_FRESH_SEARCH_RE.search(message))


def _is_breaking_news(message: str) -> bool:
    """Whether *message* is a breaking-news ask → use the news feed, not web."""
    return bool(_BREAKING_RE.search(message))


async def _search_note(brave: BraveSearchClient, query: str, *, breaking: bool) -> str | None:
    """Run a Brave search and render results as a grounding block for the reply.

    Returns ``None`` when nothing comes back, so the turn proceeds (un-grounded
    but warm) rather than blocking on a degraded search.
    """
    try:
        results = await brave.recent(query, count=5, breaking=breaking)
    except Exception as exc:  # noqa: BLE001
        logger.warning("chat_search_error", error=str(exc))
        return None
    if not results:
        return None
    lines = []
    for r in results:
        age = f" · {r['age']}" if r.get("age") else ""
        desc = (r.get("description") or "").strip()
        lines.append(f"- {r['title']}{age}: {desc} ({r['url']})")
    today = datetime.now(UTC).strftime("%A, %d %B %Y")
    return (
        f"**Live web search results (today is {today} — use these as the source of "
        "truth for current facts and prefer them over anything you remember, which "
        "may be out of date):**\n" + "\n".join(lines)
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
    # Run the DJ only on an actual music intent — or on MIXED when the user asked
    # for music alongside something else. ``needs_music`` on its own does NOT pull
    # in the DJ: the router sets it spuriously on ARTIST_INFO ("tell me about X"),
    # which used to append a stray "Playing … now" to an info answer and auto-play
    # without a request. Music stays opt-in, per the no-auto-play rule.
    if decision.intent in (IntentType.MUSIC_FIND, IntentType.MUSIC_QUEUE) or (
        decision.intent == IntentType.MIXED and decision.needs_music
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


# ── Music command fast-path (ack + speculative search) ────────────────────────

# Music is the product, so a clear play/queue command earns (a) an instant spoken
# ack while the router + search run, and (b) a Spotify search started in parallel
# with the router. Both are gated on the sub-ms keyword classifier seeing a real
# command — never fires in front of a fast chat reply.
_MUSIC_COMMAND_INTENTS = {IntentType.MUSIC_FIND, IntentType.MUSIC_QUEUE}

# An explicit action verb — a music intent only counts as a *command* when the
# user is actually asking for playback, not just mentioning music ("his music is
# fire" is chat, not a request).
_MUSIC_VERB_RE = re.compile(
    r"\b(play|queue|put (me )?on|throw on|add|skip|save|find|recommend|shuffle|resume)\b",
    re.IGNORECASE,
)

# Verbs that actually mean "start it NOW" (vs. queue/add for later). A MUSIC_QUEUE
# turn only auto-plays the seed when one of these is present ("play X and queue Y");
# a pure "queue X after this" must NOT start playback, whatever the router guessed.
_PLAY_VERB_RE = re.compile(
    r"\b(play|put (me )?on|throw on|resume|start|blast|bump)\b",
    re.IGNORECASE,
)

# Leading command verbs / trailing fillers stripped to turn a raw command into a
# cleaner speculative search query ("play some Travis Scott right now" → "some
# Travis Scott"). The result need not be perfect — it only seeds the *speculative*
# search, which is discarded unless the router's resolved query matches the text.
_SPEC_STRIP_RE = re.compile(
    r"^\s*(please\s+)?(can you\s+|could you\s+)?"
    r"(play|queue|put (me )?on|throw on|add|skip|save|find|recommend|shuffle|resume)\s+"
    r"|\b(right now|for me|please)\b\s*$",
    re.IGNORECASE,
)


def _is_music_command(message: str) -> bool:
    """Whether *message* is a clear play/queue command (keyword, sub-ms)."""
    return (
        _keyword_classify(message) in _MUSIC_COMMAND_INTENTS
        and bool(_MUSIC_VERB_RE.search(message))
    )


def _speculative_query(message: str) -> str:
    """Strip command verbs / trailing fillers to seed the speculative search."""
    return _SPEC_STRIP_RE.sub("", message).strip() or message.strip()


def _query_tokens_present(search_query: str, message: str) -> bool:
    """Whether the router's resolved query is "in" the user's words.

    The speculative search ran on the raw message, so its results are only
    representative when the router didn't resolve a reference to something *not*
    said (e.g. "play it now" → a track pulled from history). True when the query
    is empty (nothing to disagree with) or every significant token of it appears
    in the message — then the prefetched results are safe to reuse.
    """
    sq = (search_query or "").strip().lower()
    if not sq:
        return True
    words = set(re.findall(r"[a-z0-9]+", message.lower()))
    tokens = [t for t in re.findall(r"[a-z0-9]+", sq) if len(t) >= 3]
    return bool(tokens) and all(t in words for t in tokens)


async def _safe_search(
    dj_svc: DJService, query: str
) -> tuple[TrackItem, list[TrackItem]] | None:
    """Run a speculative DJ search, swallowing any error (returns ``None``)."""
    try:
        return await dj_svc.search_only(query, n=4)
    except Exception as exc:  # noqa: BLE001
        logger.debug("speculative_search_failed", query=query, error=str(exc))
        return None


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
            # A current-facts turn will search and regenerate, discarding any
            # speculative reply — so don't pay for one.
            and not _wants_fresh_search(request.message)
        ):
            general_task = asyncio.create_task(
                _speculative_general(request.message, memory_task, history_text, cfg)
            )
            bg_tasks.append(general_task)

        # ── 1c. Music command fast-path: search in parallel with the router ───
        # Music is the product, so a clear play/queue command kicks off the Spotify
        # search NOW, in parallel with the router, instead of after it. The result
        # is reused (recommend(prefetched=...)) when the router's resolved query is
        # "in" the user's words; reference commands fall back to a fresh search.
        dj_svc: DJService | None = None
        spec_search_task: asyncio.Task | None = None
        if not now_playing_query and _is_music_command(request.message):
            dj_svc = DJService(spotify=spotify, cfg=cfg)
            spec_search_task = asyncio.create_task(
                _safe_search(dj_svc, _speculative_query(request.message))
            )
            bg_tasks.append(spec_search_task)

        # ── 1d. Instant warm ack for music commands ───────────────────────────
        # A music search takes ~3-5s. Synthesise a short, content-NEUTRAL filler
        # NOW — concurrently with the router + search — so when the DJ step starts
        # we can speak it immediately and the user hears warmth instead of silence
        # while the search runs. Neutral by design ("Say less." / "On it." — never
        # "playing it now"), so it fronts a found track OR a "couldn't find it"
        # without ever contradicting the result. The [warm] tag routes it to
        # eleven_v3 for human warmth; it's emitted only if the DJ actually runs.
        ack_task: asyncio.Task | None = None
        if spec_search_task is not None:
            ack_line = get_selector().select_filler(session_id)
            ack_task = asyncio.create_task(
                synthesize(
                    f"[warm] {ack_line}",
                    provider=tts_provider, api_key=tts_api_key, voice_id=tts_voice_id,
                )
            )
            bg_tasks.append(ack_task)

        # ── 2. Route the turn (structured: intent + tone + engagement) ─────────
        # One small-model call classifies everything the turn needs. It never
        # raises — a failed call degrades to a warm GENERAL_CHAT default. Started
        # as a task so the speculative reply above overlaps the round-trip.
        #
        # Early-intent: streaming STT (Flux) fires /chat/prewarm on the eager
        # transcript a beat before the user finishes, so the router may already be
        # done. We take that result when present (the common streaming case) and
        # only classify cold on a miss (typed turns, batch STT, multi-worker) —
        # either way the input is identical, so the decision is the same.
        yield _sse("agent_start", {"agent": "router", "input": request.message})
        with trace.span("router", request.message) as span:
            router_t0 = time.monotonic()
            # Tier-1: a confident keyword read of pure conversation skips the LLM
            # router (and the prewarm) entirely — sub-ms instead of ~2s.
            decision = (
                fast_keyword_decision(request.message)
                if cfg.router_fast_path_enabled else None
            )
            fast_path = decision is not None
            prewarmed = False
            if decision is None:
                # Tier-2: reuse the decision the eager prewarm already started.
                decision = await router_prewarm.take(redis, session_id, request.message)
                prewarmed = decision is not None
            if decision is None:
                # Tier-3: classify cold. Created inside the span so the OpenAI
                # drop-in generation nests under "router" (asyncio tasks inherit
                # the current OTel context).
                router_task = asyncio.create_task(
                    classify_turn(request.message, cfg, history=router_history)
                )
                bg_tasks.append(router_task)
                decision = await router_task

            intent, confidence = decision.intent, decision.confidence
            # A pure queue request ("queue X after this") must NOT start playback —
            # the router sometimes sets start_playback anyway, which made the DJ play
            # the track now instead of lining it up. Only honour playback on a queue
            # turn when the user actually said a play verb ("play X and queue Y").
            if intent == IntentType.MUSIC_QUEUE and not _PLAY_VERB_RE.search(request.message):
                decision.start_playback = False
            steps = [] if now_playing_query else _steps_for_decision(decision)
            signals = ["weather"] if _wants_weather(request.message, steps) else []
            _src = "fast" if fast_path else "prewarmed" if prewarmed else "llm"
            span.set_output(
                f"{intent.value}/{decision.tone.value}/{decision.engagement_mode.value} ({_src})"
            )
            yield _sse("agent_done", {
                "agent": "router",
                "intent": intent.value,
                "tone": decision.tone.value,
                "engagement_mode": decision.engagement_mode.value,
                "confidence": round(confidence, 2),
                "prewarmed": prewarmed,
                "fast_path": fast_path,
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

        # Live web/news search for current-facts turns. The router answers news and
        # general questions conversationally (no specialist), so without this the
        # model replies from stale training data ("Drake's latest album" → a
        # year-old answer; "World Cup update" → "I don't know"). We search when the
        # turn is conversational AND either the router flagged needs_search or the
        # message carries recency markers, then ground the reply in the results.
        searched = False
        # Will the artist specialist actually run? It's skipped when no clean name
        # resolves ("whats Drake's newest album?" → no name), and the turn then
        # falls back to the conversational reply — which must be search-grounded.
        artist_will_run = (
            intent == IntentType.ARTIST_INFO
            and bool(extract_artist_name(request.message))
        )
        # Search-ground current-facts turns. Eligible intents answer
        # conversationally; music/mood/memory have their own data sources.
        # ARTIST_INFO is included only when its specialist WON'T run — otherwise the
        # artist agent (which does its own freshness-filtered search) handles it.
        search_eligible = (
            intent in (
                IntentType.GENERAL, IntentType.GENERAL_CHAT,
                IntentType.NEWS_QUERY, IntentType.MIXED,
            )
            or (intent == IntentType.ARTIST_INFO and not artist_will_run)
        )
        do_search = (
            not now_playing_query
            and search_eligible
            and (decision.needs_search or _wants_fresh_search(request.message))
        )
        if do_search:
            search_query = decision.search_query or request.message
            breaking = _is_breaking_news(request.message)
            yield _sse("tool_call", {
                "agent": "search", "tool": "brave_news" if breaking else "brave_web",
                "input": search_query,
            })
            with trace.span("search", search_query) as span:
                note = await _search_note(brave, search_query, breaking=breaking)
                span.set_output(f"results={'yes' if note else 'none'}")
            if note:
                user_context_text = f"{user_context_text}\n{note}".strip()
                searched = True
                yield _sse("signal", {"name": "search", "value": search_query})

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

            # Speak the pre-synthesised warm ack now, before the ~3-5s search, so
            # the user hears "Say less." while we look. One-shot (non-streaming)
            # audio → the frontend plays it on the AudioPlayer; the DJ reply
            # streams progressively after, on the StreamPlayer. The synth started
            # back in §1d, so this await is effectively instant.
            if ack_task is not None:
                try:
                    ack_audio = await ack_task
                    if ack_audio:
                        yield _sse("audio_chunk", {
                            "data": base64.b64encode(ack_audio).decode(),
                            "model": tts_provider,
                            "streaming": False,
                            "ack": True,
                        })
                except Exception as exc:  # noqa: BLE001
                    logger.warning("chat_ack_error", error=str(exc))

            with trace.span("dj", dj_query) as span:
                t0 = time.monotonic()
                try:
                    # Reuse the speculative search started alongside the router when
                    # the router's resolved query is "in" what the user said (not a
                    # reference resolved from history) and they didn't name multiple
                    # specific titles — otherwise the prefetch isn't representative
                    # and we let recommend() search the resolved query itself.
                    prefetched: tuple[TrackItem, list[TrackItem]] | None = None
                    if (
                        spec_search_task is not None
                        and len(decision.track_titles) < 2
                        and _query_tokens_present(decision.search_query or "", request.message)
                    ):
                        prefetched = await spec_search_task
                    yield _sse("tool_call", {
                        "agent": "dj", "tool": "search_tracks", "input": dj_query,
                        "prefetched": prefetched is not None,
                    })
                    if dj_svc is None:
                        dj_svc = DJService(spotify=spotify, cfg=cfg)
                    dj_result = await dj_svc.recommend(
                        query=dj_query,
                        user_context_text=user_context_text,
                        start_playback=decision.start_playback,
                        n=4,
                        requested_titles=decision.track_titles,
                        prefetched=prefetched,
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
                # The speculative reply was generated BEFORE the search ran, so on a
                # search turn it can't see the results — regenerate with the grounded
                # context. Non-search turns reuse the speculative reply as before.
                if general_task is not None and not searched:
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


@router.post("/prewarm", status_code=202, summary="Speculatively warm the router on an eager transcript")
async def prewarm(
    request: ChatRequest,
    redis: Annotated[Redis, Depends(get_redis)],
    cfg: Annotated[Settings, Depends(get_settings)],
) -> dict:
    """Start the router classification early, before the turn is final.

    Called by the streaming-STT client on Deepgram Flux's ``EagerEndOfTurn`` — a
    medium-confidence turn end whose transcript is guaranteed to match the final
    one. We kick ``classify_turn`` and stash the in-flight task; the subsequent
    ``POST /chat`` (with the final transcript) awaits it instead of starting the
    router cold, so the router latency overlaps the user's last words.

    Read-only and side-effect free — only the *classification* is precomputed;
    search/playback still happen in ``/chat`` after the final transcript. Returns
    immediately (202) without waiting for the router.

    Args:
        request: ``ChatRequest`` — ``message`` is the eager transcript;
                 ``session_id`` must match the one the client sends to ``/chat``.
    """
    session_id = request.session_id
    if not session_id:
        return {"prewarmed": False}
    # If the keyword fast-path will handle this turn without the LLM router,
    # there's nothing to prewarm — don't spend an LLM call on it.
    if cfg.router_fast_path_enabled and fast_keyword_decision(request.message) is not None:
        return {"prewarmed": False}
    turns = await get_history(redis, session_id)
    router_history = format_history(turns[-_ROUTER_HISTORY_TURNS:])
    await router_prewarm.start(
        redis,
        session_id,
        request.message,
        lambda: classify_turn(request.message, cfg, history=router_history),
    )
    return {"prewarmed": True}


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
