"""Speech-to-speech — the OpenAI Realtime (``gpt-realtime``) bridge.

This is the *realtime* counterpart to the decomposed voice pipeline, but it is a
**hybrid**, not a black-box speech-to-speech swap. The split:

  * ``gpt-realtime`` is the **ears + brain**: the browser streams mic PCM16 up,
    the model understands the speech directly (native turn-taking + barge-in),
    reasons, calls tools, and emits the reply as **text** — ``output_modalities``
    is text-only, so the model never speaks.
  * **ElevenLabs v3 is the voice.** The endpoint takes the model's text and
    synthesises it through the existing ElevenLabs streaming path so the warm,
    audio-tagged brand voice is preserved (the voice is the product). The model
    follows the Gia persona, which already prescribes ``[warm]`` / ``[laughs]``
    tags inline, and v3 renders them.

So the realtime model replaces ``STT → router → specialist`` with one
low-latency speech-understanding + tool-calling brain, while TTS stays exactly
as it is in the pipeline. The memory / Spotify / Brave / weather services are
reached through **function calling**, executed server-side with the same clients.

Two cohesive pieces live here, mirroring how ``stt_stream.py`` keeps the
provider pure-protocol while the endpoint owns the browser socket:

  * :class:`RealtimeTools` — the DI'd tool layer (schemas + dispatch together).
  * :class:`RealtimeSession` — the OpenAI Realtime WebSocket protocol: GA-shaped
    ``session.update`` (audio in, text out), the function-call round-trip, and
    barge-in. It yields normalised :class:`RealtimeEvent` text/control events; the
    endpoint turns the text into ElevenLabs audio.

GA protocol notes (changed from the 2024 beta — these bite): the model is set via
the ``?model=`` query param (no ``OpenAI-Beta`` header), audio config is nested
under ``session.audio.input``, and the audio *format* is an **object**
(``{"type": "audio/pcm", "rate": 24000}``) — not the old ``"pcm16"`` string. The
reply arrives on ``response.output_text.delta`` / ``.done`` (text modality), and
barge-in is the ``input_audio_buffer.speech_started`` event.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import websockets
from websockets.asyncio.client import ClientConnection

from backend.app.config import Settings
from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

# Input wire format: 24 kHz mono PCM16 — the same the browser already produces
# for the streaming-STT worklet, so realtime reuses that capture path unchanged.
SAMPLE_RATE = 24_000

_URL = "wss://api.openai.com/v1/realtime"


# ── Tool schemas ────────────────────────────────────────────────────────────────
# What the realtime model sees. Kept next to RealtimeTools.dispatch so a new tool
# can't be advertised without an implementation (or vice versa). Deliberately a
# small, high-signal set — tools return *facts*, not prose (the model writes the
# reply, ElevenLabs speaks it), so there's no specialist-LLM call on the hot path.

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "search_and_play_music",
        "description": (
            "Search Spotify and optionally start playback or queue tracks. Use when "
            "the user asks to play, queue, or find music. Returns the chosen track "
            "and queued tracks — describe them warmly in your own voice."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for — an artist, song, vibe, or genre.",
                },
                "start_playback": {
                    "type": "boolean",
                    "description": "Start playing the top result immediately. Only when the user clearly wants it ON now.",
                },
                "queue_more": {
                    "type": "boolean",
                    "description": "Also push the following tracks onto the Spotify queue.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "get_web_info",
        "description": (
            "Look up current, real-world facts via web search — an artist's latest "
            "release, news, who-won, anything that may have changed since your "
            "training. Returns recent results to ground your answer; prefer them "
            "over what you remember."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "breaking": {
                    "type": "boolean",
                    "description": "True for a breaking-news / headlines ask (uses the news feed).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "recall_memory",
        "description": (
            "Retrieve what you know about this listener — their taste, preferences, "
            "and past conversations — for a given topic. Use when personalising a "
            "reply or when the user refers to something from before."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "What to recall about the user (e.g. 'focus music', 'favourite artists').",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "type": "function",
        "name": "get_now_playing",
        "description": "Report what is currently playing on the user's Spotify, if anything.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "get_weather",
        "description": (
            "Get the current weather for the user's city — useful for context-aware "
            "music suggestions (energy for a hot run, something mellow for rain)."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]


@dataclass
class RealtimeTools:
    """The realtime model's hands — DI'd tool dispatch onto existing services.

    Constructed by the endpoint from the request-scoped clients so each tool runs
    with the same Spotify/Brave/Weaviate/DB/Redis wiring as the decomposed
    pipeline. :meth:`dispatch` is the single entry point the session calls; it
    never raises — a failed tool returns an ``{"error": ...}`` payload the model
    can gracefully speak around, exactly like the pipeline's per-agent try/except.

    Attributes:
        cfg:      Application settings.
        spotify:  Spotify client (live or mock).
        brave:    Brave Search client (current-facts grounding).
        store:    Weaviate memory store, or ``None`` for an anonymous session.
        db:       SQLAlchemy async session (for user-context assembly).
        redis:    Async Redis client (retrieval cache + memory dedup).
        weather:  Weather client.
        user_id:  Owning user UUID string, or ``None`` when anonymous.
    """

    cfg: Settings
    spotify: Any
    brave: Any
    store: Any
    db: Any
    redis: Any
    weather: Any
    user_id: str | None = None
    spotify_web: Any = None  # direct Web API client — the fast search path

    async def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run tool *name* with *args* and return a JSON-serialisable result.

        Returns ``{"error": ...}`` instead of raising so a degraded tool never
        kills the live session — the model just hears the failure and adapts.
        """
        try:
            handler = {
                "search_and_play_music": self._search_and_play_music,
                "get_web_info": self._get_web_info,
                "recall_memory": self._recall_memory,
                "get_now_playing": self._get_now_playing,
                "get_weather": self._get_weather,
            }.get(name)
            if handler is None:
                return {"error": f"unknown tool {name!r}"}
            return await handler(args)
        except Exception as exc:  # noqa: BLE001
            logger.warning("realtime_tool_error", tool=name, error=str(exc))
            return {"error": str(exc)}

    async def _search(self, query: str) -> list[dict[str, str]]:
        """Return ``[{uri,name,artist}]`` for *query*, fastest source first.

        Prefers the direct Web API (``spotify_web``, ~400–600 ms warm over a
        pooled connection); falls back to the MCP search (``DJService.search_only``,
        ~1.9 s) when the Web client is absent, errors, or returns nothing. Both are
        read-only, so the realtime model writes the spoken line itself.
        """
        if self.spotify_web is not None:
            try:
                results = await self.spotify_web.search_tracks(query, limit=10)
                if results:
                    return results
            except Exception as exc:  # noqa: BLE001
                logger.warning("realtime_web_search_failed", error=str(exc))
        from backend.app.agents.dj import DJService  # noqa: PLC0415

        seed, queue = await DJService(spotify=self.spotify, cfg=self.cfg).search_only(query, n=4)
        return [{"uri": t.uri, "name": t.name, "artist": t.artist} for t in [seed, *queue]]

    async def _search_and_play_music(self, args: dict[str, Any]) -> dict[str, Any]:
        """Search Spotify, optionally start/queue, return track facts.

        Search uses the fast direct Web API when available (see :meth:`_search`);
        playback/queue side effects go through the MCP client and mirror ``chat.py``.
        """
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "empty query"}
        items = await self._search(query)
        if not items:
            return {"error": f"no tracks found for {query!r}"}
        seed = items[0]
        queue_tracks = [t for t in items[1:] if t["uri"] != seed["uri"]][:4]

        playing = False
        if args.get("start_playback"):
            await self.spotify.start_playback(seed["uri"])
            playing = True
        queued: list[dict[str, str]] = []
        if args.get("queue_more"):
            # If the seed is already playing, queue only the rest; otherwise lead
            # with the seed (same rule as the pipeline's queue handling).
            to_queue = queue_tracks if playing else [seed, *queue_tracks]
            for track in to_queue:
                with contextlib.suppress(Exception):
                    await self.spotify.add_to_queue(track["uri"])
                    queued.append({"name": track["name"], "artist": track["artist"]})

        logger.info("realtime_music", query=query, seed=seed["name"], playing=playing)
        return {
            "playing": playing,
            "track": {"name": seed["name"], "artist": seed["artist"]},
            "next_up": [{"name": t["name"], "artist": t["artist"]} for t in queue_tracks[:3]],
            "queued": queued,
        }

    async def _get_web_info(self, args: dict[str, Any]) -> dict[str, Any]:
        """Brave search → recent results as grounding facts for the model."""
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "empty query"}
        results = await self.brave.recent(
            query, count=5, breaking=bool(args.get("breaking"))
        )
        return {
            "query": query,
            "results": [
                {
                    "title": r.get("title", ""),
                    "age": r.get("age", ""),
                    "summary": (r.get("description") or "").strip(),
                    "url": r.get("url", ""),
                }
                for r in (results or [])
            ],
        }

    async def _recall_memory(self, args: dict[str, Any]) -> dict[str, Any]:
        """Assemble user context for *topic* via the same retrieval the pipeline uses."""
        if not (self.user_id and self.store):
            return {"context": "", "note": "no memory for this session"}
        from backend.app.memory.retrieval import build_user_context  # noqa: PLC0415

        topic = str(args.get("topic", "")).strip() or "general preferences"
        ctx = await build_user_context(
            self.user_id, topic,
            db=self.db, store=self.store, redis=self.redis,
            spotify=self.spotify, cfg=self.cfg,
        )
        return {"context": ctx.to_prompt_text()}

    async def _get_now_playing(self, _args: dict[str, Any]) -> dict[str, Any]:
        """Report the currently-playing track (status query, not a recommendation)."""
        np = await self.spotify.get_currently_playing()
        if np and np.get("name"):
            return {"playing": True, "name": np["name"], "artist": np.get("artist", "")}
        return {"playing": False}

    async def _get_weather(self, _args: dict[str, Any]) -> dict[str, Any]:
        """Current weather at the user's configured default coordinates."""
        current = await self.weather.get_current(
            self.cfg.weather_default_lat, self.cfg.weather_default_lon
        )
        if not current:
            return {"available": False}
        return {
            "available": True,
            "location": self.cfg.weather_default_label,
            "temperature_c": round(current["temperature_c"]),
            "condition": current["condition"],
        }


# ── Normalised events the endpoint relays ────────────────────────────────────────


@dataclass(frozen=True)
class RealtimeEvent:
    """One normalised event from the realtime session, for the browser endpoint.

    The endpoint maps these to browser frames and to ElevenLabs synthesis without
    touching the OpenAI wire format — the same insulation ``TranscriptEvent`` gives
    the streaming-STT path.

    Attributes:
        kind:       One of ``user_transcript`` (finalised user words → captions +
                    memory), ``assistant_delta`` (a chunk of the reply text → live
                    captions), ``assistant_text`` (the finalised reply text →
                    ElevenLabs synthesis + persistence), ``audio`` (a chunk of the
                    model's OWN voice, base64 PCM16 in ``audio_b64`` — only in
                    ``model`` voice-source mode), ``speech_started`` (user barged in
                    → cancel synthesis + flush playback), ``tool`` (a tool is
                    running), ``response_done`` (turn complete), or ``error``.
        text:       Transcript / reply text, or the error string.
        tool_name:  Name of the tool when ``kind == "tool"``.
        audio_b64:  Base64 PCM16 chunk when ``kind == "audio"`` (forwarded verbatim).
    """

    kind: str
    text: str = ""
    tool_name: str = ""
    audio_b64: str = ""


# ── The OpenAI Realtime session ──────────────────────────────────────────────────


class RealtimeSession:
    """A live audio-in / text-out session over the OpenAI Realtime WebSocket.

    Lifecycle mirrors the streaming transcribers: an async context manager you
    feed audio with :meth:`send_audio` and drain events from via :meth:`events`,
    both running concurrently on the one socket. Tool calls run in the background
    (so a ~3-5 s Spotify search never stalls event relay or barge-in) and their
    results are sent back over the same socket.

    Args:
        api_key:             OpenAI API key (reuses ``cfg.openai_api_key``).
        model:               Realtime model id (``cfg.realtime_model``).
        vad:                 Turn detection — ``semantic_vad`` / ``server_vad``.
        transcription_model: Model for transcribing the *user's* audio (keeps the
                             observability + memory pipeline alive in realtime mode).
        instructions:        System prompt — the Gia persona plus the user's memory
                             context, assembled by the endpoint before connect.
        tools:               The DI'd :class:`RealtimeTools` dispatcher.
        voice_source:        ``elevenlabs`` (model emits text, ElevenLabs speaks) or
                             ``model`` (model speaks directly — audio out).
        voice:               Output voice when ``voice_source == "model"``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        vad: str,
        transcription_model: str,
        instructions: str,
        tools: RealtimeTools,
        voice_source: str = "elevenlabs",
        voice: str = "marin",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._vad = vad
        self._transcription_model = transcription_model
        self._instructions = instructions
        self._tools = tools
        self._voice_source = voice_source
        self._voice = voice
        self._ws: ClientConnection | None = None
        # call_id → function name, captured from response.output_item.added so the
        # name is known when we execute the call at response.done.
        self._pending_calls: dict[str, str] = {}
        self._tool_tasks: set[asyncio.Task] = set()
        # True between response.created and response.done — so a barge-in only
        # cancels when something is actually generating (cancelling nothing makes
        # the API emit a spurious error).
        self._response_active = False

    async def __aenter__(self) -> RealtimeSession:
        url = f"{_URL}?{urlencode({'model': self._model})}"
        self._ws = await websockets.connect(
            url,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
            max_size=None,
        )
        await self._configure()
        logger.info("realtime_open", model=self._model)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        for task in list(self._tool_tasks):
            task.cancel()
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

    async def _configure(self) -> None:
        """Send the GA-shaped ``session.update`` that arms the session.

        Output modality depends on the voice source: **text-only** when ElevenLabs
        is the voice (the model writes, ElevenLabs speaks), or **audio** when the
        model speaks directly (with ``audio.output`` carrying the voice). Note the
        GA nesting (``audio.input``) and the *object* audio format — the two things
        that throw type errors against the old beta shape. Input transcription is
        always on so the user's words still feed captions + the memory pipeline.
        """
        assert self._ws is not None
        audio: dict[str, Any] = {
            "input": {
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                "turn_detection": {"type": self._vad},
                "transcription": {"model": self._transcription_model},
            },
        }
        if self._voice_source == "model":
            modalities = ["audio"]
            audio["output"] = {
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                "voice": self._voice,
            }
        else:
            modalities = ["text"]
        await self._ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": modalities,
                "instructions": self._instructions,
                "audio": audio,
                "tools": TOOL_SCHEMAS,
                "tool_choice": "auto",
            },
        }))

    async def send_audio(self, pcm: bytes) -> None:
        """Forward a chunk of raw mono PCM16 to the model's input buffer.

        Server/semantic VAD commits the buffer on the detected turn end, so no
        explicit commit is needed per turn.
        """
        if self._ws is not None and pcm:
            await self._ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode(),
            }))

    async def finish(self) -> None:
        """No-op — server/semantic VAD commits each turn itself.

        An explicit ``input_audio_buffer.commit`` here just hits an already-drained
        buffer and errors with ``input_audio_buffer_commit_empty``; the turn the
        user was speaking has already been committed by VAD. Kept for symmetry with
        the streaming-STT interface.
        """
        return

    async def _cancel_response(self) -> None:
        """Tell the model to stop generating the current reply (barge-in).

        No-op when nothing is generating — cancelling an absent response makes the
        API emit a spurious error.
        """
        if self._ws is not None and self._response_active:
            self._response_active = False
            with contextlib.suppress(Exception):
                await self._ws.send(json.dumps({"type": "response.cancel"}))

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        """Yield normalised text/control events until the model closes the socket.

        Function calls are dispatched in the background and their results sent
        back here, so this loop keeps relaying text and ``speech_started``
        (barge-in) while a tool runs.
        """
        if self._ws is None:
            return
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            etype = msg.get("type", "")

            if etype == "response.created":
                self._response_active = True

            # Reply text — ElevenLabs mode (text-out).
            elif etype == "response.output_text.delta":
                delta = msg.get("delta", "")
                if delta:
                    yield RealtimeEvent(kind="assistant_delta", text=delta)

            elif etype == "response.output_text.done":
                text = (msg.get("text", "") or "").strip()
                if text:
                    yield RealtimeEvent(kind="assistant_text", text=text)

            # Model-voice mode (audio-out): the audio bytes plus their transcript
            # (the transcript still drives captions + the memory pipeline).
            elif etype == "response.output_audio.delta":
                delta = msg.get("delta", "")
                if delta:
                    yield RealtimeEvent(kind="audio", audio_b64=delta)

            elif etype == "response.output_audio_transcript.delta":
                delta = msg.get("delta", "")
                if delta:
                    yield RealtimeEvent(kind="assistant_delta", text=delta)

            elif etype == "response.output_audio_transcript.done":
                text = (msg.get("transcript", "") or "").strip()
                if text:
                    yield RealtimeEvent(kind="assistant_text", text=text)

            elif etype == "conversation.item.input_audio_transcription.completed":
                text = (msg.get("transcript", "") or "").strip()
                if text:
                    yield RealtimeEvent(kind="user_transcript", text=text)

            elif etype == "input_audio_buffer.speech_started":
                # The user started talking over the reply — stop the model
                # generating it; the endpoint will also flush ElevenLabs playback.
                await self._cancel_response()
                yield RealtimeEvent(kind="speech_started")

            elif etype == "response.output_item.added":
                item = msg.get("item", {}) or {}
                if item.get("type") == "function_call":
                    call_id = item.get("call_id", "")
                    if call_id:
                        self._pending_calls[call_id] = item.get("name", "")

            elif etype == "response.done":
                self._response_active = False
                async for ev in self._handle_response_done(msg):
                    yield ev

            elif etype == "error":
                err = msg.get("error", {})
                logger.warning("realtime_server_error", error=err)
                yield RealtimeEvent(kind="error", text=json.dumps(err))

    async def _handle_response_done(self, msg: dict) -> AsyncIterator[RealtimeEvent]:
        """Dispatch any function calls in a completed response; else signal done.

        A realtime response that wants a tool contains ``function_call`` output
        items. We run each in the background and emit a ``tool`` status; when the
        result is sent back, the model produces a fresh (text) response. A
        response with no tool calls is a finished turn.
        """
        output = (msg.get("response", {}) or {}).get("output", []) or []
        calls = [o for o in output if o.get("type") == "function_call"]
        if not calls:
            yield RealtimeEvent(kind="response_done")
            return
        for call in calls:
            name = call.get("name") or self._pending_calls.pop(call.get("call_id", ""), "")
            yield RealtimeEvent(kind="tool", tool_name=name)
            task = asyncio.create_task(
                self._run_tool(call.get("call_id", ""), name, call.get("arguments", "{}"))
            )
            self._tool_tasks.add(task)
            task.add_done_callback(self._tool_tasks.discard)

    async def _run_tool(self, call_id: str, name: str, arguments: str) -> None:
        """Execute one tool and return its output, then ask the model to respond.

        Sends ``conversation.item.create`` (a ``function_call_output`` carrying the
        JSON result) followed by ``response.create`` — the two-step that hands the
        result back and triggers the follow-up reply. Never raises: dispatch
        already converts failures into ``{"error": ...}`` payloads.
        """
        try:
            args = json.loads(arguments) if arguments else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        result = await self._tools.dispatch(name, args)
        if self._ws is None:
            return
        with contextlib.suppress(Exception):
            await self._ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                },
            }))
            # Only ask for a response if one isn't already generating — a barge-in
            # (or the model auto-responding to the tool output) can leave a
            # response active, and a second response.create errors with
            # ``conversation_already_has_active_response``.
            if not self._response_active:
                self._response_active = True
                await self._ws.send(json.dumps({"type": "response.create"}))


# ── Factory ───────────────────────────────────────────────────────────────────


def realtime_enabled(cfg: Settings) -> bool:
    """Whether realtime (speech-to-speech) mode is configured and usable.

    Requires ``voice_mode == "realtime"`` and an OpenAI key (the realtime model
    is OpenAI-only). Anything else means the endpoint should tell the client to
    use the decomposed ``/voice/stream`` + ``/chat`` path instead.
    """
    return bool(
        getattr(cfg, "voice_mode", "pipeline") == "realtime"
        and getattr(cfg, "openai_api_key", "")
    )


def build_session(cfg: Settings, *, instructions: str, tools: RealtimeTools) -> RealtimeSession:
    """Construct a :class:`RealtimeSession` from settings + a tool dispatcher."""
    return RealtimeSession(
        api_key=cfg.openai_api_key,
        model=getattr(cfg, "realtime_model", "gpt-realtime") or "gpt-realtime",
        vad=getattr(cfg, "realtime_vad", "semantic_vad") or "semantic_vad",
        transcription_model=(
            getattr(cfg, "realtime_transcription_model", "gpt-4o-mini-transcribe")
            or "gpt-4o-mini-transcribe"
        ),
        instructions=instructions,
        tools=tools,
        voice_source=getattr(cfg, "realtime_voice_source", "elevenlabs") or "elevenlabs",
        voice=getattr(cfg, "realtime_voice", "marin") or "marin",
    )


# Type alias documenting the dispatch signature the session depends on.
ToolDispatch = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
