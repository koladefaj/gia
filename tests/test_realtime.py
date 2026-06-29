"""Tests for the speech-to-speech (OpenAI Realtime) bridge.

The realtime path is a hybrid: gpt-realtime is audio-in / **text-out** (the voice
is ElevenLabs), so these cover:
  1. the GA-shaped ``session.update`` — text-only output, the nested + object
     audio-input format that silently break against the old beta shape,
  2. tool dispatch onto the existing services (and its never-raise contract),
  3. event normalisation (text deltas, user transcript, barge-in) + the
     function-call round-trip over a faked socket.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.providers.realtime import (
    SAMPLE_RATE,
    TOOL_SCHEMAS,
    RealtimeSession,
    RealtimeTools,
    build_session,
    realtime_enabled,
)


# ── Fakes ────────────────────────────────────────────────────────────────────


class FakeWS:
    """A stand-in for the OpenAI Realtime WebSocket.

    Records everything sent (so we can assert the session config + tool replies)
    and replays a scripted list of incoming JSON frames to ``async for``.
    """

    def __init__(self, incoming: list[dict] | None = None) -> None:
        self._incoming = [json.dumps(m) for m in (incoming or [])]
        self.sent: list[dict] = []

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self) -> None:
        pass

    def __aiter__(self) -> FakeWS:
        return self

    async def __anext__(self) -> str:
        if not self._incoming:
            raise StopAsyncIteration
        return self._incoming.pop(0)


def _cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.weather_default_lat = 6.5
    cfg.weather_default_lon = 3.3
    cfg.weather_default_label = "Lagos"
    return cfg


def _tools(**overrides) -> RealtimeTools:
    defaults = dict(
        cfg=_cfg(), spotify=MagicMock(), brave=MagicMock(), store=None,
        db=MagicMock(), redis=MagicMock(), weather=MagicMock(), user_id=None,
    )
    defaults.update(overrides)
    return RealtimeTools(**defaults)


def _session(
    tools: RealtimeTools | None = None, *, voice_source: str = "elevenlabs",
) -> RealtimeSession:
    return RealtimeSession(
        api_key="sk", model="gpt-realtime", vad="semantic_vad",
        transcription_model="gpt-4o-mini-transcribe", instructions="be Gia",
        tools=tools or _tools(), voice_source=voice_source, voice="marin",
    )


# ── realtime_enabled ─────────────────────────────────────────────────────────


def test_realtime_enabled_requires_mode_and_key() -> None:
    assert realtime_enabled(MagicMock(voice_mode="realtime", openai_api_key="sk")) is True
    assert realtime_enabled(MagicMock(voice_mode="pipeline", openai_api_key="sk")) is False
    assert realtime_enabled(MagicMock(voice_mode="realtime", openai_api_key="")) is False


# ── Tool schemas / dispatch contract ─────────────────────────────────────────


def test_every_schema_has_a_handler() -> None:
    """The advertised tools and the dispatch table must not drift apart."""
    advertised = {s["name"] for s in TOOL_SCHEMAS}
    assert advertised == {
        "search_and_play_music", "get_web_info", "recall_memory",
        "get_now_playing", "get_weather",
    }


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_returns_error() -> None:
    result = await _tools().dispatch("nope", {})
    assert "error" in result


@pytest.mark.asyncio
async def test_dispatch_never_raises() -> None:
    """A throwing tool yields an ``error`` payload, not an exception."""
    spotify = MagicMock()
    spotify.get_currently_playing = AsyncMock(side_effect=RuntimeError("spotify down"))
    result = await _tools(spotify=spotify).dispatch("get_now_playing", {})
    assert result == {"error": "spotify down"}


@pytest.mark.asyncio
async def test_search_and_play_music_starts_and_queues() -> None:
    spotify = MagicMock()
    spotify.search_tracks = AsyncMock(return_value=[
        {"uri": "u1", "name": "Seed", "artist": "X"},
        {"uri": "u2", "name": "Next", "artist": "Y"},
        {"uri": "u3", "name": "Third", "artist": "Z"},
    ])
    spotify.start_playback = AsyncMock()
    spotify.add_to_queue = AsyncMock()

    result = await _tools(spotify=spotify).dispatch(
        "search_and_play_music",
        {"query": "afrobeats", "start_playback": True, "queue_more": True},
    )

    assert result["playing"] is True
    assert result["track"] == {"name": "Seed", "artist": "X"}
    spotify.start_playback.assert_awaited_once_with("u1")
    # Seed is already playing, so only the rest gets queued.
    assert spotify.add_to_queue.await_count == 2
    assert {q["name"] for q in result["queued"]} == {"Next", "Third"}


@pytest.mark.asyncio
async def test_search_prefers_direct_web_api() -> None:
    """When a Web client is present, search uses it and skips the MCP search."""
    spotify = MagicMock()
    spotify.search_tracks = AsyncMock()  # MCP path — must NOT be called
    web = MagicMock()
    web.search_tracks = AsyncMock(return_value=[{"uri": "w1", "name": "Fast", "artist": "A"}])

    result = await _tools(spotify=spotify, spotify_web=web).dispatch(
        "search_and_play_music", {"query": "afrobeats"}
    )

    web.search_tracks.assert_awaited_once()
    spotify.search_tracks.assert_not_called()
    assert result["track"] == {"name": "Fast", "artist": "A"}


@pytest.mark.asyncio
async def test_search_falls_back_to_mcp_when_web_fails() -> None:
    """A failing Web search falls back to the MCP search rather than erroring."""
    spotify = MagicMock()
    spotify.search_tracks = AsyncMock(return_value=[{"uri": "m1", "name": "Mcp", "artist": "B"}])
    web = MagicMock()
    web.search_tracks = AsyncMock(side_effect=RuntimeError("web 500"))

    result = await _tools(spotify=spotify, spotify_web=web).dispatch(
        "search_and_play_music", {"query": "afrobeats"}
    )

    spotify.search_tracks.assert_awaited()  # fell back to MCP
    assert result["track"] == {"name": "Mcp", "artist": "B"}


@pytest.mark.asyncio
async def test_search_without_playback_does_not_play() -> None:
    spotify = MagicMock()
    spotify.search_tracks = AsyncMock(return_value=[{"uri": "u1", "name": "S", "artist": "A"}])
    spotify.start_playback = AsyncMock()

    result = await _tools(spotify=spotify).dispatch("search_and_play_music", {"query": "x"})

    assert result["playing"] is False
    spotify.start_playback.assert_not_called()


@pytest.mark.asyncio
async def test_get_web_info_shapes_brave_results() -> None:
    brave = MagicMock()
    brave.recent = AsyncMock(return_value=[
        {"title": "T", "age": "1h", "description": " d ", "url": "http://e"},
    ])
    result = await _tools(brave=brave).dispatch("get_web_info", {"query": "drake", "breaking": True})
    brave.recent.assert_awaited_once_with("drake", count=5, breaking=True)
    assert result["results"][0] == {"title": "T", "age": "1h", "summary": "d", "url": "http://e"}


@pytest.mark.asyncio
async def test_recall_memory_without_user_is_noop() -> None:
    result = await _tools(user_id=None, store=None).dispatch("recall_memory", {"topic": "x"})
    assert result["context"] == ""


@pytest.mark.asyncio
async def test_get_weather_rounds_and_labels() -> None:
    weather = MagicMock()
    weather.get_current = AsyncMock(return_value={"temperature_c": 30.7, "condition": "clear"})
    result = await _tools(weather=weather).dispatch("get_weather", {})
    assert result == {
        "available": True, "location": "Lagos", "temperature_c": 31, "condition": "clear",
    }


# ── Session config (GA shape, text-out) ──────────────────────────────────────


@pytest.mark.asyncio
async def test_session_update_is_text_out_ga_shape() -> None:
    ws = FakeWS()
    with patch("backend.app.providers.realtime.websockets.connect", new=AsyncMock(return_value=ws)):
        async with _session():
            pass

    update = next(m for m in ws.sent if m["type"] == "session.update")
    sess = update["session"]
    assert sess["type"] == "realtime"
    # Voice comes from ElevenLabs, so the model is TEXT-only and has no audio.output.
    assert sess["output_modalities"] == ["text"]
    assert "output" not in sess["audio"]
    # The two GA gotchas: nested audio.input, and an OBJECT audio format.
    assert sess["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": SAMPLE_RATE}
    assert sess["audio"]["input"]["turn_detection"] == {"type": "semantic_vad"}
    assert sess["audio"]["input"]["transcription"] == {"model": "gpt-4o-mini-transcribe"}
    assert sess["instructions"] == "be Gia"
    assert len(sess["tools"]) == len(TOOL_SCHEMAS)


def test_build_session_pulls_from_config() -> None:
    cfg = MagicMock(
        openai_api_key="sk", realtime_model="gpt-realtime-2",
        realtime_vad="server_vad", realtime_transcription_model="whisper-x",
        realtime_voice_source="model", realtime_voice="cedar",
    )
    session = build_session(cfg, instructions="hi", tools=_tools())
    assert session._model == "gpt-realtime-2"
    assert session._vad == "server_vad"
    assert session._transcription_model == "whisper-x"
    assert session._voice_source == "model"
    assert session._voice == "cedar"


@pytest.mark.asyncio
async def test_session_update_model_mode_is_audio_out() -> None:
    """Model voice source → audio-out modality + audio.output voice."""
    ws = FakeWS()
    with patch("backend.app.providers.realtime.websockets.connect", new=AsyncMock(return_value=ws)):
        async with _session(voice_source="model"):
            pass

    sess = next(m for m in ws.sent if m["type"] == "session.update")["session"]
    assert sess["output_modalities"] == ["audio"]
    assert sess["audio"]["output"] == {
        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE}, "voice": "marin",
    }
    # Input transcription stays on so user words still feed captions + memory.
    assert sess["audio"]["input"]["transcription"] == {"model": "gpt-4o-mini-transcribe"}


@pytest.mark.asyncio
async def test_events_model_mode_emits_audio_and_transcript() -> None:
    """Model voice source surfaces audio chunks + the assistant transcript."""
    incoming = [
        {"type": "response.created"},
        {"type": "response.output_audio.delta", "delta": "QUJD"},
        {"type": "response.output_audio_transcript.delta", "delta": "Here's "},
        {"type": "response.output_audio_transcript.done", "transcript": "Here's some jazz."},
        {"type": "response.done", "response": {"output": []}},
    ]
    ws = FakeWS(incoming)
    with patch("backend.app.providers.realtime.websockets.connect", new=AsyncMock(return_value=ws)):
        async with _session(voice_source="model") as session:
            events = [(ev.kind, ev.text or ev.audio_b64) async for ev in session.events()]

    assert ("audio", "QUJD") in events
    assert ("assistant_delta", "Here's ") in events
    assert ("assistant_text", "Here's some jazz.") in events


# ── Event normalisation + tool round-trip ────────────────────────────────────


@pytest.mark.asyncio
async def test_events_normalise_text_transcript_and_barge_in() -> None:
    incoming = [
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": "play jazz"},
        {"type": "response.created"},
        {"type": "response.output_text.delta", "delta": "Here's "},
        {"type": "input_audio_buffer.speech_started"},  # barge-in mid-reply
        {"type": "response.output_text.done", "text": "Here's some jazz."},
        {"type": "response.done", "response": {"output": []}},
    ]
    ws = FakeWS(incoming)
    with patch("backend.app.providers.realtime.websockets.connect", new=AsyncMock(return_value=ws)):
        async with _session() as session:
            kinds = [(ev.kind, ev.text) async for ev in session.events()]

    assert kinds == [
        ("user_transcript", "play jazz"),
        ("assistant_delta", "Here's "),
        ("speech_started", ""),
        ("assistant_text", "Here's some jazz."),
        ("response_done", ""),
    ]
    # Barge-in during an active response must cancel it.
    assert any(m["type"] == "response.cancel" for m in ws.sent)


@pytest.mark.asyncio
async def test_barge_in_without_active_response_does_not_cancel() -> None:
    """A speech_started with no response generating must NOT send response.cancel."""
    ws = FakeWS([{"type": "input_audio_buffer.speech_started"}])
    with patch("backend.app.providers.realtime.websockets.connect", new=AsyncMock(return_value=ws)):
        async with _session() as session:
            _ = [ev async for ev in session.events()]
    assert not any(m["type"] == "response.cancel" for m in ws.sent)


@pytest.mark.asyncio
async def test_function_call_round_trip() -> None:
    """A function_call in response.done is dispatched and its output sent back."""
    spotify = MagicMock()
    spotify.get_currently_playing = AsyncMock(return_value={"name": "Song", "artist": "Artist"})
    incoming = [
        {"type": "response.output_item.added",
         "item": {"type": "function_call", "name": "get_now_playing", "call_id": "c1"}},
        {"type": "response.done", "response": {"output": [
            {"type": "function_call", "name": "get_now_playing", "call_id": "c1", "arguments": "{}"},
        ]}},
    ]
    ws = FakeWS(incoming)
    with patch("backend.app.providers.realtime.websockets.connect", new=AsyncMock(return_value=ws)):
        async with _session(_tools(spotify=spotify)) as session:
            events = [ev async for ev in session.events()]
            await asyncio.gather(*session._tool_tasks)  # let the bg tool finish

    assert any(ev.kind == "tool" and ev.tool_name == "get_now_playing" for ev in events)
    output = next(m for m in ws.sent if m["type"] == "conversation.item.create")
    assert output["item"]["type"] == "function_call_output"
    assert output["item"]["call_id"] == "c1"
    assert json.loads(output["item"]["output"]) == {
        "playing": True, "name": "Song", "artist": "Artist",
    }
    assert any(m["type"] == "response.create" for m in ws.sent)
