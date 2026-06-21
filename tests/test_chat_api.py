"""Tests for ``POST /chat`` (SSE endpoint)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from backend.app.schemas.chat import IntentType
from backend.app.schemas.router import EngagementMode, RouterDecision, Tone


def _parse_sse(text: str) -> list[dict]:
    """Parse a raw SSE response body into a list of {event, data} dicts."""
    events = []
    current: dict = {}
    for line in text.splitlines():
        if line.startswith("event:"):
            current["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            try:
                current["data"] = json.loads(line[len("data:"):].strip())
            except json.JSONDecodeError:
                current["data"] = {}
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


def _decision(
    intent: IntentType,
    *,
    tone: Tone = Tone.WARM,
    mode: EngagementMode = EngagementMode.DIRECT_EXECUTE,
    needs_music: bool = False,
    needs_artist_lookup: bool = False,
    needs_search: bool = False,
    needs_memory: bool = False,
    search_query: str | None = None,
    start_playback: bool = False,
    confidence: float = 1.0,
) -> RouterDecision:
    """Build a RouterDecision for patching ``classify_turn`` in chat tests.

    Steps are derived in chat.py from intent + needs_*, so the intent alone
    drives DJ/Artist/Mood dispatch the way the old ExecutionPlan steps did.
    """
    return RouterDecision(
        intent=intent, tone=tone, confidence=confidence, engagement_mode=mode,
        needs_music=needs_music, needs_artist_lookup=needs_artist_lookup,
        needs_search=needs_search, needs_memory=needs_memory,
        search_query=search_query, start_playback=start_playback,
    )


def _dj_response():
    from backend.app.schemas.dj import CrossfadeQueue, DJResponse, TrackItem

    return DJResponse(
        recommendation="Here's Free Mind by Tems.",
        primary_track=TrackItem(uri="spotify:track:001", name="Free Mind", artist="Tems"),
        queue=CrossfadeQueue(seed_uri="spotify:track:001", tracks=[], crossfade_ms=3000),
        playback_started=False,
    )


@pytest.mark.asyncio
async def test_chat_streams_sse_events(client: AsyncClient) -> None:
    """``POST /chat`` returns a text/event-stream with at least router + done events."""
    with patch("backend.app.api.chat.classify_turn",
               new=AsyncMock(return_value=_decision(IntentType.MUSIC_FIND))), \
         patch("backend.app.api.chat.DJService") as mock_dj, \
         patch("backend.app.api.chat.synthesize", new=AsyncMock(return_value=b"")), \
         patch("backend.app.api.chat.pop_proactive_draft", new=AsyncMock(return_value=None)):

        mock_instance = MagicMock()
        mock_instance.recommend = AsyncMock(return_value=_dj_response())
        mock_dj.return_value = mock_instance

        response = await client.post("/chat", json={"message": "find me something chill"})

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = _parse_sse(response.text)
    event_names = [e.get("event") for e in events]
    assert "done" in event_names
    # The planner emits a richer plan event alongside the router event.
    assert "plan" in event_names


@pytest.mark.asyncio
async def test_chat_includes_router_event(client: AsyncClient) -> None:
    """The SSE stream includes a ``done`` event carrying the classified intent."""
    with patch("backend.app.api.chat.classify_turn",
               new=AsyncMock(return_value=_decision(IntentType.MUSIC_FIND))), \
         patch("backend.app.api.chat.DJService") as mock_dj, \
         patch("backend.app.api.chat.synthesize", new=AsyncMock(return_value=b"")), \
         patch("backend.app.api.chat.pop_proactive_draft", new=AsyncMock(return_value=None)):

        mock_instance = MagicMock()
        mock_instance.recommend = AsyncMock(return_value=_dj_response())
        mock_dj.return_value = mock_instance

        response = await client.post("/chat", json={"message": "play something chill"})

    events = _parse_sse(response.text)
    done_event = next((e for e in events if e.get("event") == "done"), None)
    assert done_event is not None
    assert done_event["data"]["intent"] == "MUSIC_FIND"


@pytest.mark.asyncio
async def test_chat_mood_check_intent(client: AsyncClient) -> None:
    """MOOD_CHECK plan triggers the mood agent path."""
    with patch("backend.app.api.chat.classify_turn",
               new=AsyncMock(return_value=_decision(IntentType.MOOD_CHECK))), \
         patch("backend.app.api.chat.MoodService") as mock_mood, \
         patch("backend.app.api.chat.synthesize", new=AsyncMock(return_value=b"")), \
         patch("backend.app.api.chat.pop_proactive_draft", new=AsyncMock(return_value=None)):

        from backend.app.agents.mood import MoodResult
        mock_instance = MagicMock()
        mock_instance.analyze = AsyncMock(return_value=MoodResult(
            current_label="wind-down",
            pattern_label="wind-down",
            bucket="sunday_evening",
            deviation=False,
        ))
        mock_mood.return_value = mock_instance

        response = await client.post(
            "/chat",
            json={
                "message": "what's my mood?",
                "user_id": "00000000-0000-0000-0000-000000000001",
            },
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    agent_dones = [e for e in events if e.get("event") == "agent_done"]
    mood_done = next((e for e in agent_dones if e["data"].get("agent") == "mood"), None)
    assert mood_done is not None


@pytest.mark.asyncio
async def test_chat_proactive_draft_surfaced(client: AsyncClient) -> None:
    """A pending proactive draft appears in the done event."""
    proactive = "[thoughtful] You're usually on something softer around Sunday evening."

    with patch("backend.app.api.chat.classify_turn",
               new=AsyncMock(return_value=_decision(IntentType.MUSIC_FIND))), \
         patch("backend.app.api.chat.pop_proactive_draft", new=AsyncMock(return_value=proactive)), \
         patch("backend.app.api.chat.DJService") as mock_dj, \
         patch("backend.app.api.chat.synthesize", new=AsyncMock(return_value=b"")):

        mock_instance = MagicMock()
        mock_instance.recommend = AsyncMock(return_value=_dj_response())
        mock_dj.return_value = mock_instance

        response = await client.post(
            "/chat",
            json={"message": "find me something", "user_id": "00000000-0000-0000-0000-000000000001"},
        )

    events = _parse_sse(response.text)
    done_event = next((e for e in events if e.get("event") == "done"), None)
    assert done_event is not None
    assert done_event["data"]["proactive"] == proactive


@pytest.mark.asyncio
async def test_now_playing_query_reports_track(client: AsyncClient) -> None:
    """'what's currently playing?' reports the real track, not a recommendation."""
    with patch("backend.app.api.chat.classify_turn",
               new=AsyncMock(return_value=_decision(IntentType.MUSIC_FIND))), \
         patch("backend.app.api.chat.DJService") as mock_dj, \
         patch("backend.app.api.chat.synthesize", new=AsyncMock(return_value=b"")), \
         patch("backend.app.api.chat.pop_proactive_draft", new=AsyncMock(return_value=None)):

        response = await client.post("/chat", json={"message": "what's currently playing?"})

    events = _parse_sse(response.text)
    chunks = " ".join(e["data"]["text"] for e in events if e.get("event") == "reply_chunk")
    assert "Right now you're on" in chunks
    # The DJ must NOT run for a status query.
    mock_dj.assert_not_called()
    agent_dones = [e["data"].get("agent") for e in events if e.get("event") == "agent_done"]
    assert "now_playing" in agent_dones
    assert "dj" not in agent_dones


@pytest.mark.asyncio
async def test_dj_uses_resolved_search_query_and_playback(client: AsyncClient) -> None:
    """The DJ searches the router's resolved query and honours start_playback."""
    with patch("backend.app.api.chat.classify_turn",
               new=AsyncMock(return_value=_decision(
                   IntentType.MUSIC_FIND, search_query="Fortworth Drake", start_playback=True))), \
         patch("backend.app.api.chat.DJService") as mock_dj, \
         patch("backend.app.api.chat.synthesize", new=AsyncMock(return_value=b"")), \
         patch("backend.app.api.chat.pop_proactive_draft", new=AsyncMock(return_value=None)):

        mock_instance = MagicMock()
        mock_instance.recommend = AsyncMock(return_value=_dj_response())
        mock_dj.return_value = mock_instance

        await client.post("/chat", json={"message": "just play it now"})

    kwargs = mock_instance.recommend.call_args.kwargs
    assert kwargs["query"] == "Fortworth Drake"
    assert kwargs["start_playback"] is True


@pytest.mark.asyncio
async def test_opening_returns_greeting(client: AsyncClient) -> None:
    """``GET /chat/opening`` returns Gia's opening line for an anonymous visitor."""
    with patch("backend.app.api.chat.opening_line",
               new=AsyncMock(return_value="Hey — what are we listening to?")):
        response = await client.get("/chat/opening")

    assert response.status_code == 200
    assert response.json()["greeting"] == "Hey — what are we listening to?"


@pytest.mark.asyncio
async def test_chat_general_intent_conversational(client: AsyncClient) -> None:
    """A GENERAL turn streams a conversational reply via stream_general."""
    def _fake_stream(*_args, **_kwargs):
        async def _gen():
            yield "Hey you — good to hear from me's friend."
        return _gen()

    with patch("backend.app.api.chat.classify_turn",
               new=AsyncMock(return_value=_decision(IntentType.GENERAL_CHAT))), \
         patch("backend.app.api.chat.stream_general", _fake_stream), \
         patch("backend.app.api.chat.synthesize", new=AsyncMock(return_value=b"")), \
         patch("backend.app.api.chat.pop_proactive_draft", new=AsyncMock(return_value=None)):

        response = await client.post("/chat", json={"message": "hello gia"})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    chunks = [e for e in events if e.get("event") == "reply_chunk"]
    assert any("good to hear" in e["data"]["text"] for e in chunks)


@pytest.mark.asyncio
async def test_fast_ack_precedes_router_decision(client: AsyncClient) -> None:
    """A clear retrieval intent is acknowledged before the router decision lands.

    The keyword classifier recognises "find me something" as MUSIC_FIND, so Gia
    reacts (the ``acknowledgment`` event) ahead of the ``plan`` event rather than
    waiting on the router LLM round-trip.
    """
    with patch("backend.app.api.chat.classify_turn",
               new=AsyncMock(return_value=_decision(IntentType.MUSIC_FIND))), \
         patch("backend.app.api.chat.DJService") as mock_dj, \
         patch("backend.app.api.chat.synthesize", new=AsyncMock(return_value=b"")), \
         patch("backend.app.api.chat.pop_proactive_draft", new=AsyncMock(return_value=None)):

        mock_instance = MagicMock()
        mock_instance.recommend = AsyncMock(return_value=_dj_response())
        mock_dj.return_value = mock_instance

        response = await client.post("/chat", json={"message": "find me something chill"})

    names = [e.get("event") for e in _parse_sse(response.text)]
    assert "acknowledgment" in names
    assert names.index("acknowledgment") < names.index("plan")


def test_fast_ack_skips_ambiguous_mixed_intent() -> None:
    """The fast filler fires for clear retrieval intents, not ambiguous MIXED ones."""
    from backend.app.api.chat import _fast_ack_intent

    # "tell me about X and play something" keyword-classifies to MIXED — the case
    # where the keyword guess most often disagrees with the LLM router — so no
    # fast filler should fire.
    assert _fast_ack_intent("tell me about Tems and play something") is None
    # A clear single retrieval intent still triggers a filler.
    assert _fast_ack_intent("play something chill") is not None


@pytest.mark.asyncio
async def test_greeting_does_not_fast_ack(client: AsyncClient) -> None:
    """A pure greeting carries no retrieval work, so Gia doesn't fire an ack."""
    with patch("backend.app.api.chat.classify_turn",
               new=AsyncMock(return_value=_decision(IntentType.GENERAL_CHAT))), \
         patch("backend.app.api.chat.respond_general",
               new=AsyncMock(return_value="Hey you — good to see you.")), \
         patch("backend.app.api.chat.synthesize", new=AsyncMock(return_value=b"")), \
         patch("backend.app.api.chat.pop_proactive_draft", new=AsyncMock(return_value=None)):

        response = await client.post("/chat", json={"message": "hey there"})

    names = [e.get("event") for e in _parse_sse(response.text)]
    assert "acknowledgment" not in names


@pytest.mark.asyncio
async def test_chat_weather_signal_fetches_weather(client: AsyncClient) -> None:
    """A plan with a weather signal emits a weather tool_call and signal event."""
    with patch("backend.app.api.chat.classify_turn",
               new=AsyncMock(return_value=_decision(IntentType.MUSIC_FIND, needs_music=True))), \
         patch("backend.app.api.chat.DJService") as mock_dj, \
         patch("backend.app.api.chat.synthesize", new=AsyncMock(return_value=b"")), \
         patch("backend.app.api.chat.pop_proactive_draft", new=AsyncMock(return_value=None)):

        mock_instance = MagicMock()
        mock_instance.recommend = AsyncMock(return_value=_dj_response())
        mock_dj.return_value = mock_instance

        response = await client.post("/chat", json={"message": "going for a run, play something"})

    events = _parse_sse(response.text)
    names = [e.get("event") for e in events]
    assert "signal" in names
    signal = next(e for e in events if e.get("event") == "signal")
    assert signal["data"]["name"] == "weather"
    assert "°C" in signal["data"]["value"]
