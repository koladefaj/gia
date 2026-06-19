"""Tests for ``POST /chat`` (SSE endpoint)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from backend.app.schemas.chat import ExecutionPlan, IntentType


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


def _plan(intent: IntentType, steps: list[str], signals: list[str] | None = None) -> ExecutionPlan:
    return ExecutionPlan(intent=intent, steps=steps, signals=signals or [], confidence=1.0)


def _dj_response():
    from backend.app.schemas.dj import CrossfadeQueue, DJResponse, TrackItem

    return DJResponse(
        recommendation="Here's Free Mind by Tems.",
        primary_track=TrackItem(uri="spotify:track:001", name="Free Mind", artist="Tems",
                                energy=0.38, valence=0.71, key=5, mode=0),
        queue=CrossfadeQueue(seed_uri="spotify:track:001", tracks=[], crossfade_ms=3000),
        playback_started=False,
    )


@pytest.mark.asyncio
async def test_chat_streams_sse_events(client: AsyncClient) -> None:
    """``POST /chat`` returns a text/event-stream with at least router + done events."""
    with patch("backend.app.api.chat.build_plan",
               new=AsyncMock(return_value=_plan(IntentType.MUSIC_FIND, ["dj"]))), \
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
    with patch("backend.app.api.chat.build_plan",
               new=AsyncMock(return_value=_plan(IntentType.MUSIC_FIND, ["dj"]))), \
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
    with patch("backend.app.api.chat.build_plan",
               new=AsyncMock(return_value=_plan(IntentType.MOOD_CHECK, ["mood"]))), \
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

    with patch("backend.app.api.chat.build_plan",
               new=AsyncMock(return_value=_plan(IntentType.MUSIC_FIND, ["dj"]))), \
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
async def test_chat_weather_signal_fetches_weather(client: AsyncClient) -> None:
    """A plan with a weather signal emits a weather tool_call and signal event."""
    with patch("backend.app.api.chat.build_plan",
               new=AsyncMock(return_value=_plan(IntentType.MUSIC_FIND, ["dj"], ["weather"]))), \
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
