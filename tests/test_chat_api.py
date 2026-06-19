"""Tests for ``POST /chat`` (SSE endpoint)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient


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


@pytest.mark.asyncio
async def test_chat_streams_sse_events(client: AsyncClient) -> None:
    """``POST /chat`` returns a text/event-stream with at least router + done events."""
    with patch("backend.app.api.chat.classify_intent", new=AsyncMock(return_value=("MUSIC_FIND", 1.0))), \
         patch("backend.app.api.chat.DJService") as mock_dj, \
         patch("backend.app.api.chat.synthesize", new=AsyncMock(return_value=b"")), \
         patch("backend.app.api.chat.pop_proactive_draft", new=AsyncMock(return_value=None)):

        from backend.app.schemas.chat import IntentType
        from backend.app.schemas.dj import CrossfadeQueue, DJResponse, TrackItem

        mock_instance = MagicMock()
        mock_instance.recommend = AsyncMock(return_value=DJResponse(
            recommendation="Here's Free Mind by Tems.",
            primary_track=TrackItem(uri="spotify:track:001", name="Free Mind", artist="Tems",
                                    energy=0.38, valence=0.71, key=5, mode=0),
            queue=CrossfadeQueue(seed_uri="spotify:track:001", tracks=[], crossfade_ms=3000),
            playback_started=False,
        ))
        mock_dj.return_value = mock_instance

        with patch("backend.app.api.chat.classify_intent",
                   new=AsyncMock(return_value=(IntentType.MUSIC_FIND, 1.0))):
            response = await client.post(
                "/chat",
                json={"message": "find me something chill"},
            )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = _parse_sse(response.text)
    event_names = [e.get("event") for e in events]
    assert "done" in event_names


@pytest.mark.asyncio
async def test_chat_includes_router_event(client: AsyncClient) -> None:
    """The SSE stream includes an ``agent_done`` event from the router."""
    with patch("backend.app.api.chat.classify_intent", new=AsyncMock(return_value=(__import__("backend.app.schemas.chat", fromlist=["IntentType"]).IntentType.MUSIC_FIND, 1.0))), \
         patch("backend.app.api.chat.DJService") as mock_dj, \
         patch("backend.app.api.chat.synthesize", new=AsyncMock(return_value=b"")), \
         patch("backend.app.api.chat.pop_proactive_draft", new=AsyncMock(return_value=None)):

        from backend.app.schemas.dj import CrossfadeQueue, DJResponse, TrackItem
        mock_instance = MagicMock()
        mock_instance.recommend = AsyncMock(return_value=DJResponse(
            recommendation="Chill pick.",
            primary_track=TrackItem(uri="x", name="X", artist="Y", energy=0.5, valence=0.5, key=0, mode=1),
            queue=CrossfadeQueue(seed_uri="x", tracks=[], crossfade_ms=3000),
            playback_started=False,
        ))
        mock_dj.return_value = mock_instance

        response = await client.post("/chat", json={"message": "play something chill"})

    events = _parse_sse(response.text)
    done_event = next((e for e in events if e.get("event") == "done"), None)
    assert done_event is not None
    assert done_event["data"]["intent"] == "MUSIC_FIND"


@pytest.mark.asyncio
async def test_chat_mood_check_intent(client: AsyncClient) -> None:
    """MOOD_CHECK intent triggers the mood agent path."""
    with patch("backend.app.api.chat.classify_intent", new=AsyncMock(
        return_value=(__import__("backend.app.schemas.chat", fromlist=["IntentType"]).IntentType.MOOD_CHECK, 1.0)
    )), \
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
    """A pending proactive draft appears in the reply_chunk stream."""
    proactive = "[thoughtful] You're usually on something softer around Sunday evening."

    with patch("backend.app.api.chat.classify_intent", new=AsyncMock(
        return_value=(__import__("backend.app.schemas.chat", fromlist=["IntentType"]).IntentType.MUSIC_FIND, 1.0)
    )), \
    patch("backend.app.api.chat.pop_proactive_draft", new=AsyncMock(return_value=proactive)), \
    patch("backend.app.api.chat.DJService") as mock_dj, \
    patch("backend.app.api.chat.synthesize", new=AsyncMock(return_value=b"")):

        from backend.app.schemas.dj import CrossfadeQueue, DJResponse, TrackItem
        mock_instance = MagicMock()
        mock_instance.recommend = AsyncMock(return_value=DJResponse(
            recommendation="Here's something chill.",
            primary_track=TrackItem(uri="x", name="X", artist="Y", energy=0.3, valence=0.7, key=0, mode=1),
            queue=CrossfadeQueue(seed_uri="x", tracks=[], crossfade_ms=3000),
            playback_started=False,
        ))
        mock_dj.return_value = mock_instance

        response = await client.post(
            "/chat",
            json={"message": "find me something", "user_id": "00000000-0000-0000-0000-000000000001"},
        )

    events = _parse_sse(response.text)
    done_event = next((e for e in events if e.get("event") == "done"), None)
    assert done_event is not None
    assert done_event["data"]["proactive"] == proactive
