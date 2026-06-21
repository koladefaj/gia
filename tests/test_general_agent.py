"""Tests for the General agent — Gia's conversational non-specialist replies."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.agents.general import (
    _FALLBACK_OPENINGS,
    _FALLBACK_REPLIES,
    opening_line,
    respond_general,
    stream_general,
)


def _fake_llm(text: str | Exception) -> MagicMock:
    """Build a fake LLM whose ``.call`` returns *text* or raises it."""
    llm = MagicMock()
    if isinstance(text, Exception):
        llm.call.side_effect = text
    else:
        llm.call.return_value = text
    return llm


def _chunk(content: str | None) -> SimpleNamespace:
    """Mimic one OpenAI streaming chunk: ``event.choices[0].delta.content``."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


async def _fake_stream(*parts: str | None):
    for part in parts:
        yield _chunk(part)


def _openai_client(stream) -> MagicMock:
    """Build a fake AsyncOpenAI whose chat completion returns *stream*."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=stream)
    return client


class _FakeAnthropicStream:
    """Mimic the ``client.messages.stream(...)`` async context manager."""

    def __init__(self, parts: tuple[str, ...]) -> None:
        self._parts = parts

    async def __aenter__(self) -> _FakeAnthropicStream:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    @property
    def text_stream(self):
        async def _gen():
            for part in self._parts:
                yield part
        return _gen()


def _anthropic_client(*parts: str) -> MagicMock:
    """Build a fake AsyncAnthropic whose ``messages.stream`` yields *parts*."""
    client = MagicMock()
    client.messages.stream = MagicMock(return_value=_FakeAnthropicStream(parts))
    return client


@pytest.mark.asyncio
async def test_respond_general_returns_llm_text(test_settings) -> None:
    with patch("backend.app.agents.general.get_llm", return_value=_fake_llm("Hey Kolade!")):
        reply = await respond_general("hi", "", cfg=test_settings)
    assert reply == "Hey Kolade!"


@pytest.mark.asyncio
async def test_respond_general_falls_back_on_error(test_settings) -> None:
    with patch("backend.app.agents.general.get_llm", return_value=_fake_llm(RuntimeError("down"))):
        reply = await respond_general("hi", "", cfg=test_settings)
    assert reply in _FALLBACK_REPLIES


@pytest.mark.asyncio
async def test_respond_general_falls_back_on_empty(test_settings) -> None:
    with patch("backend.app.agents.general.get_llm", return_value=_fake_llm("   ")):
        reply = await respond_general("hi", "", cfg=test_settings)
    assert reply in _FALLBACK_REPLIES


@pytest.mark.asyncio
async def test_respond_general_injects_user_context(test_settings) -> None:
    fake = _fake_llm("Welcome back.")
    with patch("backend.app.agents.general.get_llm", return_value=fake):
        await respond_general("hi", "Name: Kolade", cfg=test_settings)
    # The rendered prompt passed to the LLM must carry the user context.
    prompt = fake.call.call_args[0][0][0]["content"]
    assert "Kolade" in prompt


@pytest.mark.asyncio
async def test_stream_general_streams_openai_deltas(test_settings) -> None:
    cfg = test_settings.model_copy(update={"llm_provider": "openai"})
    stream = _fake_stream("Hey", " there", ".")
    with patch("backend.app.agents.general.get_async_openai", return_value=_openai_client(stream)):
        deltas = [d async for d in stream_general("hi", "", cfg=cfg)]
    assert "".join(deltas) == "Hey there."


@pytest.mark.asyncio
async def test_stream_general_streams_anthropic_deltas(test_settings) -> None:
    # test_settings is anthropic → stream via the Anthropic Messages API helper.
    with patch("backend.app.agents.general.get_async_anthropic",
               return_value=_anthropic_client("Hey", " there", ".")):
        deltas = [d async for d in stream_general("hi", "", cfg=test_settings)]
    assert "".join(deltas) == "Hey there."


@pytest.mark.asyncio
async def test_stream_general_falls_back_for_unsupported_provider(test_settings) -> None:
    # An unsupported provider has no streamer → degrade to one blocking chunk.
    cfg = test_settings.model_copy(update={"llm_provider": "ollama"})
    with patch("backend.app.agents.general.respond_general",
               new=AsyncMock(return_value="Blocking reply.")):
        deltas = [d async for d in stream_general("hi", "", cfg=cfg)]
    assert deltas == ["Blocking reply."]


@pytest.mark.asyncio
async def test_stream_general_falls_back_when_stream_empty(test_settings) -> None:
    # An OpenAI stream that yields no content must still produce a reply.
    cfg = test_settings.model_copy(update={"llm_provider": "openai"})
    stream = _fake_stream(None, "")
    with patch("backend.app.agents.general.get_async_openai", return_value=_openai_client(stream)), \
         patch("backend.app.agents.general.respond_general",
               new=AsyncMock(return_value="Fallback reply.")):
        deltas = [d async for d in stream_general("hi", "", cfg=cfg)]
    assert deltas == ["Fallback reply."]


@pytest.mark.asyncio
async def test_opening_line_returns_llm_text(test_settings) -> None:
    with patch("backend.app.agents.general.get_llm", return_value=_fake_llm("Hey you.")):
        line = await opening_line("", cfg=test_settings)
    assert line == "Hey you."


@pytest.mark.asyncio
async def test_opening_line_falls_back_on_error(test_settings) -> None:
    with patch("backend.app.agents.general.get_llm", return_value=_fake_llm(RuntimeError("down"))):
        line = await opening_line("", cfg=test_settings)
    assert line in _FALLBACK_OPENINGS
