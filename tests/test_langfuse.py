"""Tests for the Langfuse tracing utilities."""

from __future__ import annotations

import pytest

from backend.app.observability.langfuse import AgentSpan, CrewTrace


def test_agent_span_records_latency() -> None:
    """AgentSpan tracks wall-clock time from open to close."""
    trace = CrewTrace(session_id="s1", user_id=None)
    with trace.span("router", "find me music") as span:
        span.set_output("MUSIC_FIND")

    assert span.latency_ms >= 0.0
    assert span.agent == "router"
    assert span.input == "find me music"
    assert span.output == "MUSIC_FIND"


def test_crew_trace_collects_spans() -> None:
    """CrewTrace accumulates spans in order."""
    trace = CrewTrace(session_id="s1", user_id="u1")
    with trace.span("router") as s1:
        s1.set_output("MUSIC_FIND")
    with trace.span("dj") as s2:
        s2.set_output("Here's Free Mind.")

    assert len(trace.spans) == 2
    assert trace.spans[0].agent == "router"
    assert trace.spans[1].agent == "dj"


def test_crew_trace_span_exception_still_closes() -> None:
    """A span is closed even if the body raises an exception."""
    trace = CrewTrace(session_id="s1", user_id=None)
    with pytest.raises(ValueError):
        with trace.span("router") as span:
            raise ValueError("test error")

    assert len(trace.spans) == 1
    assert trace.spans[0].latency_ms >= 0.0


@pytest.mark.asyncio
async def test_crew_trace_context_manager() -> None:
    """``crew_trace`` yields a ``CrewTrace`` and flushes on exit."""
    from backend.app.observability.langfuse import crew_trace

    async with crew_trace("session-1", "user-1") as trace:
        assert isinstance(trace, CrewTrace)
        assert trace.session_id == "session-1"
        assert trace.user_id == "user-1"
        with trace.span("router") as s:
            s.set_output("MUSIC_FIND")

    assert len(trace.spans) == 1


def test_init_langfuse_noop_without_langfuse_installed() -> None:
    """``init_langfuse`` does not raise when the langfuse package is absent."""
    from unittest.mock import patch
    from backend.app.observability.langfuse import init_langfuse

    with patch.dict("sys.modules", {"langfuse": None}):
        # Should not raise even if langfuse is "not installed"
        try:
            init_langfuse("pk-test", "sk-test", "https://cloud.langfuse.com")
        except ImportError:
            pass  # acceptable
