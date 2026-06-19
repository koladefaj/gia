"""Langfuse AI tracing utilities.

Provides a lightweight context manager for tracing crew runs with per-agent
spans.  When Langfuse is not configured (keys absent) all calls are no-ops
so the application runs identically in dev without a Langfuse account.

Usage::

    async with crew_trace(session_id, user_id) as trace:
        with trace.span("router") as span:
            intent, _ = await classify_intent(message, cfg)
            span.set_output(intent.value)

        with trace.span("dj") as span:
            result = await dj_service.recommend(query)
            span.set_output(result.recommendation)
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

_langfuse_client: Any = None


def init_langfuse(public_key: str, secret_key: str, host: str) -> None:
    """Initialise the Langfuse SDK client.

    Called once at application startup when both keys are present.  Idempotent —
    calling again with the same keys is harmless.

    Args:
        public_key: Langfuse public key (``LANGFUSE_PUBLIC_KEY``).
        secret_key: Langfuse secret key (``LANGFUSE_SECRET_KEY``).
        host:       Langfuse host URL.
    """
    global _langfuse_client  # noqa: PLW0603
    try:
        from langfuse import Langfuse  # type: ignore[import-untyped]

        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info("langfuse_initialised", host=host)
    except ImportError:
        logger.warning("langfuse_not_installed", hint="pip install langfuse")
    except Exception as exc:  # noqa: BLE001
        logger.warning("langfuse_init_failed", error=str(exc))


@dataclass
class AgentSpan:
    """A single agent span within a crew trace.

    Collects input/output and timing.  When Langfuse is active, the span is
    sent to the Langfuse API.  When Langfuse is inactive the data is available
    locally for test assertions and the ``agent_traces`` field in ``ChatResponse``.

    Attributes:
        agent:      Agent name (e.g. ``"router"``, ``"dj"``).
        input:      Input summary stored at span start.
        output:     Output summary set via ``set_output()``.
        latency_ms: Wall-clock time from span open to close.
    """

    agent: str
    input: str = ""
    output: str = ""
    latency_ms: float = 0.0
    _start: float = field(default_factory=time.monotonic, repr=False)
    _lf_span: Any = field(default=None, repr=False)

    def set_output(self, output: str) -> None:
        """Record the agent's output.

        Args:
            output: String summary (truncated to 500 chars for Langfuse).
        """
        self.output = output
        if self._lf_span is not None:
            try:
                self._lf_span.update(output=output[:500])
            except Exception:  # noqa: BLE001
                pass

    def _close(self) -> None:
        """Finalise latency and close the Langfuse span."""
        self.latency_ms = (time.monotonic() - self._start) * 1000
        if self._lf_span is not None:
            try:
                self._lf_span.end()
            except Exception:  # noqa: BLE001
                pass


@dataclass
class CrewTrace:
    """Container for all spans in a single crew run.

    Attributes:
        session_id: Conversation session identifier.
        user_id:    User UUID string.
        spans:      Ordered list of spans collected during the run.
    """

    session_id: str
    user_id: str | None
    spans: list[AgentSpan] = field(default_factory=list)
    _lf_trace: Any = field(default=None, repr=False)

    @contextmanager
    def span(self, agent: str, input_text: str = "") -> Generator[AgentSpan, None, None]:
        """Open a span for *agent*, yield it, then close it.

        Args:
            agent:      Agent name.
            input_text: Optional input summary to record upfront.

        Yields:
            The ``AgentSpan`` instance.
        """
        lf_span = None
        if self._lf_trace is not None:
            try:
                lf_span = self._lf_trace.span(
                    name=agent,
                    input=input_text[:500] if input_text else "",
                )
            except Exception:  # noqa: BLE001
                pass

        s = AgentSpan(agent=agent, input=input_text, _lf_span=lf_span)
        try:
            yield s
        finally:
            s._close()
            self.spans.append(s)

    def flush(self) -> None:
        """Flush buffered events to Langfuse (best-effort, non-blocking)."""
        if _langfuse_client is not None:
            try:
                _langfuse_client.flush()
            except Exception:  # noqa: BLE001
                pass


@asynccontextmanager
async def crew_trace(session_id: str, user_id: str | None = None):
    """Async context manager that opens a Langfuse trace for a crew run.

    Args:
        session_id: Conversation session identifier.
        user_id:    Optional user UUID string.

    Yields:
        A ``CrewTrace`` instance.  On exit the trace is flushed to Langfuse.
    """
    lf_trace = None
    if _langfuse_client is not None:
        try:
            lf_trace = _langfuse_client.trace(
                name="gia_crew_run",
                session_id=session_id,
                user_id=user_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("langfuse_trace_open_failed", error=str(exc))

    trace = CrewTrace(session_id=session_id, user_id=user_id, _lf_trace=lf_trace)
    try:
        yield trace
    finally:
        trace.flush()
