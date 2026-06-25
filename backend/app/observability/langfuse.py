"""Langfuse AI tracing utilities (Langfuse SDK v4, OpenTelemetry-based).

Provides a lightweight context manager for tracing a chat turn with per-agent
spans.  When Langfuse is not configured (keys absent) every call is a no-op so
the application runs identically in dev without a Langfuse account — the local
``AgentSpan`` bookkeeping still works, so the ``agent_traces`` field in the
``done`` SSE event and the unit tests are unaffected.

The v4 SDK is OpenTelemetry-based: observations are created with
``start_as_current_observation(as_type=...)`` and nest automatically via OTel
context propagation.  Trace-level attributes (session/user/name/tags) are
applied with ``propagate_attributes``.  Direct OpenAI calls are traced
separately via the ``langfuse.openai`` drop-in (see ``providers/openai_client``);
those generations nest under whichever agent span is active.

Usage::

    async with crew_trace(session_id, user_id, user_input=message) as trace:
        with trace.span("router", message) as span:
            decision = await classify_turn(message, cfg)   # OpenAI drop-in
            span.set_output(decision.intent.value)         # nests under "router"
        ...
        trace.set_output(full_reply)
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any

from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

# Process-wide Langfuse client.  ``None`` until ``init_langfuse`` succeeds with
# real credentials, which is also the signal that tracing is active.
_client: Any = None


def init_langfuse(public_key: str, secret_key: str, host: str) -> None:
    """Initialise the global Langfuse SDK client (v4).

    Called once at application startup when both keys are present.  Constructing
    ``Langfuse(...)`` registers the process-wide client that ``get_client()`` and
    the ``langfuse.openai`` drop-in both pick up automatically.

    Args:
        public_key: Langfuse public key (``LANGFUSE_PUBLIC_KEY``).
        secret_key: Langfuse secret key (``LANGFUSE_SECRET_KEY``).
        host:       Langfuse host URL (``LANGFUSE_HOST``).
    """
    global _client  # noqa: PLW0603
    try:
        from langfuse import Langfuse  # noqa: PLC0415

        _client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        logger.info("langfuse_initialised", host=host)
    except ImportError:
        logger.warning("langfuse_not_installed", hint="pip install 'langfuse>=4'")
    except Exception as exc:  # noqa: BLE001
        logger.warning("langfuse_init_failed", error=str(exc))


@dataclass
class AgentSpan:
    """A single agent span within a chat-turn trace.

    Collects input/output and timing.  When Langfuse is active the span is
    backed by a live observation (``_lf_obs``) that receives the same output;
    when inactive the data is available locally for test assertions and the
    ``agent_traces`` field in ``ChatResponse``.

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
    _lf_obs: Any = field(default=None, repr=False)

    def set_output(self, output: str) -> None:
        """Record the agent's output (mirrored to Langfuse, truncated).

        Args:
            output: String summary (truncated to 1000 chars for Langfuse).
        """
        self.output = output
        if self._lf_obs is not None:
            with suppress(Exception):
                self._lf_obs.update(output=output[:1000])

    def _close(self) -> None:
        """Finalise wall-clock latency (the observation is closed by ``span``)."""
        self.latency_ms = (time.monotonic() - self._start) * 1000


@dataclass
class CrewTrace:
    """Container for all spans in a single chat turn.

    Attributes:
        session_id: Conversation session identifier.
        user_id:    User UUID string.
        spans:      Ordered list of spans collected during the turn.
    """

    session_id: str
    user_id: str | None
    spans: list[AgentSpan] = field(default_factory=list)
    _active: bool = field(default=False, repr=False)
    _root: Any = field(default=None, repr=False)

    @contextmanager
    def span(self, agent: str, input_text: str = "") -> Generator[AgentSpan, None, None]:
        """Open a child observation for *agent*, yield a span, then close it.

        Any Langfuse-instrumented call made inside the block (e.g. the
        ``langfuse.openai`` drop-in) nests under this observation automatically.

        Args:
            agent:      Agent name → observation name.
            input_text: Optional input summary recorded upfront.

        Yields:
            The ``AgentSpan`` instance.
        """
        s = AgentSpan(agent=agent, input=input_text)
        if self._active and _client is not None:
            try:
                with _client.start_as_current_observation(
                    as_type="span",
                    name=agent,
                    input=input_text[:1000] if input_text else None,
                ) as obs:
                    s._lf_obs = obs
                    try:
                        yield s
                    finally:
                        s._close()
                        self.spans.append(s)
                return
            except Exception as exc:  # noqa: BLE001
                logger.debug("langfuse_span_failed", agent=agent, error=str(exc))

        # Tracing disabled or span creation failed — local bookkeeping only.
        try:
            yield s
        finally:
            s._close()
            self.spans.append(s)

    def set_output(self, output: str) -> None:
        """Set the trace-level output (mirrored onto the root observation)."""
        if self._root is not None:
            with suppress(Exception):
                self._root.update(output=output[:2000])

    def score(
        self,
        name: str,
        value: float | str,
        *,
        data_type: str | None = None,
        comment: str | None = None,
    ) -> None:
        """Attach a self-evaluation score to this turn's trace (best-effort).

        Lightweight, deterministic feedback signals — was retrieved context used,
        the router's confidence, end-to-end latency — logged per turn so quality
        and cost are measurable over time, not guessed. No-op when tracing is off.

        Args:
            name:      Score name (e.g. ``"context_used"``).
            value:     Numeric/boolean/categorical value.
            data_type: Optional Langfuse data type (``NUMERIC`` / ``BOOLEAN`` / …).
            comment:   Optional human-readable note.
        """
        if not (self._active and _client is not None):
            return
        try:
            _client.score_current_trace(
                name=name, value=value, data_type=data_type, comment=comment
            )
        except Exception:  # noqa: BLE001
            logger.debug("langfuse_score_failed", name=name)

    def flush(self) -> None:
        """Flush buffered events to Langfuse (best-effort, non-blocking)."""
        if _client is not None:
            with suppress(Exception):
                _client.flush()


@asynccontextmanager
async def crew_trace(
    session_id: str,
    user_id: str | None = None,
    user_input: str | None = None,
):
    """Async context manager that opens a Langfuse trace for one chat turn.

    Opens a root ``gia-chat-turn`` observation and applies ``session_id`` /
    ``user_id`` / trace name / tags via ``propagate_attributes`` so every nested
    agent span and LLM generation inherits them.  On exit, buffered events are
    flushed.

    Args:
        session_id: Conversation session identifier (groups turns into a Session).
        user_id:    Optional user UUID string (enables per-user filtering).
        user_input: Optional raw user message — becomes the trace-level input.

    Yields:
        A ``CrewTrace`` instance.
    """
    trace = CrewTrace(session_id=session_id, user_id=user_id)

    if _client is None:
        # Tracing disabled — yield a local-only trace.
        try:
            yield trace
        finally:
            trace.flush()
        return

    try:
        from langfuse import propagate_attributes  # noqa: PLC0415

        with _client.start_as_current_observation(
            as_type="span",
            name="gia-chat-turn",
            input=user_input[:1000] if user_input else None,
        ) as root, propagate_attributes(
            session_id=session_id,
            user_id=user_id or None,
            trace_name="gia-chat-turn",
            tags=["chat"],
        ):
            trace._active = True
            trace._root = root
            try:
                yield trace
            finally:
                trace.flush()
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning("langfuse_trace_open_failed", error=str(exc))

    # Failed to open the Langfuse trace — degrade to local-only bookkeeping.
    try:
        yield trace
    finally:
        trace.flush()
