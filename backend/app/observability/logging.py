"""Structured logging configuration for Gia.

Uses ``structlog`` with a stdlib bridge so that third-party libraries that log
through Python's standard ``logging`` module are captured in the same pipeline.

Renderers
---------
- **Debug** (``log_level="debug"``): ``ConsoleRenderer`` with colour and
  aligned key=value output — pleasant in a local terminal.
- **Production** (any other level): ``JSONRenderer`` — one JSON object per
  line, compatible with Datadog, Cloud Logging, and most log aggregators.

Usage::

    from backend.app.observability.logging import get_logger, setup_logging

    setup_logging(log_level="info")         # called once in lifespan
    logger = get_logger(__name__)
    logger.info("track_played", uri="spotify:track:4cOd...")
"""

import logging
import sys

import structlog


def setup_logging(log_level: str = "debug") -> None:
    """Initialise structlog and the stdlib root logger.

    Should be called **once** during application startup (FastAPI lifespan).
    Subsequent calls are idempotent — structlog's ``cache_logger_on_first_use``
    means the first call wins.

    Args:
        log_level: Case-insensitive log level string (e.g. ``"debug"``,
                   ``"info"``, ``"warning"``).  Defaults to ``"debug"`` so
                   development containers are maximally verbose.
    """
    level = getattr(logging, log_level.upper(), logging.DEBUG)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if log_level.lower() == "debug":
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)

    for noisy in ("uvicorn.access", "sqlalchemy.engine", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structlog bound logger.

    Args:
        name: Logger name — conventionally ``__name__`` of the calling module.

    Returns:
        A ``structlog.stdlib.BoundLogger`` that emits structured key=value
        events through the pipeline configured by ``setup_logging``.
    """
    return structlog.get_logger(name)  # type: ignore[return-value]
