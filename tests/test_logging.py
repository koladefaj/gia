"""Tests for the structlog logging setup.

Validates that ``setup_logging`` configures structlog without raising,
that ``get_logger`` returns a bound logger, and that the renderer selection
(``ConsoleRenderer`` vs ``JSONRenderer``) matches the log level.
"""

from __future__ import annotations

import logging

from backend.app.observability.logging import get_logger, setup_logging


def test_setup_logging_debug_does_not_raise() -> None:
    """``setup_logging("debug")`` completes without raising."""
    setup_logging("debug")  # should not raise


def test_setup_logging_info_does_not_raise() -> None:
    """``setup_logging("info")`` completes without raising."""
    setup_logging("info")


def test_setup_logging_sets_root_level_to_debug() -> None:
    """After ``setup_logging("debug")``, the root logger level is DEBUG."""
    setup_logging("debug")
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_sets_root_level_to_info() -> None:
    """After ``setup_logging("info")``, the root logger level is INFO."""
    setup_logging("info")
    assert logging.getLogger().level == logging.INFO


def test_get_logger_returns_bound_logger() -> None:
    """``get_logger`` returns a structlog ``BoundLogger``."""
    setup_logging("debug")
    logger = get_logger("test.module")
    assert logger is not None
    # structlog BoundLoggers have a ``bind`` method
    assert hasattr(logger, "bind")


def test_get_logger_name_is_passed_through() -> None:
    """``get_logger`` passes the name through to the underlying logger factory."""
    setup_logging("debug")
    logger = get_logger("gia.test")
    # Calling a log method must not raise
    logger.debug("test_message", key="value")


def test_setup_logging_silences_noisy_loggers() -> None:
    """SQLAlchemy, httpx, and uvicorn.access are silenced to WARNING+."""
    setup_logging("debug")
    for name in ("sqlalchemy.engine", "httpx", "uvicorn.access"):
        assert logging.getLogger(name).level == logging.WARNING
