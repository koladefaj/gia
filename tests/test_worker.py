"""Tests for Celery tasks.

Verifies that tasks are importable, registered, and dispatch correctly.
Async implementations are patched via ``asyncio.run`` to avoid network calls.
"""

from __future__ import annotations

from unittest.mock import patch

from backend.worker.tasks.memory_extraction import extract_session_memories
from backend.worker.tasks.mood_inference import (
    run_mood_inference,
    run_mood_inference_all,
)
from backend.worker.tasks.proactive_check import check_pattern_shift


def test_extract_session_memories_dispatches_async() -> None:
    """``extract_session_memories`` calls ``asyncio.run`` with the async body."""
    expected = {"status": "ok", "stored": 1, "memory_ids": ["mem-1"]}
    with patch("backend.worker.tasks.memory_extraction.asyncio.run", return_value=expected) as mock_run:
        result = extract_session_memories("user-1", "session-1")

    assert result == expected
    mock_run.assert_called_once()


def test_run_mood_inference_dispatches_async() -> None:
    """``run_mood_inference`` calls ``asyncio.run`` with the async implementation."""
    expected = {"status": "ok", "user_id": "user-2", "patterns_stored": 1}
    with patch(
        "backend.worker.tasks.mood_inference.asyncio.run", return_value=expected
    ) as mock_run:
        result = run_mood_inference("user-2")

    assert result == expected
    mock_run.assert_called_once()


def test_run_mood_inference_all_dispatches_async() -> None:
    """``run_mood_inference_all`` calls ``asyncio.run`` for the beat task."""
    expected = {"status": "ok", "users_dispatched": 3}
    with patch(
        "backend.worker.tasks.mood_inference.asyncio.run", return_value=expected
    ) as mock_run:
        result = run_mood_inference_all()

    assert result == expected
    mock_run.assert_called_once()


def test_check_pattern_shift_dispatches_async() -> None:
    """``check_pattern_shift`` calls ``asyncio.run`` with the async implementation."""
    expected = {"status": "ok", "user_id": "user-3", "draft_produced": False}
    with patch(
        "backend.worker.tasks.proactive_check.asyncio.run", return_value=expected
    ) as mock_run:
        result = check_pattern_shift("user-3")

    assert result == expected
    mock_run.assert_called_once()


def test_tasks_are_registered_with_celery() -> None:
    """All tasks are discoverable by name in the Celery app registry."""
    from backend.worker.celery_app import celery_app

    registered = celery_app.tasks.keys()
    assert "backend.worker.tasks.memory_extraction.extract_session_memories" in registered
    assert "backend.worker.tasks.mood_inference.run_mood_inference" in registered
    assert "backend.worker.tasks.mood_inference.run_mood_inference_all" in registered
    assert "backend.worker.tasks.proactive_check.check_pattern_shift" in registered
