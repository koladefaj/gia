"""Tests for Celery task stubs.

These tasks are scaffolded for Days 3 and 6.  Tests here verify that:
  - The task is importable and registered with Celery.
  - Calling the task function directly returns the expected stub response.
  - The task name matches the registered name so ``celery_app.send_task``
    can route it correctly.
"""

from __future__ import annotations

import pytest

from backend.worker.tasks.memory_extraction import extract_session_memories
from backend.worker.tasks.mood_inference import (
    run_mood_inference,
    run_mood_inference_all,
)
from backend.worker.tasks.proactive_check import check_pattern_shift


def test_extract_session_memories_returns_stub() -> None:
    """Stub task returns a dict with status and the input IDs."""
    result = extract_session_memories("user-1", "session-1")
    assert result["status"] == "stub"
    assert result["user_id"] == "user-1"
    assert result["session_id"] == "session-1"


def test_run_mood_inference_returns_stub() -> None:
    """Stub task returns a dict with status and user_id."""
    result = run_mood_inference("user-2")
    assert result["status"] == "stub"
    assert result["user_id"] == "user-2"


def test_run_mood_inference_all_returns_stub() -> None:
    """Beat task stub returns ``{"status": "stub"}``."""
    result = run_mood_inference_all()
    assert result["status"] == "stub"


def test_check_pattern_shift_returns_stub() -> None:
    """Stub task returns a dict with status and user_id."""
    result = check_pattern_shift("user-3")
    assert result["status"] == "stub"
    assert result["user_id"] == "user-3"


def test_tasks_are_registered_with_celery() -> None:
    """All tasks are discoverable by name in the Celery app registry."""
    from backend.worker.celery_app import celery_app

    registered = celery_app.tasks.keys()
    assert "backend.worker.tasks.memory_extraction.extract_session_memories" in registered
    assert "backend.worker.tasks.mood_inference.run_mood_inference" in registered
    assert "backend.worker.tasks.mood_inference.run_mood_inference_all" in registered
    assert "backend.worker.tasks.proactive_check.check_pattern_shift" in registered
