"""Tests for conversational name capture."""

from __future__ import annotations

import pytest

from backend.app.memory.name_capture import extract_name


@pytest.mark.parametrize(
    "message,expected",
    [
        ("call me Kolade", "Kolade"),
        ("You can call me Tunde please", "Tunde"),
        ("my name is Ada", "Ada"),
        ("my name's Femi", "Femi"),
        ("I'm Zainab", "Zainab"),
        ("it's Chidi", "Chidi"),
        # Negatives — must NOT capture
        ("I'm tired", None),
        ("i'm good thanks", None),
        ("find me something chill", None),
        ("just looking for afrobeats", None),
        ("I'm Listening", None),  # stopword guard
    ],
)
def test_extract_name(message: str, expected: str | None) -> None:
    assert extract_name(message) == expected
