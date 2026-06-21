"""Tests for the mood vocabulary coercion + time bucketing."""

from __future__ import annotations

from backend.app.mood.classifier import MOOD_LABELS, coerce_label, time_bucket


class TestCoerceLabel:
    def test_exact_vocabulary_word(self) -> None:
        assert coerce_label("chill") == "chill"

    def test_vocabulary_word_inside_a_phrase(self) -> None:
        assert coerce_label("definitely a hype set") == "hype"

    def test_synonym_maps_to_vocabulary(self) -> None:
        assert coerce_label("sad") == "melancholy"
        assert coerce_label("energetic") == "hype"
        assert coerce_label("mellow") == "chill"

    def test_unknown_falls_back_to_neutral(self) -> None:
        assert coerce_label("???") == "neutral"

    def test_result_is_always_in_vocabulary(self) -> None:
        for raw in ["chill", "sad", "party", "love", "gibberish"]:
            assert coerce_label(raw) in MOOD_LABELS


class TestTimeBucket:
    def test_monday_morning(self) -> None:
        assert time_bucket(hour=8, weekday=0) == "monday_morning"

    def test_sunday_evening(self) -> None:
        assert time_bucket(hour=20, weekday=6) == "sunday_evening"

    def test_friday_night(self) -> None:
        assert time_bucket(hour=23, weekday=4) == "friday_night"

    def test_wednesday_afternoon(self) -> None:
        assert time_bucket(hour=14, weekday=2) == "wednesday_afternoon"

    def test_midnight_is_night(self) -> None:
        assert time_bucket(hour=0, weekday=5) == "saturday_night"

    def test_hour_6_is_morning(self) -> None:
        assert time_bucket(hour=6, weekday=1) == "tuesday_morning"

    def test_hour_22_is_night(self) -> None:
        # 22 is past the evening window (17 <= h < 22), so it's night.
        assert time_bucket(hour=22, weekday=3) == "thursday_night"
