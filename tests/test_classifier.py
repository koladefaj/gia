"""Tests for the mood classifier (quadrant model + time bucket)."""

from __future__ import annotations

import pytest

from backend.app.mood.classifier import (
    classify_mood,
    deviates_significantly,
    time_bucket,
)


class TestClassifyMood:
    def test_hype(self) -> None:
        assert classify_mood(0.8, 0.7) == "hype"

    def test_aggressive_focus(self) -> None:
        assert classify_mood(0.8, 0.3) == "aggressive-focus"

    def test_wind_down(self) -> None:
        assert classify_mood(0.3, 0.7) == "wind-down"

    def test_melancholic(self) -> None:
        assert classify_mood(0.3, 0.3) == "melancholic"

    def test_neutral_centre(self) -> None:
        assert classify_mood(0.5, 0.5) == "neutral"

    def test_neutral_high_energy_mid_valence(self) -> None:
        assert classify_mood(0.75, 0.5) == "neutral"

    def test_boundary_energy_high(self) -> None:
        """Exactly at the boundary (0.7) is not classified as high."""
        assert classify_mood(0.7, 0.7) == "neutral"

    def test_boundary_energy_low(self) -> None:
        """Exactly at the boundary (0.4) is not classified as low."""
        assert classify_mood(0.4, 0.7) == "neutral"


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

    def test_hour_22_is_evening(self) -> None:
        # 22 maps to evening (17 <= h < 22 is False for h=22)
        assert time_bucket(hour=22, weekday=3) == "thursday_night"


class TestDeviatesSignificantly:
    def test_large_energy_deviation(self) -> None:
        assert deviates_significantly(0.8, 0.5, 0.3, 0.5, threshold=0.2)

    def test_large_valence_deviation(self) -> None:
        assert deviates_significantly(0.5, 0.8, 0.5, 0.3, threshold=0.2)

    def test_small_deviation_no_flag(self) -> None:
        assert not deviates_significantly(0.5, 0.5, 0.55, 0.52, threshold=0.2)

    def test_exact_threshold_not_significant(self) -> None:
        """Deviation exactly equal to threshold is not significant (strict >)."""
        assert not deviates_significantly(0.7, 0.5, 0.5, 0.5, threshold=0.2)

    def test_just_over_threshold(self) -> None:
        assert deviates_significantly(0.71, 0.5, 0.5, 0.5, threshold=0.2)

    def test_both_deviate(self) -> None:
        assert deviates_significantly(0.8, 0.8, 0.3, 0.3, threshold=0.2)
