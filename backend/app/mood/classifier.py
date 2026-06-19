"""Quadrant-based mood classifier — no LLM required.

Maps (energy, valence) to a human-readable mood label using a simple
four-quadrant model.  This is intentionally not an LLM: it runs in
microseconds, is fully deterministic, and can be explained in one sentence
during a demo without looking over-engineered.

Quadrant map::

                 high valence
                      │
        wind-down  ◄──┤──►  hype
                      │
    ──────────────────┼────────────────── high energy →
                      │
      melancholic  ◄──┤──►  aggressive-focus
                      │

The ``neutral`` label is returned for the centre band where neither
extreme applies.
"""

from __future__ import annotations

_ENERGY_HIGH = 0.7
_ENERGY_LOW = 0.4
_VALENCE_HIGH = 0.6
_VALENCE_LOW = 0.4


def classify_mood(energy: float, valence: float) -> str:
    """Map (energy, valence) to a mood label.

    Args:
        energy:  Spotify audio feature 0–1 (loud / fast = high).
        valence: Spotify audio feature 0–1 (happy = high).

    Returns:
        One of ``"hype"``, ``"aggressive-focus"``, ``"wind-down"``,
        ``"melancholic"``, or ``"neutral"``.
    """
    if energy > _ENERGY_HIGH and valence > _VALENCE_HIGH:
        return "hype"
    if energy > _ENERGY_HIGH and valence < _VALENCE_LOW:
        return "aggressive-focus"
    if energy < _ENERGY_LOW and valence > _VALENCE_HIGH:
        return "wind-down"
    if energy < _ENERGY_LOW and valence < _VALENCE_LOW:
        return "melancholic"
    return "neutral"


def time_bucket(hour: int, weekday: int) -> str:
    """Convert hour (0-23) and weekday (0=Mon … 6=Sun) to a named bucket.

    The bucket is used as the key in ``MoodPattern`` Weaviate objects.
    Consistent naming lets the proactive engine match the right pattern
    against the current time without fuzzy matching.

    Args:
        hour:    Hour of the day (0–23).
        weekday: ISO weekday (0 = Monday, 6 = Sunday).

    Returns:
        Bucket string, e.g. ``"sunday_evening"`` or ``"monday_morning"``.
    """
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    day = days[weekday % 7]

    if 6 <= hour < 12:
        period = "morning"
    elif 12 <= hour < 17:
        period = "afternoon"
    elif 17 <= hour < 22:
        period = "evening"
    else:
        period = "night"

    return f"{day}_{period}"


def deviates_significantly(
    current_energy: float,
    current_valence: float,
    pattern_energy: float,
    pattern_valence: float,
    threshold: float = 0.2,
) -> bool:
    """Return ``True`` when current features deviate meaningfully from pattern.

    Args:
        current_energy:  Energy of the currently playing track.
        current_valence: Valence of the currently playing track.
        pattern_energy:  Expected energy for this time bucket.
        pattern_valence: Expected valence for this time bucket.
        threshold:       Minimum absolute difference to count as significant.

    Returns:
        ``True`` if energy or valence deviates by more than *threshold*.
    """
    return (
        abs(current_energy - pattern_energy) > threshold
        or abs(current_valence - pattern_valence) > threshold
    )
