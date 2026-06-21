"""Mood vocabulary + time bucketing.

Spotify no longer exposes audio features to new apps, so mood is no longer
derived from (energy, valence) quadrants. Instead an LLM labels the *music the
user actually plays* (track + artist names) into one of a small, fixed
vocabulary — see :mod:`backend.app.mood.labeler`. Keeping the vocabulary closed
means a current label and a stored pattern label compare cleanly (string
equality), which is what the deviation check needs.

``time_bucket`` is unchanged — it still keys patterns by ``(weekday, period)``.
"""

from __future__ import annotations

# Closed mood vocabulary. The labeler is constrained to these; pattern matching
# and deviation are plain string comparisons over this set.
MOOD_LABELS: tuple[str, ...] = (
    "hype",
    "chill",
    "melancholy",
    "focused",
    "romantic",
    "upbeat",
    "reflective",
    "neutral",
)

# Common free-text moods the model might return → canonical vocabulary label.
_SYNONYMS: dict[str, str] = {
    "sad": "melancholy",
    "down": "melancholy",
    "moody": "melancholy",
    "happy": "upbeat",
    "joyful": "upbeat",
    "energetic": "hype",
    "party": "hype",
    "dance": "hype",
    "calm": "chill",
    "mellow": "chill",
    "relaxed": "chill",
    "laid-back": "chill",
    "intense": "focused",
    "concentration": "focused",
    "study": "focused",
    "love": "romantic",
    "sensual": "romantic",
    "nostalgic": "reflective",
    "introspective": "reflective",
    "thoughtful": "reflective",
}


def coerce_label(raw: str) -> str:
    """Map a model's free-text mood answer onto the closed vocabulary.

    Tolerant by design: the labeler is asked for one vocabulary word, but models
    drift ("a chill, mellow vibe"). We scan for a vocabulary hit first, then a
    synonym, and fall back to ``"neutral"`` so a noisy answer never crashes a turn.

    Args:
        raw: The model's raw label text.

    Returns:
        One of :data:`MOOD_LABELS`.
    """
    s = raw.strip().lower()
    for label in MOOD_LABELS:
        if label in s:
            return label
    for word, label in _SYNONYMS.items():
        if word in s:
            return label
    return "neutral"


def time_bucket(hour: int, weekday: int) -> str:
    """Convert hour (0-23) and weekday (0=Mon … 6=Sun) to a named bucket.

    The bucket keys ``mood_pattern`` memories so the proactive engine can match
    the current time against a stored pattern without fuzzy matching.

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
