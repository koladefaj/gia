"""Capture a user's preferred name from a chat message.

Spotify's ``display_name`` is often a handle (e.g. ``akoladefvr``), not a name
you'd want Gia to use — so Gia asks "what should I call you?" and we capture the
answer here.  Deliberately conservative: it only fires on explicit name
statements (``call me X``, ``my name is X``, ``I'm X``) with a capitalised token,
so casual phrases like "I'm tired" don't get mistaken for a name.
"""

from __future__ import annotations

import re

# Words that look like names positionally but aren't, to avoid false captures.
_NOT_NAMES = {
    "Tired", "Good", "Fine", "Okay", "Ok", "Done", "Sure", "Here", "Back",
    "Sorry", "Listening", "Looking", "Just", "Not", "Still", "Really", "Gonna",
}

# Triggers match either case (sentence-leading "I'm" or mid-sentence "call me"),
# but the captured name stays case-sensitive ([A-Z]…) so "I'm tired" is ignored.
_NAME = r"(?P<name>[A-Z][a-zA-Z'-]{1,29})"
_PATTERNS = [
    re.compile(r"\b[Cc]all me\s+" + _NAME),
    re.compile(r"\b[Mm]y name(?:'s| is)\s+" + _NAME),
    re.compile(r"\b(?:[Ii]'?m|[Ii] am|[Ii]t's|[Tt]his is)\s+" + _NAME),
]


def extract_name(message: str) -> str | None:
    """Return a preferred name stated in *message*, or ``None``.

    Args:
        message: The user's raw message text.

    Returns:
        The captured name (capitalised), or ``None`` if no confident match.
    """
    for pattern in _PATTERNS:
        match = pattern.search(message)
        if match:
            name = match.group("name").strip()
            if name and name not in _NOT_NAMES:
                return name
    return None
