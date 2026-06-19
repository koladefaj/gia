"""Crossfade sequencing — energy-aware + Camelot key matching.

Two levels of sequencing:

Level 1 — Energy-aware (``build_energy_sequence``)
    Greedy nearest-neighbour on audio energy.  Each step targets the
    previous track's energy ± *step*, producing a smooth ramp rather than
    jarring energy jumps.

Level 2 — Camelot key matching (``build_key_matched_sequence``)
    Applies on top of Level 1.  At each step the pool is filtered to
    Camelot-compatible tracks before picking the best energy match.
    Falls back to the unfiltered pool if no compatible track remains.

Both functions are pure — they take ``TrackItem`` objects and return lists
of ``TrackItem``.  Network calls are the caller's responsibility.

References
----------
Camelot wheel: https://mixedinkey.com/camelot-wheel/
Spotify audio features: https://developer.spotify.com/documentation/web-api/reference/get-audio-features
"""

from __future__ import annotations

from backend.app.schemas.dj import TrackItem

# Camelot wheel: (spotify_key, spotify_mode) → Camelot label
# key: 0=C 1=C# 2=D 3=D# 4=E 5=F 6=F# 7=G 8=G# 9=A 10=A# 11=B
# mode: 0=minor 1=major
CAMELOT: dict[tuple[int, int], str] = {
    (0, 1): "8B",  (1, 1): "3B",  (2, 1): "10B", (3, 1): "5B",
    (4, 1): "12B", (5, 1): "7B",  (6, 1): "2B",  (7, 1): "9B",
    (8, 1): "4B",  (9, 1): "11B", (10, 1): "6B", (11, 1): "1B",
    (0, 0): "5A",  (1, 0): "12A", (2, 0): "7A",  (3, 0): "2A",
    (4, 0): "9A",  (5, 0): "4A",  (6, 0): "11A", (7, 0): "6A",
    (8, 0): "1A",  (9, 0): "8A",  (10, 0): "3A", (11, 0): "10A",
}


def camelot_key(track: TrackItem) -> str | None:
    """Return the Camelot wheel label for *track*, or ``None`` if unknown.

    Args:
        track: A ``TrackItem`` with ``key`` and ``mode`` populated.

    Returns:
        Camelot label string (e.g. ``"8B"``) or ``None``.
    """
    return CAMELOT.get((track.key, track.mode))


def camelot_compatible(a: TrackItem, b: TrackItem) -> bool:
    """Return ``True`` if tracks *a* and *b* are Camelot-wheel compatible.

    Compatible keys on the Camelot wheel are:
      - The same key (same number and letter)
      - Adjacent numbers on the same letter (e.g. 8B → 7B or 9B)
      - Same number, different letter (relative major/minor, e.g. 8B ↔ 8A)

    Unknown keys (not in the CAMELOT table) are treated as compatible so
    the sequencer never silently discards tracks with missing key data.

    Args:
        a: First track.
        b: Second track.

    Returns:
        ``True`` if the transition is DJ-safe.
    """
    ca = camelot_key(a)
    cb = camelot_key(b)
    if ca is None or cb is None:
        return True  # unknown key — allow to avoid silent drops

    num_a, let_a = int(ca[:-1]), ca[-1]
    num_b, let_b = int(cb[:-1]), cb[-1]

    # Camelot numbers wrap at 12 → 1
    same_key = ca == cb
    adjacent_same_letter = let_a == let_b and (
        abs(num_a - num_b) == 1
        or {num_a, num_b} == {1, 12}  # wrap-around
    )
    relative_key = num_a == num_b and let_a != let_b

    return same_key or adjacent_same_letter or relative_key


def build_energy_sequence(
    seed: TrackItem,
    candidates: list[TrackItem],
    n: int = 5,
    step: float = 0.07,
) -> list[TrackItem]:
    """Build an energy-aware crossfade sequence from *candidates*.

    Starts from *seed*'s energy and greedily picks the closest track, then
    nudges the target by *step* to create a gradual energy ramp.

    Args:
        seed:       Anchor track that defines the starting energy.
        candidates: Pool of tracks to sequence from (seed excluded by caller).
        n:          Maximum number of tracks to return.
        step:       Energy increment applied after each pick (positive = ramp
                    up, negative = ramp down, 0.07 = gentle upward drift).

    Returns:
        Ordered list of up to *n* ``TrackItem`` objects.
    """
    remaining = list(candidates)
    sequence: list[TrackItem] = []
    target_energy = seed.energy

    for _ in range(min(n, len(remaining))):
        best = min(remaining, key=lambda t: abs(t.energy - target_energy))
        sequence.append(best)
        remaining.remove(best)
        target_energy = best.energy + step

    return sequence


def build_key_matched_sequence(
    seed: TrackItem,
    candidates: list[TrackItem],
    n: int = 5,
    step: float = 0.07,
) -> list[TrackItem]:
    """Build an energy-aware sequence with Camelot key compatibility.

    At each step, the candidate pool is first filtered to tracks compatible
    with the *current* track on the Camelot wheel.  If no compatible track
    remains (rare with a diverse pool), the full remaining pool is used.

    Args:
        seed:       Anchor track.
        candidates: Pool of candidate tracks (seed excluded by caller).
        n:          Maximum tracks to return.
        step:       Energy step per pick.

    Returns:
        Ordered list of up to *n* ``TrackItem`` objects, Camelot-compatible
        where possible.
    """
    remaining = list(candidates)
    sequence: list[TrackItem] = []
    target_energy = seed.energy
    current = seed

    for _ in range(min(n, len(remaining))):
        compatible = [t for t in remaining if camelot_compatible(current, t)]
        pool = compatible if compatible else remaining
        best = min(pool, key=lambda t: abs(t.energy - target_energy))
        sequence.append(best)
        remaining.remove(best)
        target_energy = best.energy + step
        current = best

    return sequence


def track_from_dict(raw: dict) -> TrackItem:
    """Convert a raw Spotify track dict (with audio features merged) to ``TrackItem``.

    Args:
        raw: Dict with at minimum ``uri``, ``name``, ``artist``, ``energy``,
             ``valence``, ``key``, ``mode``.  Missing fields use safe defaults.

    Returns:
        A populated ``TrackItem`` with ``camelot_key`` filled in.
    """
    key = int(raw.get("key") or 0)
    mode = int(raw.get("mode") or 0)
    item = TrackItem(
        uri=str(raw.get("uri", "")),
        name=str(raw.get("name", "")),
        artist=str(raw.get("artist", "")),
        energy=float(raw.get("energy") or 0.5),
        valence=float(raw.get("valence") or 0.5),
        tempo=float(raw.get("tempo") or 120.0),
        key=key,
        mode=mode,
        danceability=float(raw.get("danceability") or 0.5),
        camelot_key=CAMELOT.get((key, mode)),
    )
    return item
