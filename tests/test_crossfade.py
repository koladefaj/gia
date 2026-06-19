"""Tests for the crossfade sequencing module — pure logic, no network calls."""

from __future__ import annotations

import pytest

from backend.app.schemas.dj import TrackItem
from backend.app.tools.crossfade import (
    CAMELOT,
    build_energy_sequence,
    build_key_matched_sequence,
    camelot_compatible,
    camelot_key,
    track_from_dict,
)


def _track(uri: str, energy: float, key: int = 0, mode: int = 1) -> TrackItem:
    return TrackItem(
        uri=uri,
        name=f"Track {uri}",
        artist="Artist",
        energy=energy,
        valence=0.5,
        key=key,
        mode=mode,
    )


# ── CAMELOT table ─────────────────────────────────────────────────────────────


def test_camelot_table_has_24_entries() -> None:
    """The Camelot wheel has 12 major (B) and 12 minor (A) keys."""
    assert len(CAMELOT) == 24


def test_camelot_c_major_is_8b() -> None:
    """C major (key=0, mode=1) maps to Camelot 8B."""
    assert CAMELOT[(0, 1)] == "8B"


def test_camelot_a_minor_is_8a() -> None:
    """A minor (key=9, mode=0) maps to Camelot 8A (relative of C major)."""
    assert CAMELOT[(9, 0)] == "8A"


def test_camelot_key_returns_none_for_unknown() -> None:
    """``camelot_key`` returns ``None`` for invalid (key, mode) combos."""
    t = TrackItem(uri="x", name="x", artist="x", energy=0.5, valence=0.5, key=99, mode=99)
    assert camelot_key(t) is None


# ── camelot_compatible ────────────────────────────────────────────────────────


def test_same_key_is_compatible() -> None:
    """Two tracks with the same Camelot label are compatible."""
    a = _track("a", 0.5, key=0, mode=1)  # 8B
    b = _track("b", 0.6, key=0, mode=1)  # 8B
    assert camelot_compatible(a, b)


def test_adjacent_number_same_letter_is_compatible() -> None:
    """8B → 9B is one step on the wheel — compatible."""
    a = _track("a", 0.5, key=0, mode=1)   # 8B
    b = _track("b", 0.5, key=7, mode=1)   # 9B
    assert camelot_compatible(a, b)


def test_relative_key_same_number_is_compatible() -> None:
    """8B ↔ 8A (relative major/minor) — compatible."""
    a = _track("a", 0.5, key=0, mode=1)  # 8B
    b = _track("b", 0.5, key=9, mode=0)  # 8A
    assert camelot_compatible(a, b)


def test_incompatible_keys() -> None:
    """8B → 1B is not adjacent on the wheel — incompatible."""
    a = _track("a", 0.5, key=0, mode=1)   # 8B
    b = _track("b", 0.5, key=11, mode=1)  # 1B
    assert not camelot_compatible(a, b)


def test_unknown_key_treated_as_compatible() -> None:
    """Tracks with unknown keys are always treated as compatible."""
    a = _track("a", 0.5, key=99, mode=99)
    b = _track("b", 0.5, key=0, mode=1)
    assert camelot_compatible(a, b)


def test_12_to_1_wrap_is_compatible() -> None:
    """12B → 1B is adjacent on the wheel (wraps around)."""
    a = _track("a", 0.5, key=4, mode=1)   # 12B
    b = _track("b", 0.5, key=11, mode=1)  # 1B
    assert camelot_compatible(a, b)


# ── build_energy_sequence ─────────────────────────────────────────────────────


def test_energy_sequence_returns_n_tracks() -> None:
    """Sequence returns exactly *n* tracks when enough candidates exist."""
    seed = _track("seed", energy=0.5)
    candidates = [_track(str(i), energy=0.1 * i) for i in range(1, 8)]
    result = build_energy_sequence(seed, candidates, n=4)
    assert len(result) == 4


def test_energy_sequence_closest_energy_first() -> None:
    """First picked track has energy closest to seed."""
    seed = _track("seed", energy=0.5)
    far = _track("far", energy=0.9)
    close = _track("close", energy=0.52)
    medium = _track("medium", energy=0.7)
    result = build_energy_sequence(seed, [far, close, medium], n=3)
    assert result[0].uri == "close"


def test_energy_sequence_capped_by_candidates() -> None:
    """If fewer candidates than n, returns all candidates."""
    seed = _track("seed", energy=0.5)
    candidates = [_track("a", 0.4), _track("b", 0.6)]
    result = build_energy_sequence(seed, candidates, n=10)
    assert len(result) == 2


def test_energy_sequence_no_candidates() -> None:
    """Empty candidate list returns empty sequence."""
    seed = _track("seed", energy=0.5)
    assert build_energy_sequence(seed, [], n=5) == []


# ── build_key_matched_sequence ────────────────────────────────────────────────


def test_key_matched_sequence_prefers_compatible_keys() -> None:
    """Key-matched sequencer filters for Camelot compatibility."""
    seed = _track("seed", energy=0.5, key=0, mode=1)  # 8B
    compatible = _track("compat", energy=0.52, key=7, mode=1)    # 9B — compatible
    incompatible = _track("incompat", energy=0.51, key=6, mode=1)  # 2B — not adjacent

    result = build_key_matched_sequence(seed, [incompatible, compatible], n=1)
    assert result[0].uri == "compat"


def test_key_matched_falls_back_when_no_compatible() -> None:
    """When no Camelot-compatible tracks exist, falls back to energy-only."""
    seed = _track("seed", energy=0.5, key=0, mode=1)  # 8B
    only = _track("only", energy=0.52, key=6, mode=0)  # 11A — not compatible
    result = build_key_matched_sequence(seed, [only], n=1)
    assert len(result) == 1


def test_key_matched_sequence_returns_n_tracks() -> None:
    """Sequence length equals *n* when candidates are sufficient."""
    seed = _track("seed", energy=0.5, key=0, mode=1)
    candidates = [_track(str(i), energy=0.05 * i, key=0, mode=1) for i in range(1, 10)]
    result = build_key_matched_sequence(seed, candidates, n=5)
    assert len(result) == 5


# ── track_from_dict ───────────────────────────────────────────────────────────


def test_track_from_dict_fills_camelot_key() -> None:
    """``track_from_dict`` resolves the Camelot label from key + mode."""
    raw = {
        "uri": "spotify:track:abc",
        "name": "Free Mind",
        "artist": "Tems",
        "energy": 0.38,
        "valence": 0.71,
        "tempo": 92.0,
        "key": 0,
        "mode": 1,
        "danceability": 0.62,
    }
    item = track_from_dict(raw)
    assert item.camelot_key == "8B"
    assert item.energy == pytest.approx(0.38)
    assert item.name == "Free Mind"


def test_track_from_dict_handles_missing_fields() -> None:
    """``track_from_dict`` uses safe defaults for absent fields."""
    item = track_from_dict({"uri": "x"})
    assert item.energy == pytest.approx(0.5)
    assert item.tempo == pytest.approx(120.0)
    assert item.name == ""
