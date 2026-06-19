"""Pydantic schemas for the memory engine.

These types flow through the entire memory pipeline:
  ``ExtractedMemory``  — raw output from the LLM extractor
  ``MemoryEntry``      — a record retrieved from Weaviate
  ``UserContext``      — the assembled context injected into every agent turn
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ExtractedMemory(BaseModel):
    """A preference or episode identified by the LLM extractor.

    Attributes:
        type:          Memory class — ``preference``, ``episode``, or ``mood_pattern``.
        text:          Human-readable statement of the memory.
        confidence:    Extractor's confidence that this is worth keeping (0–1).
        supersedes_id: Weaviate UUID of an older memory this one replaces, if any.
    """

    type: Literal["preference", "episode", "mood_pattern"]
    text: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    supersedes_id: str | None = None


class MemoryEntry(BaseModel):
    """A memory record retrieved from Weaviate.

    Attributes:
        id:            Weaviate UUID string.
        type:          Memory class (``preference``, ``episode``, ``mood_pattern``).
        text:          The stored memory text.
        confidence:    Confidence score at extraction time.
        created_at:    UTC timestamp of when the memory was stored.
        supersedes_id: UUID of the memory this entry replaced, if applicable.
        score:         Semantic similarity score from the near_vector query (0–1).
    """

    id: str
    type: str
    text: str
    confidence: float
    created_at: datetime
    supersedes_id: str | None = None
    score: float = 0.0


class UserContext(BaseModel):
    """Assembled context for a single agent turn.

    Built by ``build_user_context()`` at the start of every crew execution.
    Contains everything Gia knows about the user right now, formatted for
    injection into agent system prompts via ``to_prompt_text()``.

    Attributes:
        user_id:        The user this context belongs to.
        profile:        Structured Postgres facts (timezone, genres, volume).
        preferences:    Semantic preference memories from Weaviate.
        mood_patterns:  Time-indexed mood tendencies from Weaviate.
        episodes:       Episodic session summaries from Weaviate.
        session_summary: Current-session running notes from Redis (if any).
        now_playing:    Currently playing Spotify track dict (or None).
        recently_played: Up to 10 recently played Spotify track dicts.
    """

    user_id: str
    profile: dict | None = None
    preferences: list[MemoryEntry] = Field(default_factory=list)
    mood_patterns: list[MemoryEntry] = Field(default_factory=list)
    episodes: list[MemoryEntry] = Field(default_factory=list)
    session_summary: str | None = None
    now_playing: dict | None = None
    recently_played: list[dict] = Field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Render the context as structured text for LLM system-prompt injection.

        Returns:
            A markdown-lite string summarising everything Gia knows about the
            user, suitable for prepending to any agent's system prompt.
        """
        lines: list[str] = ["## What Gia knows about this user\n"]

        if self.profile:
            tz = self.profile.get("timezone", "UTC")
            genres = ", ".join(self.profile.get("preferred_genres") or []) or "none set"
            vol = self.profile.get("preferred_volume", 0.7)
            lines.append(f"**Profile:** timezone={tz} | volume={vol:.0%} | genres={genres}")

        if self.now_playing:
            name = self.now_playing.get("name", "Unknown")
            artist = self.now_playing.get("artist", "Unknown")
            energy = self.now_playing.get("energy", "?")
            lines.append(f"\n**Now Playing:** {name} — {artist} (energy={energy})")

        if self.recently_played:
            snippets = [
                f"{t.get('name', '?')} ({t.get('artist', '?')})"
                for t in self.recently_played[:3]
            ]
            lines.append(f"**Recent:** {', '.join(snippets)}")

        if self.preferences:
            lines.append("\n**Preferences:**")
            for p in self.preferences:
                lines.append(f"- {p.text} [{p.confidence:.0%} confidence]")

        if self.mood_patterns:
            lines.append("\n**Mood Patterns:**")
            for m in self.mood_patterns:
                lines.append(f"- {m.text}")

        if self.episodes:
            lines.append("\n**Recent Sessions:**")
            for e in self.episodes:
                lines.append(f"- {e.text}")

        if self.session_summary:
            lines.append(f"\n**Current Session:** {self.session_summary}")

        return "\n".join(lines)
