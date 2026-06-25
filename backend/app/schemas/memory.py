"""Pydantic schemas for the memory engine.

These types flow through the entire memory pipeline:
  ``ExtractedMemory``  — raw output from the LLM extractor
  ``MemoryEntry``      — a record retrieved from Weaviate
  ``UserContext``      — the assembled context injected into every agent turn
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


def _ago(dt: datetime) -> str:
    """Render a coarse 'how long ago' suffix for a life fact, or ``""``.

    Recency is what turns a stored fact into a natural callback — "did you ever
    finish that script?" only lands if Gia knows it was a while ago.
    """
    try:
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now(UTC).replace(tzinfo=None)
        days = (now - dt).days
    except (TypeError, ValueError):
        return ""
    if days <= 0:
        return ""
    if days == 1:
        return " (mentioned yesterday)"
    if days < 7:
        return f" (mentioned {days} days ago)"
    weeks = days // 7
    return f" (mentioned ~{weeks} week{'s' if weeks > 1 else ''} ago)"


class ExtractionRequestBody(BaseModel):
    """Body for ``POST /memory/{user_id}/extract``."""

    transcript: str


class ExtractionResponse(BaseModel):
    """Response from ``POST /memory/{user_id}/extract``."""

    user_id: str
    stored: int
    memory_ids: list[str]


class ExtractedMemory(BaseModel):
    """A preference or episode identified by the LLM extractor.

    Attributes:
        type:          Memory class — ``preference``, ``life_fact``, ``episode``,
                       or ``mood_pattern``.  ``life_fact`` is a non-music personal
                       fact (work, a project, a struggle, a plan) that makes Gia a
                       companion rather than a jukebox.
        text:          Human-readable statement of the memory.
        confidence:    Extractor's confidence that this is worth keeping (0–1).
        supersedes_id: Weaviate UUID of an older memory this one replaces, if any.
    """

    type: Literal["preference", "life_fact", "episode", "mood_pattern"]
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
        score:         Relevance score from retrieval (near_vector / hybrid /
                       rerank), higher = more relevant.
        source:        Where this memory came from (e.g. ``"extractor"``,
                       ``"seed"``, ``"mood_inference"``). Surfaced for grounding
                       so the agent can attribute, not invent, what it knows.
    """

    id: str
    type: str
    text: str
    confidence: float
    created_at: datetime
    supersedes_id: str | None = None
    score: float = 0.0
    source: str | None = None

    @property
    def ref(self) -> str:
        """Short stable reference id (first 8 chars of the UUID) for grounding."""
        return self.id.replace("-", "")[:8]


class UserContext(BaseModel):
    """Assembled context for a single agent turn.

    Built by ``build_user_context()`` at the start of every crew execution.
    Contains everything Gia knows about the user right now, formatted for
    injection into agent system prompts via ``to_prompt_text()``.

    Attributes:
        user_id:        The user this context belongs to.
        profile:        Structured Postgres facts (timezone, genres, volume).
        preferences:    Semantic preference memories from Weaviate.
        life_facts:     Non-music personal facts (work, projects, plans) — the
                        threads a companion remembers and follows up on.
        mood_patterns:  Time-indexed mood tendencies from Weaviate.
        episodes:       Episodic session summaries from Weaviate.
        session_summary: Current-session running notes from Redis (if any).
        now_playing:    Currently playing Spotify track dict (or None).
        recently_played: Up to 10 recently played Spotify track dicts.
    """

    user_id: str
    profile: dict | None = None
    insights: list[MemoryEntry] = Field(default_factory=list)
    preferences: list[MemoryEntry] = Field(default_factory=list)
    life_facts: list[MemoryEntry] = Field(default_factory=list)
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

        name = (self.profile or {}).get("display_name")
        if name:
            lines.append(f"**Name:** {name} — address them by name naturally, not every line.")
        else:
            lines.append(
                "**Name:** unknown — if it fits naturally, ask once what to call them, "
                "then use it. Don't interrogate."
            )

        if self.profile:
            tz = self.profile.get("timezone", "UTC")
            genres = ", ".join(self.profile.get("preferred_genres") or []) or "none set"
            vol = self.profile.get("preferred_volume", 0.7)
            lines.append(f"**Profile:** timezone={tz} | volume={vol:.0%} | genres={genres}")

        if self.now_playing:
            np_name = self.now_playing.get("name", "Unknown")
            np_artist = self.now_playing.get("artist", "Unknown")
            lines.append(f"\n**Now Playing:** {np_name} — {np_artist}")

        if self.recently_played:
            snippets = [
                f"{t.get('name', '?')} ({t.get('artist', '?')})"
                for t in self.recently_played[:3]
            ]
            lines.append(f"**Recent:** {', '.join(snippets)}")

        if self.insights:
            lines.append(
                "\n**Who they are** (synthesised from everything you've learned — "
                "the big picture, more reliable than any single fact below):"
            )
            for i in self.insights:
                lines.append(f"- {i.text} [ref {i.ref}]")

        if self.preferences:
            lines.append("\n**Preferences:**")
            for p in self.preferences:
                lines.append(f"- {p.text} [{p.confidence:.0%} confidence, ref {p.ref}]")

        if self.life_facts:
            lines.append(
                "\n**Life & context** (what's going on for them — you're a friend, "
                "not a jukebox; reference these naturally):"
            )
            for f in self.life_facts:
                lines.append(f"- {f.text}{_ago(f.created_at)} [ref {f.ref}]")
            lines.append(
                "If one of these is an open thread — a project, a struggle, a plan — "
                "it's natural to follow up once, warmly (\"did you ever sort out that…?\"). "
                "Not as a checklist, and only if it fits."
            )

        if self.mood_patterns:
            lines.append("\n**Mood Patterns:**")
            for m in self.mood_patterns:
                lines.append(f"- {m.text} [ref {m.ref}]")

        if self.episodes:
            lines.append("\n**Recent Sessions:**")
            for e in self.episodes:
                lines.append(f"- {e.text} [ref {e.ref}]")

        if self.session_summary:
            lines.append(f"\n**Current Session:** {self.session_summary}")

        return "\n".join(lines)
