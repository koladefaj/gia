"""Mood agent — detect mood from listening session, surface proactive insights.

``MoodService.analyze()`` is the entry point.  It:

1. Determines the current time bucket (e.g. ``"sunday_evening"``).
2. Fetches the user's known mood pattern for that bucket from Weaviate.
3. Compares the currently playing track's audio features to the pattern.
4. If deviation is significant, generates a proactive observation in Gia's voice.
5. Returns a ``MoodResult`` for injection into the crew reply.

The mood classifier is intentionally a simple quadrant model (no LLM).  Fast,
free, and explainable.  The LLM is used only to phrase the observation warmly.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from crewai import Agent

from backend.app.config import Settings
from backend.app.interfaces import SpotifyClientProtocol
from backend.app.memory.embeddings import embed
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.mood.classifier import classify_mood, deviates_significantly, time_bucket
from backend.app.mood.proactive import _parse_pattern, get_pattern_for_now
from backend.app.observability.logging import get_logger
from backend.app.prompts import PromptRegistry, get_registry
from backend.app.providers.llm import get_fast_llm
from backend.app.schemas.memory import MemoryEntry

logger = get_logger(__name__)

AGENT_KEY = "agents.mood"


@dataclass
class MoodResult:
    """Output of a ``MoodService.analyze()`` call.

    Attributes:
        current_label:   Mood label for the currently playing track.
        pattern_label:   Mood label for the known pattern (``None`` if unknown).
        bucket:          Time bucket for this analysis (e.g. ``"monday_morning"``).
        deviation:       Whether the current mood deviates from the pattern.
        proactive_draft: Ready-to-surface observation in Gia's voice (or ``None``).
    """

    current_label: str
    pattern_label: str | None = None
    bucket: str = ""
    deviation: bool = False
    proactive_draft: str | None = None


def build_mood_agent(cfg: Settings, registry: PromptRegistry | None = None) -> Agent:
    """Construct the CrewAI Mood agent from the externalised prompt registry.

    Args:
        cfg:      Application settings.
        registry: Prompt registry for the agent identity; defaults to the
                  process-wide singleton.

    Returns:
        Configured ``crewai.Agent``.
    """
    prompt = (registry or get_registry()).get(AGENT_KEY)
    return Agent(
        role=prompt.render("role"),
        goal=prompt.render("goal"),
        backstory=prompt.render("backstory"),
        llm=get_fast_llm(cfg),
        verbose=False,
        allow_delegation=False,
    )


@dataclass
class MoodService:
    """Orchestrates mood analysis and proactive draft generation.

    Attributes:
        spotify: Spotify client for ``get_currently_playing``.
        store:   Weaviate memory store for mood patterns.
        cfg:     Application settings.
    """

    spotify: SpotifyClientProtocol
    store: WeaviateMemoryStore
    cfg: Settings
    registry: PromptRegistry = field(default_factory=get_registry)

    async def analyze(self, user_id: str) -> MoodResult:
        """Run mood analysis for *user_id*.

        Fetches the currently playing track, looks up the known pattern for
        the current time bucket, and checks for deviation.

        Args:
            user_id: UUID string of the user.

        Returns:
            ``MoodResult`` populated with current label, pattern label, and
            an optional proactive draft.
        """
        now = datetime.now(timezone.utc)
        bucket = time_bucket(now.hour, now.weekday())

        # ── Get currently playing track features ──────────────────────────────
        try:
            now_playing = await self.spotify.get_currently_playing()
        except Exception as exc:  # noqa: BLE001
            logger.warning("mood_spotify_error", error=str(exc))
            now_playing = None

        if not now_playing:
            return MoodResult(current_label="neutral", bucket=bucket)

        current_energy = float(now_playing.get("energy") or 0.5)
        current_valence = float(now_playing.get("valence") or 0.5)
        current_label = classify_mood(current_energy, current_valence)

        # ── Fetch known pattern ───────────────────────────────────────────────
        try:
            pattern = await get_pattern_for_now(user_id, self.store)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mood_pattern_error", error=str(exc))
            pattern = None

        if pattern is None:
            return MoodResult(current_label=current_label, bucket=bucket)

        pattern_energy, pattern_valence = _parse_pattern(pattern.text)
        pattern_label = classify_mood(pattern_energy, pattern_valence)
        deviation = deviates_significantly(
            current_energy, current_valence, pattern_energy, pattern_valence
        )

        if not deviation:
            return MoodResult(
                current_label=current_label,
                pattern_label=pattern_label,
                bucket=bucket,
                deviation=False,
            )

        # ── Draft proactive observation ───────────────────────────────────────
        track_name = now_playing.get("name", "this track")
        draft = await self._draft_observation(
            bucket=bucket,
            pattern_label=pattern_label,
            current_label=current_label,
            track_name=track_name,
            pattern_energy=pattern_energy,
            current_energy=current_energy,
        )

        logger.info(
            "mood_deviation_detected",
            user_id=user_id,
            bucket=bucket,
            pattern_label=pattern_label,
            current_label=current_label,
        )

        return MoodResult(
            current_label=current_label,
            pattern_label=pattern_label,
            bucket=bucket,
            deviation=True,
            proactive_draft=draft,
        )

    async def _draft_observation(
        self,
        bucket: str,
        pattern_label: str,
        current_label: str,
        track_name: str,
        pattern_energy: float,
        current_energy: float,
    ) -> str:
        """Generate a warm, natural proactive observation using the LLM.

        Falls back to a template string if the LLM call fails.

        Args:
            bucket:          Time bucket string (e.g. ``"sunday_evening"``).
            pattern_label:   User's typical mood for this bucket.
            current_label:   Current mood label.
            track_name:      Name of the currently playing track.
            pattern_energy:  Expected energy level.
            current_energy:  Actual energy level.

        Returns:
            Proactive observation string in Gia's voice.
        """
        bucket_human = bucket.replace("_", " ")
        direction = "higher" if current_energy > pattern_energy else "lower"

        prompt = self.registry.get(AGENT_KEY).render(
            "observation",
            persona=self.registry.get("persona.gia").render(),
            pattern_label=pattern_label,
            bucket_human=bucket_human,
            track_name=track_name,
            current_label=current_label,
            direction=direction,
        )

        llm = get_fast_llm(self.cfg)
        try:
            draft = await asyncio.to_thread(
                llm.call, [{"role": "user", "content": prompt}]
            )
            return draft.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("mood_llm_fallback", error=str(exc))
            return (
                f"[thoughtful] Hey — you're usually on {pattern_label} stuff "
                f"around {bucket_human}. This feels a bit different. Everything okay?"
            )
