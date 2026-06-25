"""Mood agent — detect mood from recent listening, surface proactive insights.

``MoodService.analyze()`` is the entry point.  It:

1. Determines the current time bucket (e.g. ``"sunday_evening"``).
2. Labels the mood of the user's *recently played* tracks via the LLM labeler
   (Spotify no longer exposes audio features — see ``mood.labeler``).
3. Fetches the user's known mood pattern for that bucket from Weaviate.
4. If the current label differs from the pattern, drafts a proactive observation.
5. Returns a ``MoodResult`` for injection into the crew reply.

Mood is a closed-vocabulary label (``mood.classifier.MOOD_LABELS``), so current
vs. pattern is a clean string comparison; the LLM is used to label and to phrase.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

from backend.app.config import Settings
from backend.app.interfaces import SpotifyClientProtocol
from backend.app.memory.store import WeaviateMemoryStore
from backend.app.mood.classifier import time_bucket
from backend.app.mood.labeler import label_mood
from backend.app.mood.proactive import _parse_pattern, get_pattern_for_now
from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

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
        now = datetime.now(UTC)
        bucket = time_bucket(now.hour, now.weekday())

        # ── Label the current mood from what they're actually playing ─────────
        try:
            recent = await self.spotify.get_recently_played(limit=10)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mood_spotify_error", error=str(exc))
            recent = []

        current_label = await label_mood(recent, self.cfg) if recent else "neutral"

        # ── Fetch the known pattern for this time bucket ──────────────────────
        try:
            pattern = await get_pattern_for_now(user_id, self.store)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mood_pattern_error", error=str(exc))
            pattern = None

        if pattern is None:
            return MoodResult(current_label=current_label, bucket=bucket)

        pattern_label = _parse_pattern(pattern.text)
        deviation = current_label not in ("neutral", pattern_label)

        if not deviation:
            return MoodResult(
                current_label=current_label,
                pattern_label=pattern_label,
                bucket=bucket,
                deviation=False,
            )

        # ── Draft a proactive observation about the shift ─────────────────────
        draft = await self._draft_observation(bucket, pattern_label, current_label)

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
    ) -> str:
        """Generate a warm, natural observation about a mood shift via the LLM.

        Falls back to a template string if the LLM call fails.

        Args:
            bucket:        Time bucket string (e.g. ``"sunday_evening"``).
            pattern_label: User's typical mood for this bucket.
            current_label: The mood their current listening reads as.

        Returns:
            Proactive observation string in Gia's voice.
        """
        bucket_human = bucket.replace("_", " ")
        prompt = self.registry.get(AGENT_KEY).render(
            "observation",
            persona=self.registry.get("persona.gia").render(),
            pattern_label=pattern_label,
            bucket_human=bucket_human,
            current_label=current_label,
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
