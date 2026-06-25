"""DJ agent — track discovery and recommendation.

``DJService.recommend()`` is the single entry point.  It:

1. Searches Spotify for tracks matching the user's query.
2. Generates a warm, grounded recommendation via the persona LLM.
3. Optionally starts Spotify playback immediately.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from backend.app.config import Settings
from backend.app.interfaces import SpotifyClientProtocol
from backend.app.observability.logging import get_logger
from backend.app.prompts import PromptRegistry, get_registry
from backend.app.providers.llm import get_llm
from backend.app.providers.tts import has_audio_tag
from backend.app.schemas.dj import CrossfadeQueue, DJResponse, TrackItem

logger = get_logger(__name__)

AGENT_KEY = "agents.dj"


def _to_track(raw: dict) -> TrackItem:
    """Build a ``TrackItem`` from a Spotify search result (uri / name / artist)."""
    return TrackItem(
        uri=str(raw.get("uri", "")),
        name=str(raw.get("name", "")),
        artist=str(raw.get("artist", "")),
    )


@dataclass
class DJService:
    """Orchestrates track search, crossfade sequencing, and LLM recommendation.

    Attributes:
        spotify: Spotify client (live or fake).
        cfg:     Application settings.
    """

    spotify: SpotifyClientProtocol
    cfg: Settings
    registry: PromptRegistry = field(default_factory=get_registry)

    # Honour at most this many explicitly-named tracks in a per-title request.
    _MAX_NAMED_QUEUE = 10

    async def search_only(
        self, query: str, n: int = 4
    ) -> tuple[TrackItem, list[TrackItem]]:
        """Search *query* and return ``(seed, queue_tracks)`` — nothing else.

        Read-only: no playback, no LLM. Used to run the Spotify search
        *speculatively* (in parallel with the router) so a music command's
        lookup is already done by the time the router lands. The result is fed
        back into :meth:`recommend` via ``prefetched`` to skip the duplicate
        search. Safe to call without committing to anything.
        """
        return await self._search_query(query, n)

    async def recommend(
        self,
        query: str,
        user_context_text: str = "",
        start_playback: bool = False,
        n: int = 4,
        requested_titles: list[str] | None = None,
        prefetched: tuple[TrackItem, list[TrackItem]] | None = None,
    ) -> DJResponse:
        """Search for tracks, build the queue, and generate a recommendation.

        Two queue-building modes:

        - **Per-title** — when the user named two or more specific tracks
          (``requested_titles``), each title is searched on its own and the
          matches are queued in the order named. "play So Will I and queue
          Promises next" puts Promises *next*, not whatever a sequencer chose.
        - **Vibe / single** — the search's top hit is the seed and the next
          results are the queue, in relevance order.

        Spotify no longer exposes audio features to new apps, so there is no
        harmonic (Camelot/energy) sequencing to do — the real ordering signal is
        either the user's stated order or search relevance.

        Args:
            query:             Natural-language request (the router's resolved
                               search string).
            user_context_text: Rendered user context for the prompt.
            start_playback:    If ``True``, immediately start the seed track.
            n:                 Queue depth for vibe requests (ignored when the
                               user named the tracks — those are honoured in full).
            requested_titles:  Specific titles the user named, in order. The first
                               is the primary "did you mean…?" target.
            prefetched:        A ``(seed, queue_tracks)`` pair from a speculative
                               :meth:`search_only` run. When given (and the user did
                               not name multiple specific titles), it's used instead
                               of searching again — the latency win. Ignored on the
                               per-title path, which needs its own per-title search.

        Returns:
            A ``DJResponse`` with the recommendation, seed track, and queue.

        Raises:
            ValueError: If no tracks are found at all.
        """
        titles = [t.strip() for t in (requested_titles or []) if t and t.strip()]

        # ── 1. Build seed + queue ────────────────────────────────────────────
        missing: list[str] = []
        if len(titles) >= 2:
            seed, queue_tracks, missing = await self._search_named_titles(titles)
            if seed is None:  # none of the named tracks resolved → plain search
                seed, queue_tracks = await self._search_query(query, n)
        elif prefetched is not None:
            # Reuse the speculative search that ran alongside the router.
            seed, queue_tracks = prefetched
        else:
            seed, queue_tracks = await self._search_query(query, n)

        # ── 2. Generate natural language recommendation ──────────────────────
        queued_names = ", ".join(f"{t.name} by {t.artist}" for t in queue_tracks[:3]) or "none"
        prompt = self.registry.get(AGENT_KEY).render(
            "task",
            persona=self.registry.get("persona.gia").render(),
            user_context=user_context_text,
            query=query,
            seed_name=seed.name,
            seed_artist=seed.artist,
            queued_names=queued_names,
            start_playback=start_playback,
            requested_title=titles[0] if titles else None,
            missing_titles=", ".join(missing) or None,
        )

        llm = get_llm(self.cfg)
        try:
            recommendation = await asyncio.to_thread(
                llm.call, [{"role": "user", "content": prompt}]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("dj_llm_error", error=str(exc))
            recommendation = f"Here's {seed.name} by {seed.artist} — should fit the vibe."

        recommendation = recommendation.strip()
        # Music is the product moment, so it must land on the warm eleven_v3 model,
        # not the flat flash model. The TTS picker routes to v3 only when a line
        # carries an [audio tag]; a plain "Playing X now..." has none and would go
        # to flash (the robotic sound). Guarantee a delivery cue when the LLM
        # didn't add one — v3 renders it as warmth, captions strip it.
        if not has_audio_tag(recommendation):
            recommendation = f"[warm] {recommendation}"

        # ── 3. Optionally start playback ─────────────────────────────────────
        playback_started = False
        if start_playback:
            await self.spotify.start_playback(seed.uri)
            playback_started = True
            logger.info("dj_playback_started", uri=seed.uri)

        logger.info(
            "dj_recommend_done", query=query, seed=seed.name, queue_depth=len(queue_tracks)
        )

        return DJResponse(
            recommendation=recommendation.strip(),
            primary_track=seed,
            queue=CrossfadeQueue(seed_uri=seed.uri, tracks=queue_tracks, crossfade_ms=3000),
            playback_started=playback_started,
        )

    async def _search_query(self, query: str, n: int) -> tuple[TrackItem, list[TrackItem]]:
        """Search *query*; return its top hit as seed + the next *n* as the queue."""
        raw_tracks = await self.spotify.search_tracks(query, limit=20)
        items = [_to_track(t) for t in raw_tracks if t.get("uri")]
        if not items:
            raise ValueError(f"No tracks found for query: {query!r}")
        seed = items[0]
        queue_tracks = [t for t in items[1:] if t.uri != seed.uri][:n]
        return seed, queue_tracks

    async def _search_named_titles(
        self, titles: list[str]
    ) -> tuple[TrackItem | None, list[TrackItem], list[str]]:
        """Search each named title and queue the matches in the user's order.

        Returns ``(seed, queue_tracks, missing)``. ``seed`` is ``None`` when no
        named title resolved at all (the caller then falls back to a plain query
        search), and *missing* lists the titles Spotify returned nothing for.
        """
        found: list[TrackItem] = []
        missing: list[str] = []
        for title in titles[: self._MAX_NAMED_QUEUE]:
            results = await self.spotify.search_tracks(title, limit=5)
            hit = next((r for r in results if r.get("uri")), None)
            if hit is not None:
                found.append(_to_track(hit))
            else:
                missing.append(title)
        if not found:
            return None, [], missing
        return found[0], found[1:], missing
