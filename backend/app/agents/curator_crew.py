"""DJ → Curator collaboration — a real CrewAI multi-agent crew.

Two agents that genuinely hand work to each other:

  1. **Scout** — searches the catalogue with a Spotify ``search_tracks`` tool and
     returns a wide-but-relevant candidate pool. Tool-using; never invents tracks.
  2. **Curator** — reranks the Scout's candidates against the listener's taste
     profile and the current moment, dropping matches that are technically
     relevant but emotionally wrong, and returns the best few with one warm line
     each.

The hand-off is the point: ``curate_task`` declares ``context=[scout_task]``, so
CrewAI feeds the Scout's output into the Curator's prompt — that is the
"agents talking to each other" part of a collaborative crew. The Curator emits
structured ``CuratedPicks`` via ``output_pydantic`` so callers get typed data,
not prose to re-parse.

This lives **off the live conversational path** and is gated by
``cfg.crewai_curator_enabled`` (env ``CREWAI_CURATOR_ENABLED``, default off).
Inter-agent collaboration adds LLM round-trips, so it is for enrichment /
"deep pick" requests and offline synthesis — never the sub-second voice reply,
where the deterministic router + single specialists are faster and keep one
coherent voice.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from pydantic import BaseModel, Field

from backend.app.config import Settings
from backend.app.observability.logging import get_logger
from backend.app.providers.llm import get_fast_llm, get_llm

logger = get_logger(__name__)

# A track search function: (query, limit) -> list of {"name", "artist", "uri"}.
# Injected so the crew reuses our already-prewarmed Spotify bridge in production
# and a deterministic fake in tests/demos — the crew never imports Spotify itself.
SearchFn = Callable[[str, int], list[dict]]


class CuratedPick(BaseModel):
    """One curated track with the Curator's reasoning."""

    track: str = Field(description="Track title")
    artist: str = Field(description="Primary artist")
    reason: str = Field(description="One warm sentence on why it fits this moment")


class CuratedPicks(BaseModel):
    """The Curator's final, ranked selection."""

    picks: list[CuratedPick] = Field(description="Best tracks, strongest first")


def _build_search_tool(search_fn: SearchFn):
    """Wrap a track-search callable as a CrewAI tool the Scout can call.

    Kept synchronous on purpose: the crew runs via ``kickoff`` inside a worker
    thread, so a sync tool avoids nesting an event loop. The docstring is what
    the agent reads to decide how to use it, so it carries real instruction.
    """

    @tool("search_tracks")
    def search_tracks(query: str) -> str:
        """Search the music catalogue for candidate tracks matching a query.

        Returns up to 8 candidates as "Title — Artist" lines. Always use this
        before proposing tracks; never invent tracks that are not returned here.
        """
        results = search_fn(query, 8)
        if not results:
            return "No candidates found."
        return "\n".join(
            f"- {r.get('name', '?')} — {r.get('artist', '?')}" for r in results
        )

    return search_tracks


def build_curation_crew(
    search_fn: SearchFn, cfg: Settings, *, verbose: bool = False
) -> Crew:
    """Assemble the two-agent Scout → Curator crew.

    Args:
        search_fn: Track search callable injected into the Scout's tool.
        cfg:       App settings (selects the LLM provider/models).
        verbose:   Surface the agents' reasoning + tool calls (demo/debug).

    Returns:
        A ``Process.sequential`` ``Crew`` whose Curator task consumes the
        Scout task's output via ``context``.
    """
    search_tool = _build_search_tool(search_fn)

    scout = Agent(
        role="Music Scout",
        goal="Find a strong, varied pool of candidate tracks for the request.",
        backstory=(
            "You know the catalogue cold. You cast a wide-but-relevant net and "
            "never invent tracks — you only return what the search tool gives you."
        ),
        tools=[search_tool],
        llm=get_fast_llm(cfg),  # cheap/fast tier — this step is search, not taste
        max_iter=4,
        verbose=verbose,
        allow_delegation=False,
    )

    curator = Agent(
        role="Taste Curator",
        goal=(
            "Pick the few tracks that fit THIS listener and THIS moment best, "
            "and say why in one warm line each."
        ),
        backstory=(
            "You have real taste and a point of view. You weigh the candidates "
            "against the listener's history and mood, and you will drop an "
            "obvious match that is emotionally wrong for the moment."
        ),
        llm=get_llm(cfg),  # persona tier — judgement + voice live here
        max_iter=3,
        verbose=verbose,
        allow_delegation=False,
    )

    scout_task = Task(
        description=(
            "Search for candidate tracks for this request: '{query}'.\n"
            "Use the search_tracks tool, then return the raw candidate list."
        ),
        expected_output="A list of candidate tracks as 'Title — Artist' lines.",
        agent=scout,
    )

    curate_task = Task(
        description=(
            "From the Scout's candidates, choose and rank the best 3 for this "
            "listener.\n\n"
            "Listener taste profile:\n{taste_profile}\n\n"
            "Current moment:\n{moment}\n\n"
            "Drop matches that don't fit the moment emotionally. Give each pick "
            "one warm sentence on why it fits."
        ),
        expected_output=(
            "The 3 best tracks, strongest first, each with a one-line reason."
        ),
        agent=curator,
        context=[scout_task],  # ← Scout's output is handed to the Curator
        output_pydantic=CuratedPicks,  # ← typed, validated result
    )

    return Crew(
        agents=[scout, curator],
        tasks=[scout_task, curate_task],
        process=Process.sequential,
        verbose=verbose,
    )


async def curate(
    query: str,
    *,
    taste_profile: str,
    moment: str,
    search_fn: SearchFn,
    cfg: Settings,
    verbose: bool = False,
) -> CuratedPicks:
    """Run the Scout → Curator crew and return typed picks.

    Runs ``kickoff`` in a worker thread so the collaborative crew (multiple LLM
    round-trips) never blocks the event loop. Call sites must gate this on
    ``cfg.crewai_curator_enabled`` — it is deliberately off the voice path.

    Args:
        query:         The user's music request.
        taste_profile: A short text summary of the listener's taste/history.
        moment:        Current context (time, mood, weather, activity).
        search_fn:     Track search callable (real Spotify bridge or a fake).
        cfg:           App settings.
        verbose:       Surface agent reasoning (demo/debug).

    Returns:
        ``CuratedPicks`` — possibly empty if the crew returned no structured output.
    """
    crew = build_curation_crew(search_fn, cfg, verbose=verbose)
    result = await asyncio.to_thread(
        crew.kickoff,
        inputs={"query": query, "taste_profile": taste_profile, "moment": moment},
    )
    picks = result.pydantic
    logger.info(
        "curator_crew_done", query=query, n=len(picks.picks) if picks else 0
    )
    return picks or CuratedPicks(picks=[])
