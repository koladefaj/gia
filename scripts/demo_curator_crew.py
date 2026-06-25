"""Demo: the CrewAI multi-agent DJ → Curator collaboration.

Runs the real two-agent crew (Scout searches with a tool, Curator reranks the
Scout's output via context hand-off) against a deterministic fake catalogue, so
it reproduces without Spotify auth. Uses the real configured LLM.

Run inside the api container:
    docker compose exec -T api python scripts/demo_curator_crew.py
"""

from __future__ import annotations

import asyncio

from backend.app.agents.curator_crew import curate
from backend.app.config import Settings

# A small deterministic "catalogue" so the demo is reproducible. In production
# this is replaced by the real Spotify search bridge.
_CATALOGUE = [
    {"name": "Nights", "artist": "Frank Ocean", "uri": "u1"},
    {"name": "Self Control", "artist": "Frank Ocean", "uri": "u2"},
    {"name": "Redbone", "artist": "Childish Gambino", "uri": "u3"},
    {"name": "Best Part", "artist": "Daniel Caesar", "uri": "u4"},
    {"name": "Location", "artist": "Khalid", "uri": "u5"},
    {"name": "Sunday Morning", "artist": "Maroon 5", "uri": "u6"},
    {"name": "Electric Feel", "artist": "MGMT", "uri": "u7"},
    {"name": "Pink + White", "artist": "Frank Ocean", "uri": "u8"},
]


def fake_search(query: str, limit: int) -> list[dict]:
    """Pretend Spotify search — returns the whole small catalogue."""
    return _CATALOGUE[:limit]


async def main() -> None:
    cfg = Settings()
    print(f"LLM provider: {cfg.llm_provider}\n")

    picks = await curate(
        query="something warm and a little nostalgic for a rainy evening",
        taste_profile=(
            "Leans alt-R&B and soul; loves Frank Ocean and Daniel Caesar; "
            "warms to mellow, emotive vocals; cools on high-energy pop."
        ),
        moment="Tuesday evening, light rain in Lagos, winding down after work.",
        search_fn=fake_search,
        cfg=cfg,
        verbose=True,  # show both agents working + the tool call + the hand-off
    )

    print("\n================  CURATOR'S FINAL PICKS  ================")
    for i, p in enumerate(picks.picks, 1):
        print(f"{i}. {p.track} — {p.artist}\n   {p.reason}")
    print("========================================================")


if __name__ == "__main__":
    asyncio.run(main())
