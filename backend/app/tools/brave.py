"""Brave Search API client.

Thin async HTTP wrapper around the Brave Web Search REST API.
Query format from Section 9: ``"ArtistName 2026"`` or ``"artist new release"``.

Gracefully returns ``[]`` when no API key is configured so the Artist agent
degrades cleanly in dev without requiring a Brave subscription.
"""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

_BRAVE_BASE = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchClient:
    """Async Brave Web Search client.

    Attributes:
        api_key: Brave Search subscription token.  Leave empty to run in
                 no-key mode where all searches return ``[]``.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    async def search(self, query: str, count: int = 5) -> list[dict]:
        """Search the web and return the top *count* results.

        Args:
            query: Search query string.  For artists, prefer
                   ``"Odumodublvck 2026"`` or ``"Odumodublvck new album"``.
            count: Maximum number of results to return (Brave API max 20).

        Returns:
            List of dicts with ``title``, ``url``, and ``description`` keys.
            Returns ``[]`` when the API key is not configured or on error.
        """
        if not self._api_key:
            logger.debug("brave_search_no_key", query=query)
            return []

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                _BRAVE_BASE,
                params={"q": query, "count": count, "text_decorations": "false"},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self._api_key,
                },
            )
            resp.raise_for_status()

        data = resp.json()
        results = data.get("web", {}).get("results", [])
        logger.info("brave_search_done", query=query, count=len(results))

        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "description": r.get("description", ""),
            }
            for r in results[:count]
        ]
