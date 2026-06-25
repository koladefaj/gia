"""Brave Search API client.

Thin async HTTP wrapper around the Brave Web Search REST API.
Query format from Section 9: ``"ArtistName 2026"`` or ``"artist new release"``.

Gracefully returns ``[]`` when no API key is configured so the Artist agent
degrades cleanly in dev without requiring a Brave subscription.
"""

from __future__ import annotations

import httpx

from backend.app.observability.logging import get_logger
from backend.app.tools.resilience import CircuitBreaker, resilient_call

logger = get_logger(__name__)

_BRAVE_BASE = "https://api.search.brave.com/res/v1/web/search"

# Module-level breaker: BraveSearchClient is created per-request (stateless), so
# the breaker must be shared across instances to track failures meaningfully.
_BRAVE_BREAKER = CircuitBreaker("brave", threshold=5, cooldown=30.0)


class BraveSearchClient:
    """Async Brave Web Search client.

    Attributes:
        api_key: Brave Search subscription token.  Leave empty to run in
                 no-key mode where all searches return ``[]``.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def search(self, query: str, count: int = 5) -> list[dict]:
        """Search the web and return the top *count* results.

        Guarded by :func:`resilient_call` (timeout + retry + shared circuit
        breaker), so a slow or flapping Brave endpoint fails fast instead of
        stalling the Artist agent.

        Args:
            query: Search query string.  For artists, prefer
                   ``"Odumodublvck 2026"`` or ``"Odumodublvck new album"``.
            count: Maximum number of results to return (Brave API max 20).

        Returns:
            List of dicts with ``title``, ``url``, and ``description`` keys.
            Returns ``[]`` when the API key is not configured or on error.

        Raises:
            CircuitOpenError: If Brave has been failing and the breaker is open.
        """
        if not self._api_key:
            logger.debug("brave_search_no_key", query=query)
            return []

        async def _do() -> httpx.Response:
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
                return resp

        resp = await resilient_call(
            _do, name="brave.search", timeout_s=12.0, retries=1, breaker=_BRAVE_BREAKER
        )
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
