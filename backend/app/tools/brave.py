"""Brave Search API client.

Thin async HTTP wrapper around the Brave Search REST API. Exposes the two
endpoints the agents need:

* :meth:`BraveSearchClient.search` — **web** search (``/res/v1/web/search``),
  the richest result set (artist pages, Wikipedia, streaming, discussions).
* :meth:`BraveSearchClient.news` — **news** search (``/res/v1/news/search``),
  for breaking/headline queries.
* :meth:`BraveSearchClient.recent` — picks the right endpoint and applies a
  freshness window for "what's the latest…" style queries.

Both endpoints take a ``freshness`` filter (``pd``/``pw``/``pm``/``py`` or a
date range) so current-events queries don't get stale top-ranked pages.

Gracefully returns ``[]`` when no API key is configured so callers degrade
cleanly in dev without a Brave subscription.
"""

from __future__ import annotations

import httpx

from backend.app.observability.logging import get_logger
from backend.app.tools.resilience import CircuitBreaker, resilient_call

logger = get_logger(__name__)

_BRAVE_WEB = "https://api.search.brave.com/res/v1/web/search"
_BRAVE_NEWS = "https://api.search.brave.com/res/v1/news/search"

# Module-level breaker: BraveSearchClient is created per-request (stateless), so
# the breaker must be shared across instances to track failures meaningfully.
_BRAVE_BREAKER = CircuitBreaker("brave", threshold=5, cooldown=30.0)


class BraveSearchClient:
    """Async Brave Search client (web + news).

    Attributes:
        api_key: Brave Search subscription token.  Leave empty to run in
                 no-key mode where all searches return ``[]``.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def _get(self, base: str, params: dict, *, name: str) -> list[dict]:
        """Run a Brave GET request, returning the raw results list.

        Guarded by :func:`resilient_call` (timeout + retry + shared circuit
        breaker), so a slow or flapping Brave endpoint fails fast instead of
        stalling the turn. Returns ``[]`` with no key or on error.
        """
        if not self._api_key:
            logger.debug("brave_no_key", q=params.get("q"))
            return []

        async def _do() -> httpx.Response:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    base,
                    params={k: v for k, v in params.items() if v is not None},
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "X-Subscription-Token": self._api_key,
                    },
                )
                resp.raise_for_status()
                return resp

        resp = await resilient_call(
            _do, name=name, timeout_s=12.0, retries=1, breaker=_BRAVE_BREAKER
        )
        data = resp.json()
        # Web responses nest results under "web"; news responses are top-level.
        return data.get("web", {}).get("results", []) or data.get("results", [])

    @staticmethod
    def _normalize(results: list[dict], count: int) -> list[dict]:
        """Project Brave results to the fields callers use (incl. recency ``age``)."""
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "description": r.get("description", ""),
                # Human-readable recency ("2 hours ago") + ISO publish date when
                # present — both let the LLM judge how current a source is.
                "age": r.get("age", ""),
                "page_age": r.get("page_age", ""),
            }
            for r in results[:count]
        ]

    async def search(
        self, query: str, count: int = 5, *, freshness: str | None = None
    ) -> list[dict]:
        """Web search — the top *count* results.

        Args:
            query:     Search query string.
            count:     Maximum results to return (Brave web max 20).
            freshness: Optional recency filter — ``pd``/``pw``/``pm``/``py`` or a
                       ``YYYY-MM-DDtoYYYY-MM-DD`` range. ``None`` = all time.

        Returns:
            List of dicts with ``title``, ``url``, ``description``, ``age``,
            ``page_age``. ``[]`` when no key is configured or on error.
        """
        results = await self._get(
            _BRAVE_WEB,
            {
                "q": query,
                "count": count,
                "freshness": freshness,
                "text_decorations": "false",
            },
            name="brave.web",
        )
        logger.info("brave_web_done", query=query, count=len(results), freshness=freshness)
        return self._normalize(results, count)

    async def news(
        self, query: str, count: int = 5, *, freshness: str | None = "pw"
    ) -> list[dict]:
        """News search — recent articles for breaking/headline queries.

        Args:
            query:     Search query string.
            count:     Maximum results (Brave news max 50).
            freshness: Recency window; defaults to the past week (``pw``). Use
                       ``pd`` for breaking news.

        Returns:
            List of dicts (same shape as :meth:`search`). ``[]`` with no key/error.
        """
        results = await self._get(
            _BRAVE_NEWS,
            {"q": query, "count": count, "freshness": freshness},
            name="brave.news",
        )
        logger.info("brave_news_done", query=query, count=len(results), freshness=freshness)
        return self._normalize(results, count)

    async def recent(self, query: str, count: int = 5, *, breaking: bool = False) -> list[dict]:
        """Grounding search for "what's the latest…" turns.

        Web search is the default — it covers music, releases, people, and facts
        far better than the narrow news index (a new album shows up on artist /
        streaming / Wikipedia pages long before there's a news article). For an
        explicitly breaking-news query, hit the news endpoint instead.

        Args:
            query:    Search query string.
            count:    Maximum results to return.
            breaking: ``True`` for "today/headlines/breaking" queries → news feed.

        Returns:
            Normalised results, freshest first where the endpoint supports it.
        """
        if breaking:
            results = await self.news(query, count, freshness="pd")
            if results:
                return results
            # Sparse news feed (common for non-mainstream topics) → fall back to
            # web with a tight freshness window so we still ground the answer.
        return await self.search(query, count, freshness="pm")
