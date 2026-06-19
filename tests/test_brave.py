"""Tests for the Brave Search API client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_search_returns_empty_without_api_key() -> None:
    """``BraveSearchClient`` returns ``[]`` gracefully when no key is configured."""
    from backend.app.tools.brave import BraveSearchClient

    client = BraveSearchClient(api_key="")
    result = await client.search("Odumodublvck 2026")
    assert result == []


@pytest.mark.asyncio
async def test_search_returns_parsed_results() -> None:
    """Successful API response is parsed into title/url/description dicts."""
    from backend.app.tools.brave import BraveSearchClient

    fake_response = {
        "web": {
            "results": [
                {
                    "title": "Odumodublvck drops new album",
                    "url": "https://example.com/article",
                    "description": "The Nigerian rapper just released...",
                }
            ]
        }
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_response
    mock_resp.raise_for_status = MagicMock()

    with patch("backend.app.tools.brave.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        client = BraveSearchClient(api_key="test-key")
        results = await client.search("Odumodublvck 2026", count=1)

    assert len(results) == 1
    assert results[0]["title"] == "Odumodublvck drops new album"
    assert results[0]["url"] == "https://example.com/article"
    assert "rapper" in results[0]["description"]


@pytest.mark.asyncio
async def test_search_respects_count_limit() -> None:
    """Only *count* results are returned even if API returns more."""
    from backend.app.tools.brave import BraveSearchClient

    many_results = [
        {"title": f"Result {i}", "url": f"https://example.com/{i}", "description": "..."}
        for i in range(10)
    ]
    fake_response = {"web": {"results": many_results}}

    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_response
    mock_resp.raise_for_status = MagicMock()

    with patch("backend.app.tools.brave.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        client = BraveSearchClient(api_key="test-key")
        results = await client.search("query", count=3)

    assert len(results) == 3


@pytest.mark.asyncio
async def test_search_empty_web_results() -> None:
    """When Brave returns no results, ``[]`` is returned cleanly."""
    from backend.app.tools.brave import BraveSearchClient

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"web": {"results": []}}
    mock_resp.raise_for_status = MagicMock()

    with patch("backend.app.tools.brave.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        client = BraveSearchClient(api_key="test-key")
        results = await client.search("obscure query")

    assert results == []
