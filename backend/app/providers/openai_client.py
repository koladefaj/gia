"""Shared async OpenAI client + JSON helpers for the voice pipeline.

The intent-aware pipeline needs two things crewai's blocking ``LLM.call`` does
not give us cleanly: a fast structured-JSON router call and true token streaming
for the conversation agent.  Both go straight through the OpenAI SDK, so they
share one cached ``AsyncOpenAI`` instance here.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from backend.app.config import Settings


@lru_cache(maxsize=4)
def _client_for(api_key: str, base_url: str | None) -> Any:
    """Return a cached ``AsyncOpenAI`` keyed by credentials (one per loop is fine).

    Uses the Langfuse drop-in (``langfuse.openai.AsyncOpenAI``) so every call is
    traced as a generation — model name, token usage and cost are captured
    automatically and nest under whichever agent span is active.  When Langfuse
    has no credentials the drop-in is a transparent passthrough, so behaviour in
    dev is unchanged.  Falls back to the plain SDK if the drop-in is unavailable.
    """
    try:
        from langfuse.openai import AsyncOpenAI  # noqa: PLC0415
    except ImportError:
        from openai import AsyncOpenAI  # noqa: PLC0415

    # Keep connections warm between turns. The default httpx pool lets idle
    # connections expire quickly, so a conversational gap forces a fresh TCP+TLS
    # handshake (~300-460ms) on the next router/reply call — a recurring tax on
    # the critical path. A long keepalive holds the connection open across turns.
    import httpx  # noqa: PLC0415

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=5.0),
        limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=120.0),
    )
    return AsyncOpenAI(api_key=api_key, base_url=base_url or None, http_client=http_client)


def get_async_openai(cfg: Settings) -> Any:
    """Return the shared ``AsyncOpenAI`` client for the configured key."""
    return _client_for(cfg.openai_api_key, None)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from *text*, tolerating fences/prose.

    Raises:
        ValueError: If no JSON object can be parsed.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?|```$", "", stripped, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        match = _JSON_OBJ_RE.search(stripped)
        if not match:
            raise ValueError(f"No JSON object found in: {text[:200]!r}") from None
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object")
    return obj
