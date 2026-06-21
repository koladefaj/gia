"""Hybrid router — one fast structured call that classifies the whole turn.

Replaces keyword classification (which breaks on phrasing like "play something
that feels like when I used to listen to Tems") with a single small-model call
returning a validated :class:`RouterDecision`: intent, tone, confidence,
engagement mode, and which retrievers to fire.

Reliability over cleverness: when the configured provider is OpenAI we use JSON
mode for a guaranteed object; otherwise we fall back to crewai's blocking call
and parse leniently.  Any failure degrades to a warm GENERAL_CHAT default so a
flaky model never wedges a turn.
"""

from __future__ import annotations

import asyncio
import time

from pydantic import ValidationError

from backend.app.config import Settings
from backend.app.observability.logging import get_logger
from backend.app.prompts import PromptRegistry, get_registry
from backend.app.providers.openai_client import extract_json_object, get_async_openai
from backend.app.schemas.router import RouterDecision, safe_default_decision

logger = get_logger(__name__)

AGENT_KEY = "agents.hybrid_router"


async def classify_turn(
    message: str,
    cfg: Settings,
    registry: PromptRegistry | None = None,
    history: str = "",
) -> RouterDecision:
    """Classify *message* into a structured :class:`RouterDecision`.

    Args:
        message:  The user's raw message.
        cfg:      Application settings (provider, model, key).
        registry: Prompt registry; defaults to the process-wide singleton.
        history:  Recent conversation transcript (oldest→newest) so the router
                  can resolve references ("play it now") and derive a clean
                  ``search_query``. Empty when there is no prior turn.

    Returns:
        A validated ``RouterDecision``. Never raises — on any failure it returns
        the safe GENERAL_CHAT default (confidence 0.0), which the caller can
        escalate to the Planner.
    """
    reg = registry or get_registry()
    prompt = reg.get(AGENT_KEY)
    system = prompt.render("system")
    user = prompt.render("user", message=message, history=history)

    t0 = time.monotonic()
    try:
        raw = await _complete(system, user, cfg)
        decision = RouterDecision.model_validate(extract_json_object(raw))
    except (ValidationError, ValueError, KeyError) as exc:
        logger.warning("router_parse_failed", error=str(exc))
        return safe_default_decision()
    except Exception as exc:  # noqa: BLE001 — network/provider errors
        logger.warning("router_call_failed", error=str(exc))
        return safe_default_decision()

    logger.debug(
        "router_decision",
        intent=decision.intent.value,
        tone=decision.tone.value,
        mode=decision.engagement_mode.value,
        confidence=decision.confidence,
        latency_ms=round((time.monotonic() - t0) * 1000, 1),
    )
    return decision


async def _complete(system: str, user: str, cfg: Settings) -> str:
    """Run the router completion, preferring OpenAI JSON mode."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if cfg.llm_provider == "openai" and cfg.openai_api_key:
        client = get_async_openai(cfg)
        resp = await client.chat.completions.create(
            model=cfg.router_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            name="router-classify",  # Langfuse generation name (drop-in)
        )
        return resp.choices[0].message.content or ""

    # Non-OpenAI providers: use the crewai fast LLM and parse leniently.
    from backend.app.providers.llm import get_fast_llm  # noqa: PLC0415

    llm = get_fast_llm(cfg, model=cfg.router_model)
    return await asyncio.to_thread(llm.call, messages)  # type: ignore[arg-type]
