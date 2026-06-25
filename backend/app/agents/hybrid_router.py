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

from backend.app.agents.router import _keyword_classify
from backend.app.config import Settings
from backend.app.observability.logging import get_logger
from backend.app.prompts import PromptRegistry, get_registry
from backend.app.providers.openai_client import extract_json_object, get_async_openai
from backend.app.schemas.chat import IntentType
from backend.app.schemas.router import (
    EngagementMode,
    RouterDecision,
    Tone,
    safe_default_decision,
)

logger = get_logger(__name__)

AGENT_KEY = "agents.hybrid_router"


def fast_keyword_decision(message: str) -> RouterDecision | None:
    """Return a confident ``RouterDecision`` from sub-ms keywords, or ``None``.

    Tier-1 of the router: short-circuits the ~2s LLM call only for the
    unambiguous-conversation case — the keyword classifier returns ``GENERAL``
    exclusively when a greeting/small-talk keyword is present and there is *zero*
    music, artist, mood, or queue signal. Those turns run no specialist and need
    no pronoun/query resolution, so a warm ``GENERAL_CHAT`` decision is exactly
    what the LLM would return, minutes faster.

    Everything else — music, artist, mood, news, or any ambiguity — returns
    ``None`` so the caller falls through to :func:`classify_turn`, which does the
    reference resolution and ``search_query`` extraction keywords can't.
    """
    if _keyword_classify(message) is IntentType.GENERAL:
        return RouterDecision(
            intent=IntentType.GENERAL_CHAT,
            tone=Tone.WARM,
            confidence=1.0,
            engagement_mode=EngagementMode.DIRECT_EXECUTE,
        )
    return None


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
