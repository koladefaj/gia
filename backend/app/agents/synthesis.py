"""Reply synthesis — merge multiple agents' outputs into one coherent answer.

The payoff of multi-agent planning is a *single* synthesised reply, not a stack
of concatenated fragments.  When a turn fans out (e.g. a DJ pick plus an artist
aside, or a proactive mood note plus a recommendation), this module fuses the
parts into one warm Gia reply.

It is deliberately conservative:
  - With zero or one part there is nothing to merge, so it returns immediately
    (no LLM call) — which is the common, single-agent case.
  - It is gated by ``settings.synthesis_enabled`` at the call site, OFF by
    default, so the extra LLM hop never lands on the voice path unasked.
  - On any LLM failure it falls back to joining the parts, so synthesis can
    only improve a reply, never break one.
"""

from __future__ import annotations

import asyncio

from backend.app.config import Settings
from backend.app.observability.logging import get_logger
from backend.app.persona.prompt import render_persona
from backend.app.prompts import PromptRegistry, get_registry
from backend.app.providers.llm import get_fast_llm

logger = get_logger(__name__)

SYNTHESIS_KEY = "agents.synthesis"


async def synthesize_reply(
    parts: list[str],
    query: str,
    cfg: Settings,
    registry: PromptRegistry | None = None,
) -> str:
    """Combine reply *parts* into one Gia reply.

    Args:
        parts:    The individual agent outputs, in order.
        query:    The user's message (gives the merge context).
        cfg:      Application settings.
        registry: Prompt registry; defaults to the process-wide singleton.

    Returns:
        A single combined reply.  Returns the lone part (or a plain join) when
        there is nothing to synthesise or the LLM call fails.
    """
    cleaned = [p.strip() for p in parts if p and p.strip()]
    if len(cleaned) <= 1:
        return cleaned[0] if cleaned else ""

    reg = registry or get_registry()
    prompt = reg.get(SYNTHESIS_KEY).render(
        "merge",
        persona=render_persona(reg),
        query=query,
        parts="\n".join(f"- {p}" for p in cleaned),
    )

    llm = get_fast_llm(cfg)
    try:
        merged = await asyncio.to_thread(llm.call, [{"role": "user", "content": prompt}])
        return merged.strip() or " ".join(cleaned)
    except Exception as exc:  # noqa: BLE001
        logger.warning("synthesis_error", error=str(exc))
        return " ".join(cleaned)
