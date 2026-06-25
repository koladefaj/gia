"""General agent — Gia's conversational voice for non-specialist turns.

Greetings, small talk, meta-questions ("what can you do?"), and the opening line
when Gia speaks first all flow through here.  Previously these returned a single
hardcoded string, which made Gia sound like a brochure that replies identically
every time.  Routing them through the persona LLM (grounded in whatever we know
about the user) makes the same turns sound like a person.

Both helpers degrade gracefully: if the LLM is unreachable they return a short,
warm, *varied-enough* fallback rather than an error, so a flaky model never
turns a hello into a stack trace.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator

from backend.app.config import Settings
from backend.app.observability.logging import get_logger
from backend.app.prompts import PromptRegistry, get_registry
from backend.app.providers.anthropic_client import get_async_anthropic, persona_model
from backend.app.providers.llm import get_llm
from backend.app.providers.openai_client import get_async_openai

logger = get_logger(__name__)

AGENT_KEY = "agents.general"

# Last-resort lines used only when the LLM call fails. Kept varied so even the
# degraded path never sounds like the old single canned greeting.
_FALLBACK_REPLIES = [
    "Hey — good to hear from you. What are you in the mood for?",
    "Hi there. Tell me what you're feeling and I'll find something.",
    "Hey you. Music, an artist, or just checking in — I'm around.",
]
_FALLBACK_OPENINGS = [
    "Hey — Gia here. What are we listening to?",
    "Hi. I'm Gia. Tell me the vibe and I'll take it from there.",
    "Hey you. What's the mood today?",
]


async def respond_general(
    message: str,
    user_context_text: str = "",
    *,
    cfg: Settings,
    registry: PromptRegistry | None = None,
    history: str = "",
) -> str:
    """Generate Gia's conversational reply to a casual / non-specialist message.

    Args:
        message:           The user's raw message (greeting, small talk, meta).
        user_context_text: Rendered ``UserContext.to_prompt_text()`` for
                           personalisation (name, taste). ``""`` when unknown.
        cfg:               Application settings (LLM provider / model).
        registry:          Prompt registry; defaults to the process-wide singleton.
        history:           Recent conversation transcript for continuity.

    Returns:
        A warm, varied reply in Gia's voice.
    """
    reg = registry or get_registry()
    prompt = _render_task_prompt(message, user_context_text, history, reg)
    return await _call(prompt, _FALLBACK_REPLIES, cfg)


def _render_task_prompt(
    message: str, user_context_text: str, history: str, reg: PromptRegistry
) -> str:
    """Render the conversational ``task`` prompt (shared by streaming + blocking)."""
    return reg.get(AGENT_KEY).render(
        "task",
        persona=reg.get("persona.gia").render(),
        user_context=user_context_text,
        message=message,
        history=history,
    )


async def stream_general(
    message: str,
    user_context_text: str = "",
    *,
    cfg: Settings,
    registry: PromptRegistry | None = None,
    history: str = "",
) -> AsyncIterator[str]:
    """Stream Gia's conversational reply as text deltas as the model produces them.

    This is the latency-critical path for chit-chat turns: instead of waiting for
    the whole reply, the caller can reassemble deltas into sentences and start TTS
    on the first sentence while the rest is still being generated.

    True token streaming for the OpenAI and Anthropic providers; for any other
    provider (or on a streaming error before the first token) it degrades to a
    single chunk from the blocking :func:`respond_general`, so callers always get
    *something*.

    Args:
        message:           The user's raw message.
        user_context_text: Rendered user context for personalisation.
        cfg:               Application settings (provider / model / key).
        registry:          Prompt registry; defaults to the singleton.
        history:           Recent conversation transcript for continuity.

    Yields:
        Text fragments (token deltas) in order. Concatenated, they form the reply.
    """
    reg = registry or get_registry()
    prompt = _render_task_prompt(message, user_context_text, history, reg)

    streamer = _provider_streamer(prompt, cfg)
    if streamer is not None:
        emitted = False
        try:
            async for delta in streamer:
                if delta:
                    emitted = True
                    yield delta
        except Exception as exc:  # noqa: BLE001 — network/provider errors
            logger.warning("general_stream_error", error=str(exc), emitted=emitted)
            if emitted:
                # Partial reply already streamed — stopping cleanly beats either a
                # crash or a duplicated fallback on top of real content.
                return
        else:
            if emitted:
                return
        # Streaming yielded nothing (empty completion or pre-token error) — fall
        # through to the blocking path so the turn is never silent.

    text = await respond_general(
        message, user_context_text, cfg=cfg, registry=reg, history=history
    )
    if text:
        yield text


def _provider_streamer(prompt: str, cfg: Settings) -> AsyncIterator[str] | None:
    """Return a provider-specific token stream, or ``None`` if none is available."""
    if cfg.llm_provider == "openai" and cfg.openai_api_key:
        return _stream_openai(prompt, cfg)
    if cfg.llm_provider == "anthropic" and cfg.anthropic_api_key:
        return _stream_anthropic(prompt, cfg)
    return None


async def _stream_openai(prompt: str, cfg: Settings) -> AsyncIterator[str]:
    """Stream completion deltas from OpenAI (via the Langfuse drop-in client)."""
    client = get_async_openai(cfg)
    stream = await client.chat.completions.create(
        model=cfg.llm_persona_model or "gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        name="general-stream",  # Langfuse generation name (drop-in)
    )
    async for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta.content
        if delta:
            yield delta


async def _stream_anthropic(prompt: str, cfg: Settings) -> AsyncIterator[str]:
    """Stream text deltas from Anthropic's Messages API (official SDK helper)."""
    client = get_async_anthropic(cfg)
    async with client.messages.stream(
        model=persona_model(cfg),
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            if text:
                yield text


async def opening_line(
    user_context_text: str = "",
    *,
    cfg: Settings,
    registry: PromptRegistry | None = None,
) -> str:
    """Generate Gia's opening line — she speaks first, before the user types.

    Args:
        user_context_text: Rendered user context for a personalised hello.
        cfg:               Application settings.
        registry:          Prompt registry; defaults to the singleton.

    Returns:
        A short, warm opener in Gia's voice.
    """
    reg = registry or get_registry()
    prompt = reg.get(AGENT_KEY).render(
        "opening",
        persona=reg.get("persona.gia").render(),
        user_context=user_context_text,
    )
    return await _call(prompt, _FALLBACK_OPENINGS, cfg)


async def _call(prompt: str, fallbacks: list[str], cfg: Settings) -> str:
    """Run the persona LLM, returning a varied fallback line on any failure."""
    llm = get_llm(cfg)
    try:
        text = await asyncio.to_thread(llm.call, [{"role": "user", "content": prompt}])
    except Exception as exc:  # noqa: BLE001
        logger.warning("general_llm_error", error=str(exc))
        return random.choice(fallbacks)
    cleaned = text.strip()
    return cleaned or random.choice(fallbacks)
