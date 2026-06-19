"""LLM-powered memory extraction.

Given a conversation transcript and the user's existing relevant memories,
the extractor identifies *durable* preferences worth keeping and returns
them as structured ``ExtractedMemory`` objects.

One-off logistics ("play it at 8pm") are discarded.  Explicit contradictions
("actually I'm into high-energy now") are returned with a ``supersedes_id``
pointing at the old memory UUID so the caller can delete it.

The LLM call is synchronous (LiteLLM / CrewAI's LLM wrapper), so it runs
inside ``asyncio.to_thread`` to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import json
import re

from backend.app.config import Settings
from backend.app.observability.logging import get_logger
from backend.app.prompts import PromptRegistry, get_registry
from backend.app.providers.llm import get_fast_llm
from backend.app.schemas.memory import ExtractedMemory, MemoryEntry

logger = get_logger(__name__)

EXTRACTOR_KEY = "agents.memory"


def _parse_extracted_memories(raw: str) -> list[ExtractedMemory]:
    """Parse the LLM's raw text response into ``ExtractedMemory`` objects.

    Deliberately tolerant — small local models (e.g. gemma3:4b) tend to wrap
    JSON in ```json fences or add a sentence of preamble.  We strip code fences,
    then take the *outermost* ``[...]`` array (greedy) so objects containing a
    ``]`` in a string value don't truncate the parse.

    Keeping this robust and provider-agnostic is why we don't reach for a
    structured-output library here: the same parser works identically across
    gemma3:4b, GPT-4o, and Claude, with no extra dependency or model
    function-calling requirement.

    Args:
        raw: Raw string returned by the LLM.

    Returns:
        Parsed list of ``ExtractedMemory`` objects.  Returns ``[]`` if the
        response cannot be parsed or contains no valid entries.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)  # greedy → outermost array
    if not match:
        logger.warning("extractor_no_json_array", raw=raw[:200])
        return []

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        logger.warning("extractor_json_parse_error", raw=raw[:200])
        return []

    if not isinstance(data, list):
        logger.warning("extractor_not_a_list", raw=raw[:200])
        return []

    memories: list[ExtractedMemory] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            memories.append(ExtractedMemory(**item))
        except Exception:  # noqa: BLE001
            logger.debug("extractor_invalid_item", item=item)
    return memories


async def extract_memories(
    transcript: str,
    existing: list[MemoryEntry],
    cfg: Settings,
    registry: PromptRegistry | None = None,
) -> list[ExtractedMemory]:
    """Run the LLM extraction pass on *transcript*.

    Builds the prompt, calls the fast LLM model, and parses the structured
    JSON response into ``ExtractedMemory`` objects ready for storage.

    Args:
        transcript: The full conversation exchange to analyse.
        existing:   Relevant memories already stored for this user, fetched
                    before calling this function via a semantic search.
        cfg:        Application settings (provides LLM provider / model).
        registry:   Prompt registry for the extraction prompt; defaults to the
                    process-wide singleton.

    Returns:
        List of ``ExtractedMemory`` objects to store.  Empty list if nothing
        durable was found or if the LLM response could not be parsed.
    """
    existing_text = "\n".join(f"[{m.id}] {m.text}" for m in existing) or "none"
    prompt = (registry or get_registry()).get(EXTRACTOR_KEY).render(
        "extract",
        transcript=transcript,
        existing_memories=existing_text,
    )

    llm = get_fast_llm(cfg)

    def _call_llm() -> str:
        return llm.call([{"role": "user", "content": prompt}])

    try:
        raw = await asyncio.to_thread(_call_llm)
    except Exception as exc:  # noqa: BLE001
        logger.warning("extractor_llm_error", error=str(exc))
        return []

    memories = _parse_extracted_memories(raw)
    logger.info("extractor_done", found=len(memories), transcript_chars=len(transcript))
    return memories
