"""Synthetic augmentation for the router classifier.

The real Langfuse corpus is thin and badly imbalanced (MOOD_CHECK/MIXED/
ARTIST_INFO in single digits). We balance it by generating varied, realistic user
messages per intent with gpt-4o-mini, then **labelling each with the actual
production router** (``classify_turn``) — so the labels are the teacher's, not
whatever we asked the generator for. The router self-corrects ambiguous lines,
which is exactly the distillation signal we want.

Run:  uv run python ml/router/generate_synthetic.py
Out:  ml/router/synthetic.jsonl  (+ combined class-balance report)
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

from backend.app.agents.hybrid_router import classify_turn
from backend.app.config import settings
from backend.app.providers.openai_client import get_async_openai

HERE = Path(__file__).resolve().parent
REAL = HERE / "dataset.jsonl"
OUT = HERE / "synthetic.jsonl"

_BOOL_FIELDS = ("needs_search", "needs_memory", "needs_music", "needs_artist_lookup")

# How many synthetic messages to GENERATE per intent (the router may relabel some;
# the rare classes get the most so they clear a learnable floor).
_TARGETS: dict[str, int] = {
    "MOOD_CHECK": 170,
    "MIXED": 170,
    "ARTIST_INFO": 160,
    "MEMORY_QUERY": 150,
    "NEWS_QUERY": 140,
    "MUSIC_QUEUE": 120,
    "MUSIC_FIND": 80,
    "GENERAL_CHAT": 40,
}

# What each intent sounds like — steers the generator toward realistic phrasings
# for THIS product (a voice music companion; the demo user loves Afrobeats +
# Canadian hip-hop). Artists are seeds, not a script.
_GUIDANCE: dict[str, str] = {
    "MUSIC_FIND": "asking to play/find/recommend music — a vibe, genre, mood, or a named artist/song (e.g. 'play some Asake', 'find me something chill for the evening', 'put on afrobeats')",
    "MUSIC_QUEUE": "asking to QUEUE/add/save tracks for later or build a playlist (e.g. 'queue Essence after this', 'add this to my playlist', 'line up some Burna Boy next')",
    "ARTIST_INFO": "asking ABOUT a specific named artist — their latest work, news, who they are (e.g. 'tell me about Wizkid', \"what's Drake's newest album\", 'who is Tems')",
    "MOOD_CHECK": "asking about THEIR OWN mood or listening patterns/trends (e.g. \"what's my mood been like lately\", 'how have I been listening this week', 'am I in a chill phase')",
    "MEMORY_QUERY": "asking what Gia REMEMBERS about them (e.g. 'who do I usually listen to', 'what are my favorite artists', 'what did I tell you about my taste')",
    "NEWS_QUERY": "asking about current real-world events or music gossip (e.g. 'any news on the World Cup', \"what's happening with Drake\", 'did anything big drop this week')",
    "GENERAL_CHAT": "casual companion small talk — greetings, how-are-you, reactions, opinions (e.g. \"I'm doing good\", 'that song is fire', 'Drake or Asake, who you got')",
    "MIXED": "ONE message that asks BOTH about a named artist AND to play/queue music (e.g. 'tell me about Asake and put something on', \"who's PARTYNEXTDOOR — play his stuff\")",
}

_GEN_BATCH = 20
# Conservative: the router prompt is large, so high concurrency blows the 200k
# TPM cap. classify_turn swallows the 429 and returns its safe default
# (GENERAL_CHAT, confidence 0.0), which would poison the labels — so we detect
# that and retry with backoff instead.
_CONCURRENCY = 4


async def _generate_for_intent(client, intent: str, n: int) -> list[str]:
    """Ask gpt-4o-mini for *n* varied user messages fitting *intent*."""
    out: list[str] = []
    seen: set[str] = set()
    guidance = _GUIDANCE[intent]
    while len(out) < n:
        prompt = (
            f"You write realistic user messages for a voice music companion named Gia. "
            f"The user loves Afrobeats (Asake, Wizkid, Burna Boy, Tems, BNXN) and Canadian "
            f"hip-hop (Drake, PARTYNEXTDOOR). Generate {_GEN_BATCH} DIFFERENT, natural things "
            f"a real user would SAY OUT LOUD that fit this case: {guidance}. "
            f"Vary length, slang, and formality; some short, some longer; include filler like "
            f"'um' and 'can you' sometimes. Return ONLY JSON: {{\"messages\": [\"...\", ...]}}."
        )
        try:
            resp = await client.chat.completions.create(
                model=settings.router_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=1.0,
            )
            batch = json.loads(resp.choices[0].message.content or "{}").get("messages", [])
        except Exception as exc:  # noqa: BLE001
            print(f"  gen error ({intent}): {exc}")
            break
        for m in batch:
            key = str(m).strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(str(m).strip())
    return out[:n]


async def _label(sem: asyncio.Semaphore, msg: str) -> dict | None:
    """Label one message with the production router (the teacher).

    classify_turn never raises — on a rate-limit/error it returns the safe
    default (confidence 0.0). We treat conf-0.0 as a failure and retry with
    backoff so a transient 429 doesn't silently mislabel the row as GENERAL_CHAT.
    Returns ``None`` if it never succeeds (skipped rather than poisoned).
    """
    async with sem:
        for attempt in range(5):
            d = await classify_turn(msg, settings, history="")
            if d.confidence > 0.0:  # a real decision (safe default is exactly 0.0)
                return {
                    "message": msg,
                    "intent": d.intent.value,
                    "tone": d.tone.value,
                    "engagement_mode": d.engagement_mode.value,
                    **{f: bool(getattr(d, f)) for f in _BOOL_FIELDS},
                }
            await asyncio.sleep(0.5 * (2**attempt))  # 0.5,1,2,4,8s backoff
    return None


async def main() -> None:
    client = get_async_openai(settings)

    # 1. Generate candidate messages per intent (in parallel across intents).
    print("generating candidates...")
    gen_results = await asyncio.gather(
        *(_generate_for_intent(client, intent, n) for intent, n in _TARGETS.items())
    )
    candidates: list[str] = []
    real_msgs = {
        json.loads(line)["message"].strip().lower()
        for line in REAL.read_text(encoding="utf-8").splitlines() if line.strip()
    } if REAL.exists() else set()
    seen = set(real_msgs)
    for msgs in gen_results:
        for m in msgs:
            k = m.strip().lower()
            if k not in seen:
                seen.add(k)
                candidates.append(m)
    print(f"  {len(candidates)} unique candidates")

    # 2. Label every candidate with the real router (the distillation step).
    print("labelling with the production router (teacher)...")
    sem = asyncio.Semaphore(_CONCURRENCY)
    labelled = await asyncio.gather(*(_label(sem, m) for m in candidates))
    rows = [r for r in labelled if r]

    OUT.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    # 3. Report the COMBINED (real + synthetic) balance.
    real_rows = [
        json.loads(line) for line in REAL.read_text(encoding="utf-8").splitlines() if line.strip()
    ] if REAL.exists() else []
    combined = Counter(r["intent"] for r in real_rows) + Counter(r["intent"] for r in rows)
    print(f"\nsynthetic rows written: {len(rows)}  ->  {OUT.name}")
    print("combined intent balance (real + synthetic):")
    for intent, n in combined.most_common():
        print(f"  {intent:14} {n}")


if __name__ == "__main__":
    asyncio.run(main())
