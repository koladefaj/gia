"""Pull every `router-classify` generation from Langfuse into a training set.

This is step 1 of distilling the gpt-4o-mini router into a fast local classifier:
the production router's decisions are our labels. Each Langfuse `router-classify`
observation is a (prompt → RouterDecision JSON) pair; we recover the raw user
message from the prompt and the structured labels from the output, and write a
clean JSONL plus a class-balance report so we can see whether there's enough
real signal to train on (or whether we need synthetic augmentation).

Run:  uv run python ml/router/extract_dataset.py
Out:  ml/router/dataset.jsonl  + a printed summary
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import httpx

from backend.app.config import settings

OUT = Path(__file__).resolve().parent / "dataset.jsonl"

# The current user message sits at the end of the router prompt as: User: "<msg>"
# (history turns, if any, come before it). Grab the LAST such quoted message.
_USER_RE = re.compile(r'User:\s*"(.+?)"', re.DOTALL)

# Categorical label fields the encoder classifier will predict (search_query /
# track_titles are free-form and stay with the LLM fallback).
_BOOL_FIELDS = ("needs_search", "needs_memory", "needs_music", "needs_artist_lookup")


def _message_from_input(inp: object) -> str | None:
    """Recover the raw user message from a logged router-classify input."""
    text = ""
    if isinstance(inp, list):  # [{role, content}, ...]
        for m in inp:
            if isinstance(m, dict) and m.get("role") == "user":
                text = str(m.get("content", ""))
    elif isinstance(inp, str):
        text = inp
    matches = _USER_RE.findall(text)
    return matches[-1].strip() if matches else None


def _decision_from_output(out: object) -> dict | None:
    """Parse the RouterDecision JSON from a logged output."""
    raw = out
    if isinstance(out, dict):
        raw = out.get("content", out)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return None
    return None


def fetch_all() -> list[dict]:
    host = settings.langfuse_host.rstrip("/")
    auth = (settings.langfuse_public_key, settings.langfuse_secret_key)
    obs: list[dict] = []
    page = 1
    with httpx.Client(auth=auth, timeout=30.0) as client:
        while True:
            resp = client.get(
                f"{host}/api/public/observations",
                params={"name": "router-classify", "limit": 100, "page": page,
                        "fields": "core,io"},
            )
            resp.raise_for_status()
            body = resp.json()
            batch = body.get("data", [])
            obs.extend(batch)
            meta = body.get("meta", {})
            total_pages = meta.get("totalPages", page)
            if not batch or page >= total_pages:
                break
            page += 1
    return obs


def main() -> None:
    obs = fetch_all()
    rows: list[dict] = []
    skipped = 0
    seen: set[str] = set()
    for o in obs:
        msg = _message_from_input(o.get("input"))
        dec = _decision_from_output(o.get("output"))
        if not msg or not dec or "intent" not in dec:
            skipped += 1
            continue
        key = msg.strip().lower()
        if key in seen:  # dedupe identical phrasings
            continue
        seen.add(key)
        rows.append({
            "message": msg,
            "intent": dec.get("intent"),
            "tone": dec.get("tone"),
            "engagement_mode": dec.get("engagement_mode"),
            **{f: bool(dec.get(f, False)) for f in _BOOL_FIELDS},
        })

    OUT.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    print(f"observations fetched : {len(obs)}")
    print(f"usable rows (deduped): {len(rows)}")
    print(f"skipped (unparseable): {skipped}")
    print(f"\nintent distribution:")
    for intent, n in Counter(r["intent"] for r in rows).most_common():
        print(f"  {intent:14} {n}")
    print(f"\ntone distribution:")
    for tone, n in Counter(r["tone"] for r in rows).most_common():
        print(f"  {tone:14} {n}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
