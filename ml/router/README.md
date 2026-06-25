# Distilled router classifier

A fast, local classifier that takes the `gpt-4o-mini` router (~1.4s, on the
critical path of every turn) off the hot path for the turns that don't need an
LLM's reasoning. It slots into the existing cascade:

```
keyword fast-path (sub-ms)   → explicit greetings / commands
distilled classifier (~20-40ms, CPU)  → the confident majority   ← this
gpt-4o-mini (~1.4s)          → the ambiguous tail + any turn needing search_query
```

## Why this shape (the honest engineering call)

- **Frozen encoder + linear heads, not end-to-end DistilBERT fine-tuning.**
  Training here is CPU-only (no CUDA in this env) and the real data is thin and
  skewed, so full fine-tuning would be slow and overfit. A frozen distilled
  sentence-encoder (`all-MiniLM-L6-v2`, itself a distilled BERT) + small sklearn
  linear heads is sample-efficient, trains in seconds, serves in ~20-40ms on CPU,
  and is robust to small data. The MiniLM embeddings carry the semantics; the
  linear heads just learn the boundaries.

- **Distilled from the production router.** Every `router-classify` trace in
  Langfuse is a `(message → RouterDecision)` label, so `gpt-4o-mini` is the
  teacher. The classifier only predicts the **categorical** fields it can learn
  (`intent`, `tone`, `engagement_mode`, and the `needs_*` flags). It does NOT
  produce `search_query` / `track_titles` (free-form) — any turn that needs those
  falls back to the LLM, by design.

- **Confidence-gated.** The classifier is used only when its top-class
  probability clears a threshold; below it, the turn falls back to the LLM. So
  net accuracy stays ~the teacher's, with most turns much faster.

## Data honesty

The real corpus (410 deduped Langfuse examples) is heavily imbalanced — 59%
`GENERAL_CHAT`, with `MOOD_CHECK` / `MIXED` / `ARTIST_INFO` in the single digits.
Those classes are unlearnable from real data alone, so we **augment with
teacher-labeled synthetic phrasings** to balance the classes. This is a standard
bootstrap, and it's only honest if stated plainly: early accuracy is partly
trained on synthetic data and would be retrained as real traffic accumulates.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Pull real labels | `extract_dataset.py` | `dataset.jsonl` |
| 2. Synthetic augment | `generate_synthetic.py` | `synthetic.jsonl` |
| 3. Train heads | `train.py` | `heads.joblib`, `labelmaps.json`, `eval_report.txt` |
| 4. Serve | `backend/app/agents/router_local.py` | wired into the chat cascade |

```bash
uv run python ml/router/extract_dataset.py
uv run python ml/router/generate_synthetic.py
uv run python ml/router/train.py
```
