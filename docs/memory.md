# The memory system (why she feels like she knows you)

Memory is a real pipeline, not a chat-history window:

- **Extraction** — a background worker distils durable `preference` and `life_fact` memories from conversations (throttled, batched embeddings — one API call per pass).
- **Consolidation (the reflection loop)** — periodically, an LLM reads the *whole* set of raw facts and synthesises 2–4 higher-order **insights** ("uses music to focus; reaches for lyric-light tracks while working"). Insights are derived, so each run fully supersedes the last. They're injected *above* raw facts as the big-picture summary.
- **Retrieval** — hybrid search (BM25 for exact artist/track tokens + dense vectors for semantic intent), reranked, cached in Redis, assembled in parallel into one `UserContext`.
- **Mood, reflected from behavior** — recently-played tracks are ingested into history; a worker LLM-labels each `(weekday × time-of-day)` bucket into a closed mood vocabulary; when current listening drifts from the bucket's pattern, a proactive note is drafted for the next turn.

Everything degrades quietly — a flaky Weaviate or Spotify yields an empty slice, never a failed turn.

For the full storage-layer breakdown and the consolidation/retrieval internals, see [architecture.md](architecture.md#7-memory-system).
