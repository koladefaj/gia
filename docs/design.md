# Design decisions, tradeoffs & limitations

## Design decisions & tradeoffs

**LLM provider abstraction via litellm directly.** OpenAI, Anthropic, and Ollama all route through a single `LLM` wrapper in `providers/llm.py` that calls `litellm.completion()` — one place to change the model, one place to add a provider, no vendor lock-in. The hot path orchestration is plain `asyncio`: the voice turn is async, streaming, and intent-driven, so the right tool is Python, not a framework.

**Spotify deprecated audio features mid-build — so I deleted them.** The DJ originally key-matched a crossfade queue using Camelot wheel + energy/valence. Spotify killed `/audio-features` for new apps, so those values became constants and the "harmonic sequencing" was a no-op. Rather than leave dead code computing on placeholders, I **removed the entire machinery** (crossfade module, audio-feature fetch, the feature fields on the track schema) and rebuilt queueing on signals that still exist: the user's stated track order, or search relevance. *Noticing the platform changed under me and re-architecting is the decision I'm most proud of here.*

**Mood, rebuilt the same way.** With audio features gone, mood couldn't be `(energy, valence)` quadrants. It's now an LLM labeling the *artists and track names* you actually play into a **closed vocabulary** — which keeps "current mood vs. your pattern" a clean string comparison instead of fuzzy numeric deviation.

**`played_at` is approximated.** Spotify's MCP recently-played carries no per-track timestamps, so ingestion stamps the poll time. With frequent use it's accurate to the time-bucket; I documented the approximation rather than pretend it's exact.

**Self-eval is deterministic, not an LLM judge.** I log cheap, free signals per turn instead of spending an extra LLM call (and latency) grading every reply. An LLM-as-judge is the obvious next step *once there's real traffic to sample.*

**Reflection runs in the background, never on the hot path.** Consolidation and mood inference are Celery jobs triggered after a turn streams — the user never waits on "analyze six months of history."

**Voice mode is a flag, not a rewrite.** The decomposed pipeline (A) and the speech-to-speech paths (B/C on `gpt-realtime`) coexist behind `VOICE_MODE`, sharing memory, Spotify, Brave, and persistence. Realtime ships *beside* the pipeline rather than replacing it — so the router cascade, distilled classifier, and speculative-execution work stay intact, and the *comparison* (when decomposed control wins vs. when native end-to-end latency wins) is itself the artefact. See [benchmarks.md](benchmarks.md) and [architecture.md](architecture.md#6b-speech-to-speech--the-realtime-path).

---

## What I deliberately did *not* build (and why)

A portfolio is as much about scope judgment as features. Things I chose to leave out, with the reason:

- **Episodic memory, user embeddings, predictive recommendations** — these need *real usage data* to be anything but theater. With one seeded demo user I'd be tuning against synthetic data. The right time is after launch, when there's behavior to learn from.
- **Multi-modal memory & a social graph** — different products. Out of scope for a focused voice companion.
- **Full real-time barge-in (WebRTC)** — streaming STT already moved capture to a WebSocket and Deepgram Flux exposes the `StartOfTurn` signal barge-in needs, so the groundwork is in; wiring interrupt-while-Gia-speaks is a focused follow-up, not the transport rewrite it was before.
- **A general speculative-execution framework** — I built *targeted* speculation where it pays (the conversational reply and the Spotify search, both gated so they never affect correctness). A generic "speculate every tool call" engine would add double-spend and reconciliation complexity the rest of the pipeline doesn't need.

The discipline of *not* building these is the point: I'd rather ship a focused system I can stand behind than a sprawling one that's 60% done.

---

## Known limitations

Stated plainly, because honest scope reads better than a flawless pitch:

- **TTFA is still above the conversational threshold (in the pipeline).** The target for voice AI that feels natural is **700–1000ms** TTFA. The pipeline's ~4–6s (chat) / ~5–7s (music) is a real improvement from ~10s but isn't there yet; the realtime path (B) reaches ~1.0–1.1s, into the band. The pipeline bottleneck isn't transport — it's inference. Even with everything parallel and speculation hitting, the irreducible floor is LLM TTFT (~600–1200ms for gpt-4o) plus TTS first-byte (~800–1000ms for ElevenLabs streamed), a hard minimum around 1.5–2s on cloud APIs. Switching to WebRTC (LiveKit/Pipecat) saves HTTP transport (~100–200ms) but not those inference costs. The real path to sub-1s is a faster LLM provider (Groq/Cerebras get TTFT to ~150–200ms) paired with a lower-latency TTS (Cartesia or local Kokoro under 200ms first-byte) — a model-quality tradeoff, not just infra.
- **Browser playback is verified by types/tests, not a full cross-browser pass.** The progressive `MediaSource` playback is unit-tested and type-checked end-to-end; a manual smoke test across browsers is still pending. Safari's `MediaSource` support for `audio/mpeg` is limited — Chrome/Edge/Firefox are the supported targets.
- **Speculation only helps "guessable" music commands.** *"play some Drake"* lets the search run early; *"yeah, that one"* / *"land on something"* resolve from conversation history, so the query can't be pre-guessed and those turns pay the full serial `router → search → phrase`.
- **The router was the latency wild card — now largely tamed.** `gpt-4o-mini` is ~1.1–1.7s (occasional spike to ~3.5s+). Three tiers keep it off the hot path: keyword fast-path, distilled local classifier (~10ms for the confident chat majority), and eager prewarm for the rest — the cold LLM only runs on the genuinely ambiguous tail or turns needing a resolved `search_query`. The caveat is the classifier's data: held-out accuracy 0.78, trained partly on synthetic phrasings, so confidence-gated (misses fall back to the LLM) and OFF by default until there's real traffic to retrain on.
- **Streaming STT is validated against the live API, not yet browser-measured.** The Deepgram Flux pipe and the prewarm reuse are proven end-to-end server-side; the full in-browser TTFA with AudioWorklet capture is the pending measurement. If streaming is unavailable, the client falls back to the batch `whisper-1` path automatically.
- **Realtime (B/C) validated against the live API + mocked socket, not full load.** The session config, tool round-trip, and event normalisation are unit-tested against a mocked socket and confirmed against the live OpenAI Realtime API; per-turn Langfuse tracing for realtime is turn-level (not full span nesting), and typed turns are voice-first only.
- **One seeded demo user.** The memory and mood systems are structurally complete but validated against synthetic history, not real usage at scale — by design (see above).
- **Dev ergonomics traded for stability.** `uvicorn --reload` is disabled in Docker because the file watcher is unstable on Windows bind mounts; code changes need a container recreate locally.
