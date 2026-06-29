# Latency — the thing I obsessed over: time-to-first-audio

The metric isn't total response time — it's **TTFA**, when the user first hears Gia. I drove it from **~10s p99 to ~4–5s** (chat) by attacking the serial dead time on the critical path, profiling each stage in Langfuse, and removing the biggest blocks one at a time.

> The speech-to-speech paths (B/C) cut this further — see [benchmarks.md](benchmarks.md) for the A/B/C comparison. This page is the engineering story of the **decomposed pipeline (A)**.

**1. Stream the TTS instead of waiting for the whole file.** This was the single biggest win. The pipeline had a complete sentence-streaming machine that the chat path *threw away* — it streamed the text, then synthesised the entire reply in one blocking call before sending a single audio byte. That was **~3.3s of dead silence** per turn. The fix sends the full reply text up-front to ElevenLabs' `/stream` endpoint (so `eleven_v3` keeps the ~250-char context it needs for natural prosody and audio-tag rendering) but **forwards the MP3 bytes as they render** — and the Next.js frontend plays them progressively through a `MediaSource` buffer. First audio now lands well before the file is finished.

**2. Take the router off the critical path.** A music product still needs the `gpt-4o-mini` router (~2s) to resolve intent, but most turns are chit-chat. So the conversational reply is **generated speculatively, concurrently with the router**, and emitted *only if* the router confirms a conversational intent. Correctness is identical — nothing is spoken before the router lands — but the router latency now overlaps the reply generation instead of preceding it.

**3. Speculative Spotify search for music commands.** When a play/queue command is detected, the Spotify search fires **in parallel with the router** (read-only — no playback). Its result is reused when the router's resolved query is "in" the user's words (token-overlap check); reference commands like *"just play it now"* (resolved from history) safely fall back to a fresh search. Playback side-effects always wait for the router, so nothing ever plays the wrong thing.

**4. Replies are spoken, so they're short.** A prompt that's tuned for *reading* writes paragraphs; spoken, that's slow to generate **and** slow to synthesise. Enforcing 1–2 sentences cut both — observed reply lengths dropped from ~200–530 chars to ~120–200.

**5. Small router, big thinker.** `gpt-4o-mini` classifies intent/tone/plan in one structured JSON call; the expensive persona model only runs when the turn actually needs it. Connection pools (a keep-alive httpx client for ElevenLabs) and a startup prewarm keep per-turn overhead off the hot path.

**6. Streaming STT, and a router that starts before you finish.** The front of the pipe was the last serial block: record → upload → transcribe meant **~1.5–2.8s of dead air** before `/chat` could even begin. The fix is a transport change — the browser captures the mic through an **AudioWorklet** and streams 24 kHz PCM frames over a WebSocket to a provider-agnostic streaming-STT layer (behind the same `STT_PROVIDER` switch as everything else). I chose **Deepgram Flux** (`/v2/listen`), the conversational model that does end-of-turn detection itself — so the client-side silence timer goes away, and the turn fires the instant Flux emits a confirmed `EndOfTurn`.

The same model unlocks **early-intent**: Flux emits an `EagerEndOfTurn` a beat *before* the user actually stops, with a transcript it guarantees will match the final. So the client fires `/chat/prewarm` on that eager text, the `gpt-4o-mini` router starts immediately, and when `/chat` arrives with the final transcript it **reuses the in-flight decision** instead of classifying cold — moving the router's ~2s off the critical path for the music/specialist turns where it was still serial. It's gated exactly like the speculative reply: classification is read-only, the prewarm key is the *exact* normalised transcript, and all search/playback still waits for the final, so a corrected utterance never acts on the wrong words. The prewarm result is cached in Redis, so the head start survives even when prewarm and `/chat` land on different workers. Both the streaming pipe and the prewarm reuse are validated end-to-end against the live Deepgram API; a full browser TTFA re-measure is the next step.

**7. A four-tier router cascade — and a distilled local classifier for the rest.** Profiling the traces showed the router was *still* the floor: even prewarmed, `gpt-4o-mini` is ~1.1–1.7s, and the eager lead time often isn't long enough to fully hide it. So the router became a cascade, fastest tier first:

```
keyword (sub-ms)           → explicit greetings / play commands
distilled classifier (~40–60ms, CPU)  → the confident chat/mood/memory majority   ← new
eager prewarm reuse        → music/specialist turns the eager signal got ahead of
cold gpt-4o-mini (~1.1–1.7s) → the genuinely ambiguous tail + anything needing a query
```

The new tier is a **distilled classifier**: every `router-classify` decision Langfuse has logged is a `(message → RouterDecision)` label, so the production `gpt-4o-mini` router is the **teacher** and a tiny local model is the **student**. The student is a *frozen* `all-MiniLM-L6-v2` sentence encoder + small `scikit-learn` linear heads — **not** an end-to-end fine-tune, which would be slow on CPU and overfit thin data. It predicts only the **categorical** fields (`intent`, `tone`, `engagement`, the `needs_*` flags) in **~7–10ms pure inference on CPU** (~40–60ms end-to-end including cascade overhead, measured from Langfuse); it never produces the free-form `search_query`, so it's gated to the intents that don't need one (`GENERAL_CHAT` / `MOOD_CHECK` / `MEMORY_QUERY`) and only when its top-class confidence clears a threshold. Everything else — music, artist, news, mixed, or low-confidence — falls through to the LLM, so **net accuracy stays the teacher's; only latency changes**. On the confident chat majority that's **~40–60ms instead of ~1.1–1.7s**.

The honest part is the data: the real corpus was 410 examples, 59% `GENERAL_CHAT`, with `MOOD_CHECK`/`MIXED`/`ARTIST_INFO` in single digits — unlearnable as-is. I balanced it by generating varied phrasings per intent and **labelling each with the production router** (teacher labels, not my guesses), to ~1,440 examples. Held-out intent accuracy is **0.78** — fine *because* it's confidence-gated. It's trained partly on synthetic data, so it's OFF by default and would be retrained as real traffic accumulates; the whole pipeline (extract → augment → train → serve) lives in [`ml/router/`](../ml/router/). The point isn't the model — it's the cascade design, the confidence gating, and being straight about the data.

## The feature I built, measured, and deleted

Earlier the headline trick was an **acknowledgment** — a sub-ms keyword pass that spoke a neutral filler ("On it.") before the router returned. It demoed well in theory. In practice the data killed it:

- On fast **chat** turns it filled a gap of only ~0.3s, and on short text it sounded **robotic** — ElevenLabs v3 needs context to sound human, and a two-word filler gives it none.
- I tried scoping it to **music** turns only (where there's a real ~3–4s wait, and the filler's flash model matched the DJ reply's voice so they'd blend). It was better — but still added a synthetic "Gia talks at you" beat the user didn't want.

So I kept the mechanisms that reduce *real* latency (speculative reply + search) and **removed the one that only masked *perceived* latency.** Knowing the difference — and being willing to delete my own clever feature when it didn't earn its place — is the decision I'd most want a reviewer to see.

> Note: the realtime paths (B/C) reintroduce a *spoken* acknowledgment for tool turns — but model-native this time: `gpt-realtime` says "one sec" and emits the tool call in the same turn, so the ack is real speech overlapping a real wait, not a synthetic filler. See [benchmarks.md](benchmarks.md).
