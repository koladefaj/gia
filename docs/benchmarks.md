# Benchmarks

All numbers are end-to-end, measured from Langfuse traces and container-level micro-benchmarks (single dev box, RTX 4060, `gpt-4o`/`gpt-4o-mini`, ElevenLabs v3/flash).

**Time-to-first-audio**

| Turn type | Before | After |
|---|---:|---:|
| Chat / conversational | ~10s p99 | **~4–6s** |
| Music command | ~10s p99 | **~5–7s** |

**Per-stage (where the time goes now)**

| Stage | Latency | Notes |
|---|---:|---|
| STT — batch (OpenAI `whisper-1`) | ~1.5–2.8s | the old serial cost, before `/chat`; now the **fallback** path |
| STT — streaming (Deepgram Flux) | first interim <~0.3s, no serial wait | model does end-of-turn; the turn fires on `EndOfTurn`, the **new default** |
| Router — LLM (`gpt-4o-mini`, JSON) | ~1.1–1.7s | occasional spike to ~3.5s+; **overlapped via eager prewarm**; the tail tier |
| Router — distilled (MiniLM + linear heads, CPU) | **~40–60ms** end-to-end (pure inference ~7–10ms) | the confident chat/mood/memory majority; 0.78 held-out intent acc, confidence-gated |
| Conversational reply (`gpt-4o` stream) | TTFT ~0.6–1.2s | **overlapped under the router** via speculation |
| DJ specialist | ~2.8–4.5s | Spotify search ~2s + recommendation LLM ~1–2s; search removed from the path when speculation hits |
| TTS | was ~3.3s blocking → **streamed first byte <1s** | the headline win |

**Three voice paths, side by side**

The same turn can run three ways. End-to-end on a single dev box (`gpt-realtime`, `gpt-4o`/`gpt-4o-mini`, ElevenLabs v3); directional, not p99.

| | **A · Decomposed pipeline** | **B · Speech-to-speech (model voice)** | **C · Speech-to-speech + ElevenLabs** |
|---|---|---|---|
| Path | STT → router cascade → specialist → streaming TTS | audio → `gpt-realtime` → audio | audio → `gpt-realtime` (text) → ElevenLabs v3 |
| Picks intent / tools | **4-tier router** (keyword · distilled · prewarm · `gpt-4o-mini`) | the model, native function-calling | the model, native function-calling |
| Voice | ElevenLabs v3 (brand) | `gpt-realtime` (Marin) | ElevenLabs v3 (brand) |
| Speech-end → model start | n/a (serial STT first) | ~40 ms | ~40 ms |
| **TTFA — chat turn** | ~4–6 s | **~1.0–1.1 s** | **~1.4–1.6 s** |
| **TTFA — music / tool turn** | ~5–7 s | ~1.4 s | ~1.9–2.1 s |
| Config | default | `VOICE_MODE=realtime` `REALTIME_VOICE_SOURCE=model` | `VOICE_MODE=realtime` `REALTIME_VOICE_SOURCE=elevenlabs` |

**Takeaway:** moving intent + tool selection *into* the model (B and C) deletes the STT→router→specialist serialization and cuts chat TTFA **~4×** (≈4–6 s → ≈1.1 s), landing B at the top of the 700–1000 ms conversational band the pipeline couldn't reach. C buys the brand voice back for ~0.4–0.5 s more than B — the cost of ElevenLabs' first byte over the model's own — via sentence-streaming so the first sentence speaks while the rest generates. **The router only exists in A.** In B/C, `gpt-realtime` emits tool-call arguments ~0.55 s into the turn on its own, so a separate router would only add latency. Music turns get two extra cuts in B/C: the search runs through the **direct Spotify Web API** (~0.4 s warm vs ~1.9 s through the MCP stdio server), and the model speaks a short acknowledgement *before* the search so the user hears Gia at ~first-token latency while it runs behind her voice.

**Is the router even needed in the decomposed path (A)?** Honestly — *not strictly*. Its real job (which intent → which specialist/tool, and a clean `search_query`) is something the reply LLM can do itself with **native function-calling**, exactly as B/C do. So the router could be collapsed into the conversational `gpt-4o` call. It earns its place on *one* axis: **latency on the chat majority** — the keyword and distilled tiers answer "this is just chat" in ~40 ms and skip the ~1.4 s LLM intent step entirely, and the prewarm tier overlaps it for the rest. Drop the router and every pipeline turn pays full-LLM intent latency; keep it and the common case stays cheap. With B/C now doing model-native tool-calling, the cascade is best read as a *latency optimization for the text pipeline*, not a structural necessity — which is why it's absent from the realtime paths rather than ported to them.

**STT micro-benchmark — why I *didn't* switch to local or `large-v3-turbo`**

I A/B'd the OpenAI Whisper API against local `faster-whisper` on the GPU, then benchmarked the model in isolation to decide whether `large-v3-turbo` was worth baking:

| Measurement (RTX 4060, `large-v3`, `int8_float16`) | Result |
|---|---:|
| Warm inference, 7s clip | ~1.1–1.8s |
| Warm inference, 15s clip | ~1.8–2.0s (barely scales) |
| webm/opus decode overhead | ~0.03s (negligible) |
| Idle-downclock penalty | ~0.35s |
| End-to-end `/voice/transcribe` in practice | ~2.0–3.8s |

The model floor (~1.1–2s) is real, but ~1–2s of production STT is **non-model overhead** that `turbo` can't touch (turbo only shrinks the decoder). So `turbo`'s realistic gain is ~0.5–1s — landing roughly *on par* with the OpenAI API's ~2s, for a 1.6 GB model bake. **Conclusion at the time: stay on `whisper-1`.** The interesting part isn't the answer, it's that the decision was *measured* instead of assumed.

> **Update:** that benchmark is what motivated the move to **streaming STT**. The real problem wasn't *which* batch model — every batch path pays ~1–2s of serial transcription before `/chat` starts. Streaming dissolves that wait entirely (and its interim results are what enable early-intent), so Deepgram Flux is now the default and the `whisper-1` path above is the graceful fallback.
