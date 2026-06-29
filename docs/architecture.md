# Gia — Architecture Deep Dive

A reference document covering every major subsystem, the engineering decisions behind each one, the tradeoffs made, what was tried and changed, and what the known limitations are. Written to give a complete picture of the system for interviews or code reviews.

---

## Table of Contents

1. [High-level architecture](#1-high-level-architecture)
2. [The hot path — one voice turn end to end](#2-the-hot-path--one-voice-turn-end-to-end)
3. [Speech-to-text — streaming vs batch](#3-speech-to-text--streaming-vs-batch)
4. [The router cascade — four tiers](#4-the-router-cascade--four-tiers)
5. [Specialist agents](#5-specialist-agents)
6. [Text-to-speech and streaming audio](#6-text-to-speech-and-streaming-audio)
6b. [Speech-to-speech — the realtime path](#6b-speech-to-speech--the-realtime-path)
7. [Memory system](#7-memory-system)
8. [Background workers — Celery](#8-background-workers--celery)
9. [CrewAI — what it does and what it doesn't](#9-crewai--what-it-does-and-what-it-doesnt)
10. [The curator crew — proper CrewAI multi-agent](#10-the-curator-crew--proper-crewai-multi-agent)
11. [Observability — Langfuse](#11-observability--langfuse)
12. [Frontend — Next.js voice UI](#12-frontend--nextjs-voice-ui)
13. [Data stores](#13-data-stores)
14. [Provider abstraction](#14-provider-abstraction)
15. [Features built, measured, and changed](#15-features-built-measured-and-changed)
16. [Known limitations](#16-known-limitations)
17. [Design decisions and tradeoffs — quick reference](#17-design-decisions-and-tradeoffs--quick-reference)

---

## 1. High-level architecture

```
Browser mic (AudioWorklet, 24 kHz PCM16)
        │
        ▼ WebSocket
Streaming STT (Deepgram Flux)
        │  interim transcripts
        ├──────────────────► POST /chat/prewarm   → router starts mid-utterance
        │  final transcript
        ▼ SSE
POST /chat (FastAPI)
        │
        ├── Memory retrieval (parallel fan-out)
        ├── Speculative reply generation (concurrent with router)
        ├── Speculative Spotify search (concurrent with router)
        │
        ├── Router cascade (4 tiers, fastest first)
        │       keyword → local classifier → prewarm reuse → gpt-4o-mini
        │
        └── Specialist (DJ / Artist / Mood / Chat)
                │
                ▼ full reply text
        ElevenLabs /stream  →  MP3 chunks  →  MediaSource buffer  →  speaker
```

**Why SSE instead of WebSocket for the reply?** SSE is unidirectional server-push over HTTP, which maps cleanly onto "server streams a reply." WebSockets would require the server to manage a bidirectional connection for something that is fundamentally one-way. SSE also degrades gracefully and is trivially proxied.

**Why WebSocket for audio capture?** Mic audio is continuous binary data. HTTP request/response would require buffering the full recording before sending. WebSocket keeps the socket open so PCM frames are forwarded as they are captured, which is what makes streaming STT possible.

---

## 2. The hot path — one voice turn end to end

`backend/app/api/chat.py` is the orchestrator. The key engineering insight is that everything that can run concurrently does. Here is what happens the moment `/chat` receives a request:

```
t=0ms   Request arrives
        │
        ├── asyncio.create_task: memory retrieval (Weaviate + Postgres + Redis)
        ├── asyncio.create_task: speculative reply (general reply generated under the router)
        ├── asyncio.create_task: speculative Spotify search (for music commands only)
        │
        ├── Tier-1 router: keyword fast-path (sub-ms) — or fall through
        ├── Tier-2 router: local MiniLM classifier (~20-40ms CPU) — or fall through
        ├── Tier-3 router: prewarm reuse (result from /chat/prewarm, already running) — or fall through
        └── Tier-4 router: cold gpt-4o-mini (~1.4s) — only for ambiguous/music turns
                │
                ▼ router decision
        Specialist runs (DJ / Artist / Mood / general)
                │
                ▼ full reply text assembled
        ElevenLabs /stream — MP3 bytes forwarded as they render
                │
                ▼ audio_chunk SSE events
        Browser MediaSource — plays audio before file is complete
```

**The speculative reply** — a general conversational reply is generated concurrently with the router. If the router confirms a conversational intent, that reply is emitted immediately. If the router returns a music/specialist intent, the speculative reply is discarded. Net correctness is identical; the router latency now overlaps reply generation instead of preceding it.

**The speculative Spotify search** — when the message contains clear play/queue signals, a Spotify search fires in parallel with the router using the raw message text. When the router's resolved `search_query` overlaps the user's actual words (token-overlap check), the speculative result is reused and the search round-trip disappears from the critical path. Reference commands like *"yeah, that one"* always fall back to a fresh search with the resolved query.

---

## 3. Speech-to-text — streaming vs batch

### Why streaming STT was built

The original pipeline was: record → upload → transcribe → `/chat`. That serial wait was **~1.5–2.8s** of dead air before the turn even started, independent of which batch model was used. Profiling confirmed the problem was the pattern, not the model — switching from OpenAI `whisper-1` to local `large-v3-turbo` would have saved ~0.5–1s max, while the overhead floor (~1–2s of serialization + HTTP) was unmovable.

The fix is a transport change: the browser streams 24 kHz mono PCM16 over a WebSocket, and the server pipes it to a streaming ASR provider in real time.

### Deepgram Flux (`backend/app/providers/stt_stream.py`)

Deepgram's conversational model (`flux-general-en`) does end-of-turn detection internally. It emits:

| Event | Meaning |
|---|---|
| `StartOfTurn` | User started speaking |
| `Update` | Interim transcript, ~every 250ms |
| `EagerEndOfTurn` | Medium-confidence turn end — the early-intent signal |
| `TurnResumed` | Eager guess was wrong, user kept talking — cancel speculative work |
| `EndOfTurn` | High-confidence final transcript — fire `/chat` |

**Why Deepgram Flux specifically?** It does end-of-turn detection itself, so the client-side silence timer goes away. The `EagerEndOfTurn` event is what enables prewarm (the router starts before the user finishes). Nova (Deepgram's other model) doesn't emit `EagerEndOfTurn`.

### OpenAI Realtime (`openai_stream` provider)

An alternative adapter is implemented using `gpt-4o-mini-transcribe` over the Realtime WebSocket API. It works but is the secondary tested path. Sits behind the same `STT_PROVIDER` switch, same `TranscriptEvent` interface.

### Batch fallback

If `STT_PROVIDER` is not `deepgram` or `openai_stream`, the turn falls back to `POST /voice/transcribe` → OpenAI `whisper-1`. This is the original serial path. Local `faster-whisper` is not baked into the Docker image by default (`INSTALL_LOCAL_STT=false`) — it pulled ~1.3GB of CUDA wheels and a ~3GB model, and streaming makes it unnecessary.

### Provider abstraction

Both streaming adapters implement `StreamingTranscriber` (a `Protocol`) with `send_audio`, `finish`, and `events()`. The WebSocket endpoint and the rest of the system never branch on which ASR is behind the switch.

---

## 4. The router cascade — four tiers

Every turn must be classified before the right specialist runs. The router is the latency wildcard. The cascade attacks it from fastest to slowest:

```
Tier 1: keyword fast-path   sub-ms        explicit greetings / small talk
Tier 2: local classifier    ~20-40ms CPU  confident chat/mood/memory majority
Tier 3: prewarm reuse       ~0ms          router already ran on the eager transcript
Tier 4: cold gpt-4o-mini    ~1.4s         ambiguous turns + anything needing search_query
```

### Tier 1 — keyword (`backend/app/agents/router.py`)

A simple keyword match. Returns `GENERAL_CHAT` only when a greeting/small-talk keyword is present and there is zero music, artist, mood, or queue signal. Returns `None` (fall through) for everything else.

### Tier 2 — distilled local classifier (`backend/app/agents/router_local.py`)

A frozen `all-MiniLM-L6-v2` sentence encoder + small scikit-learn linear heads trained in `ml/router/`. **~20-40ms on CPU.**

**Why this architecture (frozen encoder + linear heads, not end-to-end fine-tuning)?**
- Training is CPU-only; full DistilBERT fine-tuning would be slow and overfit on thin data
- The frozen MiniLM embeddings already carry the semantics; the linear heads just learn boundaries
- Sample-efficient: trains in seconds, 1,440 examples total

**The data honesty problem:** The real Langfuse corpus was 410 examples, 59% `GENERAL_CHAT`, with `MOOD_CHECK`/`MIXED`/`ARTIST_INFO` in single digits — unlearnable as-is. Balanced by generating synthetic phrasings per intent and **labelling each with the production `gpt-4o-mini` router** (teacher labels, not manual guesses). Held-out intent accuracy: **0.78**.

**Two safety guardrails:**
1. **Confidence gate** (default threshold 0.75) — only fires when the top-class probability clears the threshold; below it, falls through to the LLM
2. **Safe intents only** — only returns a decision for `GENERAL_CHAT`, `MOOD_CHECK`, `MEMORY_QUERY` (intents that need no `search_query`/`track_titles`). Music/artist/news always go to the LLM

**Off by default** (`router_local_enabled=False`) until enough real traffic accumulates to retrain on.

### Tier 3 — prewarm reuse (`backend/app/agents/router_prewarm.py`)

Deepgram's `EagerEndOfTurn` fires a beat before the user finishes. The client sends `POST /chat/prewarm` on that eager text, which starts the `gpt-4o-mini` router immediately. When `/chat` arrives with the final transcript, it looks up the in-flight result instead of starting cold.

**Two-tier hand-off (same worker vs different worker):**
- **Same worker:** the in-flight `asyncio.Task` is in a process-local dict; `take()` awaits it directly
- **Different worker:** the completed `RouterDecision` is written to Redis (keyed by normalised transcript SHA1); a cross-worker `take()` reads from Redis, or polls briefly on an in-flight marker

**Correctness guarantees:**
- The key is the exact normalised transcript — `prewarm` and `/chat` must have identical text to share a result
- Classification is read-only (no playback/search side effects)
- A corrected utterance (Flux's `TurnResumed` event) signals the client to abandon the prewarm

### Tier 4 — cold LLM (`backend/app/agents/hybrid_router.py`)

`gpt-4o-mini` in JSON mode (OpenAI) or via `crewai.LLM.call()` (other providers). Returns a full `RouterDecision`: `intent`, `tone`, `confidence`, `engagement_mode`, `needs_*` flags, `search_query`, `track_titles`, `start_playback`. This is the only tier that can resolve references ("just play it now" → the track discussed earlier) and produce a clean `search_query`.

---

## 5. Specialist agents

Each specialist is a plain Python service class that constructs a `crewai.Agent` for its persona but **never uses `crew.kickoff()`** — it makes direct async LLM calls. The agent object is a configured persona container (role/goal/backstory/LLM), not an executor.

| Agent | File | What it does |
|---|---|---|
| DJ | `agents/dj.py` | Spotify search → recommendation LLM → playback |
| Artist | `agents/artist.py` | Brave search → artist info synthesis |
| Mood | `agents/mood.py` | Mood pattern analysis from listening history |
| General | `agents/general.py` | Conversational reply, opening line |
| Memory | `agents/memory.py` | Assembles user context from all memory sources |
| Planner | `agents/planner.py` | Handles weather and complex mixed intents |

**Why not `crew.kickoff()` on the hot path?**
- `kickoff()` is synchronous and blocking — you can't `yield` SSE chunks from inside it
- The speculative tasks and prewarm reuse require `asyncio` control that `kickoff()` abstracts away
- The set of agents that run per turn changes based on the router decision — `@CrewBase` defines a fixed crew at class definition time; conditional composition requires plain Python

---

## 6. Text-to-speech and streaming audio

### ElevenLabs (`backend/app/providers/tts.py`)

The production TTS path. Two models:

| Model | When used | Why |
|---|---|---|
| `eleven_v3` | Emotional sentences (audio tags, questions), or `TTS_FORCE_V3=true` | Full context prosody, audio tag rendering |
| `eleven_flash_v2_5` | Logistics sentences | Faster, cheaper |

**The headline latency win:** The original pipeline synthesised the entire reply as one blocking call before sending a single audio byte (~3.3s dead silence). The fix forwards MP3 bytes from the `/stream` endpoint **as they are rendered**, so first audio arrives well before the file is complete.

**Sentence-streaming (`tts_stream_sentences`, default on):** The reply is split on punctuation boundaries and each sentence is synthesised the moment it's ready — so the first sentence's audio plays while the rest is still being generated/synthesised (`synthesize_sentence_stream` in `voice/streaming.py`). This is a deliberate reversal of the earlier whole-reply default: it trades a little per-sentence v3 prosody context for markedly lower first-audio latency. v3 still gets a *whole sentence* per call (tags stay attached to their sentence), and `tts_stream_sentences=false` restores the single-pass whole-reply synthesis (warmer prosody, slower first audio). The wire contract is identical either way (`audio_start` → N `audio_chunk` with monotonic `seq` → `audio_end`), so the frontend is unchanged. Both the decomposed pipeline and the realtime path use the same helper.

**Connection pooling:** A single `httpx.AsyncClient` with `keepalive_expiry=60s` is reused across turns. Without this, every turn paid a fresh TLS handshake.

**Audio tags:** `eleven_v3` interprets `[warm]`, `[laughs]`, `[sighs]` etc. as delivery cues. The TTS provider strips them for Kokoro (which would read them aloud).

### Kokoro (local dev)

A local TTS pipeline (zero cost, zero latency on the network). Used for testing agent logic without burning ElevenLabs credits. Gracefully degrades to silence if not installed. Set `TTS_PROVIDER=kokoro` (or leave unset) in dev.

### Frontend progressive playback

Audio chunks arrive as base64-encoded SSE `audio_chunk` events. The Next.js frontend decodes them and appends to a `MediaSource` buffer — the browser plays audio while chunks are still arriving. The user hears Gia before the full reply exists.

---

## 6b. Speech-to-speech — the realtime path

Everything above (sections 2–6) describes the **decomposed pipeline**:
`streaming STT → router cascade → specialist → streaming TTS`. Every stage is
observable, individually swappable, and individually optimised — that
decomposition is what made the TTFA story measurable. But it has a structural
latency floor: even with each stage streamed, a turn still serialises
transcribe → classify → generate → synthesise, and turn-taking/barge-in are
bolted on client-side (the VAD loop, the grace window).

The **realtime path** (`VOICE_MODE=realtime`) is a **hybrid** speech-to-speech
alternative — and the split is the important design decision:

- **`gpt-realtime` is the ears + brain.** The browser streams PCM16 to
  `WS /voice/realtime`; the model understands the audio directly (native
  turn-taking + barge-in), reasons, calls tools, and emits the reply as **text**
  (`output_modalities: ["text"]` — it never speaks). This collapses
  `STT → router → specialist` into one low-latency speech-understanding brain.
- **ElevenLabs v3 is the voice.** The endpoint takes the model's text and streams
  it through the existing ElevenLabs path, so the warm, audio-tagged brand voice
  is preserved. **The voice is the product** — handing TTS to the realtime
  model's own voice would throw that away. The model follows the Gia persona,
  which already prescribes `[warm]`/`[laughs]` tags inline, and v3 renders them.

So the realtime model replaces speech-understanding and routing, while TTS stays
exactly as it is in the pipeline.

**Voice source is a toggle (`REALTIME_VOICE_SOURCE`).** The default `elevenlabs`
is the hybrid above. Set `model` and gpt-realtime speaks **directly** (pure
speech-to-speech, `output_modalities: ["audio"]`, billed under OpenAI) — lower
latency, no TTS provider needed, at the cost of the brand voice. It exists as
graceful degradation: when ElevenLabs is unavailable (e.g. an expired plan
returns 401), flipping to `model` keeps the voice working. The provider switches
the session shape and event parsing (`response.output_audio.delta` + the audio
transcript vs. `response.output_text.*`); the endpoint either relays the model's
PCM frames or runs ElevenLabs synthesis; the frontend plays PCM via a Web Audio
queue (`realtimePlayer.ts`) or MP3 via `StreamPlayer`, picked per frame type.
Input transcription stays on in both modes, so captions + memory are unaffected.

### Why it's a parallel mode, not a replacement

This was a deliberate call. A full swap to a speech-to-speech model would
dissolve the parts of this system that are worth showing — the router cascade,
the distilled classifier, speculative execution. So realtime ships **behind a
flag, beside the pipeline**, exactly like `STT_PROVIDER` / `TTS_PROVIDER`: the
two modes share memory, Spotify, Brave, and persistence, and differ only in who
orchestrates the turn. The point isn't "realtime is better" — it's that the
*comparison* is the interesting artefact (when decomposed control wins vs. when
native end-to-end latency wins).

### The architecture survives as tools

The realtime model reaches the existing services through **function calling**
(`backend/app/providers/realtime.py`):

| Tool | Wraps | Notes |
|---|---|---|
| `search_and_play_music` | `DJService.search_only` + Spotify | Read-only search — **no** specialist-LLM call; the realtime model writes the line itself |
| `get_web_info` | `BraveSearchClient.recent` | Returns facts, not prose — grounds the model on current events |
| `recall_memory` | `build_user_context` | The same hybrid retrieval the pipeline uses |
| `get_now_playing` / `get_weather` | Spotify / Open-Meteo | Status + context signals |

`RealtimeTools.dispatch` never raises — a failed tool returns `{"error": …}` the
model speaks around, mirroring the pipeline's per-agent try/except.

### Keeping observability + memory alive

The session enables **input transcription** (`gpt-4o-mini-transcribe`) for the
user's audio; the assistant side is already text. So finalised user *and*
assistant text still flow into the session history (for "play it now" reference
resolution) and the Celery memory extractor — on the same throttle as `chat.py`.
Memory context is injected once, up front, as the session `instructions` (Gia
persona + `build_user_context`), with `recall_memory` for deeper mid-turn lookups.

### Transport: server-side WebSocket bridge

`browser → WS /voice/realtime → OpenAI Realtime WS`. The bridge (not a direct
browser→OpenAI WebRTC connection) is what lets tools run **server-side** with the
real clients and the API key never reach the browser. WebRTC would push tool
execution to the client and gut that reuse — it's noted as a future latency
optimisation, the same way local faster-whisper is.

### GA protocol notes (the bits that bite)

The 2025 GA shape differs from the 2024 beta in three ways that fail silently if
you carry the old shape over: the model is selected via the `?model=` query param
(no `OpenAI-Beta` header), audio config is nested under `session.audio.input`,
and the audio **format is an object** (`{"type": "audio/pcm", "rate": 24000}`) —
not the old `"pcm16"` string. Because the model is text-out, the reply arrives on
`response.output_text.delta` / `.done` (and there is no `audio.output` block).
Barge-in is the `input_audio_buffer.speech_started` event, which the backend
turns into a `response.cancel` (stop the model) + a `flush` frame (drop buffered
ElevenLabs playback in the browser).

### Frontend

Because the voice is ElevenLabs MP3, the realtime path reuses the **same audio
frames and `StreamPlayer`** as `/chat` — no new audio engine. `lib/realtimeSession.ts`
owns the one socket (reusing the 24 kHz capture worklet) and forwards
`audio_start`/`audio_chunk`/`audio_end` + `reply_chunk` captions + the `flush`
barge-in to the hook; `useVoiceSession` branches on `NEXT_PUBLIC_VOICE_MODE` and
keeps its public interface identical, so `VoiceScreen` is untouched.

### Latency: sentence-streaming masks the TTS cost

The reply text is **streamed**: as gpt-realtime emits text deltas, the backend
splits them on punctuation boundaries (`stream_sentences`) and synthesises each
sentence the moment it lands (`synthesize_sentence_stream`). So the first
sentence's ElevenLabs audio plays while the model is still generating the next —
generation, synthesis, and playback overlap instead of serialising. Combined with
gpt-realtime understanding the speech directly, this keeps the brand voice *and*
gets first audio out fast. Gated by `tts_stream_sentences` (shared with the
pipeline); turning it off falls back to one whole-reply v3 pass per turn. Barge-in
cancels the in-flight sentence synthesis and flushes browser playback.

### Known limitations (realtime)

| Limitation | Detail |
|---|---|
| Per-turn Langfuse tracing | The decomposed pipeline traces every span; the realtime turn is currently logged structurally but not wrapped in a per-turn Langfuse trace. Transcripts still feed the memory pipeline. |
| Typed turns | Realtime is voice-first — typed input is surfaced in the transcript but not injected into the live session in this version. |
| Validated against the GA shape, not end-to-end load | The session config, tool round-trip, and event normalisation are unit-tested against a mocked socket; full in-browser TTFA + barge-in feel is the pending live measurement. |

---

## 7. Memory system

Memory is a real pipeline, not a chat-history window. The system stores, retrieves, and synthesises information across turns.

### Storage layers

| Store | What lives there | Why |
|---|---|---|
| Weaviate | Semantic memories (preferences, life facts, insights) | Hybrid search (BM25 + dense vectors) |
| Postgres | User profile, conversation history, mood buckets | Relational, queryable, authoritative |
| Redis | Session state, retrieval cache, prewarm results, throttles | Fast, ephemeral |

### Memory extraction (`backend/app/memory/extractor.py`)

A background Celery worker reads the conversation turn and distils durable `preference` and `life_fact` memories. Throttled and batched — one embedding API call per pass, not one per sentence.

### Consolidation / reflection (`backend/app/memory/consolidation.py`)

Periodically a worker LLM reads the entire set of raw facts and synthesises 2–4 higher-order **insights** ("uses music to focus; reaches for lyric-light tracks while working"). Each run fully supersedes the last. Insights are injected above raw facts in the context block so the persona model gets the big picture first.

**Why run in the background?** Consolidation reads potentially hundreds of facts and makes one large LLM call. Putting that on the hot path would add 2–5s to every turn. Off-path, the user never waits on it.

### Retrieval (`backend/app/memory/retrieval.py`)

**Hybrid search:** BM25 for exact artist/track token matches + dense vectors for semantic intent. Results are reranked (`backend/app/memory/reranker.py`) and cached in Redis. The retrieval fan-out runs in parallel with the router so memory context is ready when the specialist needs it.

### Mood inference (`backend/app/mood/`)

Recently-played tracks are ingested from Spotify. A worker LLM labels each `(weekday × time-of-day)` bucket into a closed mood vocabulary. When current listening drifts from the bucket's pattern, a proactive draft is prepared for the next turn. `played_at` is approximated (Spotify's MCP recently-played carries no per-track timestamps) — documented rather than hidden.

### Why not a chat-history window?

A sliding window of the last N messages gives no insight between sessions, can't synthesise patterns, and grows stale. The pipeline here extracts durable facts, periodically synthesises them into insights, and retrieves only what's relevant — so the persona model gets a compact, high-signal context rather than raw transcript.

---

## 8. Background workers — Celery

`backend/worker/tasks/` — all tasks are Celery workers triggered after a turn streams, never on the hot path.

| Task | What it does |
|---|---|
| `memory_extraction` | Extracts `preference`/`life_fact` memories from the turn |
| `memory_consolidation` | Synthesises raw facts into higher-order insights |
| `mood_inference` | LLM-labels recently-played buckets into mood vocabulary |
| `proactive_check` | Checks whether a drift notification should be drafted |
| `session_flush` | Persists session history to Postgres |

**Why Celery?** Reliable task queuing with retry, backoff, and dead-letter. The alternative (fire-and-forget `asyncio.create_task`) would lose tasks on process restart.

---

## 9. CrewAI — what it does and what it doesn't

### What CrewAI contributes

| What | Where | Value |
|---|---|---|
| `crewai.LLM` | `providers/llm.py` | Provider abstraction (OpenAI/Anthropic/Ollama via litellm) |
| `crewai.Agent` | Every specialist | Structured role/goal/backstory/LLM container |
| `Crew`/`Task`/`Process` | `curator_crew.py` | Real multi-agent orchestration with hand-off |
| `@CrewBase`/`@agent`/`@task`/`@crew` | `curator_crew.py` | Standard framework pattern — YAML config, self.agents/self.tasks |
| `@tool` | `curator_crew.py` | Tool definition for the Scout agent |

### Why `@CrewBase` is not on the hot path

`@CrewBase` is designed for classes that end with `crew.kickoff()`. The hot path can't use `kickoff()` because:
1. It is synchronous and blocking — incompatible with `yield`-based SSE streaming
2. The turn's agent set is decided at runtime by the router — `@CrewBase` defines a fixed crew at class definition time
3. Speculative tasks and prewarm reuse require `asyncio` control that the framework abstracts away

### Why CrewAI Flows were not used

CrewAI recommends Flows (`@start`/`@listen` state machines) for routing between crews. This doesn't fit because:
1. Flow's event model is sequential — the hot path runs speculative tasks concurrently
2. Flow has no concept of a streaming HTTP response
3. The prewarm result from Redis can't be natively wired into a Flow state

### Could CrewAI be removed entirely?

**Yes, at some cost.** `crewai.LLM` could be replaced with direct litellm calls (~20 lines). `crewai.Agent` on the hot path is a persona container — a dataclass would hold the same fields. The curator would need manual task hand-off logic. The dependency is most justified by the curator crew and the upgrade path it enables; if the system stays at its current scale and no more crews are added, a sharp CTO would rightly call it an over-dependency.

---

## 10. The curator crew — proper CrewAI multi-agent

`backend/app/agents/curator_crew.py` — the one place the full CrewAI execution engine runs.

### Pattern used: `@CrewBase` with YAML config

Agent personas (`role`, `goal`, `backstory`) and task prompts (`description`, `expected_output`) live in:
- `backend/app/agents/config/agents.yaml`
- `backend/app/agents/config/tasks.yaml`

The `@agent` and `@task` methods use `config=self.agents_config['scout']` and `config=self.tasks_config['scout_task']` to pull from YAML. Runtime wiring (`llm`, `tools`, `context`, `output_pydantic`) is set in Python because it can't live in a static file.

**Why YAML for the curator but not other agents?** The curator is a `@CrewBase` class that uses `kickoff()` — the YAML pattern is part of `@CrewBase`'s infrastructure and works naturally here. The other agents are plain Python service classes that don't use `kickoff()`, so `@CrewBase` doesn't apply.

### How the hand-off works

```
Scout agent
  └── calls search_tracks tool → 8 Spotify candidates
  └── returns candidate list as text

curate_task declares context=[scout_task]
  └── CrewAI injects Scout's output into the Curator's prompt

Curator agent
  └── receives candidates + taste_profile + moment
  └── reranks, filters emotionally wrong matches
  └── returns CuratedPicks (typed Pydantic output via output_pydantic)
```

### Why off by default

Two chained LLM round-trips = 3–5s minimum. TTFA target for the voice path is 4–5s total. The curator is for a future "deep pick" mode, not the instant voice turn.

### Why `asyncio.to_thread`

`kickoff()` is synchronous. Running it on the event loop directly would block all other coroutines. `asyncio.to_thread` offloads it to a thread pool worker — the event loop stays free.

---

## 11. Observability — Langfuse

`backend/app/observability/langfuse.py` — every turn is a Langfuse trace.

**What is traced:**
- Router decision (intent, confidence, latency)
- Each agent span (start, end, latency)
- LLM generations (model, tokens, cost)
- Self-evaluation scores per turn: `context_used`, `retrieval_used`, `router_confidence`, `turn_latency_ms`

**Self-eval is deterministic, not an LLM judge.** Scores are computed from cheap signals (did retrieval return anything? did the router clear the threshold?) rather than spending an extra LLM call grading every reply. An LLM-as-judge would be the next step once there is real traffic to sample.

**Why Langfuse?** It provides structured trace → span → generation nesting, a UI for browsing traces by session/user, and a dataset API that was used to extract training data for the local classifier (`ml/router/extract_dataset.py` pulls `router-classify` generations from Langfuse).

---

## 12. Frontend — Next.js voice UI

`frontend/nextjs/`

### Audio capture — AudioWorklet

The browser captures mic audio through an `AudioWorklet` (runs in a dedicated audio thread, avoids main-thread jank). Frames are converted to 24 kHz mono PCM16 and forwarded over a WebSocket to `backend/app/api/voice_stream.py`.

**Why AudioWorklet over MediaRecorder?** `MediaRecorder` produces chunked blobs in a container format (webm/opus). The streaming STT providers want raw PCM. AudioWorklet gives frame-level access to the raw samples without a decode step on the server.

### Progressive playback — MediaSource

Audio chunks arrive as base64 SSE events. The frontend appends decoded MP3 bytes to a `MediaSource` buffer. The browser starts playing as soon as the buffer has enough data — before the full reply exists.

**Safari caveat:** Safari's `MediaSource` support for `audio/mpeg` is limited. Chrome/Edge/Firefox are the supported targets.

### State machine — `useVoiceSession`

The hook manages the voice turn lifecycle: `idle → listening → thinking → speaking → idle`. It owns the WebSocket (STT), the SSE stream (reply), and the audio playback, and exposes a clean `{ phase, start, stop, sendText }` interface to the UI.

### Theme system

Dual theme (light/dark) via CSS custom properties. Switching is instantaneous — no re-render, no flash.

---

## 13. Data stores

### Weaviate

Hybrid vector store. Stores memory objects with both dense embeddings (for semantic search) and BM25-indexed text (for exact artist/track name matches). Reranker sits on top. Gracefully degrades — a flaky Weaviate yields an empty memory slice, never a failed turn.

**Schema initialised at startup** (`backend/app/db/weaviate_init.py`) using `create_all`-style logic. Alembic (`alembic stamp head`) is used to prevent crash-loops when the schema already exists.

### Postgres

SQLAlchemy async. Stores: users, conversation history, user profiles, mood buckets. Two migrations (`alembic/versions/`): initial schema and `profile_display_name`.

### Redis

Session state, retrieval cache, prewarm results, throttles, in-flight prewarm markers. All keys are TTL'd. The prewarm result TTL is 30s — long enough to bridge eager→final, short enough to self-clean abandoned turns.

---

## 14. Provider abstraction

### LLM — `backend/app/providers/llm.py`

`get_llm(cfg)` returns the persona-tier model (expensive, expressive). `get_fast_llm(cfg)` returns the logistics-tier model (cheap, fast). Model resolution order:

1. Explicit `model=` argument (per-call override)
2. `cfg.llm_persona_model` / `cfg.llm_fast_model` (deploy-time env var)
3. Built-in provider default (`gpt-4o` / `gpt-4o-mini` for OpenAI; `claude-sonnet-4-6` / `claude-haiku-4-5` for Anthropic)

For Ollama, both tiers use the same local model — running two models locally is wasteful.

### STT — `backend/app/providers/stt.py` + `stt_stream.py`

Batch path (`stt.py`): OpenAI `whisper-1` or local faster-whisper.
Streaming path (`stt_stream.py`): Deepgram or OpenAI Realtime, behind a `StreamingTranscriber` Protocol.

### TTS — `backend/app/providers/tts.py`

ElevenLabs (production streaming) or Kokoro (local dev). Switched by `TTS_PROVIDER`.

---

## 15. Features built, measured, and changed

### Instant acknowledgment — built, scoped, kept

Originally a sub-ms keyword pass that spoke a neutral filler ("On it.") before the router returned, on all turns. Removed from the general path because on fast chat turns it filled a gap of only ~0.3s and on short text ElevenLabs v3 sounded robotic (needs context for good prosody).

**Scoped version still runs:** when a music command is detected (speculative search fires), a short neutral filler is synthesised concurrently so the user hears warmth during the ~3–5s search+recommendation wait. This is where the gap is real enough to justify it.

### Local faster-whisper — benchmarked, not baked

Benchmarked `large-v3` on an RTX 4060: warm inference 7s clip = ~1.1–1.8s, but end-to-end in practice = 2.0–3.8s due to non-model overhead (idle-downclock penalty, decode). `turbo` would save ~0.5–1s — landing roughly on par with the OpenAI API, for a 1.6GB model bake. Conclusion: the problem wasn't which batch model, it was the batch pattern. Streaming STT dissolved the problem entirely. Local faster-whisper is opt-in (`INSTALL_LOCAL_STT=true`) for GPU development only.

### Spotify audio features — removed

The DJ originally key-matched a crossfade queue using Camelot wheel + energy/valence (harmonic sequencing). Spotify deprecated `/audio-features` for new apps mid-build. Rather than leave dead code computing on placeholder constants, the entire machinery was removed: crossfade module, audio-feature fetch, feature fields on the track schema. Queueing is now search-relevance based. Mood is LLM-labelled from artist/track names into a closed vocabulary.

---

## 16. Known limitations

| Limitation | Detail |
|---|---|
| Classifier trained partly on synthetic data | 410 real examples, balanced with teacher-labelled synthetic phrasings to ~1,440. Held-out accuracy 0.78 but generalization to out-of-distribution phrasings is uncertain. Off by default until real traffic accumulates. |
| Streaming STT validated server-side only | Deepgram Flux end-to-end pipe and prewarm reuse are proven against the live API. Full in-browser TTFA with AudioWorklet capture is the pending measurement. |
| One seeded demo user | Memory and mood systems are structurally complete but validated against synthetic history. Real usage patterns will look different. |
| MediaSource / Safari | `audio/mpeg` via MSE is limited on Safari. Chrome/Edge/Firefox are the supported targets. |
| `played_at` approximation | Spotify's recently-played via MCP carries no per-track timestamps. Ingestion stamps the poll time. Accurate to the time-bucket with frequent use; documented rather than hidden. |
| `uvicorn --reload` disabled in Docker | File watcher is unstable on Windows bind mounts. Code changes need a container recreate. |
| 85% coverage gate not met | Tests are integration/unit tests that mock external deps; 472 pass. The gate requires 85% line coverage which the background worker tasks don't reach. |

---

## 17. Design decisions and tradeoffs — quick reference

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| STT | Streaming (Deepgram Flux) | Batch (whisper-1) | Eliminates ~1.5–2.8s serial wait; enables early-intent |
| Router | 4-tier cascade | Single LLM call | LLM is the latency floor; keyword + classifier + prewarm cut it for the majority of turns |
| Local classifier | Frozen encoder + linear heads | End-to-end fine-tune | CPU-only training, thin data — linear heads are sample-efficient and fast to serve |
| TTS | Sentence-streaming (default), bytes forwarded as rendered | Whole reply in one v3 pass | Per-sentence synthesis gets first audio out while the rest still generates; flag (`tts_stream_sentences`) reverts to whole-reply for max prosody context |
| Orchestration | Custom asyncio | CrewAI Flow / kickoff | Streaming + async parallelism + prewarm reuse are incompatible with kickoff()'s blocking model |
| CrewAI `@CrewBase` | Curator only | All agents | Hot-path agents are composed dynamically at runtime based on router decision; @CrewBase requires a fixed class definition |
| Memory | Extract → consolidate → hybrid retrieval | Sliding chat window | Windows give no cross-session insight; extraction + synthesis gives compact, high-signal context |
| Self-eval | Deterministic signals | LLM-as-judge | No extra latency or cost; LLM judge deferred until there is real traffic to sample |
| Background work | Celery | asyncio fire-and-forget | Celery survives process restarts; fire-and-forget drops tasks on crash |
| Spotify mood features | Removed | Keep with placeholders | Deprecated API returned constants; dead code computing on constants is worse than honest removal |
