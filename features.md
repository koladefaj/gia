# Gia — Intent-Aware Acknowledgment & Latency-Hiding Voice Pipeline


> write any code" section first — I want you to explore and plan before building.

---

## Context


The problem: the user hears silence while retrieval / search / tool calls / LLM
reasoning run. We're going to make Gia **engage immediately** — react or
acknowledge in <1s — while expensive work happens in the background. The user
should feel like Aria is *thinking while talking*, not waiting and then talking.

This is an additive feature on top of an existing codebase. **Do not rewrite
working STT/TTS/agent code unless it's required for integration.** Adapt to what's
already there.

---

## Objective (north star)

When the user says *"Did you see what Drake said about Ice Man?"*, Gia should:

1. Classify intent + tone + engagement mode in <150ms.
2. Speak a personality-rich reaction within ~1s: *"[surprised] Wait, for real? Let me check."*
3. Run Brave Search (and any other needed retrieval) **concurrently** while that audio plays.
4. Stream the real answer back **sentence-by-sentence** into TTS as the LLM generates it.
5. Never repeat the same acknowledgment twice in a row.
6. Emit a full trace for the turn.

No noticeable silence at any point.

---

## Before you write any code

1. Map the existing pipeline: where STT hands off, how the current router works,
   how tool calls are invoked, how TTS is fed, and what the async model is.
2. Identify the seams where this feature plugs in. List them back to me.
3. Note what already exists that we can reuse (TTS adapter? async runner? trace setup?).
4. Propose a short build plan (phases below are a starting point) and **flag any
   assumptions or decision points** before implementing. In particular confirm:
   - Which TTS provider(s) are wired today (ElevenLabs / OpenAI / Cartesia / Kokoro?).
   - Whether retrieval functions (Brave, Spotify, memory/RAG) are already async.
   - Whether there's existing Langfuse/Prometheus instrumentation to extend.

Build in vertical slices that run end-to-end, not one giant change.

---

## Target architecture

```
User Speech
   │
   ▼
  STT
   │
   ▼
Hybrid Router ──► {intent, tone, confidence, engagement_mode, needs_*}
   │
   ├─ intent confidence ≥ threshold ──► proceed
   └─ intent confidence <  threshold ──► Planner ──► agent/capability selection
   │
   ▼
Acknowledgment Selection (local, deterministic, <10ms)
   │
   ▼
Voice Adapter (tone → provider tags)
   │
   ▼
Immediate TTS  ──────────────────►  Audio (plays NOW)
   │
   │  (in parallel, only if engagement_mode executes)
   ▼
Parallel Retrieval (asyncio.gather): brave_search · memory · spotify · rag
   │
   ▼
Conversation Agent (context build → token stream)
   │
   ▼
Sentence Buffer → TTS → Playback   (per sentence, never wait for full completion)
   │
   ▼
Observability (trace the whole turn)
```

---

## Components to build

### 1. Hybrid Router
- **Model:** small/fast (config: `ROUTER_MODEL`, default `gpt-4o-mini`).
- **Output:** structured JSON only (schema below). No prose, no markdown fences.
- **Behavior:** classify intent + tone + engagement_mode + retrieval needs + confidence.
- **Escalation:** if intent confidence < `ROUTER_CONFIDENCE_THRESHOLD` (default 0.8),
  call the **Planner** instead of dispatching directly.
- Replace any keyword/`if "play" in query` logic with model classification. Keywords
  break on phrasing like *"play something that feels like when I used to listen to Tems."*
- **Latency target:** <150ms.

### 2. Planner (fallback only)
- **Model:** config `PLANNER_MODEL`, default `gpt-5.5`.
- Invoked only on low-confidence/ambiguous/mixed queries.
- Returns the resolved intent **and** the set of capabilities/agents to run, e.g.
  `{"intent": "MIXED", "agents": ["memory", "weather", "dj", "calendar"]}`.
- Keep it off the hot path — most turns must never touch it (it's the "LLM tax").

### 3. Acknowledgment System
- **No LLM.** Templates stored locally in `acknowledgements.json`.
- Keyed by `intent → tone → [list of strings]`.
- Selects a template matching intent+tone (and engagement_mode where relevant),
  **avoids the last 5 used**, returns a string.
- **Latency target:** <10ms.

### 4. Voice Adapter
- Converts an abstract `tone` into provider-specific tags.
- **The router MUST NOT emit provider tags** (`[light laugh]` is wrong; `playful` is right).
- Single abstraction so we can swap ElevenLabs / OpenAI Voice / Cartesia / Kokoro
  without touching router logic.
- Interface:
  ```python
  class VoiceAdapter:
      def convert_tone_to_tags(self, tone: str) -> str: ...
  ```
- Mappings (extend as needed): `playful → [light laugh]`, `warm → [gentle]`,
  `thoughtful → [thoughtful pause]`, `surprised → [surprised]`.

### 5. Parallel Retrieval Orchestrator
- All needed retrieval runs **concurrently** via `asyncio.gather`. No sequential calls.
  ```python
  results = await asyncio.gather(brave_search(...), memory_lookup(...), spotify_lookup(...))
  ```
- Only kick off the retrievers the router flagged (`needs_search`, `needs_memory`, etc.).
- Runs *while the acknowledgment audio is already playing*.

### 6. Streaming Conversation Pipeline
- **Model:** config `CONVERSATION_MODEL`, default `gpt-5.5` (start on `gpt-4o` if 5.5 unavailable).
- Builds context: `{query, search_results, memory_context, spotify_context, tone}`.
- Streams tokens → **sentence buffer** → flush each complete sentence to TTS → playback.
- Never wait for the full completion before speaking.

### 7. State Tracking
- Track `last_5_acknowledgments` per session; don't reuse until they age out.
- Track per-turn timing for observability.

### 8. Observability
- Langfuse traces per turn. Prometheus metrics for:
  `router_latency`, `ack_selection_latency`, `search_latency`, `memory_latency`,
  `llm_first_token_latency`, `tts_first_audio_latency`, `time_to_first_audio`.

---

## Router output schema

```json
{
  "intent": "NEWS_QUERY",
  "tone": "surprised",
  "confidence": 0.94,
  "engagement_mode": "react_then_execute",
  "needs_search": true,
  "needs_memory": false,
  "needs_music": false,
  "needs_artist_lookup": false
}
```

**Intents:** `MUSIC_FIND`, `MUSIC_QUEUE`, `ARTIST_INFO`, `MOOD_CHECK`,
`MEMORY_QUERY`, `NEWS_QUERY`, `GENERAL_CHAT`, `MIXED`.

**Tones:** `curious`, `surprised`, `warm`, `playful`, `thoughtful`, `excited`,
`empathetic`, `confident`.

---

## Engagement mode logic

The router sets `engagement_mode`. This decides whether Aria reacts, clarifies, or
executes — it's what makes her a *participant* in the conversation rather than a
search box.

| Mode | When | Gia's behavior | Retrieval? |
|---|---|---|---|
| `direct_execute` | Unambiguous action ("Play Tems") | Brief ack, act immediately | Yes, now |
| `react_then_execute` | Enough info + emotional hook ("Did you see what Drake said about Ice Man?") | React with personality ("Wait, for real? Let me check") **and** execute | Yes, now |
| `clarify` | Genuinely underspecified ("Did you see what Drake said?" with no topic) | Ask a short question ("Nah, what'd he say?") | **No** — wait for next turn |
| `confirm_action` | Destructive / high-consequence ("Delete all my playlists") | Confirm before doing anything | **No** until confirmed |

Rules of thumb to encode:
- Intent confidence ≥ 0.8 → lean toward executing. < 0.8 → escalate to Planner.
- **Do not over-clarify.** A second question on a clear request is annoying
  ("Play Tems" → "Which Tems song?" → "Are you sure?" is a failure mode).
- `react_then_execute` is the sweet spot for most query-style turns: personality
  *without* burning an extra turn.

---

## Model assignments (config-driven; names are env values, not hardcoded)

| Component | Default model | Env var |
|---|---|---|
| Router Agent | `gpt-4o-mini` | `ROUTER_MODEL` |
| Memory Extractor | `gpt-4o-mini` | `MEMORY_MODEL` |
| Artist Agent | `gpt-4o` | `ARTIST_MODEL` |
| Planner Agent | `gpt-5.5` | `PLANNER_MODEL` |
| Conversation Agent | `gpt-5.5` | `CONVERSATION_MODEL` |
| Mood Engine | none (no LLM) | — |
| DJ Logic | none (deterministic) | — |

Put all model names in config so they swap without code changes. If `gpt-5.5`
isn't available in the environment, fall back to `gpt-4o` and log it.

---

## Latency budget (per turn)

| Stage | Target |
|---|---|
| STT | <300ms |
| Router | <150ms |
| Acknowledgment selection | <10ms |
| Acknowledgment TTS start | <300ms |
| Brave Search | <1000ms |
| Memory retrieval | <100ms |
| Conversation model first token | <500ms |
| **Time to first audio** | **<1000ms** |

---

## Acceptance criteria

Walkthrough — user says *"Did you see what Drake said about Ice Man?"*:

- [ ] Router classifies `NEWS_QUERY`, tone `surprised`, `engagement_mode: react_then_execute`, `needs_search: true`.
- [ ] An acknowledgment is selected (no LLM call) in <10ms.
- [ ] Acknowledgment audio begins within 1s, with the correct voice tag applied.
- [ ] Brave Search runs concurrently with the acknowledgment audio.
- [ ] The real response streams sentence-by-sentence into TTS.
- [ ] No noticeable silence at any point.
- [ ] The acknowledgment is not a repeat of the last 5.
- [ ] A full trace for the turn is visible in Langfuse with all metrics populated.

Also verify the clarify path: *"Did you see what Drake said?"* (no topic) →
`engagement_mode: clarify` → Gia asks a question → **no search fires** until the
follow-up turn provides the topic.

---

## Constraints & non-goals

- **No LLM for acknowledgments.** Local templates only.
- **Router never emits provider-specific voice tags** — only abstract tones.
- **Retrieval is concurrent** (`asyncio.gather`), never sequential.
- **Don't over-clarify.** Default to reacting + executing when confidence is high.
- **Keep the Planner off the hot path** — most turns shouldn't pay for it.
- Everything provider/model-specific goes behind config/adapters.
- Don't break the existing pipeline — extend it.

---


---

## Build in phases

1. **Router + schema** (pydantic models, JSON-only output, confidence) + unit tests
   on sample utterances covering every intent/tone/engagement_mode.
2. **Acknowledgment system + voice adapter** — selection <10ms, last-5 avoidance,
   tone→tag mapping. Wire immediate TTS of the acknowledgment.
3. **Parallel retrieval orchestrator** — `asyncio.gather`, only firing flagged
   retrievers, gated by engagement_mode.
4. **Streaming conversation pipeline** — sentence buffer → TTS → playback.
5. **Planner fallback** on low confidence.
6. **Observability** — traces + metrics across all of the above.

After each phase, show me a runnable slice and the timing numbers before moving on.