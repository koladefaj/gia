# Gia — Voice Music Companion

> [orb GIF here — record on Day 14]

"A voice companion that knows your music taste, sounds like a warm human, sequences playlists like a DJ, and notices your mood before you mention it."

> [Demo video link here]

---

## What it feels like

- You speak. She responds — warm, specific, grounded in what she knows about you.
- She recommends one track, not ten. She explains why. The crossfade is smooth.
- She notices you're listening to something different than usual and gently asks about it.

---

## Architecture

> [diagram — fill Day 6]

---

## Four things I designed for

**Memory continuity** — three-tier: Postgres facts, Weaviate semantic preferences, Redis session. Extraction, decay, supersede. She remembers because of a real system, not a chat history.

**Emotional prosody** — ElevenLabs v3 with audio tags. v3/Flash hybrid for latency. The voice is the product.

**Proactive mood awareness** — audio feature time-series, quadrant classifier, Celery-driven pattern detection. No LLM needed here. Fast, free, interpretable.

**DJ-quality sequencing** — energy-aware ordering + Camelot wheel key matching. Each track leads naturally into the next.

---

## Engineering highlights

> [fill Day 13]

---

## Design decisions and tradeoffs

> [fill Day 13]

---

## Responsible design

Gia is designed to help and let you go, not to maximise engagement. The memory journal makes everything she has learned visible. She never auto-saves, auto-queues, or creates playlists without a confirmed yes in the same turn. If asked whether she is an AI, she says she is. The mood inference is transparent — the quadrant classifier is documented here, not hidden behind "AI magic."

---

## Run it

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY (or set LLM_PROVIDER=ollama for local)
docker compose up
# First run: seed the demo user
python scripts/seed_user.py
# Open http://localhost:8000/health
```

---

## Roadmap

- Real-time voice with barge-in (WebRTC)
- User-editable memory ("Gia, forget that")
- Scene mode: Google Places + social discovery
- Mood ML model trained on personal listening history
- Shared listening: two users, one queue
- Multilingual (ElevenLabs v3 supports 70+ languages)

---

*Memory pipeline reuses patterns from [Engram](../engram). LLM factory from Phalanx. Worker patterns from Relier.*
