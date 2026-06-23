# Gia — Demo Script

A spoken walkthrough that exercises every feature, in a good order for a video.
Each line is something to **say out loud**; the arrow notes **what to listen/look
for** so you can confirm it on camera.

> Seeded demo user: `47309d9c-b862-52f4-b809-545a3e7a4194` (afrobeats/Drake taste,
> 34 listening events, taste memories). Personalisation below assumes you're
> signed in as this user.

## Before you hit record
1. Stack up & healthy: `docker compose ps` (api `healthy`), `curl localhost:8000/health` → 200.
2. Open **http://localhost:3000**, be signed in (Spotify), and have an **active Spotify device** (open Spotify on your phone/desktop and hit play once so a device exists).
3. Allow the **mic**. Quiet room. One browser tab.
4. Optional: open the api logs in another pane to show the pipeline live —
   `docker compose logs -f api` (you'll see `deepgram_flux_open`, `prewarmed`, `fast_path`, `brave_*`).

---

## Act 1 — She wakes up and speaks first
- **Tap the ring (or the mic).**
  → A soft 2-second "waking up" swell plays, the HUD says **Waking up…**, then Gia
  **greets you first** — and the greeting nods to your taste (chill/afrobeats), not a generic hello.

*Talking point: she opens the conversation; you didn't have to say anything.*

## Act 2 — She's a companion, not a search box
- "Hey, how's your day going?"
  → Warm, short, human reply. **No** music pushed.
- "Quick one — Drake or Asake, who you got?"
  → She **actually picks one** and says why (she has opinions; she won't fence-sit).
- "What can you do?"
  → Mentions music / artists / noticing your patterns — lightly, not a feature list.

## Act 3 — Play music (the acknowledgment moment)
- "Play some chill afrobeats."
  → You hear an **instant warm "okay"** — *"Say less." / "On it." / "Bet."* — **while she searches**,
  then she names the track and a few more lined up, and it **starts playing**.

*Talking point: the ack fills the ~4s search; it's neutral, so it never promises a
track before she's found one.*

- "Actually put on Essence by Wizkid."
  → Plays the specific track (or, if she can't find that exact one, says so and offers the closest — she won't fake it).
- "Queue up some Burna Boy after this."
  → Adds to the queue (doesn't interrupt what's playing).

## Act 4 — She knows the *world* (live, grounded — not from memory)
- "What's Drake's latest album?"
  → **Iceman** (current), not an old answer — pulled live from search.
- "Tell me about Drake."
  → Grounded recent facts (Iceman topping charts), and she **offers** to play something
    rather than auto-playing.
- "Any news on the World Cup today?"
  → A current, real result — not "I'm not sure."

*Talking point: current-facts turns hit live web/news search and she answers from the
results, not stale training data.*

## Act 5 — She knows *you* (memory + mood)
- "What do I usually listen to?"
  → Recalls your taste from memory (afrobeats, the artists you lean on).
- "What's my mood been like lately?"
  → Reads your listening patterns and gives a read, tied to your usual.
- "Recommend something for me right now."
  → A pick that fits *your* taste — and she **waits for a yes** before playing.
- "Yeah, go for it."
  → *Now* it plays.

## Act 6 — Status & restraint
- "What's playing right now?"
  → Reports the actual current track (doesn't guess).
- "I'm a Davido fan for life."
  → She **reacts** to the statement like a friend — agrees/riffs — and does **not** auto-play Davido.

## Act 7 — Close
- "Alright, that's it for now — thanks Gia."
  → Warm sign-off. If you stay silent ~10s the ring goes calm on its own.

---

## Under the hood (caption/narration ideas)
- **Streaming STT (Deepgram Flux):** she starts processing while you're still talking — no record-then-upload wait.
- **Early-intent:** the router starts on the *eager* end-of-turn, so music/specialist turns don't pay the full ~2s router serially (watch `prewarmed`/`fast_path` in the logs).
- **Streaming TTS:** audio streams out as it's generated — she's mid-sentence before the full reply exists.
- **Grounded answers:** live Brave web/news search for current facts; she prefers it over memory.
- **Warm ack:** neutral filler on music turns only, on the expressive v3 voice.

## If something's off
- No greeting on tap → check `curl localhost:8000/chat/opening` returns a `greeting`.
- "Nothing's playing" on a play command → no active Spotify device; open Spotify and press play once.
- Stale answers → confirm `BRAVE_API_KEY` is set and `STT_PROVIDER=deepgram` in `.env`.
- api won't start with "relation users already exists" → `docker compose run --rm api alembic stamp head` then `docker compose up -d api`.
