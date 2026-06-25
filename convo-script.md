# Gia — 2-minute conversation eval script

A natural, ~2-minute spoken flow for testing Gia end-to-end. It's a real
conversation, not a checklist — but every line is chosen to exercise a specific
path (latency, memory, music intelligence, voice quality). Annotations tell you
what each turn tests and what to watch/listen for.

**Setup**
- Open the web app with the **seeded identity** (once — it's then remembered):
  `http://localhost:3000/?user_id=47309d9c-b862-52f4-b809-545a3e7a4194`
- Use a Chrome/Edge/Firefox browser (Safari's MediaSource audio is limited).
- Tap the ring to start a hands-free session, then just talk. Let Gia finish before the next line.
- Have a second tab on your **Langfuse** project to read per-turn timings after.

> Seeded profile (so you know what she should recall): Kolade, 21, Abuja, dog
> Rex, ex-footballer thinking of a comeback. Central Cee on morning runs, Drake
> at night, Afrobeats (Mavo / Suono Sai / Zaylevelten / Monochrome) when vibing.

> Speak naturally and a little elliptically (like the real traces) — that's the
> harder, more honest test. Don't over-enunciate "play artist X" every time.

---

## The flow

**0 · Gia greets first (on load).**
🎧 *Listen:* warm and alive, with a single tag (`[warm]`/`[softly]`). Reload once or twice — the phrasing **and** the tag should vary, never the same hello twice.

---

**1 · You:** "Hey Gia, I'm back — been heads-down on that AI project all day."
- *Tests:* conversational path + speculative reply (runs under the router); memory (the project).
- 🎧 *Listen:* she just **responds** — no "On it." filler in front. Audio starts streaming within a couple of seconds. Reply is short (1–2 sentences), not a paragraph.

**2 · You:** "Yeah, the latency stuff — finally getting somewhere."
- *Tests:* continuity / reference resolution from the previous turn.
- 🎧 *Listen:* she picks up the thread, doesn't restart with "Hey" again.

**3 · You:** "Anyway, play me something chill to wind down."
- *Tests:* `MUSIC_FIND`. "chill" is vibe-y (no artist in the words) → speculative search likely *misses* → normal search.
- 🎧 *Listen:* **one** track with a short reason — not a recited list. She plays it (no "want me to?").

**4 · You:** "Hmm, not feeling it — put on some Central Cee instead."
- *Tests:* `MUSIC_FIND` with the artist **in your words** → speculative search should **hit** (search ran under the router). Bonus: she may tie it to your morning-run habit.
- 🎧 *Compare:* this one should feel a touch snappier to music than #3. (Confirm in Langfuse: the `dj` span is shorter / `prefetched: true`.)

**5 · You:** "Yeah, that's the vibe. What's playing right now?"
- *Tests:* now-playing status query — answered from Spotify, **not** a recommendation.
- 🎧 *Listen:* she reports the actual track fast; the DJ does **not** run.

**6 · You:** "Cool. After this, queue some Asake."
- *Tests:* `MUSIC_QUEUE` + reference ("after this") → speculative search *miss* (elliptical) → fallback search; queue side-effect waits for the router.
- 🎧 *Listen:* she confirms it's queued (lead track + "a few more"), doesn't read the whole list.

**7 · You:** "What've you got lined up?"
- *Tests:* queue read-back.
- 🎧 *Listen:* concise — names what's next, not a 5-item recital.

**8 · You:** "Honestly I've been deep in the Mavo / burti wave lately."
- *Tests:* companion behavior on a **statement** + whether she gets the niche ("burti" / Mavo are in your memory) — she should react like a friend, **not** auto-queue.
- 🎧 *Listen:* she engages with the opinion (ideally nods to the new-wave sound); no track fires off a bare statement.

**9 · You:** "Who's better though — Central Cee or Drake?"
- *Tests:* persona spine — she must **pick one** with a reason, not fence-sit.
- 🎧 *Listen:* an actual take ("Drake, because…"), not "they both have their own styles."

**10 · You:** "Alright — what's my mood been like lately?"
- *Tests:* `MOOD_CHECK` (mood agent, reflected from listening).
- 🎧 *Listen:* a mood read tied to your pattern, degrades gracefully if thin.

**11 · You:** "Nice. Remember I'm working on that AI project — wish me luck."
- *Tests:* **memory recall** across the session (the project from turn 1).
- 🎧 *Listen:* she calls back the project specifically — proof the thread held.

*(Let her sign off. ~2 minutes total.)*

---

## Eval rubric

Score each ✅ / ⚠️ / ❌ as you go.

**Latency (the main event)**
- [ ] First audio on most turns lands in ~**3–5s** (chat) / ~**5–7s** (music).
- [ ] Audio **streams** — she's mid-sentence before the reply is "done"; no long silence then a wall of speech.
- [ ] Direct artist command (#4) feels faster to music than the vibe one (#3).
- [ ] **No** robotic "On it." filler in front of replies anywhere.

**Voice quality**
- [ ] Greeting tag sounds *alive*, not forced; varies across reloads.
- [ ] No jarring voice change mid-reply; chunks play gaplessly (MediaSource).
- [ ] Emotional lines (questions/tagged) sound richer than logistics lines.

**Music intelligence**
- [ ] One track + a reason, not a dumped list (#3, #4).
- [ ] Queue read-back is concise (#7); no full tracklist recited.
- [ ] Now-playing (#5) reports reality, doesn't invent.
- [ ] A bare statement (#8) does **not** auto-queue.

**Companion / memory**
- [ ] Picks a side on Dave vs Drake (#9) — no fence-sitting.
- [ ] Resolves references ("after this", "that one") correctly.
- [ ] Recalls the AI project at the end (#11).
- [ ] Replies stay short and spoken, not essay-like.

---

## What to capture (for the write-up)

Pull these from **Langfuse** for 2–3 representative turns:
- `turn_latency_ms` (self-eval score) — the end-to-end number.
- Span breakdown: `router` (~2s), `general` / `dj`, and confirm the conversational
  reply overlaps the router (speculation working).
- `dj` span on #4 vs #6 — shorter / `prefetched` on the direct command.
- `context_used` / `retrieval_used` / `router_confidence` scores.

From the **api logs** (`docker compose logs api`), the live signals:
- `elevenlabs_stream_ok` — streaming TTS firing (`model=eleven_v3` vs `eleven_flash_v2_5`).
- `dj_recommend_done` — DJ completed.
- `voice_transcribe_*` — STT (OpenAI `whisper-1`) timing.

> Note: STT runs on OpenAI `whisper-1` (~1.5–2.8s, serial before `/chat`) and
> normalises your "uh"/"um" out of the transcript — that's expected, not a bug.
