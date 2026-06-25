'use client';

/**
 * A procedural "wake" earcon — played the instant the user taps to start, to
 * cover the ~2-3s while the greeting is synthesized. A soft ascending swell
 * (root + fifth + octave) with a low-pass filter opening up, so it reads as
 * "powering on" rather than a beep.
 *
 * Synthesized with Web Audio (no asset). Created inside the tap gesture so it's
 * allowed to sound; uses its own short-lived context and closes itself after.
 */

export function playBootEarcon(): void {
  try {
    const Ctor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return;
    const ctx = new Ctor();
    void ctx.resume?.();

    const now = ctx.currentTime;
    const DUR = 2.0;

    // Master envelope — quiet; this is an earcon, not music.
    const master = ctx.createGain();
    master.gain.setValueAtTime(0, now);
    master.gain.linearRampToValueAtTime(1, now + 0.06);
    master.gain.setValueAtTime(1, now + DUR - 0.5);
    master.gain.linearRampToValueAtTime(0, now + DUR);

    // Warmth that "opens" as it wakes.
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(420, now);
    filter.frequency.exponentialRampToValueAtTime(2400, now + 0.95);
    filter.Q.value = 0.6;

    master.connect(filter);
    filter.connect(ctx.destination);

    // A soft chord: A3, E4, A4 — each with a gentle attack, slightly staggered,
    // and a small upward glide so the whole thing feels like it rises into being,
    // then drifts down over a couple of seconds.
    const partials = [220, 330, 440];
    partials.forEach((f, i) => {
      const osc = ctx.createOscillator();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(f * 0.99, now);
      osc.frequency.exponentialRampToValueAtTime(f, now + 0.7);

      const g = ctx.createGain();
      const peak = 0.16 / (i + 1);
      g.gain.setValueAtTime(0.0001, now);
      g.gain.linearRampToValueAtTime(peak, now + 0.2 + i * 0.07);
      g.gain.exponentialRampToValueAtTime(0.0008, now + DUR);

      osc.connect(g);
      g.connect(master);
      osc.start(now);
      osc.stop(now + DUR + 0.05);
    });

    // Free the context once the sound has finished.
    window.setTimeout(() => {
      void ctx.close?.();
    }, (DUR + 0.4) * 1000);
  } catch {
    /* audio is a nicety — never let it break the session start */
  }
}
