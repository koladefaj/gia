'use client';

/**
 * VoiceScreen — the post-auth, hands-free session. The ring fills the surface
 * and reacts to the live conversation; a glass HUD carries status and the
 * type-or-talk controls. The voice is the product, so captions stay small and
 * ephemeral and never become a chat log.
 *
 * All session behaviour comes from useVoiceSession — this file only presents it.
 */

import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';

import { getOpening } from '@/lib/api';
import { playBootEarcon } from '@/lib/earcon';
import { useTheme } from '@/lib/theme';
import { useVoiceSession, type VoicePhase } from '@/lib/useVoiceSession';
import VoiceWaves from './VoiceWaves';
import ThemeToggle from './ThemeToggle';
import { Microphone, SignOut, Stop, User } from './Icons';

const PHASE_LABEL: Record<VoicePhase, string> = {
  idle: 'Tap to talk',
  listening: 'Listening',
  thinking: 'Thinking',
  speaking: 'Gia is speaking',
  error: 'Something went wrong',
};

function ProfileMenu({ onSignOut }: { onSignOut: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-label="Account menu"
        className="flex h-9 w-9 items-center justify-center rounded-full border border-hairline/10 bg-[var(--glass-bg)] text-ink-soft backdrop-blur-md transition-colors hover:text-ink"
      >
        <User size={17} weight="bold" />
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.96 }}
            transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
            className="absolute right-0 top-11 min-w-[150px] rounded-2xl border border-hairline/10 bg-[var(--glass-bg)] p-1 shadow-[var(--glass-shadow)] backdrop-blur-xl"
          >
            <button
              onClick={() => {
                setOpen(false);
                onSignOut();
              }}
              className="flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-[13px] font-medium text-ink-soft transition-colors hover:bg-[rgb(var(--ink)/0.06)] hover:text-ink"
            >
              <SignOut size={16} />
              Sign out
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function VoiceScreen({
  userId,
  sessionId,
  onSignOut,
}: {
  userId: string | null;
  sessionId: string;
  onSignOut: () => void;
}) {
  const { theme } = useTheme();
  const session = useVoiceSession(userId, sessionId);
  const { phase, status, transcript, error, levelRef, start, stop, sendText } = session;
  const [opening, setOpening] = useState('');
  const [draft, setDraft] = useState('');
  const greetedRef = useRef(false);

  // Gia greets first — fetch a warm opening line on mount.
  useEffect(() => {
    let alive = true;
    getOpening(userId).then((g) => {
      if (alive && g) setOpening(g);
    });
    return () => {
      alive = false;
    };
  }, [userId]);

  const active = phase !== 'idle';

  // "Waking up" — the gap between the first tap and Gia's greeting audio, while
  // the TTS is synthesized. A wake earcon + this state cover the dead air.
  const [booting, setBooting] = useState(false);

  // The first tap also unlocks audio (autoplay policy), so Gia speaks her
  // opening here, not on load. She greets on the first interaction of EVERY
  // session (every login), not just the very first sign-up.
  const beginSession = async () => {
    const isFirst = !greetedRef.current && transcript.length === 0;
    if (!isFirst) {
      void start();
      return;
    }
    // Commit to greeting now. Crucially, mark greeted only because we ARE about to
    // greet — and if the prefetched opening hasn't landed yet (returning users
    // hydrate instantly and often tap first), fetch it on demand rather than
    // silently skipping the greeting for the whole session.
    greetedRef.current = true;
    setBooting(true);
    playBootEarcon();
    const greet = opening || (await getOpening(userId));
    void start(greet || undefined);
  };
  const toggle = () => {
    if (active) stop();
    else void beginSession();
  };

  // Clear the waking state as soon as the session actually moves (greeting
  // starts speaking, listening begins, or it errors). Safety-timeout in case
  // the start stalls, so the HUD never sticks on "Waking up".
  useEffect(() => {
    if (phase !== 'idle') setBooting(false);
  }, [phase]);
  useEffect(() => {
    if (!booting) return;
    const t = setTimeout(() => setBooting(false), 6000);
    return () => clearTimeout(t);
  }, [booting]);

  // Voice-first: nothing Gia says is rendered on screen. The opening greeting is
  // still spoken (passed to start), and `transcript` still gates the greet, but
  // no captions are shown — the voice is the whole experience.
  const showDots = booting || phase === 'listening' || phase === 'thinking' || phase === 'speaking';
  const statusLabel = booting ? 'Waking up…' : status || PHASE_LABEL[phase];

  return (
    <main className="relative min-h-[100dvh] w-full overflow-hidden">
      <div className="gia-grain" aria-hidden />

      {/* Top controls */}
      <div className="absolute right-5 top-5 z-30 flex items-center gap-3">
        <ThemeToggle />
        <ProfileMenu onSignOut={onSignOut} />
      </div>

      {/* Audio-reactive waveform — tap anywhere on it to start/stop */}
      <button onClick={toggle} aria-label="Toggle Gia voice" className="absolute inset-0 cursor-pointer">
        <VoiceWaves levelRef={levelRef} active={active} theme={theme} />
      </button>

      {/* Glass HUD — status + the type-or-talk controls */}
      <div className="absolute bottom-[6%] left-1/2 z-20 w-[min(560px,90vw)] -translate-x-1/2">
        <div className="flex flex-col items-center gap-4 rounded-panel border border-hairline/[0.08] bg-[var(--glass-bg)] px-5 py-5 shadow-[var(--glass-shadow)] backdrop-blur-xl">
          <div className="flex h-[16px] items-center gap-[11px]">
            {showDots && (
              <div className="flex h-[13px] items-end gap-[3px]">
                {[0, 0.15, 0.3, 0.45].map((delay) => (
                  <span
                    key={delay}
                    className="w-[2.5px] rounded-[2px] bg-accent"
                    style={{ height: '100%', animation: `giaDot 0.7s ease-in-out ${delay}s infinite` }}
                  />
                ))}
              </div>
            )}
            <span
              className="text-[13.5px] font-medium tracking-[0.04em] text-ink-soft"
              style={active || booting ? { animation: 'giaPulse 2.4s ease-in-out infinite' } : undefined}
            >
              {statusLabel}
            </span>
          </div>

          {error && <span className="text-[12px] text-accent">{error}</span>}

          {/* Type or talk — both run the same pipeline */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              const text = draft.trim();
              if (!text) return;
              setDraft('');
              void sendText(text);
            }}
            className="flex w-full items-center gap-2.5"
          >
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder={active ? 'Type to Gia' : 'Type, or tap the mic to talk'}
              className="h-11 flex-1 rounded-full border border-hairline/10 bg-[rgb(var(--field)/0.6)] px-5 text-[14px] text-ink outline-none transition-colors placeholder:text-ink-faint focus:border-accent/40"
            />
            <button
              type="button"
              onClick={toggle}
              aria-label={active ? 'Stop' : 'Start talking'}
              title={active ? 'Stop' : 'Start talking'}
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-ink text-field shadow-[0_6px_16px_rgba(0,0,0,0.25)] transition-transform hover:scale-105 active:scale-95"
            >
              {active ? <Stop size={16} weight="fill" /> : <Microphone size={18} weight="fill" />}
            </button>
          </form>
        </div>
      </div>
    </main>
  );
}
