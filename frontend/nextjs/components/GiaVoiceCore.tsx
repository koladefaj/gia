'use client';

/**
 * GiaVoiceCore — the full product surface.
 *
 * Pre-auth  : a living landing — audio-reactive ring, a typing tagline, value
 *             props, and real "Continue with Spotify" OAuth.
 * Post-auth : a hands-free voice session — tap the ring to talk, live captions,
 *             agent status, and a ring that reacts to the real conversation.
 *
 * Stack: Next.js App Router · Tailwind · @react-three/fiber · postprocessing
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Bloom, EffectComposer } from '@react-three/postprocessing';
import * as THREE from 'three';

import { getOpening, spotifyLoginUrl } from '@/lib/api';
import { useIdentity } from '@/lib/identity';
import { useVoiceSession, type VoicePhase } from '@/lib/useVoiceSession';
import TypingText from './TypingText';

/* -------------------------------------------------------------------------- */
/* Tunables + helpers                                                          */
/* -------------------------------------------------------------------------- */
const CONFIG = {
  ringRadius: 1.0,
  tubeRadius: 0.011,
  tubularSegments: 520,
  radialSegments: 14,
  bloomIdle: 0.4,
  bloomActive: 1.25,
} as const;

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

// Shared white theme — a clean near-white field with a faint cool falloff so
// the glowing ring still reads against it. One source of truth for every screen.
const BG =
  'bg-[radial-gradient(circle_at_50%_42%,#ffffff_0%,#f6f7f9_58%,#eef0f3_100%)]';

/* -------------------------------------------------------------------------- */
/* VoiceRing — pulses to a live audio level (0..1) via levelRef                */
/* -------------------------------------------------------------------------- */
function VoiceRing({
  levelRef,
  active,
  bloomRef,
}: {
  levelRef: React.MutableRefObject<number>;
  active: boolean;
  bloomRef: React.MutableRefObject<{ intensity: number } | null>;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const energy = useRef(0);

  // Build the torus once, and precompute — per tubular segment (≈520, not per
  // vertex ≈7800) — the ring angle and outward radial direction, plus scratch
  // offset buffers. Keeps the per-frame wobble cheap.
  const ring = useMemo(() => {
    const geometry = new THREE.TorusGeometry(
      CONFIG.ringRadius,
      CONFIG.tubeRadius,
      CONFIG.radialSegments,
      CONFIG.tubularSegments,
    );
    const base = Float32Array.from(geometry.attributes.position.array);
    const len = CONFIG.tubularSegments + 1; // vertices per radial loop
    const angle = new Float32Array(len);
    const dirX = new Float32Array(len);
    const dirY = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      const u = (i / CONFIG.tubularSegments) * Math.PI * 2;
      angle[i] = u;
      dirX[i] = Math.cos(u);
      dirY[i] = Math.sin(u);
    }
    return { geometry, base, len, angle, dirX, dirY, offX: new Float32Array(len), offY: new Float32Array(len) };
  }, []);

  useFrame((state) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const t = state.clock.getElapsedTime();

    // Smoothly track the live level; fall back to a serene breath when idle.
    const target = active ? levelRef.current : 0;
    energy.current = lerp(energy.current, target, 0.12);
    const e = energy.current;

    if (bloomRef.current) {
      const targetBloom = active
        ? CONFIG.bloomIdle + e * (CONFIG.bloomActive - CONFIG.bloomIdle)
        : CONFIG.bloomIdle + Math.sin(t * 1.1) * 0.06;
      bloomRef.current.intensity = lerp(bloomRef.current.intensity, targetBloom, 0.08);
    }

    // ── Wobble: undulate the ring's radius around its circumference so the
    // outline is organic, never a perfect circle. A few drifting sine waves,
    // gentle at rest and swelling with the voice. Offset is per segment, then
    // applied to that segment's radial vertices.
    const { geometry, base, len, angle, dirX, dirY, offX, offY } = ring;
    const amp = 0.03 + e * 0.13;
    for (let i = 0; i < len; i++) {
      const u = angle[i];
      const w =
        (Math.sin(3 * u + t * 0.9) +
          0.5 * Math.sin(5 * u - t * 0.7) +
          0.3 * Math.sin(8 * u + t * 1.35)) *
        amp;
      offX[i] = dirX[i] * w;
      offY[i] = dirY[i] * w;
    }
    const pos = geometry.attributes.position as THREE.BufferAttribute;
    const arr = pos.array as Float32Array;
    for (let k = 0; k < pos.count; k++) {
      const i = k % len;
      arr[k * 3] = base[k * 3] + offX[i];
      arr[k * 3 + 1] = base[k * 3 + 1] + offY[i];
    }
    pos.needsUpdate = true;

    const breathe = Math.sin(t * 0.9) * 0.012;
    const scale = active ? 1 + breathe + e * 0.1 : 1 + breathe;
    mesh.scale.setScalar(scale);
    mesh.rotation.z = Math.sin(t * 0.18) * 0.05;
  });

  return (
    <mesh ref={meshRef} geometry={ring.geometry} frustumCulled={false}>
      <meshBasicMaterial color="#23272e" toneMapped={false} transparent />
    </mesh>
  );
}

function RingCanvas({
  levelRef,
  active,
}: {
  levelRef: React.MutableRefObject<number>;
  active: boolean;
}) {
  const bloomRef = useRef<{ intensity: number } | null>(null);
  return (
    <Canvas camera={{ position: [0, 0, 5.6], fov: 35 }} gl={{ antialias: true, alpha: true }} dpr={[1, 2]}>
      <VoiceRing levelRef={levelRef} active={active} bloomRef={bloomRef} />
      <EffectComposer>
        <Bloom
          ref={bloomRef as never}
          intensity={CONFIG.bloomIdle}
          radius={0.08}
          luminanceThreshold={0.9}
          luminanceSmoothing={0.4}
          mipmapBlur
        />
      </EffectComposer>
    </Canvas>
  );
}

/* -------------------------------------------------------------------------- */
/* Icons                                                                       */
/* -------------------------------------------------------------------------- */
function SpotifyIcon({ size = 22 }: { size?: number }) {
  // Official Spotify glyph (simpleicons path) — a filled green disc whose three
  // sound-bars are negative space, so on the dark button they read as the brand
  // mark. Replaces the hand-rolled icon that looked slightly off.
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="#1ED760" aria-hidden xmlns="http://www.w3.org/2000/svg">
      <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.42 1.56-.299.421-1.02.599-1.559.3z" />
    </svg>
  );
}

function MicIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden>
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M6 11a6 6 0 0 0 12 0M12 17v3" />
    </svg>
  );
}

/* -------------------------------------------------------------------------- */
/* Profile menu                                                                 */
/* -------------------------------------------------------------------------- */
function ProfileAvatar({ onSignOut }: { onSignOut: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="absolute right-6 top-6 z-20">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-label="Account menu"
        className="flex h-9 w-9 items-center justify-center rounded-full bg-[#1a1c1f] text-[13px] font-semibold text-[#f4f5f6] shadow-[0_2px_8px_rgba(0,0,0,0.25)] ring-1 ring-white/10 transition-transform hover:scale-105"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
          <path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z" />
        </svg>
      </button>
      {open && (
        <div className="absolute right-0 top-11 min-w-[130px] rounded-xl border border-white/10 bg-[#1a1c1f]/90 p-1 shadow-xl backdrop-blur-xl">
          <button
            onClick={() => {
              setOpen(false);
              onSignOut();
            }}
            className="w-full rounded-lg px-3 py-2 text-left text-[13px] font-medium text-[rgba(244,245,246,0.72)] transition-colors hover:bg-white/[0.08] hover:text-[#f4f5f6]"
          >
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Screen 1 — Landing (pre-auth)                                               */
/* -------------------------------------------------------------------------- */
const TAGLINES = [
  'Just talk to it.',
  'It learns what you love.',
  'Reads the moment, plays what fits.',
  'Always listening, never pushy.',
];

function LandingScreen({ userId }: { userId: string | null }) {
  const idleLevel = useRef(0);
  return (
    <main className={`relative h-dvh w-full overflow-hidden ${BG}`}>
      {/* Ambient ring — fills the screen; everything else is placed against it */}
      <div className="pointer-events-none absolute inset-0">
        <RingCanvas levelRef={idleLevel} active={false} />
      </div>

      {/* Inside the ring: wordmark, the typed line, and the one action */}
      <div className="pointer-events-none absolute inset-0 z-10 flex flex-col items-center justify-center px-6">
        <h1 className="text-[44px] font-[300] uppercase tracking-[0.46em] text-[#1b1e23] select-none">
          Gia
        </h1>
        <p className="mt-3 text-[12px] uppercase tracking-[0.36em] text-[rgba(27,30,35,0.42)]">
          Hands-free
        </p>
        <p className="mt-2 h-[22px] text-center text-[13.5px] tracking-[0.01em] text-[rgba(27,30,35,0.6)]">
          <TypingText phrases={TAGLINES} />
        </p>

        {/* Spotify — a logo circle that expands into a pill on hover */}
        <a
          href={spotifyLoginUrl(userId)}
          aria-label="Continue with Spotify"
          className="group pointer-events-auto mt-8 inline-flex h-12 items-center overflow-hidden rounded-full bg-[#121212] shadow-[0_10px_30px_rgba(0,0,0,0.16)] transition-all duration-300 ease-out hover:shadow-[0_14px_34px_rgba(0,0,0,0.22)]"
        >
          <span className="flex h-12 w-12 shrink-0 items-center justify-center">
            <SpotifyIcon size={24} />
          </span>
          <span className="max-w-0 whitespace-nowrap text-[14px] font-[600] text-white opacity-0 transition-all duration-300 ease-out group-hover:max-w-[230px] group-hover:pr-6 group-hover:opacity-100">
            Continue with Spotify
          </span>
        </a>
      </div>

      {/* Footer */}
      <p className="absolute bottom-7 left-1/2 z-10 -translate-x-1/2 whitespace-nowrap text-[11px] uppercase tracking-[0.18em] text-[rgba(27,30,35,0.36)]">
        We use Spotify to learn your taste · No posts, ever
      </p>
    </main>
  );
}

/* -------------------------------------------------------------------------- */
/* Screen 2 — Voice session (post-auth)                                        */
/* -------------------------------------------------------------------------- */
const PHASE_LABEL: Record<VoicePhase, string> = {
  idle: 'Tap to talk',
  listening: 'Listening…',
  thinking: 'Thinking…',
  speaking: 'Gia is speaking',
  error: 'Something went wrong',
};

function VoiceScreen({
  userId,
  sessionId,
  onSignOut,
}: {
  userId: string | null;
  sessionId: string;
  onSignOut: () => void;
}) {
  const session = useVoiceSession(userId, sessionId);
  const { phase, status, transcript, error, levelRef, start, stop, sendText, beginCapture, endCapture } =
    session;
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
  // The first tap also unlocks audio (browser autoplay policy), so Gia speaks
  // her opening greeting here — not on page load, where the browser mutes it.
  const beginSession = () => {
    // Greet only on the very first start AND only when no conversation has
    // happened yet — so typing first, then tapping the mic, never replays it.
    const greet =
      !greetedRef.current && opening && transcript.length === 0 ? opening : undefined;
    greetedRef.current = true;
    void start(greet);
  };
  const toggle = () => (active ? stop() : beginSession());

  // Ephemeral captions — show the current exchange while Gia thinks and speaks,
  // then let it fade away. The goal is for Gia to feel like a voice you're
  // talking to, not a chat log you're reading, so nothing persists on screen.
  const lastGia = useMemo(
    () => [...transcript].reverse().find((t) => t.role === 'gia')?.text ?? '',
    [transcript],
  );

  const [captionShown, setCaptionShown] = useState(false);
  useEffect(() => {
    // Visible while she's working; held briefly after, then faded out.
    if (phase === 'thinking' || phase === 'speaking') {
      setCaptionShown(true);
      return;
    }
    if (transcript.length === 0) {
      setCaptionShown(false);
      return;
    }
    const t = setTimeout(() => setCaptionShown(false), 2600);
    return () => clearTimeout(t);
  }, [phase, transcript.length]);

  return (
    <main className={`relative h-dvh w-full overflow-hidden ${BG}`}>
      <ProfileAvatar onSignOut={onSignOut} />

      {/* Audio-reactive ring — tap to start/stop the session */}
      <button onClick={toggle} aria-label="Toggle Gia voice" className="absolute inset-0 cursor-pointer">
        <RingCanvas levelRef={levelRef} active={active} />
      </button>

      {/* Caption lives inside the ring, centered and restrained — the voice is
          the product, so text stays small and never crowds the circle. Greeting
          and Gia's words cross-fade in the same spot. */}
      <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center px-8">
        {/* Warm opening line, before the first turn */}
        <p
          className={`absolute max-w-[300px] text-center text-[15px] font-[300] leading-[1.65] text-[#2b2f36] transition-opacity duration-700 ${
            opening && transcript.length === 0 ? 'opacity-100' : 'opacity-0'
          }`}
        >
          {opening}
        </p>

        {/* Gia's current words — fade in while she speaks, fade out after */}
        <p
          className={`max-w-[290px] text-center text-[15px] font-[300] leading-[1.65] text-[#23272d] [-webkit-box-orient:vertical] [-webkit-line-clamp:4] [display:-webkit-box] overflow-hidden transition-opacity duration-500 ease-out ${
            captionShown && lastGia ? 'opacity-100' : 'opacity-0'
          }`}
        >
          {lastGia}
        </p>
      </div>

      {/* Status + controls */}
      <div className="absolute bottom-[7%] left-1/2 z-10 flex w-[min(560px,88vw)] -translate-x-1/2 flex-col items-center gap-4">
        <div className="flex h-[15px] items-center gap-[11px]">
          {(phase === 'listening' || phase === 'thinking' || phase === 'speaking') && (
            <div className="flex h-[13px] items-end gap-[3px]">
              {[0, 0.15, 0.3, 0.45].map((delay) => (
                <span
                  key={delay}
                  className="w-[2.5px] rounded-[2px] bg-[#3a3f47]"
                  style={{ height: '100%', animation: `giaDot 0.7s ease-in-out ${delay}s infinite` }}
                />
              ))}
            </div>
          )}
          <span
            className="text-[13.5px] font-medium tracking-[0.06em] text-[#3a3f47]"
            style={active ? { animation: 'giaPulse 2.4s ease-in-out infinite' } : undefined}
          >
            {status || PHASE_LABEL[phase]}
          </span>
        </div>

        {error && <span className="text-[12px] text-[#b4452f]">{error}</span>}

        {/* Type or talk — both run the same pipeline */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            const text = draft.trim();
            if (!text) return;
            setDraft('');
            void sendText(text);
          }}
          className="flex w-full items-center gap-2"
        >
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={active ? 'Type to Gia…' : 'Type, or tap the mic to talk…'}
            className="h-11 flex-1 rounded-full border border-black/10 bg-white/70 px-5 text-[14px] text-[#2b2f36] shadow-[0_2px_10px_rgba(20,23,26,0.05)] outline-none backdrop-blur-sm transition-colors placeholder:text-[rgba(58,63,71,0.4)] focus:border-black/25 focus:bg-white/85"
          />
          {active ? (
            <button
              type="button"
              onPointerDown={beginCapture}
              onPointerUp={endCapture}
              onPointerLeave={endCapture}
              aria-label="Hold to talk"
              title="Hold to talk"
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[#15171a] text-[#f4f5f6] shadow-[0_6px_16px_rgba(20,23,26,0.28)] transition-transform active:scale-95"
            >
              <MicIcon />
            </button>
          ) : (
            <button
              type="button"
              onClick={beginSession}
              aria-label="Start talking"
              title="Start talking"
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[#15171a] text-[#f4f5f6] shadow-[0_6px_16px_rgba(20,23,26,0.28)] transition-transform hover:scale-105 active:scale-95"
            >
              <MicIcon />
            </button>
          )}
        </form>
      </div>
    </main>
  );
}

/* -------------------------------------------------------------------------- */
/* Root — routes on identity                                                   */
/* -------------------------------------------------------------------------- */
export default function GiaVoiceCore() {
  const { userId, signedIn, sessionId, hydrated, signOut } = useIdentity();

  // Avoid a hydration flash: render nothing meaningful until we've read storage.
  if (!hydrated) {
    return <main className={`h-dvh w-full ${BG}`} />;
  }

  if (!signedIn) {
    return <LandingScreen userId={userId} />;
  }
  return <VoiceScreen userId={userId} sessionId={sessionId} onSignOut={signOut} />;
}
