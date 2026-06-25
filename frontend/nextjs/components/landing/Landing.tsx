'use client';

/**
 * Landing — the pre-auth marketing surface. The living ring anchors the hero;
 * below it, real sections explain a voice-first music companion and lead to the
 * single action: Continue with Spotify.
 *
 * Premium-consumer overhaul · VARIANCE 8 · MOTION 7 · DENSITY 3.
 */

import { useRef } from 'react';

import { spotifyLoginUrl } from '@/lib/api';
import { useTheme } from '@/lib/theme';
import VoiceWaves from '@/components/VoiceWaves';
import TypingText from '@/components/TypingText';
import MagneticButton from '@/components/MagneticButton';
import ThemeToggle from '@/components/ThemeToggle';
import Reveal from './Reveal';
import { Clock, Heart, Microphone, Sparkle, SpotifyMark, Waveform } from '@/components/Icons';

/* -------------------------------------------------------------------------- */
/* Shared CTA — one intent, one label, used in nav, hero, and closing.         */
/* -------------------------------------------------------------------------- */
function SpotifyCTA({ userId, size = 'md' }: { userId: string | null; size?: 'sm' | 'md' }) {
  const pad = size === 'sm' ? 'h-10 px-4 text-[13px]' : 'h-12 px-6 text-[15px]';
  return (
    <a
      href={spotifyLoginUrl(userId)}
      className={`group inline-flex ${pad} items-center gap-2.5 rounded-full bg-ink font-[600] text-field shadow-[0_10px_30px_rgba(0,0,0,0.16)] transition-all duration-300 ease-out hover:-translate-y-[2px] hover:shadow-[0_16px_38px_rgba(0,0,0,0.22)] active:translate-y-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-field`}
    >
      <SpotifyMark size={size === 'sm' ? 18 : 20} />
      Continue with Spotify
    </a>
  );
}

/* -------------------------------------------------------------------------- */
/* Nav                                                                          */
/* -------------------------------------------------------------------------- */
function Nav({ userId }: { userId: string | null }) {
  return (
    <header className="fixed inset-x-0 top-0 z-40">
      <div className="mx-auto flex h-16 max-w-[1200px] items-center justify-between px-6">
        <span className="font-display text-[19px] font-[600] tracking-[0.02em] text-ink">Gia</span>
        <div className="flex items-center gap-3">
          <ThemeToggle />
          <div className="hidden sm:block">
            <SpotifyCTA userId={userId} size="sm" />
          </div>
        </div>
      </div>
    </header>
  );
}

/* -------------------------------------------------------------------------- */
/* Hero — the living ring, a headline, the typed line, the one action.         */
/* -------------------------------------------------------------------------- */
const TAGLINES = [
  'Say what you feel, hear what fits.',
  'It learns what you love.',
  'Reads the moment, plays the song.',
  'Hands-free. Always listening.',
];

function Hero({ theme }: { theme: 'light' | 'dark' }) {
  const idle = useRef(0);
  return (
    <section className="relative flex min-h-[100dvh] flex-col items-center justify-center overflow-hidden px-6 text-center">
      {/* Ambient depth — a soft ember glow, no ring. */}
      <div
        aria-hidden
        className="pointer-events-none absolute left-1/2 top-1/2 h-[440px] w-[640px] max-w-[92vw] -translate-x-1/2 -translate-y-1/2 rounded-full blur-[100px]"
        style={{ background: 'radial-gradient(circle, var(--hero-glow), transparent 70%)' }}
      />

      {/* The brand motif: a quiet voice band, sitting low beneath the words. */}
      <div aria-hidden className="pointer-events-none absolute inset-x-0 top-[62%] h-[200px] -translate-y-1/2 opacity-60">
        <VoiceWaves levelRef={idle} active={false} theme={theme} variant="ambient" />
      </div>

      <div className="relative z-10 flex flex-col items-center -translate-y-[7vh]">
        <Reveal>
          <h1 className="font-display text-[44px] font-[500] leading-[1.04] tracking-[-0.02em] text-ink sm:text-6xl md:text-7xl">
            Talk to your music.
          </h1>
        </Reveal>
        <Reveal i={1}>
          <p className="mt-5 h-[26px] text-[15px] tracking-[0.01em] text-ink-soft md:text-lg">
            <TypingText phrases={TAGLINES} />
          </p>
        </Reveal>
      </div>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* How it works — staggered moments, not equal cards.                          */
/* -------------------------------------------------------------------------- */
const STEPS = [
  {
    icon: Microphone,
    title: 'Tap and talk',
    body: 'Touch the ring and just speak. No wake word, no menus, no typing.',
    offset: 'md:mt-0',
  },
  {
    icon: Sparkle,
    title: 'It reads the moment',
    body: 'Gia hears the mood behind the words, not only the words themselves.',
    offset: 'md:mt-16',
  },
  {
    icon: Waveform,
    title: 'The song starts',
    body: 'The right track plays on your own Spotify. Skip or steer it by voice.',
    offset: 'md:mt-8',
  },
];

function HowItWorks() {
  return (
    <section className="mx-auto max-w-[1200px] px-6 py-28 md:py-40">
      <Reveal>
        <h2 className="max-w-prose font-display text-3xl font-[500] tracking-[-0.01em] text-ink md:text-5xl">
          A conversation, not a search box.
        </h2>
      </Reveal>
      <div className="mt-16 grid grid-cols-1 gap-10 md:grid-cols-3 md:gap-8">
        {STEPS.map((s, i) => {
          const Icon = s.icon;
          return (
            <Reveal key={s.title} i={i} className={s.offset}>
              <div className="flex flex-col items-start">
                <span className="flex h-12 w-12 items-center justify-center rounded-full border border-hairline/10 text-accent">
                  <Icon size={22} weight="duotone" />
                </span>
                <h3 className="mt-5 font-display text-xl font-[600] text-ink">{s.title}</h3>
                <p className="mt-2 max-w-[34ch] text-[15px] leading-relaxed text-ink-soft">{s.body}</p>
              </div>
            </Reveal>
          );
        })}
      </div>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* What you can say — overheard requests, scattered.                           */
/* -------------------------------------------------------------------------- */
const UTTERANCES: { text: string; size: string; tone: string }[] = [
  { text: 'Play something for a slow Sunday morning.', size: 'text-lg md:text-2xl', tone: 'text-ink' },
  { text: 'Something like Khruangbin, but darker.', size: 'text-base md:text-xl', tone: 'text-ink-soft' },
  { text: 'I need to lock in for two hours.', size: 'text-lg md:text-3xl', tone: 'text-ink' },
  { text: 'Skip this, too sleepy.', size: 'text-base md:text-lg', tone: 'text-ink-soft' },
  { text: 'Who is this? Add it to my liked songs.', size: 'text-base md:text-xl', tone: 'text-ink-soft' },
  { text: 'Wind me down for the night.', size: 'text-lg md:text-2xl', tone: 'text-ink' },
];

function WhatYouCanSay() {
  return (
    <section className="border-y border-hairline/[0.07] bg-[rgb(var(--ink)/0.015)]">
      <div className="mx-auto max-w-[1100px] px-6 py-28 md:py-40">
        <Reveal>
          <p className="text-[12px] font-[600] uppercase tracking-[0.2em] text-accent">In your words</p>
          <h2 className="mt-3 max-w-prose font-display text-3xl font-[500] tracking-[-0.01em] text-ink md:text-5xl">
            However you'd say it to a friend.
          </h2>
        </Reveal>
        <div className="mt-14 columns-1 gap-x-10 sm:columns-2 [&>*]:mb-8 [&>*]:break-inside-avoid">
          {UTTERANCES.map((u, i) => (
            <Reveal key={u.text} i={i % 3}>
              <p className={`font-display font-[400] leading-snug ${u.size} ${u.tone}`}>
                <span className="mr-1 text-accent">“</span>
                {u.text}
                <span className="ml-0.5 text-accent">”</span>
              </p>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* Why it's different — bento with real visual variation.                      */
/* -------------------------------------------------------------------------- */
function Bento() {
  return (
    <section className="mx-auto max-w-[1200px] px-6 py-28 md:py-40">
      <Reveal>
        <h2 className="max-w-prose font-display text-3xl font-[500] tracking-[-0.01em] text-ink md:text-5xl">
          It actually knows you.
        </h2>
      </Reveal>

      <div className="mt-14 grid grid-cols-1 gap-4 md:auto-rows-[210px] md:grid-cols-3">
        {/* Big feature — taste memory */}
        <Reveal className="md:col-span-2 md:row-span-2" y={32}>
          <div className="flex h-full flex-col justify-between rounded-panel border border-hairline/[0.08] bg-[var(--glass-bg)] p-8 backdrop-blur-md">
            <span className="flex h-11 w-11 items-center justify-center rounded-full bg-accent/10 text-accent">
              <Heart size={22} weight="duotone" />
            </span>
            <div>
              <h3 className="font-display text-2xl font-[600] text-ink md:text-3xl">It remembers your taste</h3>
              <p className="mt-3 max-w-prose text-[15px] leading-relaxed text-ink-soft">
                Gia keeps a long memory of what you reach for, what you skip, and how your week sounds. The more you
                talk, the closer it lands.
              </p>
            </div>
          </div>
        </Reveal>

        {/* Image — atmospheric, speaks alone */}
        <Reveal className="md:row-span-1" y={32}>
          <img
            src="https://picsum.photos/seed/gia-late-night-room/700/500?grayscale"
            alt="A dim room lit by a single warm lamp at night"
            loading="lazy"
            className="h-56 w-full rounded-panel object-cover opacity-90 md:h-full"
          />
        </Reveal>

        {/* Reads the moment */}
        <Reveal className="md:row-span-1" y={32}>
          <div className="flex h-full flex-col justify-between rounded-panel border border-hairline/[0.08] bg-[var(--glass-bg)] p-7 backdrop-blur-md">
            <span className="flex h-11 w-11 items-center justify-center rounded-full bg-accent/10 text-accent">
              <Clock size={20} weight="duotone" />
            </span>
            <div>
              <h3 className="font-display text-lg font-[600] text-ink">Reads the moment</h3>
              <p className="mt-1.5 text-[14px] leading-relaxed text-ink-soft">Morning focus and midnight wind-down are not the same request.</p>
            </div>
          </div>
        </Reveal>

        {/* Spotify-native */}
        <Reveal className="md:row-span-1" y={32}>
          <div className="flex h-full flex-col justify-between rounded-panel border border-hairline/[0.08] bg-[var(--glass-bg)] p-7 backdrop-blur-md">
            <SpotifyMark size={26} />
            <div>
              <h3 className="font-display text-lg font-[600] text-ink">Your Spotify, your library</h3>
              <p className="mt-1.5 text-[14px] leading-relaxed text-ink-soft">It plays on your account, with your saved songs and playlists.</p>
            </div>
          </div>
        </Reveal>

        {/* Wide image — atmospheric */}
        <Reveal className="md:col-span-2 md:row-span-1" y={32}>
          <img
            src="https://picsum.photos/seed/gia-headphones-city/1000/500?grayscale"
            alt="City lights blurred through a window at night"
            loading="lazy"
            className="h-56 w-full rounded-panel object-cover opacity-90 md:h-full"
          />
        </Reveal>
      </div>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* Closing                                                                      */
/* -------------------------------------------------------------------------- */
function Closing({ userId }: { userId: string | null }) {
  return (
    <section className="mx-auto max-w-[900px] px-6 py-32 text-center md:py-44">
      <Reveal>
        <h2 className="mx-auto max-w-[16ch] font-display text-4xl font-[500] leading-[1.06] tracking-[-0.02em] text-ink md:text-6xl">
          Your next song is one sentence away.
        </h2>
      </Reveal>
      <Reveal i={1}>
        <div className="mt-10 flex justify-center">
          <MagneticButton>
            <SpotifyCTA userId={userId} />
          </MagneticButton>
        </div>
      </Reveal>
      <Reveal i={2}>
        <p className="mt-6 text-[13px] text-ink-faint">We use Spotify to learn your taste. No posts, ever.</p>
      </Reveal>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* Footer                                                                        */
/* -------------------------------------------------------------------------- */
function Footer() {
  return (
    <footer className="border-t border-hairline/[0.08]">
      <div className="mx-auto flex max-w-[1200px] flex-col items-center justify-between gap-3 px-6 py-8 text-[13px] text-ink-faint sm:flex-row">
        <span className="font-display text-[15px] font-[600] text-ink-soft">Gia</span>
        <span>A voice companion for your music.</span>
      </div>
    </footer>
  );
}

/* -------------------------------------------------------------------------- */
/* Page                                                                         */
/* -------------------------------------------------------------------------- */
export default function Landing({ userId }: { userId: string | null }) {
  const { theme } = useTheme();
  return (
    <main className="relative w-full overflow-x-hidden">
      <div className="gia-grain" aria-hidden />
      <Nav userId={userId} />
      <Hero theme={theme} />
      <HowItWorks />
      <WhatYouCanSay />
      <Bento />
      <Closing userId={userId} />
      <Footer />
    </main>
  );
}
