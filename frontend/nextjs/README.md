# Gia — Voice Music Companion (frontend)

A voice-first music companion. Pre-auth is an editorial landing that leads to
Spotify OAuth; post-auth is a hands-free voice session with a live, layered
sine-wave visualizer. Dual light (warm ivory) / dark (near-black) theme.

## Stack

- **Next.js 14** App Router, React 18, TypeScript
- **Tailwind v3** with CSS-variable design tokens (`app/globals.css`), flipped by
  `data-theme` on `<html>`
- **Motion** (`motion/react`) for scroll reveals, the magnetic CTA, and the theme
  toggle
- **@phosphor-icons/react** for UI glyphs
- **Canvas 2D** for the voice visualizer (no WebGL / Three.js)
- Fonts via `next/font`: Bricolage Grotesque (display) + Manrope (body)

## Run

```bash
npm install
npm run dev   # http://localhost:3000
```

Point the client at the backend with `NEXT_PUBLIC_API_URL` (see `.env.example`).

## Architecture

```
app/                layout (fonts + pre-hydration theme init), globals (tokens)
components/
  GiaVoiceCore.tsx  routes on identity → Landing or VoiceScreen
  landing/          editorial landing: hero, how-it-works, utterances, bento, cta
  VoiceScreen.tsx   post-auth hands-free session + glass HUD
  VoiceWaves.tsx    layered sine-wave visualizer (amplitude rides levelRef)
  ThemeToggle, MagneticButton, Icons
lib/
  useVoiceSession   the voice engine — mic VAD, STT, /chat stream, TTS playback
  audio             Web Audio players + shared analyser feeding the visualizer
  api               typed client (/chat SSE, /voice, /chat/opening, Spotify auth)
  identity          user id (Spotify OAuth) + per-load session id
  theme             light/dark state, persistence, cross-instance sync
```

The visualizer reads a live audio level (`levelRef`, 0..1) every frame — mic
energy while listening, Gia's TTS output while speaking — and reads its colour
and glow from the theme tokens (`--ring-color`, `--accent`, `--ring-bloom`).

## Theme

One `data-theme` attribute on `<html>` drives every surface through CSS
variables. The value is set before first paint by an inline script in
`layout.tsx` (no flash), persisted to `localStorage`, and respects
`prefers-color-scheme`. All animation honors `prefers-reduced-motion`.
