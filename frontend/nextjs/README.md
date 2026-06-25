# Gia — Voice AI Core

Ultra-premium, light-mode conversational voice AI landing core: glassmorphic
floating nav, a glowing audio-reactive 3D ring, and an Idle ⇄ Streaming toggle.

## Install

```bash
npm i three @react-three/fiber @react-three/drei @react-three/postprocessing lucide-react
```

> React 19 / Next.js App Router compatible. `@react-three/fiber` v9+ and
> `@react-three/postprocessing` v3+ support React 19.

## Use

```tsx
// app/page.tsx
import GiaVoiceCore from '@/components/GiaVoiceCore';
export default function Page() {
  return <GiaVoiceCore />;
}
```

`components/GiaVoiceCore.tsx` is a single self-contained `'use client'` component.
Click anywhere on the canvas (or wire up the mic pill) to flip `streaming`.

## How it works

- **Ring** — a very thin `TorusGeometry` viewed head-on reads as a brilliant
  filament. Per-vertex base position / radial direction / angle are cached once.
- **Liquid ripple** — in `useFrame`, each vertex is displaced radially by a sum
  of ring-periodic harmonics, each harmonic tied to a different mock frequency
  band, so the edge ripples organically instead of uniformly scaling.
- **Bloom** — `@react-three/postprocessing` `<Bloom mipmapBlur>` with a high
  `luminanceThreshold` so only the HDR-white ring (`color={[1.5,1.5,1.5]}`,
  `toneMapped={false}`) glows; the radial-gradient backdrop stays calm.
- **Reactivity** — ripple amplitude and bloom intensity lerp toward an `energy`
  value each frame; Idle falls back to a serene breathing pulse.

## Swap in real audio

Replace `useMockAudio` with a Web Audio `AnalyserNode`:

```ts
const data = new Uint8Array(analyser.frequencyBinCount);
analyser.getByteFrequencyData(data);
// bucket `data` into 3 bands, normalise to 0..1, feed the same `bands` array.
```

Everything downstream (ripple + bloom) already consumes the band array, so no
other changes are needed.

## Note on transparency

This build uses a transparent `<Canvas alpha>` over a Tailwind radial gradient.
If you see dark bloom fringing on some GPUs, render an in-scene gradient quad
instead (see the raw-three reference build) and keep the canvas opaque.
