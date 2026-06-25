'use client';

/**
 * VoiceWaves — the voice made visible. Several overlapping sine curves drawn on
 * a 2D canvas; their amplitude rides the live audio level (`levelRef`, 0..1).
 * Edge-tapered so the band reads as a contained voice, not screen-wide lines.
 *
 * Colour and glow come from theme tokens (`--ring-color`, `--accent`,
 * `--ring-bloom`). Idle is a slow breath; active swells with speech. Reduced
 * motion collapses to a single still wave.
 *
 * Light, dependency-free (no Three.js). Reads `levelRef` directly — never sets
 * React state in the animation loop.
 */

import { useEffect, useRef } from 'react';

import { readToken, type Theme } from '@/lib/theme';

interface Layer {
  freq: number; // cycles across the width
  speed: number; // horizontal drift
  scale: number; // amplitude multiplier
  width: number; // line width (css px)
  use: 'ink' | 'accent';
  alpha: number;
}

const LAYERS: Layer[] = [
  { freq: 1.6, speed: 0.55, scale: 1.0, width: 2.2, use: 'ink', alpha: 0.92 },
  { freq: 2.3, speed: -0.4, scale: 0.66, width: 1.6, use: 'accent', alpha: 0.8 },
  { freq: 3.1, speed: 0.8, scale: 0.42, width: 1.2, use: 'ink', alpha: 0.45 },
];

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/** "224 92 56" -> "rgb(224,92,56)" with optional alpha. */
function rgb(triplet: string, alpha = 1): string {
  const p = triplet.trim().split(/\s+/).map(Number);
  if (p.length < 3 || p.some(Number.isNaN)) return `rgba(120,120,120,${alpha})`;
  return alpha >= 1 ? `rgb(${p[0]},${p[1]},${p[2]})` : `rgba(${p[0]},${p[1]},${p[2]},${alpha})`;
}

export default function VoiceWaves({
  levelRef,
  active,
  theme,
  variant = 'full',
}: {
  levelRef: React.MutableRefObject<number>;
  active: boolean;
  theme: Theme;
  /** `full` = the voice screen; `ambient` = a quiet brand band on the landing. */
  variant?: 'full' | 'ambient';
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const activeRef = useRef(active);
  activeRef.current = active;

  // Palette is re-read whenever the theme flips (CSS stays the source of truth).
  const palette = useRef({ ink: '#25232a', accent: '224 92 56', glow: 0.16 });
  useEffect(() => {
    palette.current = {
      ink: readToken('--ring-color') || '#25232a',
      accent: readToken('--accent') || '224 92 56',
      glow: parseFloat(readToken('--ring-bloom')) || 0.16,
    };
  }, [theme]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const ambient = variant === 'ambient';
    let dpr = Math.min(window.devicePixelRatio || 1, 2);
    let cssW = 0;
    let cssH = 0;
    // energy = live audio level (smoothed); act = idle↔active envelope so the
    // wave eases between its contained idle size and its expanded active size.
    const env = { energy: 0, act: 0 };

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      cssW = rect.width;
      cssH = rect.height;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.floor(cssW * dpr));
      canvas.height = Math.max(1, Math.floor(cssH * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const draw = (timeMs: number) => {
      const t = timeMs / 1000;
      if (cssW < 2 || cssH < 2) return;

      // Smoothly track the live level and the active envelope.
      const target = activeRef.current ? levelRef.current : 0;
      env.energy = lerp(env.energy, target, 0.12);
      env.act = lerp(env.act, activeRef.current ? 1 : 0, 0.07);

      const midY = cssH / 2;
      const breath = (Math.sin(t * 1.1) * 0.5 + 0.5) * 0.18;

      // Contained idle (the same restrained band as the hero) that has room to
      // grow: when listening / speaking it expands and rides the live voice.
      const idleAmp = Math.min(cssH * 0.16, 46) * (0.28 + breath);
      const activeAmp = Math.min(cssH * 0.3, 150) * (0.5 + env.energy * 0.5);
      // The ambient (hero) variant has no active state, so it stays contained.
      const amplitude = ambient ? idleAmp : lerp(idleAmp, activeAmp, env.act);

      const { ink, accent, glow } = palette.current;
      const globalAlpha = ambient ? 0.5 : 1;

      ctx.clearRect(0, 0, cssW, cssH);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      const step = 4;
      for (const layer of LAYERS) {
        const amp = amplitude * layer.scale;
        ctx.beginPath();
        for (let x = 0; x <= cssW; x += step) {
          const u = x / cssW; // 0..1
          // Raised-cosine window: taper to zero at both edges.
          const env = Math.pow(Math.sin(Math.PI * u), 1.5);
          const phase = u * Math.PI * 2 * layer.freq + t * layer.speed * 2;
          const y = midY + Math.sin(phase) * amp * env;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        const color = layer.use === 'accent' ? rgb(accent, 1) : ink;
        ctx.strokeStyle = color;
        ctx.lineWidth = layer.width;
        ctx.globalAlpha = layer.alpha * globalAlpha;
        // Soft glow — stronger on the dark field where it reads as light.
        ctx.shadowColor = layer.use === 'accent' ? rgb(accent, 1) : color;
        ctx.shadowBlur = glow * (ambient ? 8 : 22);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      ctx.shadowBlur = 0;
    };

    let raf = 0;
    if (reduce) {
      // One still frame at a calm amplitude.
      draw(700);
    } else {
      const loop = (ts: number) => {
        draw(ts);
        raf = requestAnimationFrame(loop);
      };
      raf = requestAnimationFrame(loop);
    }

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
    // theme is intentionally a dep so the palette ref is refreshed before drawing.
  }, [levelRef, variant, theme]);

  return <canvas ref={canvasRef} className="h-full w-full" aria-hidden />;
}
