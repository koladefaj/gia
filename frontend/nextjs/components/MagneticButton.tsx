'use client';

/**
 * MagneticButton — the element drifts toward the cursor inside a small radius
 * and springs back on leave. Pure motion values (no React re-renders per frame).
 * Collapses to a static element under reduced motion.
 */

import { useRef } from 'react';
import { motion, useMotionValue, useReducedMotion, useSpring } from 'motion/react';

interface Props {
  children: React.ReactNode;
  className?: string;
  /** Max pixels the element drifts toward the cursor. */
  strength?: number;
}

export default function MagneticButton({ children, className = '', strength = 14 }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const reduce = useReducedMotion();
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const sx = useSpring(x, { stiffness: 180, damping: 15, mass: 0.4 });
  const sy = useSpring(y, { stiffness: 180, damping: 15, mass: 0.4 });

  const onMove = (e: React.PointerEvent) => {
    if (reduce || !ref.current) return;
    const r = ref.current.getBoundingClientRect();
    const dx = e.clientX - (r.left + r.width / 2);
    const dy = e.clientY - (r.top + r.height / 2);
    x.set(Math.max(-strength, Math.min(strength, dx * 0.35)));
    y.set(Math.max(-strength, Math.min(strength, dy * 0.35)));
  };

  const reset = () => {
    x.set(0);
    y.set(0);
  };

  return (
    <motion.div
      ref={ref}
      onPointerMove={onMove}
      onPointerLeave={reset}
      style={reduce ? undefined : { x: sx, y: sy }}
      className={`inline-flex ${className}`}
    >
      {children}
    </motion.div>
  );
}
