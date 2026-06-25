'use client';

/**
 * Reveal — enter-on-scroll for a section or item. Motion's `whileInView` (no
 * scroll listener); collapses to static under reduced motion.
 */

import { motion, useReducedMotion } from 'motion/react';

interface Props {
  children: React.ReactNode;
  className?: string;
  /** Stagger index — shifts the delay for sequential items. */
  i?: number;
  y?: number;
}

export default function Reveal({ children, className = '', i = 0, y = 24 }: Props) {
  const reduce = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial={reduce ? false : { opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.3 }}
      transition={{ duration: 0.7, delay: i * 0.08, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  );
}
