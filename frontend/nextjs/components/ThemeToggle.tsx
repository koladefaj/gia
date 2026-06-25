'use client';

import { AnimatePresence, motion } from 'motion/react';

import { useTheme } from '@/lib/theme';
import { MoonStars, Sun } from './Icons';

export default function ThemeToggle({ className = '' }: { className?: string }) {
  const { theme, toggle, mounted } = useTheme();

  return (
    <button
      onClick={toggle}
      aria-label={theme === 'dark' ? 'Switch to light' : 'Switch to dark'}
      className={`relative flex h-9 w-9 items-center justify-center rounded-full border border-hairline/10 bg-[var(--glass-bg)] text-ink-soft backdrop-blur-md transition-colors hover:text-ink ${className}`}
    >
      {/* Until mounted, render nothing inside to avoid a hydration mismatch. */}
      <AnimatePresence mode="wait" initial={false}>
        {mounted && (
          <motion.span
            key={theme}
            initial={{ opacity: 0, rotate: -40, scale: 0.6 }}
            animate={{ opacity: 1, rotate: 0, scale: 1 }}
            exit={{ opacity: 0, rotate: 40, scale: 0.6 }}
            transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
            className="flex"
          >
            {theme === 'dark' ? <Sun size={17} weight="bold" /> : <MoonStars size={17} weight="bold" />}
          </motion.span>
        )}
      </AnimatePresence>
    </button>
  );
}
