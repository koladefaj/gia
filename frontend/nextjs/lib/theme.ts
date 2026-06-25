'use client';

/**
 * Theme state — light / dark with one rule: the whole surface is one theme.
 *
 * The initial value is set before React hydrates by an inline script in
 * layout.tsx (so there's no flash), reading the same localStorage key and
 * `prefers-color-scheme` fallback used here. This hook keeps React in sync and
 * lets the user flip it; the choice persists across reloads.
 */

import { useCallback, useEffect, useState } from 'react';

export type Theme = 'light' | 'dark';

const KEY = 'gia_theme';

/** Resolve the active theme the way the pre-hydration script does. */
export function resolveTheme(): Theme {
  if (typeof window === 'undefined') return 'light';
  const stored = localStorage.getItem(KEY);
  if (stored === 'light' || stored === 'dark') return stored;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

/** Read a CSS custom property off <html> (kept as the single source of truth). */
export function readToken(name: string): string {
  if (typeof window === 'undefined') return '';
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>('light');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setThemeState(resolveTheme());
    setMounted(true);
    // Keep every hook instance (toggle, ring, HUD) in sync when any one flips,
    // and across tabs. Not a scroll listener — cheap, event-driven.
    const sync = () => setThemeState(resolveTheme());
    window.addEventListener('gia-theme', sync);
    window.addEventListener('storage', sync);
    return () => {
      window.removeEventListener('gia-theme', sync);
      window.removeEventListener('storage', sync);
    };
  }, []);

  const apply = useCallback((next: Theme) => {
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem(KEY, next);
    setThemeState(next);
    window.dispatchEvent(new Event('gia-theme'));
  }, []);

  const toggle = useCallback(() => {
    apply(theme === 'dark' ? 'light' : 'dark');
  }, [theme, apply]);

  return { theme, setTheme: apply, toggle, mounted };
}
