'use client';

/**
 * Client-side identity.
 *
 * After Spotify OAuth the backend redirects to `/?user_id=…&connected=1`. We
 * adopt that id, persist it in localStorage (so it survives reloads and powers
 * long-term memory), and scrub it from the URL. A per-load `session_id` threads
 * multi-turn continuity for one conversation.
 */

import { useCallback, useEffect, useState } from 'react';

const USER_KEY = 'gia_user_id';

function newId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export interface Identity {
  /** Resolved Gia user id, or null before sign-in. */
  userId: string | null;
  /** True once a user id is known (returning or freshly connected). */
  signedIn: boolean;
  /** Per-page-load conversation id. */
  sessionId: string;
  /** True once the client has read localStorage/URL (avoids hydration flash). */
  hydrated: boolean;
  signOut: () => void;
}

export function useIdentity(): Identity {
  const [userId, setUserId] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);
  const [sessionId] = useState(newId);

  useEffect(() => {
    const url = new URL(window.location.href);
    const fromUrl = url.searchParams.get('user_id');

    if (fromUrl) {
      localStorage.setItem(USER_KEY, fromUrl);
      setUserId(fromUrl);
      // Scrub OAuth params so a reload/share doesn't re-trigger anything.
      url.searchParams.delete('user_id');
      url.searchParams.delete('connected');
      window.history.replaceState({}, '', url.pathname + url.search + url.hash);
    } else {
      setUserId(localStorage.getItem(USER_KEY));
    }
    setHydrated(true);
  }, []);

  const signOut = useCallback(() => {
    localStorage.removeItem(USER_KEY);
    setUserId(null);
  }, []);

  return { userId, signedIn: !!userId, sessionId, hydrated, signOut };
}
