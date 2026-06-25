'use client';

/**
 * GiaVoiceCore — routes on identity.
 *
 * Pre-auth  : a living landing that explains a voice-first music companion and
 *             leads to real "Continue with Spotify" OAuth.
 * Post-auth : a hands-free voice session — tap the ring to talk.
 */

import { useIdentity } from '@/lib/identity';
import Landing from './landing/Landing';
import VoiceScreen from './VoiceScreen';

export default function GiaVoiceCore() {
  const { userId, signedIn, sessionId, hydrated, signOut } = useIdentity();

  // Avoid a hydration flash: render the bare field until storage is read.
  if (!hydrated) {
    return <main className="min-h-[100dvh] w-full" />;
  }

  if (!signedIn) {
    return <Landing userId={userId} />;
  }
  return <VoiceScreen userId={userId} sessionId={sessionId} onSignOut={signOut} />;
}
