/**
 * Typed client for the Gia backend.
 *
 * The base URL is configurable via NEXT_PUBLIC_API_URL so the same build runs
 * against localhost in dev and a real origin in prod. CORS is open on the API
 * (`allow_origins=["*"]`), so the browser can call it directly.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '') ?? 'http://localhost:8000';

/* -------------------------------------------------------------------------- */
/* /chat — Server-Sent Events                                                  */
/* -------------------------------------------------------------------------- */

export type ChatEventName =
  | 'agent_start'
  | 'agent_done'
  | 'tool_call'
  | 'plan'
  | 'signal'
  | 'acknowledgment'
  | 'reply_chunk'
  | 'audio_chunk'
  | 'error'
  | 'done';

export interface ChatEvent {
  event: ChatEventName | string;
  // The payload shape varies by event; callers narrow on `event`.
  data: Record<string, unknown>;
}

export interface ChatBody {
  message: string;
  user_id?: string | null;
  session_id?: string | null;
}

/**
 * POST /chat and yield each SSE frame as it arrives.
 *
 * Frames are `event: <name>\ndata: <json>\n\n`. We buffer partial reads and
 * split on the blank-line delimiter, mirroring the proven vanilla client.
 */
export async function* chatStream(
  body: ChatBody,
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const resp = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!resp.body) throw new Error('No response body from /chat');

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    const parts = buf.split('\n\n');
    buf = parts.pop() ?? '';

    for (const part of parts) {
      let event = '';
      let data = '';
      for (const line of part.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data += line.slice(5).trim();
      }
      if (!event || !data) continue;
      try {
        yield { event, data: JSON.parse(data) };
      } catch {
        /* skip malformed frame */
      }
    }
  }
}

/* -------------------------------------------------------------------------- */
/* /voice — STT + TTS                                                          */
/* -------------------------------------------------------------------------- */

/** Transcribe a recorded audio blob to text. Returns "" on any failure. */
export async function transcribe(blob: Blob, language = 'en'): Promise<string> {
  const fd = new FormData();
  fd.append('audio', blob, 'turn.webm');
  fd.append('language', language);
  try {
    const resp = await fetch(`${API_BASE}/voice/transcribe`, {
      method: 'POST',
      body: fd,
    });
    if (!resp.ok) return '';
    const data = (await resp.json()) as { transcript?: string };
    return (data.transcript ?? '').trim();
  } catch {
    return '';
  }
}

/** Synthesise *text* to speech. Returns audio bytes, or null when unavailable. */
export async function speak(text: string): Promise<ArrayBuffer | null> {
  try {
    const resp = await fetch(`${API_BASE}/voice/speak`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!resp.ok) return null;
    const buf = await resp.arrayBuffer();
    return buf.byteLength ? buf : null;
  } catch {
    return null;
  }
}

/* -------------------------------------------------------------------------- */
/* /chat/opening — Gia speaks first                                            */
/* -------------------------------------------------------------------------- */

/** Fetch Gia's warm opening line. Best-effort — returns "" on failure. */
export async function getOpening(userId?: string | null): Promise<string> {
  const url = userId
    ? `${API_BASE}/chat/opening?user_id=${encodeURIComponent(userId)}`
    : `${API_BASE}/chat/opening`;
  try {
    const resp = await fetch(url);
    if (!resp.ok) return '';
    const data = (await resp.json()) as { greeting?: string };
    return (data.greeting ?? '').trim();
  } catch {
    return '';
  }
}

/* -------------------------------------------------------------------------- */
/* /auth/spotify — OAuth entry point                                           */
/* -------------------------------------------------------------------------- */

/**
 * Full-page navigation target for Spotify sign-in. Pass an existing user id to
 * *link* an account; omit it for a fresh sign-in (the backend creates the user
 * and redirects back with `?user_id=…&connected=1`).
 */
export function spotifyLoginUrl(userId?: string | null): string {
  return userId
    ? `${API_BASE}/auth/spotify/login?user_id=${encodeURIComponent(userId)}`
    : `${API_BASE}/auth/spotify/login`;
}
