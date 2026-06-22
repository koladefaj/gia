'use client';

/**
 * useVoiceSession — hands-free conversational loop.
 *
 * Tap to start a live session. The hook captures the mic, uses voice-activity
 * detection (energy + a silence timer) to auto-segment each turn, transcribes
 * it, streams it through /chat, plays Gia's TTS, then resumes listening. Tap
 * to end. A `levelRef` exposes live audio energy so the visual ring reacts to
 * the real conversation (mic while listening, Gia's voice while speaking).
 *
 * Fallbacks for noisy rooms / no-mic: `beginCapture`/`endCapture` (push-to-talk)
 * and `sendText` (typed turns) reuse the same chat pipeline.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { chatStream, speak as speakApi, transcribe } from './api';
import { AudioPlayer, StreamPlayer, createAnalyser, stripTags } from './audio';

export type VoicePhase = 'idle' | 'listening' | 'thinking' | 'speaking' | 'error';

export interface Turn {
  role: 'user' | 'gia';
  text: string;
}

export interface VoiceSession {
  phase: VoicePhase;
  /** Short status line, e.g. "Listening…", "dj → search_tracks". */
  status: string;
  transcript: Turn[];
  error: string | null;
  /** Live audio energy 0..1 for the ring (read in useFrame). */
  levelRef: React.MutableRefObject<number>;
  start: (greeting?: string) => Promise<void>;
  stop: () => void;
  /** Push-to-talk: begin/end a manual capture within a running session. */
  beginCapture: () => void;
  endCapture: () => void;
  /** Send a typed turn through the same pipeline. */
  sendText: (text: string) => Promise<void>;
  /** Speak a line (e.g. Gia's opening greeting) through the TTS player. */
  speak: (text: string) => Promise<void>;
}

// VAD tuning (time-domain RMS on a 0..~0.5 scale for normal speech).
const SPEECH_RMS = 0.025; // above this = speaking
const SILENCE_MS = 1000; // trailing silence that ends a turn
const MIN_SPEECH_MS = 350; // ignore blips shorter than this

// Tap-to-talk: tapping the ring IS the intent signal, so there's no wake word.
// After a tap (and after each Gia reply) the session is "engaged" — fully
// hands-free, every utterance is sent. If the user stays silent for GRACE_MS,
// the session closes itself and the ring goes calm; a tap reopens it. This is
// the continuous-conversation model people expect from modern voice AI
// (ChatGPT/Gemini Live), not a say-the-name-every-time gate.
const GRACE_MS = 10000; // silent window after talking before the session closes

function pickMime(): string {
  if (typeof MediaRecorder === 'undefined') return '';
  for (const m of ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg']) {
    if (MediaRecorder.isTypeSupported(m)) return m;
  }
  return '';
}

export function useVoiceSession(
  userId: string | null,
  sessionId: string,
): VoiceSession {
  const [phase, setPhase] = useState<VoicePhase>('idle');
  const [status, setStatus] = useState('');
  const [transcript, setTranscript] = useState<Turn[]>([]);
  const [error, setError] = useState<string | null>(null);

  const levelRef = useRef(0);
  const phaseRef = useRef<VoicePhase>('idle');
  const activeRef = useRef(false);

  const ctxRef = useRef<AudioContext | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  const streamPlayerRef = useRef<StreamPlayer | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const timeData = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const rafRef = useRef<number | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const capturingRef = useRef(false);
  const manualRef = useRef(false); // capture driven by push-to-talk
  const speechStartRef = useRef(0);
  const lastVoiceRef = useRef(0);
  const streamingRef = useRef(false); // a /chat stream is in flight
  const engagedUntilRef = useRef(0); // silence deadline; past it the session closes
  const stopRef = useRef<(() => void) | null>(null); // stable stop() handle for auto-close

  const setPhaseBoth = useCallback((p: VoicePhase) => {
    phaseRef.current = p;
    setPhase(p);
  }, []);

  /* ---- chat turn -------------------------------------------------------- */

  const runChat = useCallback(
    async (text: string) => {
      setTranscript((t) => [...t, { role: 'user', text }]);
      setPhaseBoth('thinking');
      setStatus('Thinking…');

      // Add an empty Gia turn we append reply chunks into.
      let giaIdx = -1;
      setTranscript((t) => {
        giaIdx = t.length;
        return [...t, { role: 'gia', text: '' }];
      });
      const appendGia = (chunk: string) =>
        setTranscript((t) => {
          const next = [...t];
          const cur = next[giaIdx];
          if (cur) next[giaIdx] = { role: 'gia', text: (cur.text ? cur.text + ' ' : '') + chunk };
          return next;
        });

      streamingRef.current = true;
      const player = playerRef.current;
      try {
        for await (const { event, data } of chatStream({
          message: text,
          user_id: userId,
          session_id: sessionId,
        })) {
          switch (event) {
            case 'agent_start':
              setStatus(`${data.agent as string}…`);
              break;
            case 'tool_call':
              setStatus(`${data.agent as string} → ${data.tool as string}`);
              break;
            case 'agent_done':
              setStatus('');
              break;
            case 'reply_chunk': {
              const clean = stripTags(String(data.text ?? ''));
              if (clean) {
                appendGia(clean);
                if (phaseRef.current !== 'speaking') setPhaseBoth('speaking');
              }
              break;
            }
            case 'audio_start':
              // A progressive (MediaSource) reply is starting.
              if (streamPlayerRef.current) {
                streamPlayerRef.current.begin();
                setPhaseBoth('speaking');
              }
              break;
            case 'audio_chunk':
              if (typeof data.data === 'string') {
                // `streaming` chunks are MP3 fragments for the MediaSource buffer;
                // a plain chunk (Kokoro blob, or MSE unsupported) is a complete
                // file for the one-shot decoder.
                if (data.streaming && streamPlayerRef.current) {
                  streamPlayerRef.current.pushBase64(data.data);
                } else if (player) {
                  player.enqueueBase64(data.data);
                }
                setPhaseBoth('speaking');
              }
              break;
            case 'audio_end':
              streamPlayerRef.current?.end();
              break;
            case 'error':
              setStatus(`error: ${String(data.error ?? 'unknown')}`);
              break;
            case 'done':
              setStatus('');
              break;
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'connection error');
      } finally {
        streamingRef.current = false;
        // Resume listening once any trailing audio finishes (or immediately when
        // there was none). Either player may be carrying this turn's audio; the
        // active one's onDrained resumes listening when it stops, so we only
        // resume here when nothing is sounding. resumeListening starts the grace
        // window, so it counts from when she actually stops speaking.
        const sounding = player?.isActive || streamPlayerRef.current?.isActive;
        if (!sounding) resumeListening();
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [userId, sessionId, setPhaseBoth],
  );

  /* ---- turn capture ----------------------------------------------------- */

  const finalizeTurn = useCallback(
    async (blob: Blob) => {
      capturingRef.current = false;
      if (!activeRef.current || blob.size < 1200) {
        resumeListening();
        return;
      }
      setPhaseBoth('thinking');
      setStatus('Transcribing…');
      const text = await transcribe(blob);
      if (!text.trim()) {
        resumeListening();
        return;
      }
      // No wake word — the user tapped to talk, so every captured utterance is
      // theirs to send. (Push-to-talk and typed turns reach runChat directly.)
      await runChat(text.trim());
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [runChat, setPhaseBoth],
  );

  const startRecorder = useCallback(() => {
    const stream = streamRef.current;
    if (!stream || capturingRef.current) return;
    const mime = pickMime();
    const rec = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
    chunksRef.current = [];
    rec.ondataavailable = (e) => {
      if (e.data.size) chunksRef.current.push(e.data);
    };
    rec.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: mime || 'audio/webm' });
      void finalizeTurn(blob);
    };
    recorderRef.current = rec;
    capturingRef.current = true;
    speechStartRef.current = performance.now();
    lastVoiceRef.current = performance.now();
    rec.start();
  }, [finalizeTurn]);

  const stopRecorder = useCallback(() => {
    const rec = recorderRef.current;
    if (rec && rec.state !== 'inactive') rec.stop();
  }, []);

  /* ---- VAD / level loop ------------------------------------------------- */

  const loop = useCallback(() => {
    if (!activeRef.current) return;
    const analyser = analyserRef.current;
    const player = playerRef.current;
    const buf = timeData.current;

    if (phaseRef.current === 'speaking' && player) {
      // Drive the ring from Gia's actual output.
      levelRef.current = player.sampleLevel();
    } else if (analyser && buf) {
      analyser.getByteTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) {
        const v = (buf[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / buf.length);
      levelRef.current = Math.min(1, rms * 4);

      // Hands-free VAD while listening. With no wake word, speech auto-starts a
      // capture; a silent grace window auto-closes the whole session.
      if (phaseRef.current === 'listening' && !manualRef.current) {
        const now = performance.now();
        if (!capturingRef.current) {
          if (rms > SPEECH_RMS) {
            startRecorder();
          } else if (!streamingRef.current && now > engagedUntilRef.current) {
            stopRef.current?.(); // grace elapsed in silence — close the session
            return;
          }
        } else {
          if (rms > SPEECH_RMS) lastVoiceRef.current = now;
          const long = now - speechStartRef.current > MIN_SPEECH_MS;
          const silent = now - lastVoiceRef.current > SILENCE_MS;
          if (long && silent) stopRecorder();
        }
      }
    }

    rafRef.current = requestAnimationFrame(loop);
  }, [startRecorder, stopRecorder]);

  function resumeListening() {
    if (!activeRef.current) return;
    capturingRef.current = false;
    manualRef.current = false;
    levelRef.current = 0;
    // The grace window starts NOW — when Gia stops speaking and listening
    // resumes — not when she started. A long reply no longer eats the window.
    engagedUntilRef.current = performance.now() + GRACE_MS;
    phaseRef.current = 'listening';
    setPhase('listening');
    setStatus('Listening…');
  }

  /* ---- lifecycle -------------------------------------------------------- */

  const ensureAudio = useCallback(async () => {
    if (!ctxRef.current) {
      const Ctor =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      ctxRef.current = new Ctor();
      // One analyser shared by both players so the ring reacts to whichever is
      // sounding (the one-shot greeting or the streamed reply).
      const analyser = createAnalyser(ctxRef.current);
      const onDrained = () => {
        if (!streamingRef.current) resumeListening();
      };
      playerRef.current = new AudioPlayer(ctxRef.current, analyser);
      playerRef.current.onDrained = onDrained;
      if (StreamPlayer.supported) {
        streamPlayerRef.current = new StreamPlayer(ctxRef.current, analyser);
        streamPlayerRef.current.onDrained = onDrained;
      }
    }
    if (ctxRef.current.state === 'suspended') await ctxRef.current.resume();
  }, []);

  const start = useCallback(async (greeting?: string) => {
    if (activeRef.current) return;
    setError(null);
    try {
      await ensureAudio();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const ctx = ctxRef.current!;
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      src.connect(analyser); // analyser only — never to destination (no echo)
      analyserRef.current = analyser;
      timeData.current = new Uint8Array(analyser.fftSize);

      activeRef.current = true;
      rafRef.current = requestAnimationFrame(loop);

      // Gia greets first, if asked to. The browser only allows this audio now,
      // because the tap is a user gesture. While it plays the phase is 'speaking'
      // so VAD won't capture it; onDrained resumes listening when she finishes.
      const buf = greeting?.trim() ? await speakApi(greeting.trim()) : null;
      if (buf && playerRef.current) {
        setPhaseBoth('speaking');
        playerRef.current.enqueueBuffer(buf);
      } else {
        resumeListening();
      }
    } catch (err) {
      setError(
        err instanceof DOMException
          ? 'Microphone access denied. Use text instead.'
          : 'Could not start the mic.',
      );
      setPhaseBoth('error');
    }
  }, [ensureAudio, loop, setPhaseBoth]);

  const stop = useCallback(() => {
    activeRef.current = false;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    streamPlayerRef.current?.clear();
    stopRecorder();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    analyserRef.current = null;
    capturingRef.current = false;
    manualRef.current = false;
    levelRef.current = 0;
    engagedUntilRef.current = 0;
    setStatus('');
    setPhaseBoth('idle');
  }, [stopRecorder, setPhaseBoth]);

  const beginCapture = useCallback(() => {
    if (!activeRef.current || phaseRef.current !== 'listening') return;
    manualRef.current = true;
    startRecorder();
  }, [startRecorder]);

  const endCapture = useCallback(() => {
    if (manualRef.current) stopRecorder();
  }, [stopRecorder]);

  const sendText = useCallback(
    async (text: string) => {
      const clean = text.trim();
      if (!clean || streamingRef.current) return;
      await ensureAudio();
      await runChat(clean);
    },
    [ensureAudio, runChat],
  );

  const speak = useCallback(
    async (text: string) => {
      const clean = text.trim();
      if (!clean) return;
      await ensureAudio(); // resumes the AudioContext (must be inside a gesture)
      const player = playerRef.current;
      if (!player) return;
      const buf = await speakApi(clean);
      if (buf) {
        setPhaseBoth('speaking');
        player.enqueueBuffer(buf); // onDrained → resumeListening when it finishes
      }
    },
    [ensureAudio, setPhaseBoth],
  );

  // Keep a stable handle to stop() so the rAF loop can auto-close on silence
  // without taking stop() as a dependency (which would re-run the loop).
  useEffect(() => {
    stopRef.current = stop;
  }, [stop]);

  // Clean up on unmount.
  useEffect(() => () => stop(), [stop]);

  return {
    phase,
    status,
    transcript,
    error,
    levelRef,
    start,
    stop,
    beginCapture,
    endCapture,
    sendText,
    speak,
  };
}
