/**
 * RealtimeClient — browser end of the speech-to-speech session (/voice/realtime).
 *
 * Hybrid: gpt-realtime is the ears + brain (we stream mic PCM16 up; it
 * understands + reasons + calls tools + writes the reply as text), and
 * ElevenLabs v3 is the voice (the backend synthesises the text and streams MP3
 * back). So this client sends audio up and receives the SAME audio frames as the
 * /chat pipeline — it reuses PcmCapture for the mic and hands playback frames to
 * the caller's StreamPlayer via callbacks (no separate audio engine).
 *
 * Wire contract (mirrors backend/app/api/voice_realtime.py):
 *   ready           — session armed; audio may flow
 *   user_transcript — finalised user words (captions)
 *   reply_chunk     — a chunk of the reply text (captions)
 *   audio_start / audio_chunk / audio_end — ElevenLabs MP3 (→ StreamPlayer)
 *   flush           — barge-in: drop buffered playback
 *   speech_started  — user is talking
 *   tool            — a tool is running (status)
 *   response_done   — turn complete
 *   error           — session unavailable / failed
 */

import { wsUrl } from './api';
import { PcmCapture } from './pcm';
import { PcmPlayer } from './realtimePlayer';

export interface RealtimeCallbacks {
  onReady?: () => void;
  onUserTranscript?: (text: string) => void;
  onReplyChunk?: (text: string) => void;
  onAudioStart?: () => void;
  onAudioChunk?: (b64: string) => void;
  onAudioEnd?: () => void;
  onFlush?: () => void;
  onSpeechStarted?: () => void;
  onTool?: (name: string) => void;
  onResponseDone?: () => void;
  onError?: (error: string) => void;
  onClose?: () => void;
}

interface Frame {
  type:
    | 'ready'
    | 'user_transcript'
    | 'reply_chunk'
    | 'audio' // model-voice mode: raw PCM16 chunk
    | 'audio_start'
    | 'audio_chunk'
    | 'audio_end'
    | 'flush'
    | 'speech_started'
    | 'tool'
    | 'response_done'
    | 'error';
  data?: string;
  text?: string;
  name?: string;
  error?: string;
}

export class RealtimeClient {
  private ws: WebSocket | null = null;
  private capture: PcmCapture | null = null;
  /** Model-voice PCM playback (unused in the ElevenLabs path). */
  readonly player = new PcmPlayer();
  private ready = false;

  constructor(private readonly cb: RealtimeCallbacks) {}

  get isReady(): boolean {
    return this.ready;
  }

  /** Open the socket and start mic capture. Never throws. */
  async start(
    stream: MediaStream,
    params?: { user_id?: string | null; session_id?: string },
  ): Promise<void> {
    await this.player.resume();
    const ws = new WebSocket(
      wsUrl('/voice/realtime', {
        user_id: params?.user_id ?? undefined,
        session_id: params?.session_id,
      }),
    );
    ws.binaryType = 'arraybuffer';
    ws.onmessage = (e) => this.onMessage(e);
    ws.onerror = () => this.cb.onError?.('websocket error');
    ws.onclose = () => {
      this.ready = false;
      this.cb.onClose?.();
    };
    this.ws = ws;

    const capture = new PcmCapture();
    capture.onFrame = (frame) => {
      if (this.ready && this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(frame);
      }
    };
    try {
      await capture.start(stream);
      this.capture = capture;
    } catch {
      this.cb.onError?.('microphone unavailable');
    }
  }

  private onMessage(e: MessageEvent): void {
    let frame: Frame;
    try {
      frame = JSON.parse(e.data as string) as Frame;
    } catch {
      return;
    }
    switch (frame.type) {
      case 'ready':
        this.ready = true;
        this.cb.onReady?.();
        break;
      case 'user_transcript':
        if (frame.text) this.cb.onUserTranscript?.(frame.text);
        break;
      case 'reply_chunk':
        if (frame.text) this.cb.onReplyChunk?.(frame.text);
        break;
      case 'audio':
        // Model-voice mode: play the model's own PCM16 directly.
        if (frame.data) this.player.pushBase64(frame.data);
        break;
      case 'audio_start':
        this.cb.onAudioStart?.();
        break;
      case 'audio_chunk':
        if (frame.data) this.cb.onAudioChunk?.(frame.data);
        break;
      case 'audio_end':
        this.cb.onAudioEnd?.();
        break;
      case 'flush':
        // Barge-in: drop both playback paths.
        this.player.flush();
        this.cb.onFlush?.();
        break;
      case 'speech_started':
        this.cb.onSpeechStarted?.();
        break;
      case 'tool':
        if (frame.name) this.cb.onTool?.(frame.name);
        break;
      case 'response_done':
        this.cb.onResponseDone?.();
        break;
      case 'error':
        this.cb.onError?.(frame.error ?? 'realtime unavailable');
        break;
    }
  }

  /** Tear down the socket and capture. */
  async close(): Promise<void> {
    this.ready = false;
    if (this.ws?.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify({ type: 'stop' }));
      } catch {
        /* noop */
      }
    }
    try {
      this.ws?.close();
    } catch {
      /* noop */
    }
    this.ws = null;
    await this.capture?.stop();
    this.capture = null;
    await this.player.close();
  }
}
