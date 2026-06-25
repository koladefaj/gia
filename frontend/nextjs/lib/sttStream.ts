/**
 * FluxStream — client for the backend's streaming-STT WebSocket (/voice/stream).
 *
 * Sends raw PCM16 audio frames up and receives normalised transcript frames
 * back. The backend hides which ASR provider is behind the switch, so this
 * client only knows the wire contract:
 *
 *   ready   — provider socket is open; start sending audio
 *   partial — interim transcript (revisable)
 *   eager   — Deepgram Flux EagerEndOfTurn (Phase 2 early-intent hook)
 *   resumed — Flux TurnResumed (cancel any speculative work)
 *   final   — committed transcript for this turn → run it
 *   error   — streaming unavailable; caller should fall back to one-shot STT
 */

import { wsUrl } from './api';

export interface FluxCallbacks {
  onReady?: () => void;
  onPartial?: (text: string) => void;
  onEager?: (text: string) => void;
  onResumed?: () => void;
  onFinal?: (text: string) => void;
  onError?: (error: string) => void;
  onClose?: () => void;
}

interface Frame {
  type: 'ready' | 'partial' | 'eager' | 'resumed' | 'final' | 'error';
  text?: string;
  error?: string;
}

export class FluxStream {
  private ws: WebSocket | null = null;
  private ready = false;

  constructor(private readonly cb: FluxCallbacks) {}

  /** True once the provider socket reported ready and audio can flow. */
  get isReady(): boolean {
    return this.ready;
  }

  /** Open the socket. `language`/`provider` are optional overrides. */
  connect(params?: { language?: string; provider?: string }): void {
    const ws = new WebSocket(wsUrl('/voice/stream', params));
    ws.binaryType = 'arraybuffer';

    ws.onmessage = (e) => {
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
        case 'partial':
          if (frame.text) this.cb.onPartial?.(frame.text);
          break;
        case 'eager':
          if (frame.text) this.cb.onEager?.(frame.text);
          break;
        case 'resumed':
          this.cb.onResumed?.();
          break;
        case 'final':
          if (frame.text) this.cb.onFinal?.(frame.text);
          break;
        case 'error':
          this.cb.onError?.(frame.error ?? 'streaming unavailable');
          break;
      }
    };

    ws.onerror = () => this.cb.onError?.('websocket error');
    ws.onclose = () => {
      this.ready = false;
      this.cb.onClose?.();
    };

    this.ws = ws;
  }

  /** Forward a PCM16 audio frame (no-op until ready / after close). */
  sendAudio(pcm: ArrayBuffer): void {
    if (this.ready && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(pcm);
    }
  }

  /** Ask the provider to flush the final transcript for the current turn. */
  stop(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'stop' }));
    }
  }

  /** Tear down the socket. */
  close(): void {
    this.ready = false;
    try {
      this.ws?.close();
    } catch {
      /* noop */
    }
    this.ws = null;
  }
}
