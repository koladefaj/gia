/**
 * PcmPlayer — gapless playback of streamed 24 kHz mono PCM16 chunks.
 *
 * Used only in the model-voice path (REALTIME_VOICE_SOURCE=model), where
 * gpt-realtime speaks directly and the backend relays raw PCM16 audio deltas.
 * (The ElevenLabs path stays on the MP3 `StreamPlayer`.) Each base64 chunk is
 * decoded to a small AudioBuffer and scheduled back-to-back on a running time
 * cursor so the voice plays smoothly as bytes arrive.
 *
 *  - `flush()` — barge-in: stop every scheduled source and reset the cursor.
 *  - `sampleLevel()` — live output energy (time-domain RMS) for the voice ring.
 */

const PLAYBACK_RATE = 24_000;

export class PcmPlayer {
  private ctx: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private buf: Uint8Array<ArrayBuffer> | null = null;
  private nextTime = 0;
  private sources = new Set<AudioBufferSourceNode>();

  private ensure(): AudioContext {
    if (!this.ctx) {
      const Ctor =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.ctx = new Ctor({ sampleRate: PLAYBACK_RATE });
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 1024;
      this.analyser.connect(this.ctx.destination);
      this.buf = new Uint8Array(this.analyser.fftSize);
    }
    return this.ctx;
  }

  /** Resume the context if the browser suspended it (call inside a gesture). */
  async resume(): Promise<void> {
    const ctx = this.ensure();
    if (ctx.state === 'suspended') await ctx.resume();
  }

  /** True while audio is scheduled to play (or playing). */
  get isActive(): boolean {
    return this.sources.size > 0;
  }

  /** Decode and schedule one base64 PCM16 chunk for gapless playback. */
  pushBase64(b64: string): void {
    const ctx = this.ensure();
    const bytes = base64ToBytes(b64);
    if (bytes.byteLength < 2) return;

    const samples = bytes.byteLength >> 1;
    const view = new DataView(bytes.buffer, bytes.byteOffset, samples * 2);
    const audio = ctx.createBuffer(1, samples, PLAYBACK_RATE);
    const channel = audio.getChannelData(0);
    for (let i = 0; i < samples; i++) {
      channel[i] = view.getInt16(i * 2, true) / 32768;
    }

    const src = ctx.createBufferSource();
    src.buffer = audio;
    src.connect(this.analyser!);

    const now = ctx.currentTime;
    if (this.nextTime < now) this.nextTime = now + 0.02;
    src.start(this.nextTime);
    this.nextTime += audio.duration;

    this.sources.add(src);
    src.onended = () => this.sources.delete(src);
  }

  /** Barge-in: stop everything scheduled and reset the cursor. */
  flush(): void {
    this.sources.forEach((src) => {
      try {
        src.onended = null;
        src.stop();
      } catch {
        /* already stopped */
      }
    });
    this.sources.clear();
    this.nextTime = 0;
  }

  /** Current output energy 0..1 for the ring (0 when idle). */
  sampleLevel(): number {
    if (!this.analyser || !this.buf || this.sources.size === 0) return 0;
    this.analyser.getByteTimeDomainData(this.buf);
    let sum = 0;
    for (let i = 0; i < this.buf.length; i++) {
      const v = (this.buf[i] - 128) / 128;
      sum += v * v;
    }
    return Math.min(1, Math.sqrt(sum / this.buf.length) * 4);
  }

  /** Tear down the playback context. */
  async close(): Promise<void> {
    this.flush();
    try {
      await this.ctx?.close();
    } catch {
      /* already closed */
    }
    this.ctx = null;
    this.analyser = null;
    this.buf = null;
  }
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
