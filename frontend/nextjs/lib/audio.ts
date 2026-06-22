/**
 * Web Audio playback queue for Gia's streamed TTS.
 *
 * Chunks arrive base64-encoded over SSE (`audio_chunk` frames) and must play
 * gaplessly in arrival order. Playback is routed through an AnalyserNode so the
 * visual ring can react to Gia's actual voice while she speaks.
 */

/** Strip ElevenLabs-style `[audio tags]` — delivery cues, not display text. */
export function stripTags(text: string): string {
  return text
    .replace(/\[[a-z][a-z ]*\]/gi, '')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

export class AudioPlayer {
  private readonly ctx: AudioContext;
  private readonly analyser: AnalyserNode;
  private readonly freq: Uint8Array<ArrayBuffer>;
  private readonly queue: ArrayBuffer[] = [];
  private playing = false;

  /** Called when the queue empties and the last buffer finishes. */
  onDrained?: () => void;
  /** Called with `true` when playback starts, `false` when it fully stops. */
  onPlayingChange?: (playing: boolean) => void;

  constructor(ctx: AudioContext) {
    this.ctx = ctx;
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 256;
    this.analyser.smoothingTimeConstant = 0.8;
    this.analyser.connect(ctx.destination);
    this.freq = new Uint8Array(this.analyser.frequencyBinCount);
  }

  /** Queue a base64-encoded audio chunk from an `audio_chunk` SSE frame. */
  enqueueBase64(b64: string): void {
    const bin = atob(b64);
    const buf = new ArrayBuffer(bin.length);
    const view = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
    this.queue.push(buf);
    void this.playNext();
  }

  /** Queue a raw audio buffer (e.g. a one-shot /voice/speak response). */
  enqueueBuffer(buf: ArrayBuffer): void {
    this.queue.push(buf);
    void this.playNext();
  }

  private async playNext(): Promise<void> {
    if (this.playing) return;
    const buf = this.queue.shift();
    if (!buf) {
      this.onPlayingChange?.(false);
      this.onDrained?.();
      return;
    }
    this.playing = true;
    this.onPlayingChange?.(true);
    try {
      // decodeAudioData detaches the buffer — pass a copy so retries are safe.
      const decoded = await this.ctx.decodeAudioData(buf.slice(0));
      const src = this.ctx.createBufferSource();
      src.buffer = decoded;
      src.connect(this.analyser);
      src.onended = () => {
        this.playing = false;
        void this.playNext();
      };
      src.start();
    } catch {
      this.playing = false;
      void this.playNext();
    }
  }

  /** Current normalised output energy (0..1) — for the audio-reactive ring. */
  sampleLevel(): number {
    this.analyser.getByteFrequencyData(this.freq);
    let sum = 0;
    for (let i = 0; i < this.freq.length; i++) sum += this.freq[i];
    return sum / this.freq.length / 255;
  }

  /** True while audio is playing or still queued. */
  get isActive(): boolean {
    return this.playing || this.queue.length > 0;
  }

  /** Drop anything queued (does not stop the buffer already sounding). */
  clear(): void {
    this.queue.length = 0;
  }
}
