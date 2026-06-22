/**
 * Web Audio playback for Gia's TTS.
 *
 * Two playback paths share one AnalyserNode (so the audio-reactive ring reacts
 * to whichever is sounding):
 *
 *  - `AudioPlayer` — one-shot, fully-decoded buffers (the opening greeting and
 *    `/voice/speak`). Each chunk is a complete audio file decoded with
 *    `decodeAudioData` and played gaplessly in arrival order.
 *
 *  - `StreamPlayer` — progressive MP3 streaming for chat replies. ElevenLabs'
 *    `/stream` endpoint returns a single continuous MP3 whose chunks are NOT
 *    independently decodable, so we feed them into a MediaSource buffer on an
 *    `<audio>` element and play as bytes arrive. This is what makes Gia start
 *    talking before the whole reply is rendered.
 */

/** Strip ElevenLabs-style `[audio tags]` — delivery cues, not display text. */
export function stripTags(text: string): string {
  return text
    .replace(/\[[a-z][a-z ]*\]/gi, '')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

/** Read normalised output energy (0..1) from a shared analyser — for the ring. */
function readLevel(analyser: AnalyserNode, freq: Uint8Array<ArrayBuffer>): number {
  analyser.getByteFrequencyData(freq);
  let sum = 0;
  for (let i = 0; i < freq.length; i++) sum += freq[i];
  return sum / freq.length / 255;
}

/** Create the shared analyser used by both players (wired to destination). */
export function createAnalyser(ctx: AudioContext): AnalyserNode {
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 256;
  analyser.smoothingTimeConstant = 0.8;
  analyser.connect(ctx.destination);
  return analyser;
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

  constructor(ctx: AudioContext, analyser: AnalyserNode) {
    this.ctx = ctx;
    this.analyser = analyser;
    this.freq = new Uint8Array(this.analyser.frequencyBinCount);
  }

  /** Queue a base64-encoded audio chunk (a complete audio file). */
  enqueueBase64(b64: string): void {
    this.enqueueBuffer(base64ToBuffer(b64));
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
    return readLevel(this.analyser, this.freq);
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

const STREAM_MIME = 'audio/mpeg';

/**
 * Progressive MP3 playback via MediaSource. One `<audio>` element and one
 * MediaElementSource are created once and reused across turns (a
 * MediaElementSource can only be created once per element); each reply gets a
 * fresh MediaSource.
 */
export class StreamPlayer {
  private readonly analyser: AnalyserNode;
  private readonly freq: Uint8Array<ArrayBuffer>;
  private readonly el: HTMLAudioElement;

  private mediaSource: MediaSource | null = null;
  private sourceBuffer: SourceBuffer | null = null;
  private readonly pending: ArrayBuffer[] = [];
  private ended = false; // server sent audio_end; flush then endOfStream
  private active = false;
  private objectUrl: string | null = null;

  onDrained?: () => void;
  onPlayingChange?: (playing: boolean) => void;

  /** True when MediaSource MP3 streaming is supported in this browser. */
  static get supported(): boolean {
    return (
      typeof MediaSource !== 'undefined' &&
      MediaSource.isTypeSupported(STREAM_MIME)
    );
  }

  constructor(ctx: AudioContext, analyser: AnalyserNode) {
    this.analyser = analyser;
    this.freq = new Uint8Array(this.analyser.frequencyBinCount);
    this.el = new Audio();
    this.el.preload = 'auto';
    // Route the element through the SAME analyser the ring reads, then to output.
    const src = ctx.createMediaElementSource(this.el);
    src.connect(this.analyser);
    this.el.addEventListener('ended', () => this.finish());
  }

  /** Begin a new streamed reply. Resets any prior MediaSource. */
  begin(): void {
    this.reset();
    this.active = true;
    this.ended = false;
    this.onPlayingChange?.(true);

    const ms = new MediaSource();
    this.mediaSource = ms;
    this.objectUrl = URL.createObjectURL(ms);
    this.el.src = this.objectUrl;
    ms.addEventListener('sourceopen', () => {
      // sourceopen can fire after a later reply already replaced ms — guard.
      if (this.mediaSource !== ms) return;
      try {
        const sb = ms.addSourceBuffer(STREAM_MIME);
        sb.addEventListener('updateend', () => this.flush());
        this.sourceBuffer = sb;
        this.flush();
      } catch {
        this.finish();
      }
    });
  }

  /** Append a base64-encoded MP3 chunk and start playback on the first one. */
  pushBase64(b64: string): void {
    this.pending.push(base64ToBuffer(b64));
    this.flush();
    // Autoplay is permitted: a streamed reply only follows a user gesture.
    void this.el.play().catch(() => {});
  }

  /** Signal that no more chunks are coming; close the stream once flushed. */
  end(): void {
    this.ended = true;
    this.flush();
  }

  private flush(): void {
    const sb = this.sourceBuffer;
    const ms = this.mediaSource;
    if (!sb || !ms || sb.updating) return;

    const next = this.pending.shift();
    if (next) {
      try {
        sb.appendBuffer(next);
      } catch {
        // QuotaExceeded or invalid state — re-queue is pointless mid-MP3; bail.
        this.finish();
      }
      return;
    }
    // Nothing pending: if the server is done and the buffer is idle, close it so
    // the element fires 'ended' and the ring goes calm.
    if (this.ended && ms.readyState === 'open') {
      try {
        ms.endOfStream();
      } catch {
        /* already closing */
      }
    }
  }

  private finish(): void {
    if (!this.active) return;
    this.active = false;
    this.onPlayingChange?.(false);
    this.onDrained?.();
  }

  private reset(): void {
    this.pending.length = 0;
    this.sourceBuffer = null;
    try {
      this.el.pause();
    } catch {
      /* noop */
    }
    if (this.objectUrl) {
      URL.revokeObjectURL(this.objectUrl);
      this.objectUrl = null;
    }
    this.mediaSource = null;
  }

  sampleLevel(): number {
    return readLevel(this.analyser, this.freq);
  }

  get isActive(): boolean {
    return this.active;
  }

  /** Stop and tear down the current stream (e.g. on session stop). */
  clear(): void {
    this.reset();
    this.active = false;
  }
}

function base64ToBuffer(b64: string): ArrayBuffer {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return buf;
}
