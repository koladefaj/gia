/**
 * PcmCapture — taps a mic MediaStream and emits 24 kHz mono PCM16 frames.
 *
 * Uses a dedicated AudioContext at 24 kHz (the streaming-STT wire rate) so the
 * pcm-worklet receives audio already at the target rate — no resampling. This
 * context is separate from the playback context, so capture and Gia's TTS never
 * interfere. The same MediaStream can feed both (each context makes its own
 * source node), so we don't open the mic twice.
 */

const WORKLET_URL = '/pcm-worklet.js';
const CAPTURE_RATE = 24_000;

export class PcmCapture {
  private ctx: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private src: MediaStreamAudioSourceNode | null = null;

  /** Called with each ~80 ms PCM16 frame (transferred ArrayBuffer). */
  onFrame?: (pcm: ArrayBuffer) => void;

  /** True once the worklet is wired and frames are flowing. */
  get active(): boolean {
    return this.node !== null;
  }

  /** Begin capturing from *stream*. Safe to call once per session. */
  async start(stream: MediaStream): Promise<void> {
    const Ctor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new Ctor({ sampleRate: CAPTURE_RATE });
    await ctx.audioWorklet.addModule(WORKLET_URL);

    const src = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, 'pcm-worklet');
    node.port.onmessage = (e: MessageEvent) => this.onFrame?.(e.data as ArrayBuffer);

    // source → worklet → destination. The worklet emits silence, so routing it
    // to the destination (needed for the graph to run) causes no echo.
    src.connect(node);
    node.connect(ctx.destination);

    this.ctx = ctx;
    this.node = node;
    this.src = src;
  }

  /** Stop capture and release the audio nodes/context. */
  async stop(): Promise<void> {
    try {
      this.node?.port.close();
      this.node?.disconnect();
      this.src?.disconnect();
      await this.ctx?.close();
    } catch {
      /* already torn down */
    }
    this.node = null;
    this.src = null;
    this.ctx = null;
  }
}
