/**
 * pcm-worklet — mic capture for streaming STT.
 *
 * Runs on a 24 kHz AudioContext (set by the caller), so the input is already at
 * the wire rate and no resampling is needed here. It batches the render-quantum
 * Float32 blocks into ~80 ms frames (Deepgram Flux's recommended chunk size),
 * converts them to signed PCM16, and transfers each frame to the main thread.
 *
 * The node produces no audio output (its outputs stay silent), so connecting it
 * onward to the destination — required for the graph to pull input — is safe.
 */
class PcmWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this._chunks = [];
    this._count = 0;
    // ~80 ms at 24 kHz = 1920 samples.
    this._target = 1920;
  }

  process(inputs) {
    const input = inputs[0];
    const channel = input && input[0];
    if (channel && channel.length) {
      this._chunks.push(channel.slice(0));
      this._count += channel.length;
      if (this._count >= this._target) {
        const merged = new Float32Array(this._count);
        let offset = 0;
        for (const c of this._chunks) {
          merged.set(c, offset);
          offset += c.length;
        }
        this._chunks = [];
        this._count = 0;

        const pcm = new Int16Array(merged.length);
        for (let i = 0; i < merged.length; i++) {
          const s = Math.max(-1, Math.min(1, merged[i]));
          pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        this.port.postMessage(pcm.buffer, [pcm.buffer]);
      }
    }
    return true;
  }
}

registerProcessor('pcm-worklet', PcmWorklet);
