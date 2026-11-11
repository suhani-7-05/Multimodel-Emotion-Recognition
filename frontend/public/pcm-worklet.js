// Collects mono PCM from the microphone and posts ~100 ms chunks.
// processorOptions.format: "s16" -> Int16 (compact, for a remote backend) or "f32".
// Runs on the audio rendering thread, so it only copies/converts samples.
class PcmCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.format = opts.format === "s16" ? "s16" : "f32";
    this.size = Math.max(256, Math.round((opts.chunkSeconds || 0.1) * sampleRate));
    this.buf = new Float32Array(this.size);
    this.len = 0;
  }

  flush() {
    let out;
    if (this.format === "s16") {
      out = new Int16Array(this.size);
      for (let i = 0; i < this.size; i++) {
        const s = Math.max(-1, Math.min(1, this.buf[i]));
        out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
    } else {
      out = this.buf;
      this.buf = new Float32Array(this.size);
    }
    this.port.postMessage(out.buffer, [out.buffer]);
    this.len = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) {
      let i = 0;
      while (i < channel.length) {
        const n = Math.min(channel.length - i, this.size - this.len);
        this.buf.set(channel.subarray(i, i + n), this.len);
        this.len += n;
        i += n;
        if (this.len === this.size) this.flush();
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
