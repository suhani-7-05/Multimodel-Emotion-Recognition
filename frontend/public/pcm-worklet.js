// Collects mono float32 PCM from the microphone and posts it in ~2048-sample chunks.
// Runs on the audio rendering thread, so it only copies samples - no processing.
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.size = 2048;
    this.buf = new Float32Array(this.size);
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
        if (this.len === this.size) {
          this.port.postMessage(this.buf.buffer, [this.buf.buffer]);
          this.buf = new Float32Array(this.size);
          this.len = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
