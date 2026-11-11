import { useCallback, useEffect, useRef, useState } from "react";
import { liveSocketUrl } from "../api.js";

const MIC_CONSTRAINTS = {
  audio: {
    // Voice-call DSP flattens the prosody the emotion model relies on.
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
    channelCount: 1,
  },
};

/**
 * Browser mic -> AudioWorklet -> PCM over the WebSocket.
 * Preferred: an AudioContext at 16 kHz (the browser resamples properly) sending Int16,
 * 32 KB/s - fine even when the backend runs on a remote server. Browsers that cannot
 * connect a mic to a 16 kHz context fall back to the native rate in float32.
 */
async function startBrowserCapture(onChunk) {
  const stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
  const attempt = async (opts, format) => {
    const ctx = new AudioContext(opts);
    try {
      await ctx.audioWorklet.addModule("/pcm-worklet.js");
      const source = ctx.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(ctx, "pcm-capture", { processorOptions: { format, chunkSeconds: 0.1 } });
      const mute = ctx.createGain();
      mute.gain.value = 0;
      node.port.onmessage = (e) => onChunk(e.data);
      source.connect(node);
      node.connect(mute).connect(ctx.destination);
      return { ctx, node, format };
    } catch (err) {
      ctx.close();
      throw err;
    }
  };
  let cap;
  try {
    cap = await attempt({ sampleRate: 16000 }, "s16");
  } catch {
    cap = await attempt(undefined, "f32");
  }
  return {
    sampleRate: cap.ctx.sampleRate,
    format: cap.format,
    stop() {
      cap.node.port.onmessage = null;
      stream.getTracks().forEach((t) => t.stop());
      cap.ctx.close();
    },
  };
}

/** status: idle | connecting | running | stopping */
export function useLiveSession({ onSegment } = {}) {
  const [status, setStatus] = useState("idle");
  const [errors, setErrors] = useState([]);
  const [stats, setStats] = useState(null);
  const [sourceInfo, setSourceInfo] = useState("");
  const [startedAt, setStartedAt] = useState(null);
  const wsRef = useRef(null);
  const captureRef = useRef(null);
  const sendingRef = useRef(false);
  const onSegmentRef = useRef(onSegment);
  onSegmentRef.current = onSegment;

  const addError = useCallback((msg) => setErrors((e) => [...e, msg].slice(-10)), []);

  const stopCapture = () => {
    sendingRef.current = false;
    captureRef.current?.stop();
    captureRef.current = null;
  };

  const start = useCallback(
    async ({ source, device, settings, debugWav }) => {
      if (wsRef.current) return;
      setStatus("connecting");
      setStats(null);
      setErrors([]);
      let rate = null;
      let format = null;
      if (source === "browser") {
        try {
          captureRef.current = await startBrowserCapture((buf) => {
            const ws = wsRef.current;
            if (sendingRef.current && ws?.readyState === WebSocket.OPEN) ws.send(buf);
          });
          rate = captureRef.current.sampleRate;
          format = captureRef.current.format;
        } catch (err) {
          addError(`Microphone access failed: ${err.message || err}`);
          setStatus("idle");
          return;
        }
      }

      const ws = new WebSocket(liveSocketUrl());
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      ws.onopen = () =>
        ws.send(JSON.stringify({
          type: "start", source, device: device ?? null, sample_rate: rate, format,
          settings, debug_wav: !!debugWav,
        }));
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
          case "started":
            setSourceInfo(msg.source_info);
            setStatus("running");
            setStartedAt(Date.now());
            sendingRef.current = true;
            break;
          case "segment":
            onSegmentRef.current?.(msg.segment);
            break;
          case "stats":
            setStats(msg);
            break;
          case "error":
            addError(msg.message);
            break;
          case "stopping":
            stopCapture();
            setStatus("stopping");
            break;
          case "stopped":
            ws.close();
            break;
          default:
            break;
        }
      };
      ws.onerror = () => addError("Lost the connection to the backend.");
      ws.onclose = () => {
        stopCapture();
        wsRef.current = null;
        setStatus("idle");
        setStartedAt(null);
      };
    },
    [addError]
  );

  const stop = useCallback(() => {
    stopCapture();
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
      setStatus("stopping");
    } else {
      ws?.close();
    }
  }, []);

  useEffect(() => () => {
    stopCapture();
    wsRef.current?.close();
  }, []);

  return { status, errors, stats, sourceInfo, startedAt, start, stop };
}
