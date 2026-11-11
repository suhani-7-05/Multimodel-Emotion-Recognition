import { useCallback, useEffect, useRef, useState } from "react";
import { liveSocketUrl } from "./api.js";

// Browser mic -> AudioWorklet -> raw float32 PCM over the WebSocket. Voice-call DSP
// (echo cancellation, noise suppression, AGC) is off: it flattens the prosody the
// emotion model relies on.
async function startBrowserCapture(onChunk) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
    },
  });
  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule("/pcm-worklet.js");
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "pcm-capture");
  const mute = ctx.createGain();
  mute.gain.value = 0;
  node.port.onmessage = (e) => onChunk(e.data);
  source.connect(node);
  node.connect(mute).connect(ctx.destination); // keeps the node pulled, but silent
  return {
    sampleRate: ctx.sampleRate,
    stop() {
      node.port.onmessage = null;
      stream.getTracks().forEach((t) => t.stop());
      ctx.close();
    },
  };
}

/** status: idle | connecting | running | stopping */
export function useLiveSession() {
  const [status, setStatus] = useState("idle");
  const [segments, setSegments] = useState([]);
  const [errors, setErrors] = useState([]);
  const [stats, setStats] = useState(null);
  const [sourceInfo, setSourceInfo] = useState("");
  const wsRef = useRef(null);
  const captureRef = useRef(null);
  const sendingRef = useRef(false);

  const addError = useCallback((msg) => setErrors((e) => [...e, msg].slice(-20)), []);

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
      let sampleRate = null;
      if (source === "browser") {
        try {
          captureRef.current = await startBrowserCapture((buf) => {
            const ws = wsRef.current;
            if (sendingRef.current && ws?.readyState === WebSocket.OPEN) ws.send(buf);
          });
          sampleRate = captureRef.current.sampleRate;
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
        ws.send(
          JSON.stringify({
            type: "start",
            source,
            device: device ?? null,
            sample_rate: sampleRate,
            settings,
            debug_wav: !!debugWav,
          })
        );
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
          case "started":
            setSourceInfo(msg.source_info);
            setStatus("running");
            sendingRef.current = true;
            break;
          case "segment":
            setSegments((s) => [...s, msg.segment]);
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
      ws.onerror = () => addError("Lost connection to the backend live socket.");
      ws.onclose = () => {
        stopCapture();
        wsRef.current = null;
        setStatus("idle");
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

  const clear = useCallback(() => {
    setSegments([]);
    setErrors([]);
  }, []);

  useEffect(() => () => {
    stopCapture();
    wsRef.current?.close();
  }, []);

  return { status, segments, errors, stats, sourceInfo, start, stop, clear };
}
