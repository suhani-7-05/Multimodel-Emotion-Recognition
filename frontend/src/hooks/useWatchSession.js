import { useCallback, useEffect, useRef, useState } from "react";
import { liveSocketUrl } from "../api.js";

/**
 * Live analysis of a playing <video>. The viewer hears and sees the film normally; a copy of
 * its soundtrack (downmixed to mono, int16 at the context rate) is streamed to /ws/live.
 *
 * Time mapping: the server reports turn times on its own stream clock (seconds of audio it
 * received). For every chunk we send we record [streamSec, mediaSec], so a turn maps back to
 * the exact moment in the film even after pauses and seeks.
 *
 * Seeks: audio is only sent for parts of the film not analysed yet ("frontier").
 *  - Rewinding and rewatching sends nothing: earlier results are simply shown again.
 *  - Playback that catches up with the frontier continues seamlessly.
 *  - Jumping ahead past the frontier sends a "cut" so the server never glues audio from both
 *    sides of the jump into one turn, and starts a fresh dialogue context.
 */
export function useWatchSession({ onSegment }) {
  const [status, setStatus] = useState("idle"); // idle | connecting | running | stopping
  const [errors, setErrors] = useState([]);
  const [stats, setStats] = useState(null);
  const [rewatching, setRewatching] = useState(false);

  const audioRef = useRef(null); // { ctx, el } - one MediaElementSource per element, ever
  const wsRef = useRef(null);
  const sendingRef = useRef(false);
  const mapRef = useRef([]); // [streamSec, mediaSec] per chunk sent
  const sentRef = useRef(0); // samples sent
  const frontierRef = useRef(0); // media seconds analysed up to
  const rewatchRef = useRef(false);
  const onSegmentRef = useRef(onSegment);
  onSegmentRef.current = onSegment;

  const addError = useCallback((m) => setErrors((e) => [...e, m].slice(-10)), []);

  const mediaTime = useCallback((streamSec) => {
    const map = mapRef.current;
    if (!map.length) return streamSec;
    let lo = 0, hi = map.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (map[mid][0] <= streamSec) lo = mid; else hi = mid - 1;
    }
    return map[lo][1] + (streamSec - map[lo][0]);
  }, []);

  const onChunk = useCallback((buf) => {
    const a = audioRef.current;
    const ws = wsRef.current;
    if (!a || !sendingRef.current || ws?.readyState !== WebSocket.OPEN) return;
    const el = a.el;
    if (el.paused || el.seeking) return;
    const rate = a.ctx.sampleRate;
    const n = buf.byteLength / 2;
    const start = el.currentTime - n / rate; // media time this chunk began at
    const behind = start < frontierRef.current - 0.25;
    if (behind !== rewatchRef.current) {
      rewatchRef.current = behind;
      setRewatching(behind);
    }
    if (behind) return;
    if (sentRef.current > 0 && start > frontierRef.current + 0.6) {
      ws.send(JSON.stringify({ type: "cut", reset_context: true }));
    }
    mapRef.current.push([sentRef.current / rate, start]);
    ws.send(buf);
    sentRef.current += n;
    frontierRef.current = el.currentTime;
  }, []);

  /** Route the element through Web Audio once (it can only be done once per element). */
  const attach = useCallback(async (el) => {
    if (audioRef.current?.el === el) {
      await audioRef.current.ctx.resume();
      return audioRef.current;
    }
    const ctx = new AudioContext({ latencyHint: "playback" });
    await ctx.audioWorklet.addModule("/pcm-worklet.js");
    const src = ctx.createMediaElementSource(el);
    src.connect(ctx.destination); // the viewer still hears the film
    const node = new AudioWorkletNode(ctx, "pcm-capture", {
      processorOptions: { format: "s16", chunkSeconds: 0.1 },
      channelCount: 1, channelCountMode: "explicit", channelInterpretation: "speakers", // proper stereo -> mono
    });
    const mute = ctx.createGain();
    mute.gain.value = 0;
    node.port.onmessage = (e) => onChunk(e.data);
    src.connect(node);
    node.connect(mute).connect(ctx.destination);
    audioRef.current = { ctx, el, node };
    await ctx.resume();
    return audioRef.current;
  }, [onChunk]);

  const start = useCallback(async (el, settings) => {
    if (wsRef.current) return;
    setStatus("connecting");
    setErrors([]);
    setStats(null);
    let a;
    try {
      a = await attach(el);
    } catch (err) {
      addError(`Could not read the video's audio: ${err.message || err}`);
      setStatus("idle");
      return;
    }
    mapRef.current = [];
    sentRef.current = 0;
    // Keep what earlier sessions on this video already covered; start where playback is.
    frontierRef.current = Math.max(frontierRef.current, el.currentTime);
    rewatchRef.current = false;
    setRewatching(false);

    const ws = new WebSocket(liveSocketUrl());
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;
    ws.onopen = () => ws.send(JSON.stringify({
      type: "start", source: "browser", sample_rate: a.ctx.sampleRate, format: "s16",
      label: "Video soundtrack", settings,
    }));
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "started") {
        setStatus("running");
        sendingRef.current = true;
      } else if (msg.type === "segment") {
        const seg = msg.segment;
        const mStart = mediaTime(seg.start_sec);
        onSegmentRef.current?.({ ...seg, media_start: mStart, media_end: mStart + (seg.end_sec - seg.start_sec) });
      } else if (msg.type === "stats") {
        setStats(msg);
      } else if (msg.type === "error") {
        addError(msg.message);
      } else if (msg.type === "stopping") {
        sendingRef.current = false;
        setStatus("stopping");
      } else if (msg.type === "stopped") {
        ws.close();
      }
    };
    ws.onerror = () => addError("Lost the connection to the backend.");
    ws.onclose = () => {
      sendingRef.current = false;
      wsRef.current = null;
      setStatus("idle");
    };
  }, [attach, addError, mediaTime]);

  const stop = useCallback(() => {
    sendingRef.current = false;
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
      setStatus("stopping");
    } else {
      ws?.close();
    }
  }, []);

  /** New video: forget what was analysed. */
  const reset = useCallback(() => {
    frontierRef.current = 0;
    rewatchRef.current = false;
    setRewatching(false);
    setErrors([]);
    setStats(null);
  }, []);

  useEffect(() => () => {
    wsRef.current?.close();
    audioRef.current?.ctx.close();
  }, []);

  return { status, errors, stats, rewatching, frontierRef, start, stop, reset };
}
