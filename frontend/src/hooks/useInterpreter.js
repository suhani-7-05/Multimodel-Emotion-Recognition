import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api.js";

const CONTEXT_TURNS = 8;

/**
 * Explains turns one at a time, oldest first, so each explanation sees the turns before it.
 * results: { [audio_url]: { status: "pending"|"loading"|"done"|"error", data, error, latency } }
 */
export function useInterpreter(segments, { enabled, onResult }) {
  const [results, setResults] = useState({});
  const busyRef = useRef(false);
  const resultsRef = useRef(results);
  resultsRef.current = results;
  const onResultRef = useRef(onResult);
  onResultRef.current = onResult;
  const abortRef = useRef(null);

  const next = useCallback(async () => {
    if (busyRef.current || !enabled) return;
    const idx = segments.findIndex((s) => !resultsRef.current[s.audio_url]);
    if (idx === -1) return;
    const key = segments[idx].audio_url;
    busyRef.current = true;
    setResults((r) => ({ ...r, [key]: { status: "loading" } }));
    const lo = Math.max(0, idx - CONTEXT_TURNS);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const res = await api.interpret(segments.slice(lo, idx + 1), idx - lo, ctrl.signal);
      setResults((r) => ({ ...r, [key]: { status: "done", data: res.interpretation, latency: res.latency_ms } }));
      onResultRef.current?.(segments[idx], res.interpretation);
    } catch (err) {
      if (err.name === "AbortError") {
        setResults((r) => {
          const { [key]: _, ...rest } = r;
          return rest;
        });
      } else {
        setResults((r) => ({ ...r, [key]: { status: "error", error: err.message } }));
      }
    } finally {
      busyRef.current = false;
      abortRef.current = null;
      setResults((r) => ({ ...r })); // nudge the effect to pick the next turn
    }
  }, [segments, enabled]);

  useEffect(() => {
    next();
  }, [next, results]);

  const retry = useCallback((key) => {
    setResults((r) => {
      const { [key]: _, ...rest } = r;
      return rest;
    });
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setResults({});
  }, []);

  return { results, retry, reset };
}
