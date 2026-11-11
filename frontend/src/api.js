async function request(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  status: () => request("/api/status"),
  config: () => request("/api/config"),
  loadModels: () => request("/api/models/load", { method: "POST" }),
  devices: () => request("/api/devices"),
  batch: (file, settings, transcriptOverride) => {
    const form = new FormData();
    form.append("file", file);
    form.append("speaker_threshold", String(settings.speaker_threshold));
    if (transcriptOverride) form.append("transcript_override", transcriptOverride);
    return request("/api/batch", { method: "POST", body: form });
  },
};

export function liveSocketUrl() {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/live`;
}
