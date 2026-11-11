async function request(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
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
  batch: (file, settings) => {
    const form = new FormData();
    form.append("file", file);
    for (const [k, v] of Object.entries(settings)) form.append(k, String(v));
    return request("/api/batch", { method: "POST", body: form });
  },
  interpret: (turns, target, signal) =>
    request("/api/interpret", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ turns, target }),
      signal,
    }),
};

export function liveSocketUrl() {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/live`;
}
