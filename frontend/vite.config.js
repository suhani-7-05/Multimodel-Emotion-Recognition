import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend runs separately (uvicorn on :8000). In dev, Vite proxies the API and
// the live WebSocket so the browser only ever talks to one origin.
const BACKEND = process.env.BACKEND_URL || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: BACKEND, changeOrigin: true },
      "/ws": { target: BACKEND.replace(/^http/, "ws"), ws: true },
    },
  },
});
