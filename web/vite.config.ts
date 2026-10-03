import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config"; // vite's config type plus the `test` block

// In development the UI talks to the demo server (backend/scripts/demo_server.py), which signs
// requests on behalf of a demo user. That proxy is dev-only: production needs SSO (PRD 16, FR-42).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/demo": "http://127.0.0.1:8000",
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
