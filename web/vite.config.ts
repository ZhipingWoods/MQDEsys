import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `pnpm dev` proxies API calls to a running `mqde serve` (default 127.0.0.1:18080).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: process.env.MQDE_API ?? "http://127.0.0.1:18080", changeOrigin: true } },
  },
  build: { outDir: "dist", chunkSizeWarningLimit: 1500 },
});
