import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5174,
    proxy: { "/api": "http://127.0.0.1:8010" },
  },
  test: { include: ["src/**/*.test.ts"] },
  build: {
    rollupOptions: {
      output: {
        manualChunks: { react: ["react", "react-dom"], charts: ["recharts"] },
      },
    },
  },
});
