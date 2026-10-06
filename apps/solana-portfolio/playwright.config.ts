import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests", testMatch: "*.spec.ts", timeout: 30000,
  use: { baseURL: "http://127.0.0.1:5174", headless: true },
  webServer: { command: "npm run dev:portfolio", cwd: "../..", url: "http://127.0.0.1:5174/api/health", reuseExistingServer: true, timeout: 30000 },
  reporter: "list",
});
