import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  workers: 1,
  fullyParallel: false,
  use: {
    baseURL: "http://127.0.0.1:8011",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "node scripts/browser-server.mjs",
    url: "http://127.0.0.1:8011/api/health",
    reuseExistingServer: false,
    timeout: 30000,
  },
});
