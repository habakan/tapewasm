import { defineConfig, devices } from "@playwright/test";

// The page is static: the fixtures are files a build step wrote, and `ts/pkg/`
// is served alongside them, which is the shape a page embedding a precompiled
// model would have.
export default defineConfig({
  testDir: ".",
  timeout: 120_000,
  fullyParallel: false,
  reporter: [["list"]],
  webServer: {
    command: "node serve.mjs",
    url: "http://127.0.0.1:8123/page.html",
    reuseExistingServer: false,
    timeout: 30_000,
  },
  use: { baseURL: "http://127.0.0.1:8123" },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "firefox", use: { ...devices["Desktop Firefox"] } },
    { name: "webkit", use: { ...devices["Desktop Safari"] } },
  ],
});
