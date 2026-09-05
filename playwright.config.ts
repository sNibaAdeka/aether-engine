import { defineConfig, devices } from '@playwright/test';

/**
 * E2E config. Scoped to `tests/e2e` so Playwright's default testMatch does not
 * pick up the Vitest unit suite in `tests/unit` — configless Playwright globs
 * the repo root and happily "collects" `noise.test.ts`, then fails on the
 * missing `test()` import.
 */
const ANGLE =
  process.platform === 'darwin' ? 'metal' : process.platform === 'win32' ? 'd3d11' : 'vulkan';

export default defineConfig({
  testDir: './tests/e2e',
  testMatch: '**/*.spec.ts',
  fullyParallel: false,
  workers: 1,
  // A cold WebGPU boot plus shader compilation is genuinely slow.
  timeout: 120_000,
  expect: { timeout: 30_000 },
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'retain-on-failure',
    video: 'off',
  },
  projects: [
    {
      name: 'chromium-webgpu',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1280, height: 720 },
        launchOptions: {
          args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            `--use-angle=${ANGLE}`,
            '--ignore-gpu-blocklist',
            '--enable-gpu-rasterization',
            '--force-color-profile=srgb',
          ],
        },
      },
    },
  ],
  webServer: {
    command: 'npm run dev -- --port 5173 --strictPort',
    url: 'http://localhost:5173',
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
