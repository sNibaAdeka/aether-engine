/**
 * Boot smoke test.
 *
 * Guards the things that are cheap to break and expensive to notice: that the
 * app reaches a first frame at all, that it does so on WebGPU rather than
 * silently degrading to WebGL2, that the console stays clean, and that the
 * frame-graph plumbing actually renders geometry rather than an empty pass.
 */

import { test, expect, type Page } from '@playwright/test';

async function boot(page: Page): Promise<void> {
  await page.goto('/', { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__AETHER_READY__ === true, undefined, {
    timeout: 90_000,
  });
  // Let streaming and the first shader compiles settle.
  await page.waitForTimeout(1500);
}

test.describe('boot', () => {
  test('reaches a first frame on the WebGPU backend', async ({ page }) => {
    const pageErrors: string[] = [];
    page.on('pageerror', (e) => pageErrors.push(e.message));

    await boot(page);

    const backend = await page.evaluate(() => window.__AETHER_DEBUG__?.backend());
    expect(backend, 'must not silently fall back to WebGL2').toBe('webgpu');
    expect(pageErrors).toEqual([]);
  });

  test('logs no console errors or warnings', async ({ page }) => {
    await boot(page);
    const errors = await page.evaluate(() => window.__AETHER_DEBUG__?.errors() ?? []);
    // The spec's phase-0 gate: zero errors AND zero warnings for a session.
    expect(errors, `console was not clean:\n${errors.join('\n')}`).toEqual([]);
  });

  test('actually renders geometry', async ({ page }) => {
    await boot(page);
    const stats = await page.evaluate(() => window.__AETHER_DEBUG__?.frameStats());
    expect(stats).toBeTruthy();
    // A frame graph that runs but draws nothing would still "pass" a screenshot
    // check against a blank sky.
    expect(stats!.draws).toBeGreaterThan(0);
    expect(stats!.tris).toBeGreaterThan(0);
    expect(stats!.fps).toBeGreaterThan(20);
  });

  test('camera control surface works', async ({ page }) => {
    await boot(page);
    await page.evaluate(() =>
      window.__AETHER_DEBUG__?.setCamera({
        position: [12, 6, 24],
        target: [0, 2, 0],
        fov: 55,
      }),
    );
    await page.evaluate(() => window.__AETHER_DEBUG__?.waitFrames(5));

    const pose = await page.evaluate(() => window.__AETHER_DEBUG__?.getCamera());
    expect(pose!.position[0]).toBeCloseTo(12, 3);
    expect(pose!.position[1]).toBeCloseTo(6, 3);
    expect(pose!.position[2]).toBeCloseTo(24, 3);
    expect(pose!.fov).toBeCloseTo(55, 3);
  });

  test('quality presets change the render resolution', async ({ page }) => {
    await boot(page);

    const read = (): Promise<{ scale: number; shadowRes: number; grass: number }> =>
      page.evaluate(() => {
        const s = window.__AETHER__.engine.settings;
        return {
          scale: s.effectiveRenderScale,
          shadowRes: s.quality.shadowResolution,
          grass: s.quality.grassRadius,
        };
      });

    await page.evaluate(() => window.__AETHER_DEBUG__?.setPreset('ultra'));
    await page.evaluate(() => window.__AETHER_DEBUG__?.waitFrames(3));
    const ultra = await read();

    await page.evaluate(() => window.__AETHER_DEBUG__?.setPreset('low'));
    await page.evaluate(() => window.__AETHER_DEBUG__?.waitFrames(3));
    const low = await read();

    expect(low.scale).toBeLessThan(ultra.scale);
    expect(low.shadowRes).toBeLessThan(ultra.shadowRes);
    expect(low.grass).toBeLessThan(ultra.grass);
  });

  test('settings panel stays in sync after a programmatic preset change', async ({ page }) => {
    // Regression guard: lil-gui captures the bound object by reference, so
    // replacing settings.quality would leave every slider writing to a
    // detached object while the engine read a different one.
    await boot(page);
    await page.evaluate(() => window.__AETHER_DEBUG__?.setPreset('ultra'));
    await page.evaluate(() => window.__AETHER_DEBUG__?.waitFrames(3));

    const inSync = await page.evaluate(() => {
      const s = window.__AETHER__.engine.settings;
      const gui = window.__AETHER__.settingsPanel.root;
      const shown = gui
        .controllersRecursive()
        .find((c: { property: string }) => c.property === 'shadowResolution');
      return {
        engine: s.quality.shadowResolution,
        widget: shown ? shown.getValue() : null,
      };
    });
    expect(inSync.widget).toBe(inSync.engine);
    expect(inSync.engine).toBe(4096);
  });
});
