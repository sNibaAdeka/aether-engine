/**
 * Screenshot harness.
 *
 *   npm run shot                      → all viewpoints
 *   npm run shot -- --preset=valley-dawn
 *   npm run shot -- --quality=ultra --scale=1
 *   npm run shot -- --preset=mountain-noon --tod=5,8,12,17,19.5,23
 *
 * Boots the dev server (or reuses a running one), drives the camera through
 * `window.__AETHER_DEBUG__`, waits for TAA to converge, and writes PNGs to
 * `.screenshots/`. It also dumps the profiler report and any console errors
 * next to the images, so "it looks fine" and "it is fast and clean" are checked
 * in the same pass.
 */

import { chromium, type Browser, type Page } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import { mkdir, writeFile, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import viewpointsData from './viewpoints.json' with { type: 'json' };

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const OUT_DIR = join(ROOT, '.screenshots');

interface Viewpoint {
  name: string;
  position: [number, number, number];
  target: [number, number, number];
  timeOfDay: number;
  weather: string;
  fov: number;
  note: string;
}

const VIEWPOINTS = viewpointsData.viewpoints as Viewpoint[];

interface Args {
  preset?: string;
  quality: string;
  width: number;
  height: number;
  settle: number;
  keepServer: boolean;
  url?: string;
  seed?: string;
  /**
   * Hours of the game day to sweep, replacing each viewpoint's own time.
   *
   * Phase 3's acceptance criterion is "no frame with an unnatural colour" over a
   * whole day, and that is not a thing one screenshot can answer. With this set,
   * every viewpoint is captured once per listed hour, into
   * `<name>@<hour>.png`.
   */
  tod?: number[];
}

function parseArgs(): Args {
  const out: Args = {
    quality: 'high',
    width: 1920,
    height: 1080,
    settle: 30,
    keepServer: false,
  };
  for (const raw of process.argv.slice(2)) {
    const [k, v] = raw.replace(/^--/, '').split('=');
    switch (k) {
      case 'preset':
      case 'viewpoint':
        out.preset = v;
        break;
      case 'quality':
        out.quality = v;
        break;
      case 'width':
        out.width = Number(v);
        break;
      case 'height':
        out.height = Number(v);
        break;
      case 'settle':
        out.settle = Number(v);
        break;
      case 'url':
        out.url = v;
        break;
      case 'seed':
        out.seed = v;
        break;
      case 'tod':
        out.tod = v
          .split(',')
          .map((x) => Number(x))
          .filter((x) => Number.isFinite(x));
        break;
      case 'keep-server':
        out.keepServer = true;
        break;
      default:
        break;
    }
  }
  return out;
}

async function waitForServer(url: string, timeoutMs = 60000): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url, { method: 'GET' });
      if (res.ok) return true;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  return false;
}

async function startDevServer(): Promise<{ proc: ChildProcess | null; url: string }> {
  const url = 'http://localhost:5173/';
  if (await waitForServer(url, 1200)) {
    console.log('[shot] reusing running dev server');
    return { proc: null, url };
  }
  console.log('[shot] starting dev server…');
  const proc = spawn('npm', ['run', 'dev', '--', '--port', '5173', '--strictPort'], {
    cwd: ROOT,
    stdio: 'pipe',
    env: { ...process.env, FORCE_COLOR: '0' },
  });
  proc.stdout?.on('data', (d: Buffer) => {
    const s = d.toString();
    if (s.includes('error')) process.stdout.write(`[vite] ${s}`);
  });
  proc.stderr?.on('data', (d: Buffer) => process.stderr.write(`[vite] ${d.toString()}`));

  if (!(await waitForServer(url))) {
    proc.kill();
    throw new Error('dev server did not become ready in 60s');
  }
  return { proc, url };
}

/**
 * Chromium flags needed for real WebGPU in headless.
 *
 * The ANGLE backend must be named explicitly per platform: on macOS
 * `--use-angle=default` resolves to a path where `navigator.gpu` is absent
 * entirely, so the harness silently falls back to WebGL2 and every screenshot
 * quietly measures the wrong renderer. Verified working combination per
 * platform below; the harness also asserts `backend === 'webgpu'` after boot so
 * a future regression fails loudly instead of producing plausible-looking PNGs.
 */
function chromeFlags(): string[] {
  const angle =
    process.platform === 'darwin' ? 'metal' : process.platform === 'win32' ? 'd3d11' : 'vulkan';
  return [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    `--use-angle=${angle}`,
    '--ignore-gpu-blocklist',
    '--enable-gpu-rasterization',
    '--force-color-profile=srgb',
    '--disable-frame-rate-limit',
  ];
}

async function capture(
  page: Page,
  vp: Viewpoint,
  args: Args,
  hourOverride?: number,
): Promise<string[]> {
  const problems: string[] = [];
  const hour = hourOverride ?? vp.timeOfDay;
  const label = hourOverride === undefined ? vp.name : `${vp.name}@${formatHour(hourOverride)}`;

  await page.evaluate(
    ({ pose, tod, weather, quality }) => {
      const api = window.__AETHER_DEBUG__;
      if (!api) throw new Error('__AETHER_DEBUG__ missing');
      api.setPreset(quality as 'low' | 'medium' | 'high' | 'ultra');
      api.setTimeOfDay(tod);
      api.setWeather(weather);
      api.setCamera(pose);
      api.setUIVisible(false);
    },
    {
      pose: { position: vp.position, target: vp.target, fov: vp.fov },
      tod: hour,
      weather: vp.weather,
      quality: args.quality,
    },
  );

  // Let streaming settle, then let TAA converge — and then let the *eye*
  // converge. Auto-exposure adapts over 2.5 s of wall clock when it is
  // darkening, so a shot taken right after a jump from noon to midnight records
  // the adaptation, not the night.
  await page.waitForTimeout(1200);
  await page.evaluate((n) => window.__AETHER_DEBUG__?.waitFrames(n), args.settle);
  await page.waitForTimeout(hourOverride === undefined ? 150 : 6000);

  const file = join(OUT_DIR, `${label}.png`);
  await page.screenshot({ path: file, type: 'png' });

  const stats = await page.evaluate(() => ({
    frame: window.__AETHER_DEBUG__?.frameStats(),
    report: window.__AETHER_DEBUG__?.profileReport(),
    errors: window.__AETHER_DEBUG__?.errors() ?? [],
    backend: window.__AETHER_DEBUG__?.backend(),
  }));

  await writeFile(
    join(OUT_DIR, `${label}.md`),
    [
      `# ${label}`,
      '',
      `${vp.note}`,
      '',
      `backend: **${stats.backend}** · quality: **${args.quality}** · ` +
        `time ${hour}h · weather ${vp.weather}`,
      '',
      stats.report ?? '(no profiler data)',
      '',
      stats.errors.length ? `## console\n\n\`\`\`\n${stats.errors.join('\n')}\n\`\`\`` : '',
    ].join('\n'),
    'utf8',
  );

  const fps = stats.frame?.fps ?? 0;
  console.log(
    `[shot] ${label.padEnd(22)} ${fps.toFixed(0).padStart(3)} FPS  ` +
      `p95 ${(stats.frame?.p95 ?? 0).toFixed(1)}ms  draws ${stats.frame?.draws ?? 0}`,
  );

  if (stats.errors.length > 0) {
    problems.push(`${label}: ${stats.errors.length} console message(s)`);
  }
  return problems;
}

/** 19.5 → "19h30", so the filenames sort in clock order. */
function formatHour(h: number): string {
  const hh = Math.floor(h);
  const mm = Math.round((h - hh) * 60);
  return `${String(hh).padStart(2, '0')}h${String(mm).padStart(2, '0')}`;
}

async function main(): Promise<void> {
  const args = parseArgs();

  // Only wipe on a full run. A single-viewpoint run is an iteration loop —
  // deleting the other five images would make "compare against the last full
  // pass" impossible exactly when it is most useful.
  if (!args.preset && existsSync(OUT_DIR)) {
    await rm(OUT_DIR, { recursive: true, force: true });
  }
  await mkdir(OUT_DIR, { recursive: true });

  let server: { proc: ChildProcess | null; url: string } = { proc: null, url: args.url ?? '' };
  if (!args.url) server = await startDevServer();

  let browser: Browser | null = null;
  const problems: string[] = [];

  try {
    browser = await chromium.launch({ args: chromeFlags(), headless: true });
    const page = await browser.newPage({
      viewport: { width: args.width, height: args.height },
      deviceScaleFactor: 1,
    });

    page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
    page.on('console', (m) => {
      if (m.type() === 'error') problems.push(`console.error: ${m.text()}`);
    });

    const url = new URL(server.url);
    if (args.seed) url.searchParams.set('seed', args.seed);
    await page.goto(url.toString(), { waitUntil: 'domcontentloaded' });

    await page.waitForFunction(() => window.__AETHER_READY__ === true, undefined, {
      timeout: 90000,
    });

    const backend = await page.evaluate(() => window.__AETHER_DEBUG__?.backend());
    console.log(`[shot] backend: ${backend}`);
    if (backend !== 'webgpu') {
      problems.push(`headless browser fell back to ${backend} — GPU flags may not be taking`);
    }

    const list = args.preset ? VIEWPOINTS.filter((v) => v.name === args.preset) : VIEWPOINTS;
    if (list.length === 0) {
      throw new Error(
        `unknown viewpoint "${args.preset}". Known: ${VIEWPOINTS.map((v) => v.name).join(', ')}`,
      );
    }

    for (const vp of list) {
      if (args.tod) {
        for (const hour of args.tod) {
          problems.push(...(await capture(page, vp, args, hour)));
        }
      } else {
        problems.push(...(await capture(page, vp, args)));
      }
    }

    await writeFile(
      join(OUT_DIR, 'INDEX.md'),
      [
        `# Screenshots — quality ${args.quality} · ${args.width}×${args.height}`,
        '',
        ...list.flatMap((v) =>
          args.tod
            ? args.tod.map((h) => {
                const n = `${v.name}@${formatHour(h)}`;
                return `## ${n}\n\n![${n}](./${n}.png)\n\n${v.note}\n`;
              })
            : [`## ${v.name}\n\n![${v.name}](./${v.name}.png)\n\n${v.note}\n`],
        ),
        problems.length ? `## Problems\n\n${problems.map((p) => `- ${p}`).join('\n')}` : '',
      ].join('\n'),
      'utf8',
    );
  } finally {
    await browser?.close();
    if (server.proc && !args.keepServer) server.proc.kill();
  }

  if (problems.length > 0) {
    console.error(`\n[shot] ${problems.length} problem(s):`);
    for (const p of problems) console.error(`  - ${p}`);
    process.exitCode = 1;
  } else {
    console.log(`\n[shot] clean. PNGs in ${OUT_DIR}`);
  }
}

main().catch((err) => {
  console.error('[shot] failed:', err);
  process.exit(1);
});
