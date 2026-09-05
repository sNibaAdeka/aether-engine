/**
 * `window.__AETHER_DEBUG__` — the control surface the Playwright screenshot
 * harness and the e2e tests drive.
 *
 * Keeping it in one typed module (rather than sprinkling `(window as any)`
 * around) means the harness and the engine cannot drift apart silently.
 */

import { Vector3 } from 'three/webgpu';
import type { Engine } from '@/engine/Engine';
import type { FlyCamera } from '@/player/FlyCamera';
import type { HUD } from '@/ui/HUD';
import type { DebugOverlay } from '@/ui/DebugOverlay';
import type { QualityPreset } from '@/engine/Settings';

export interface CameraPose {
  position: [number, number, number];
  target: [number, number, number];
  fov?: number;
}

export interface AetherDebugAPI {
  version: string;
  /** Move the debug camera. Resolves once the pose is applied. */
  setCamera(pose: CameraPose): void;
  getCamera(): CameraPose;
  /** Wait for N rendered frames — used to let TAA converge before capture. */
  waitFrames(n: number): Promise<void>;
  setPreset(preset: QualityPreset): void;
  setTimeOfDay(hours: number): void;
  setWeather(name: string): void;
  setSeed(seed: number): void;
  setUIVisible(visible: boolean): void;
  /** Current profiler snapshot as markdown — dumped into PROGRESS.md. */
  profileReport(): string;
  /** Structured timings for automated budget assertions. */
  timings(): Record<string, { p50: number; p95: number; gpu: number }>;
  frameStats(): { fps: number; p50: number; p95: number; draws: number; tris: number };
  /** Terrain height at a world position — used by the CPU/GPU parity test. */
  sampleHeight(x: number, z: number): number;
  backend(): string;
  errors(): string[];
}

export interface DebugBridgeDeps {
  engine: Engine;
  flyCamera: FlyCamera;
  hud: HUD;
  overlay: DebugOverlay;
  setTimeOfDay?: (hours: number) => void;
  setWeather?: (name: string) => void;
  setSeed?: (seed: number) => void;
  sampleHeight?: (x: number, z: number) => number;
}

const _pos = new Vector3();
const _target = new Vector3();

/** Console errors and warnings collected since boot, for the smoke test. */
const collectedErrors: string[] = [];

export function installErrorCollector(): void {
  const origError = console.error.bind(console);
  const origWarn = console.warn.bind(console);
  console.error = (...args: unknown[]): void => {
    collectedErrors.push(`error: ${args.map(String).join(' ')}`);
    origError(...args);
  };
  console.warn = (...args: unknown[]): void => {
    collectedErrors.push(`warn: ${args.map(String).join(' ')}`);
    origWarn(...args);
  };
  window.addEventListener('error', (e) => {
    collectedErrors.push(`uncaught: ${e.message} @ ${e.filename}:${e.lineno}`);
  });
  window.addEventListener('unhandledrejection', (e) => {
    collectedErrors.push(`unhandled rejection: ${String(e.reason)}`);
  });
}

export function installDebugBridge(deps: DebugBridgeDeps): AetherDebugAPI {
  const { engine, flyCamera, hud, overlay } = deps;

  const api: AetherDebugAPI = {
    version: '0.1.0',

    setCamera(pose: CameraPose): void {
      _pos.set(pose.position[0], pose.position[1], pose.position[2]);
      _target.set(pose.target[0], pose.target[1], pose.target[2]);
      flyCamera.enabled = true;
      flyCamera.setPose(_pos, _target);
      if (pose.fov !== undefined) {
        engine.settings.fov = pose.fov;
        engine.settings.emit();
      }
    },

    getCamera(): CameraPose {
      const c = engine.camera;
      c.getWorldDirection(_target);
      _target.add(c.position);
      return {
        position: [c.position.x, c.position.y, c.position.z],
        target: [_target.x, _target.y, _target.z],
        fov: c.fov,
      };
    },

    waitFrames(n: number): Promise<void> {
      return new Promise((resolve) => {
        let left = Math.max(1, n);
        const step = (): void => {
          left--;
          if (left <= 0) resolve();
          else requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
      });
    },

    setPreset(preset: QualityPreset): void {
      engine.settings.applyPreset(preset);
    },

    setTimeOfDay(hours: number): void {
      deps.setTimeOfDay?.(hours);
    },

    setWeather(name: string): void {
      deps.setWeather?.(name);
    },

    setSeed(seed: number): void {
      deps.setSeed?.(seed);
    },

    setUIVisible(visible: boolean): void {
      hud.setVisible(visible);
      overlay.setVisible(visible);
    },

    profileReport(): string {
      return engine.profiler.report();
    },

    timings(): Record<string, { p50: number; p95: number; gpu: number }> {
      const out: Record<string, { p50: number; p95: number; gpu: number }> = {};
      for (const s of engine.profiler.stats()) {
        out[s.name] = { p50: s.cpuP50, p95: s.cpuP95, gpu: s.gpuP50 };
      }
      return out;
    },

    frameStats() {
      const p = engine.profiler;
      return {
        fps: p.fps,
        p50: p.frameMs,
        p95: p.frameMsP95,
        draws: p.drawCalls,
        tris: p.triangles,
      };
    },

    sampleHeight(x: number, z: number): number {
      return deps.sampleHeight?.(x, z) ?? 0;
    },

    backend(): string {
      return engine.rendererInfo.backend;
    },

    errors(): string[] {
      return [...collectedErrors];
    },
  };

  (window as unknown as Record<string, unknown>).__AETHER_DEBUG__ = api;
  return api;
}

declare global {
  interface Window {
    __AETHER_DEBUG__?: AetherDebugAPI;
    __AETHER_READY__?: boolean;
    /**
     * Live object graph, for console poking and e2e assertions that need to
     * reach past the curated debug API. Loosely typed on purpose — anything
     * worth depending on should be promoted to `AetherDebugAPI`.
     */
    __AETHER__: {
      engine: Engine;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      settingsPanel: { root: any };
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      [key: string]: any;
    };
  }
}
