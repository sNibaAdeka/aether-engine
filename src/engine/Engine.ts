/**
 * Owner of the main loop, the renderer and the system list.
 *
 * Frame shape:
 *   beginFrame → drain resource queue → N fixed physics steps → system update
 *   → system prepare → frame graph → render → endFrame → dynamic resolution
 *
 * Nothing here allocates per frame. `FrameContext` is a single mutable object
 * reused every frame; systems read it and must not retain it.
 */

import * as THREE from 'three/webgpu';
import { PerspectiveCamera, Scene, type WebGPURenderer } from 'three/webgpu';
import { Clock, FIXED_DT } from './Clock';
import { Profiler } from './Profiler';
import { Settings } from './Settings';
import { FrameGraph } from './FrameGraph';
import { ResourceManager } from './ResourceManager';
import { createRenderer, ResizeController, type RendererInfo } from './Renderer';
import type { EngineContext, FrameContext, System } from './types';

export interface EngineOptions {
  container: HTMLElement;
  seed?: number;
  forceWebGL?: boolean;
}

export class Engine {
  readonly scene = new Scene();
  readonly camera: PerspectiveCamera;
  readonly clock = new Clock();
  readonly profiler = new Profiler();
  readonly settings = new Settings();
  readonly resources = new ResourceManager();
  readonly frameGraph: FrameGraph;

  renderer!: WebGPURenderer;
  rendererInfo!: RendererInfo;
  resize!: ResizeController;

  readonly systems: System[] = [];
  private systemsByName = new Map<string, System>();

  seed: number;
  running = false;
  paused = false;
  /** Set once init + first frame have completed; the screenshot tool waits on it. */
  ready = false;

  private rafHandle = 0;
  private ctx!: EngineContext;
  private frameCtx: {
    fixedDt: number;
    dt: number;
    alpha: number;
    elapsed: number;
    frame: number;
    cameraPosition: { x: number; y: number; z: number };
  };

  // Dynamic-resolution state.
  private dynResAccum = 0;
  private dynResFrames = 0;
  private dynResCooldown = 0;

  constructor(private options: EngineOptions) {
    this.seed = options.seed ?? 1337;
    this.camera = new PerspectiveCamera(this.settings.fov, 1, 0.1, 30000);
    this.camera.position.set(0, 40, 0);
    this.frameGraph = new FrameGraph(this.profiler);
    this.frameCtx = {
      fixedDt: FIXED_DT,
      dt: 0,
      alpha: 0,
      elapsed: 0,
      frame: 0,
      cameraPosition: { x: 0, y: 0, z: 0 },
    };
  }

  async init(): Promise<void> {
    const { renderer, info } = await createRenderer({
      forceWebGL: this.options.forceWebGL,
    });
    this.renderer = renderer;
    this.rendererInfo = info;

    this.options.container.appendChild(renderer.domElement);

    this.resize = new ResizeController(renderer, this.options.container);
    this.resize.setRenderScale(this.settings.effectiveRenderScale);
    this.resize.start((w, h) => {
      this.camera.aspect = this.resize.width / Math.max(1, this.resize.height);
      this.camera.updateProjectionMatrix();
      for (const s of this.systems) {
        const r = s as System & { onResize?: (w: number, h: number) => void };
        r.onResize?.(w, h);
      }
    });

    this.settings.onChange((s) => {
      this.camera.fov = s.fov;
      this.camera.far = Math.max(2000, s.quality.drawDistance * 1.2);
      this.camera.updateProjectionMatrix();
      this.resize.setRenderScale(s.effectiveRenderScale);
    });

    this.ctx = {
      renderer,
      scene: this.scene,
      camera: this.camera,
      settings: this.settings,
      profiler: this.profiler,
      resources: this.resources,
      seed: this.seed,
      hasCompute: info.backend === 'webgpu',
    };

    document.addEventListener('visibilitychange', this.onVisibility);
    window.addEventListener('aether:context-lost', this.onContextLost);
  }

  get context(): EngineContext {
    return this.ctx;
  }

  async addSystem(system: System): Promise<void> {
    if (this.systemsByName.has(system.name)) {
      throw new Error(`Engine: duplicate system name "${system.name}"`);
    }
    await system.init(this.ctx);
    this.systems.push(system);
    this.systemsByName.set(system.name, system);
  }

  getSystem<T extends System>(name: string): T | undefined {
    return this.systemsByName.get(name) as T | undefined;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.clock.reset(performance.now());
    this.rafHandle = requestAnimationFrame(this.tick);
  }

  stop(): void {
    this.running = false;
    if (this.rafHandle) cancelAnimationFrame(this.rafHandle);
    this.rafHandle = 0;
  }

  private onVisibility = (): void => {
    if (document.hidden) {
      this.paused = true;
    } else {
      this.paused = false;
      // Reset the accumulator or we simulate the entire time the tab was away.
      this.clock.reset(performance.now());
      this.profiler.resumeFromPause();
    }
  };

  private onContextLost = (): void => {
    this.stop();
  };

  private tick = (now: number): void => {
    if (!this.running) return;
    this.rafHandle = requestAnimationFrame(this.tick);

    if (this.paused) return;

    const profiler = this.profiler;
    profiler.beginFrame(now);

    const steps = this.clock.advance(now);

    const fc = this.frameCtx;
    fc.dt = this.clock.dt;
    fc.alpha = this.clock.alpha;
    fc.elapsed = this.clock.elapsedShader;
    fc.frame = this.clock.frame;
    fc.cameraPosition.x = this.camera.position.x;
    fc.cameraPosition.y = this.camera.position.y;
    fc.cameraPosition.z = this.camera.position.z;

    profiler.begin('resources');
    this.resources.drain();
    profiler.end('resources');

    // Fixed-step simulation.
    if (steps > 0) {
      profiler.begin('fixed-update');
      for (let i = 0; i < steps; i++) {
        for (let s = 0; s < this.systems.length; s++) {
          const sys = this.systems[s] as System & { fixedUpdate?: (dt: number) => void };
          sys.fixedUpdate?.(FIXED_DT);
        }
      }
      profiler.end('fixed-update');
    }

    // Variable-step update.
    profiler.begin('update');
    for (let i = 0; i < this.systems.length; i++) {
      this.systems[i].update(this.clock.dt, fc as FrameContext);
    }
    profiler.end('update');

    // GPU resource preparation.
    profiler.begin('prepare');
    for (let i = 0; i < this.systems.length; i++) {
      this.systems[i].prepare?.(fc as FrameContext);
    }
    profiler.end('prepare');

    // Explicit render pass order.
    this.frameGraph.execute(fc as FrameContext);

    // Counters for the overlay. `drawCalls`/`triangles` are per-frame;
    // `compute.calls` is cumulative since boot, so use `frameCalls`.
    const rInfo = this.renderer.info;
    profiler.drawCalls = rInfo.render.drawCalls;
    profiler.triangles = rInfo.render.triangles;
    profiler.computeCalls = rInfo.compute.frameCalls;
    profiler.gpuMemoryMB = this.resources.megabytes;

    this.pollGpuTimestamps();

    profiler.endFrame();

    this.updateDynamicResolution();

    if (!this.ready && this.clock.frame > 2) {
      this.ready = true;
      window.dispatchEvent(new CustomEvent('aether:ready'));
      (window as unknown as Record<string, unknown>).__AETHER_READY__ = true;
    }
  };

  /**
   * Read back GPU pass durations.
   *
   * `resolveTimestampsAsync` resolves a query written several frames ago, so
   * the numbers lag the CPU timings slightly — that is fine for budgeting and
   * far better than reporting CPU submit time as if it were GPU cost. Only one
   * resolve is in flight at a time; awaiting every frame would serialise the
   * pipeline, which is exactly the stall this whole design avoids.
   */
  private gpuResolveInFlight = false;
  /** Frame index of the last resolve, so a timestamp can be divided by its span. */
  private gpuResolveFrame = 0;

  private pollGpuTimestamps(): void {
    if (this.gpuResolveInFlight) return;
    if (this.rendererInfo.backend !== 'webgpu' || !this.rendererInfo.hasTimestampQuery) return;
    // Throttle: the resolve costs a buffer map, and three's query pool holds
    // 2048 entries — polling every frame at several hundred FPS churns it for
    // no extra information.
    if (this.clock.frame % 15 !== 0) return;

    // How many frames this resolve will be summing over.
    //
    // three's `resolveQueriesAsync` resets its query index and returns the
    // TOTAL of every query written since the previous resolve — so at a poll
    // interval of 15 frames it returns fifteen frames of GPU time, and
    // reporting it as-is overstates per-frame cost by that factor. This is why
    // 'terrain-generate' read 2.80 ms against a 1.0 ms budget while the pass it
    // measures was costing about 0.19 ms a frame, and why the same counter read
    // 1.33 ms with erosion completely idle. The span is not always exactly 15:
    // a resolve still in flight skips a poll. Divide by what actually elapsed.
    const span = Math.max(1, this.clock.frame - this.gpuResolveFrame);
    this.gpuResolveFrame = this.clock.frame;

    this.gpuResolveInFlight = true;
    void this.renderer
      .resolveTimestampsAsync(THREE.TimestampQuery.RENDER)
      .then((ms) => {
        if (typeof ms === 'number' && ms > 0) this.profiler.recordGpu('gpu-render', ms / span);
        return this.renderer.resolveTimestampsAsync(THREE.TimestampQuery.COMPUTE);
      })
      .then((ms) => {
        if (typeof ms === 'number' && ms > 0) {
          const perFrame = ms / span;
          this.profiler.recordGpu('gpu-compute', perFrame);
          // three aggregates every compute pass of the frame into one timestamp
          // query, and terrain tile baking is currently the only compute work
          // in the engine — so attributing it to 'terrain-generate' is accurate
          // today. It stops being accurate the moment a second compute system
          // exists, and then this needs per-pass queries rather than a rename.
          //
          // Deliberately a GPU-only series: nothing CPU-times the same name, so
          // every ring slot holds a real sample and the p50 is a percentile
          // rather than a median over mostly-empty frames.
          this.profiler.recordGpu('terrain-generate', perFrame);
        }
      })
      .catch(() => {
        // A dropped timestamp is not worth a console error every frame.
      })
      .finally(() => {
        this.gpuResolveInFlight = false;
      });
  }

  /**
   * Nudge render scale to hold the frame budget. Hysteresis and a cooldown stop
   * it oscillating: drop fast when over budget, creep back up only when there
   * is comfortable headroom.
   */
  private updateDynamicResolution(): void {
    if (!this.settings.dynamicResolution) return;

    this.dynResAccum += this.profiler.frameMsLast;
    this.dynResFrames++;
    if (this.dynResFrames < 30) return;

    const avg = this.dynResAccum / this.dynResFrames;
    this.dynResAccum = 0;
    this.dynResFrames = 0;

    if (this.dynResCooldown > 0) {
      this.dynResCooldown--;
      return;
    }

    const target = this.settings.targetFrameMs;
    const current = this.settings.effectiveRenderScale;

    if (avg > target * 1.05) {
      this.settings.setEffectiveRenderScale(current - 0.05);
      this.dynResCooldown = 2;
    } else if (avg < target * 0.8 && current < this.settings.quality.renderScale) {
      this.settings.setEffectiveRenderScale(current + 0.05);
      this.dynResCooldown = 4;
    }
  }

  /** Render one frame synchronously — used by the screenshot harness. */
  renderOnce(): void {
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.stop();
    document.removeEventListener('visibilitychange', this.onVisibility);
    window.removeEventListener('aether:context-lost', this.onContextLost);
    for (const s of this.systems) s.dispose();
    this.systems.length = 0;
    this.systemsByName.clear();
    this.frameGraph.clear();
    this.resources.disposeAll();
    this.resize?.dispose();
    this.renderer?.dispose();
  }
}

export { THREE };
