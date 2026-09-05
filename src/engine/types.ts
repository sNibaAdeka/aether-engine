/**
 * Core engine contracts. Everything the world is made of implements `System`.
 *
 * Deliberately tiny: no ECS, no event bus, no dependency injection container.
 * `World` owns an explicit ordered list of systems and calls them in that
 * order. When the order matters — and in a renderer it always matters — an
 * explicit array beats any amount of priority magic.
 */

import type { Camera, Scene, WebGPURenderer } from 'three/webgpu';
import type { Profiler } from './Profiler';
import type { Settings } from './Settings';
import type { ResourceManager } from './ResourceManager';

export interface EngineContext {
  readonly renderer: WebGPURenderer;
  readonly scene: Scene;
  readonly camera: Camera;
  readonly settings: Settings;
  readonly profiler: Profiler;
  /**
   * GPU resource ownership. Systems that allocate textures or buffers register
   * them here — the F3 memory readout is only as honest as this is complete.
   */
  readonly resources: ResourceManager;
  /** Global world seed. Everything procedural derives from this. */
  readonly seed: number;
  /** True when running on the real WebGPU backend (not the WebGL2 fallback). */
  readonly hasCompute: boolean;
}

export interface FrameContext {
  /** Fixed physics timestep, seconds. Constant 1/60. */
  readonly fixedDt: number;
  /** Wall-clock delta for this frame, seconds, clamped. */
  readonly dt: number;
  /** Interpolation alpha between the last two physics states, [0,1). */
  readonly alpha: number;
  /** Seconds since start, wrapped to keep f32 shader precision usable. */
  readonly elapsed: number;
  /** Monotonically increasing frame counter. */
  readonly frame: number;
  /** Camera world position, do not mutate. */
  readonly cameraPosition: Readonly<{ x: number; y: number; z: number }>;
}

export interface System {
  readonly name: string;
  init(ctx: EngineContext): Promise<void>;
  /** CPU logic. Runs every frame. Must not allocate. */
  update(dt: number, ctx: FrameContext): void;
  /** Optional per-frame GPU resource preparation, runs after all updates. */
  prepare?(ctx: FrameContext): void;
  dispose(): void;
}

/** A system that also contributes named passes to the frame graph. */
export interface RenderSystem extends System {
  readonly passes: readonly string[];
}
