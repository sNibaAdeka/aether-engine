/**
 * Fixed-timestep clock with an accumulator.
 *
 * Physics runs at exactly 60 Hz regardless of display refresh; rendering
 * interpolates between the last two physics states using `alpha`. The substep
 * cap prevents the spiral of death when a frame takes catastrophically long
 * (tab restored from background, shader compile hitch, GC pause).
 */

export const FIXED_DT = 1 / 60;
const MAX_SUBSTEPS = 5;
/** Ignore frame deltas above this — they mean the tab was suspended. */
const MAX_FRAME_DT = 0.25;
/**
 * Shader time wraps here. f32 has ~7 significant digits, so by 3600 s a
 * `sin(time * 40)` term has lost all sub-frame precision and vegetation starts
 * to jitter. Wrapping at a multiple of 2π keeps every periodic term continuous
 * across the seam.
 */
const SHADER_TIME_WRAP = Math.PI * 2 * 573; // ≈ 3600.4 s

export class Clock {
  /** Seconds since start, unwrapped. Use for gameplay/day cycle. */
  elapsedTotal = 0;
  /** Seconds since start, wrapped to `SHADER_TIME_WRAP`. Use for shaders. */
  elapsedShader = 0;
  /** Last frame's wall-clock delta, clamped. */
  dt = 0;
  /** Interpolation factor between physics states, [0,1). */
  alpha = 0;
  /** Number of fixed steps consumed by the last `advance` call. */
  steps = 0;
  /** Monotonic frame counter. */
  frame = 0;
  /** True when the last frame was clamped, i.e. we dropped simulation time. */
  didClamp = false;

  private accumulator = 0;
  private lastTime = 0;
  private started = false;

  /** Number of times the shader clock has wrapped. Systems that need absolute
   *  phase continuity across the seam can use this to compensate. */
  wrapCount = 0;

  reset(now: number): void {
    this.lastTime = now;
    this.accumulator = 0;
    this.started = true;
    this.didClamp = false;
  }

  /**
   * Advance to `now` (milliseconds, from `performance.now`).
   * Returns the number of fixed steps that should be simulated this frame.
   */
  advance(now: number): number {
    if (!this.started) {
      this.reset(now);
      return 0;
    }

    let delta = (now - this.lastTime) / 1000;
    this.lastTime = now;

    this.didClamp = false;
    if (delta > MAX_FRAME_DT) {
      delta = MAX_FRAME_DT;
      this.didClamp = true;
    }
    if (delta < 0) delta = 0;

    this.dt = delta;
    this.elapsedTotal += delta;
    this.elapsedShader += delta;
    if (this.elapsedShader >= SHADER_TIME_WRAP) {
      this.elapsedShader -= SHADER_TIME_WRAP;
      this.wrapCount++;
    }

    this.accumulator += delta;

    let steps = 0;
    while (this.accumulator >= FIXED_DT && steps < MAX_SUBSTEPS) {
      this.accumulator -= FIXED_DT;
      steps++;
    }
    // Hit the cap: discard the backlog rather than trying to catch up forever.
    if (steps === MAX_SUBSTEPS && this.accumulator > FIXED_DT) {
      this.accumulator = 0;
      this.didClamp = true;
    }

    this.steps = steps;
    this.alpha = this.accumulator / FIXED_DT;
    this.frame++;
    return steps;
  }
}
