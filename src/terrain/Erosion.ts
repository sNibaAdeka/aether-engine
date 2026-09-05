/**
 * Erosion scheduler: runs the droplet and talus kernels for one tile at a time,
 * a few dispatches per frame, inside the generation budget.
 *
 * Erosion is not a per-frame effect and it is not a one-shot bake either. One
 * 129² tile is 296k droplets over a 385² scratch domain — measured at 12.5 ms of
 * GPU time, i.e. three quarters of a frame at 60 Hz — so it can neither run
 * every frame nor be finished in one. It is a *job*: the tile is claimed, 93
 * small dispatches are run across however many frames the budget allows (about
 * 25 at the measured 0.5 ms/frame), and the result is written into the layer the
 * un-eroded bake already filled.
 *
 * That ordering is deliberate and it is what keeps the streamer honest. The
 * analytic bake marks a tile ready immediately, so a node always has a surface
 * to draw; erosion later replaces the contents of the same layer. A tile is
 * therefore never a hole and never waits — it starts smooth and gains its
 * channels a fraction of a second later. The cost is a visible change on tiles
 * that erode while on screen, which is the honest trade for never stalling.
 *
 * Job state lives here rather than in `TilePool` because the pool is an
 * allocator: it knows about layers and recency, not about a multi-frame pipeline
 * that can be invalidated halfway through by an eviction.
 */

import type * as THREE from 'three/webgpu';

import type { Profiler } from '@/engine/Profiler';
import type { ResourceManager } from '@/engine/ResourceManager';
import { TILE_CELLS, type TilePool } from './TilePool';
import {
  APRON,
  BAND_GROUPS,
  CORE_GROUPS,
  DROPLET_BANDS,
  EROSION_TUNING,
  PHASES,
  ROUNDS,
  SCRATCH_GROUPS,
  THERMAL_ITERATIONS,
  WORKGROUP,
  buildErosionProgram,
  type ErosionProgram,
} from './shaders/erosion.wgsl';

/** Dispatches making up one checkerboard phase: N droplet bands, then an apply. */
const STEPS_PER_PHASE = DROPLET_BANDS + 1;
const STEPS_PER_ROUND = PHASES * STEPS_PER_PHASE;
const HYDRAULIC_STEPS = ROUNDS * STEPS_PER_ROUND;
/** Each talus pass is a scatter plus the same apply the droplets use. */
const THERMAL_STEPS = THERMAL_ITERATIONS * 2;
/** seed · hydraulics · talus · smoothDelta · applySmooth · blurX · resolve */
export const STEPS_PER_TILE = 1 + HYDRAULIC_STEPS + THERMAL_STEPS + 4;

/**
 * Estimated GPU cost of each dispatch kind, milliseconds, on this machine.
 *
 * Measured, not guessed. Method: pin the scheduler to a fixed number of
 * dispatches per frame and read the p50 *frame interval* against an idle
 * baseline, frame limiter off, streamer settled. Mountain viewpoint, 1600×900:
 *
 *   dispatches/frame   1      2      4      5      7      8
 *   frame p50 (ms)     0.90   1.00   1.90   2.70   3.60   3.90
 *   over idle (0.80)  +0.10  +0.20  +1.10  +1.90  +2.80  +3.10
 *
 * The marginal cost above two dispatches is ≈0.50 ms; the first one or two are
 * nearly free because they overlap the render. 0.30 is the value that makes the
 * scheduler pick three, which measures +0.60 ms — inside the 1.0 ms generation
 * slice with margin, and not so conservative that a tile takes forever.
 *
 * The number this replaced (0.4 for a droplet band) was a whole-tile total
 * divided by a guessed mix, and the loop it fed tested the budget *after*
 * spending it, so the pass ran at roughly 2× its allowance. Both halves of that
 * are fixed; this is the half that needed a measurement.
 *
 * TODO(quality): still open loop. three exposes one aggregate COMPUTE timestamp
 * per frame rather than one per pass, so nothing here can self-correct; on much
 * slower hardware this needs re-measuring by the same method.
 */
const COST_DISPATCH = 0.3;
const COST_SEED = COST_DISPATCH;
const COST_DROPLET = COST_DISPATCH;
const COST_APPLY = COST_DISPATCH;
const COST_THERMAL = COST_DISPATCH;
const COST_SMOOTH = COST_DISPATCH;
const COST_BLUR = COST_DISPATCH;
const COST_RESOLVE = COST_DISPATCH;

/**
 * Hard ceiling on dispatches per frame, whatever the budget says.
 *
 * A safety rail against a mis-estimated cost, not a tuning knob: at 0.175 ms
 * apiece the 1.0 ms generation slice buys five, so this only binds if someone
 * raises the budget.
 */
const MAX_STEPS_PER_FRAME = 8;

/** Reused dispatch descriptors — `renderer.compute` must not allocate. */
const _dispatchScratch: number[] = [SCRATCH_GROUPS, SCRATCH_GROUPS, 1];
const _dispatchBand: number[] = [SCRATCH_GROUPS, BAND_GROUPS, 1];
const _dispatchCore: number[] = [CORE_GROUPS, CORE_GROUPS, 1];

export interface ErosionStats {
  /** Tiles whose erosion has completed and been written back. */
  completed: number;
  /** Tiles abandoned because their layer was recycled mid-job. */
  abandoned: number;
  /** Dispatches issued in the most recent frame. */
  stepsLastFrame: number;
  /** Progress of the tile currently in the pipeline, 0..1. Negative when idle. */
  progress: number;
  /**
   * Non-zero flow texels the resolve pass reported writing, read back off the
   * GPU after the first tile. -1 until the readback lands.
   */
  flowTexelsWritten: number;
}

/** A quarter of one tile. See `Erosion.verifyFlowMap` for why that number. */
const MIN_VERIFIED_TEXELS = Math.floor((TILE_CELLS + 1) ** 2 / 4);

export class Erosion {
  /** GPU milliseconds erosion may consume this frame. Set by the caller. */
  gpuBudgetMs = 0.8;

  readonly stats: ErosionStats = {
    completed: 0,
    abandoned: 0,
    stepsLastFrame: 0,
    progress: -1,
    flowTexelsWritten: -1,
  };

  /** Set only by the readback in `verifyFlowMap`. */
  private verified = false;
  private verifyRequested = false;

  private program: ErosionProgram;

  /** Layer currently being eroded, or -1. */
  private slot = -1;
  /** Pool key the job was started for; the layer is only written if it still matches. */
  private slotKey = -1;
  private step = 0;

  constructor(
    private renderer: THREE.WebGPURenderer,
    private pool: TilePool,
    private profiler: Profiler,
    resources: ResourceManager,
    seed: number,
  ) {
    this.program = buildErosionProgram(pool, seed);
    // The scratch buffers are ~2.5 MB and permanent; the F3 memory readout is
    // only as honest as this registration is complete.
    for (let i = 0; i < this.program.attributes.length; i++) {
      resources.track(
        this.program.attributes[i],
        `terrain/erosionScratch${i}`,
        this.program.bytes / this.program.attributes.length,
      );
    }
  }

  /**
   * Issue as many dispatches as the budget allows.
   *
   * Always issues at least one when there is work: refusing entirely would let a
   * slow frame stall erosion permanently, and one small dispatch is cheaper than
   * the stutter of a tile that never finishes.
   */
  update(): void {
    this.profiler.begin('terrain-erode');
    let spent = 0;
    let steps = 0;

    for (;;) {
      // Look before leaping. The old loop tested `spent < budget` *after* the
      // fact, so it always issued the dispatch that broke the budget — with a
      // 0.4 ms estimate against a 0.8 ms allowance that is a guaranteed 50 %
      // overshoot every frame, and the estimate was itself 2.3× low. Charging
      // the cost first is the whole difference between a budget and a wish.
      if (steps > 0 && spent + this.nextCost() > this.gpuBudgetMs) break;
      if (!this.ensureJob()) break;
      const cost = this.runStep(this.step);
      this.step++;
      spent += cost;
      steps++;
      if (this.step >= STEPS_PER_TILE) this.finish();
      // A hard ceiling so a mis-estimated cost cannot spin a whole tile into one
      // frame if someone raises the budget.
      if (steps >= MAX_STEPS_PER_FRAME) break;
    }

    this.stats.stepsLastFrame = steps;
    this.stats.progress = this.slot >= 0 ? this.step / STEPS_PER_TILE : -1;
    this.profiler.end('terrain-erode');
  }

  /** Estimated cost of the dispatch `runStep` would issue next. */
  private nextCost(): number {
    // Uniform by measurement — see COST_DISPATCH. Kept as a function rather
    // than inlined so that re-measuring per-kernel costs later is a local edit
    // to the table and not a change of shape here.
    return COST_DISPATCH;
  }

  /**
   * Make sure a valid job is in flight, claiming a new tile if needed.
   *
   * The mid-job validity check is not defensive padding: the pool may recycle a
   * layer between frames (a settings change flushes it outright), and writing
   * the resolve pass into a layer that now holds a different tile would paint
   * one part of the world with another part's terrain.
   */
  private ensureJob(): boolean {
    if (this.slot >= 0) {
      if (this.pool.isErosionJobValid(this.slot, this.slotKey)) return true;
      this.pool.abortErosion(this.slot);
      this.stats.abandoned++;
      this.slot = -1;
    }

    const slot = this.pool.beginErosion();
    if (slot < 0) return false;

    this.slot = slot;
    this.slotKey = this.pool.keyOfSlot(slot);
    this.step = 0;
    this.bindTile(slot);
    return true;
  }

  /** Point the uniforms at a tile. Runs once per job, not per dispatch. */
  private bindTile(slot: number): void {
    const u = this.program.uniforms;
    const size = this.pool.slotSize(slot);
    const spacing = size / TILE_CELLS;
    const originX = this.pool.slotOriginX(slot) - APRON * spacing;
    const originZ = this.pool.slotOriginZ(slot) - APRON * spacing;

    u.origin.value.set(originX, originZ);
    u.spacing.value = spacing;
    // Erosion depth in metres is capped so a coarse tile does not carve a
    // 200 m canyon where its own children carve a 3 m gully.
    u.outScale.value = Math.min(spacing, EROSION_TUNING.REFERENCE_CELL_M);
    u.layer.value = slot;
    u.level.value = this.pool.levelOfSlot(slot);
    // Absolute world cell index of the scratch corner. The droplet hash and the
    // checkerboard phase both key off this, which is what makes two neighbouring
    // tiles simulate bit-identical droplets in their overlap.
    u.cellBaseX.value = this.pool.gxOfSlot(slot) * TILE_CELLS - APRON;
    u.cellBaseZ.value = this.pool.gzOfSlot(slot) * TILE_CELLS - APRON;
  }

  /** Run dispatch `k` of the current tile. Returns its estimated cost in ms. */
  private runStep(k: number): number {
    const u = this.program.uniforms;
    const kernels = this.program.kernels;

    let index = k;
    if (index === 0) {
      this.renderer.compute(kernels.seed, _dispatchScratch);
      return COST_SEED;
    }
    index -= 1;

    if (index < HYDRAULIC_STEPS) {
      const round = Math.floor(index / STEPS_PER_ROUND);
      const inRound = index % STEPS_PER_ROUND;
      const phase = Math.floor(inRound / STEPS_PER_PHASE);
      const inPhase = inRound % STEPS_PER_PHASE;

      if (inPhase < DROPLET_BANDS) {
        u.round.value = round;
        u.phase.value = phase;
        u.rowOffset.value = inPhase * BAND_GROUPS * WORKGROUP;
        // The last band is short when the scratch edge is not a multiple of the
        // band height; the kernel bounds-checks anyway, but dispatching fewer
        // groups is free.
        _dispatchBand[1] = Math.min(BAND_GROUPS, SCRATCH_GROUPS - inPhase * BAND_GROUPS);
        this.renderer.compute(kernels.droplet, _dispatchBand);
        return COST_DROPLET;
      }
      u.genClamp.value = EROSION_TUNING.GEN_CLAMP_CELLS;
      this.renderer.compute(kernels.apply, _dispatchScratch);
      return COST_APPLY;
    }
    index -= HYDRAULIC_STEPS;

    if (index < THERMAL_STEPS) {
      if (index % 2 === 0) {
        this.renderer.compute(kernels.thermal, _dispatchScratch);
        return COST_THERMAL;
      }
      u.genClamp.value = EROSION_TUNING.THERMAL_CLAMP_CELLS;
      this.renderer.compute(kernels.apply, _dispatchScratch);
      return COST_APPLY;
    }
    index -= THERMAL_STEPS;

    // Tail: smooth the erosion delta, write it back, blur the moisture in X,
    // then resolve the core into the array layers. Order matters — `blurX`
    // overwrites the scratch `smoothDelta` parked its result in.
    if (index === 0) {
      this.renderer.compute(kernels.smoothDelta, _dispatchScratch);
      return COST_SMOOTH;
    }
    if (index === 1) {
      this.renderer.compute(kernels.applySmooth, _dispatchScratch);
      return COST_SMOOTH;
    }
    if (index === 2) {
      this.renderer.compute(kernels.blurX, _dispatchScratch);
      return COST_BLUR;
    }
    this.renderer.compute(kernels.resolve, _dispatchCore);
    return COST_RESOLVE;
  }

  private finish(): void {
    this.pool.finishErosion(this.slot);
    this.slot = -1;
    this.step = 0;
    this.stats.completed++;
    if (this.stats.completed === 1) this.verifyFlowMap();
  }

  /**
   * Read back the resolve pass's own tally of non-zero flow texels.
   *
   * Fired once, after the first tile finishes. `stats.completed > 0` — what
   * this flag used to be — says the scheduler reached the last dispatch of a
   * job, which is a claim about control flow and stayed true while the flow
   * texture was empty. The two debug views gated on it then rendered a
   * confident nothing, and that is worse than rendering an error. So the flag
   * now waits for a number that came back off the GPU.
   *
   * A whole tile is 129² = 16,641 texels. Requiring a quarter of them is far
   * above anything a partial or misaddressed write could produce and far below
   * the ~100 % a healthy resolve gives, so the threshold does not need tuning.
   */
  private verifyFlowMap(): void {
    if (this.verifyRequested) return;
    this.verifyRequested = true;
    void this.renderer
      .getArrayBufferAsync(this.program.verifyAttribute)
      .then((buffer) => {
        this.stats.flowTexelsWritten = new Uint32Array(buffer)[0];
        this.verified = this.stats.flowTexelsWritten >= MIN_VERIFIED_TEXELS;
        if (!this.verified) {
          console.warn(
            `[aether] erosion resolve wrote ${this.stats.flowTexelsWritten} non-zero flow ` +
              `texels for a completed tile (expected ≥ ${MIN_VERIFIED_TEXELS}); ` +
              'flow and moisture debug views stay disabled',
          );
        }
      })
      .catch(() => {
        // Readback is a diagnostic, not a dependency. If it fails the flag
        // stays false and the two views stay locked, which is the safe way
        // round: an unverified map must not be shown as a result.
      });
  }

  /**
   * True once the GPU has confirmed the flow map holds data.
   *
   * Deliberately NOT `stats.completed > 0` — see `verifyFlowMap`.
   */
  get hasResults(): boolean {
    return this.verified;
  }

  dispose(): void {
    const k = this.program.kernels;
    k.seed.dispose();
    k.droplet.dispose();
    k.apply.dispose();
    k.thermal.dispose();
    k.smoothDelta.dispose();
    k.applySmooth.dispose();
    k.blurX.dispose();
    k.resolve.dispose();
  }
}
