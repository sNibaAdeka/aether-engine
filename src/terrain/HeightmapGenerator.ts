/**
 * Compute-shader bake of terrain tiles: one dispatch pair per frame, batched
 * over every tile the streamer asked for.
 *
 * This is the whole point of phase 1b. Before it, the vertex shader evaluated
 * the height function five times per vertex — once for the position and four
 * more for central-difference normals — every frame, for every one of ~156k
 * vertices. The height function is ~32 Perlin lookups deep, so that was ~780k
 * evaluations per frame for a surface that does not change. Baking moves the
 * cost to once per tile: ~16.6k evaluations, amortised over the hundreds of
 * frames the tile stays resident, and the vertex shader degenerates to four
 * texture fetches.
 *
 * Two dispatches, not one, because the normal pass has to read the *baked*
 * heights — the same values the vertex shader will sample — rather than
 * re-deriving them analytically. That is what keeps normals attached to the
 * surface actually rendered, and it is the only formulation that still works
 * once erosion (phase 1b, task 2) starts moving the heightfield.
 */

import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  computeKernel,
  float,
  globalId,
  int,
  length,
  normalize,
  storage,
  storageTexture,
  textureStore,
  uint,
  uniform,
  uvec2,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

import type { Profiler } from '@/engine/Profiler';
import { fnTerrainHeight } from './shaders/heightfield.wgsl';
import { TILE_TEXELS, type TilePool } from './TilePool';

/** Workgroup edge. 8×8 = 64 invocations, comfortably inside every device's 256. */
const WORKGROUP = 8;
/**
 * 129 is not a multiple of 8, so the dispatch covers 136×136 and the kernel
 * discards the 15 % overhang. Padding the tile to 136 would waste more memory
 * than the discarded threads cost.
 */
const GROUPS = Math.ceil(TILE_TEXELS / WORKGROUP);

/**
 * Upper bound on tiles per dispatch. The quality presets ask for 1–3; the
 * headroom exists so the root bake at init and any future burst mode do not
 * need a second code path.
 */
const MAX_BATCH = 8;

/**
 * Cost of baking one 129² tile, both passes, on this machine (Apple M-series).
 *
 * Measured by differencing frame time with the frame limiter off, between the
 * settled state and a forced continuous bake: 0.20 ms per frame for two tiles.
 * three's own COMPUTE timestamp reports 0.62 ms for the same work, but phase 0
 * already established that those timestamps are uncalibrated on this backend,
 * and 0.62 ms cannot be reconciled with an 833 FPS frame time.
 *
 * TODO(quality): this is an open-loop estimate, not a feedback controller. The
 * ceiling below divides the budget by it to decide how many tiles may go in a
 * frame, so on much slower hardware the generation pass will overshoot its
 * 1.0 ms slice until someone re-measures. A closed loop needs per-pass GPU
 * timestamps, which three only exposes as one aggregate COMPUTE query.
 */
const EST_MS_PER_TILE = 0.1;

/** Reused dispatch descriptor — `renderer.compute` must not allocate per frame. */
const _dispatch: number[] = [GROUPS, GROUPS, 1];

/**
 * three r185 does not re-export `ComputeNode` from `three/webgpu`, so the type
 * is taken from the factory that produces it.
 */
type ComputeKernelNode = ReturnType<typeof computeKernel>;

export class HeightmapGenerator {
  /** GPU milliseconds the bake may consume in one frame. */
  gpuBudgetMs = 1.0;

  /** Exposed so the erosion pass can bill itself against what the bake spent. */
  readonly estimatedMsPerTile = EST_MS_PER_TILE;

  /** Tiles baked since boot, for the overlay. */
  bakedTotal = 0;
  /** Tiles baked in the most recent frame. */
  bakedLastFrame = 0;

  private uSeed = uniform(1337);

  /**
   * Per-tile parameters, indexed by the dispatch's z axis: xy = tile world
   * origin, z = texel spacing in metres, w = destination array layer.
   *
   * A storage buffer rather than a uniform mutated between dispatches: one
   * buffer means one dispatch for the whole batch, one bind group and one
   * submit. Mutating a uniform between `compute()` calls also works (queue
   * order guarantees it), but it costs a submit per tile.
   */
  private paramData = new Float32Array(MAX_BATCH * 4);
  private paramAttr: THREE.StorageInstancedBufferAttribute;

  private bakeNode: ComputeKernelNode;
  private surfaceNode: ComputeKernelNode;

  constructor(
    private renderer: THREE.WebGPURenderer,
    private pool: TilePool,
    private profiler: Profiler,
    seed: number,
  ) {
    this.uSeed.value = seed;

    this.paramAttr = new THREE.StorageInstancedBufferAttribute(this.paramData, 4);
    const tileParams = storage(this.paramAttr, 'vec4', MAX_BATCH);

    const seedNode = this.uSeed;

    // --- pass 1: height -------------------------------------------------
    const bake = Fn(() => {
      const gx = globalId.x;
      const gy = globalId.y;
      const p = tileParams.element(globalId.z);

      If(gx.lessThan(uint(TILE_TEXELS)).and(gy.lessThan(uint(TILE_TEXELS))), () => {
        // Shared-edge convention: texel i sits exactly at origin + i*spacing,
        // so the last texel of a tile and the first of its neighbour are the
        // same world point and evaluate to the same height. No gutter needed.
        const world = p.xy.add(vec2(float(gx), float(gy)).mul(p.z));
        const h = fnTerrainHeight({ pos: world, seed: seedNode });
        // `.depth()` must be applied to the target *before* textureStore():
        // textureStore pushes the node onto the shader stack immediately while
        // .depth() returns a clone, so the reverse order silently drops the
        // array index and emits a 2-argument store.
        textureStore(
          storageTexture(this.pool.heightArray).depth(int(p.w)),
          uvec2(gx, gy),
          vec4(h, 0, 0, 1),
        );
      });
    });

    this.bakeNode = computeKernel(bake(), [WORKGROUP, WORKGROUP, 1]);
    this.bakeNode.setName('terrainBakeHeight');

    // --- pass 2: normal + slope ------------------------------------------
    const heightRead = storageTexture(this.pool.heightArray).toReadWrite();

    const surface = Fn(() => {
      const gx = globalId.x;
      const gy = globalId.y;
      const p = tileParams.element(globalId.z);

      If(gx.lessThan(uint(TILE_TEXELS)).and(gy.lessThan(uint(TILE_TEXELS))), () => {
        const layer = int(p.w);
        const origin = p.xy;
        const spacing = p.z;
        const ix = int(gx);
        const iy = int(gy);

        // Reads the baked map inside the tile. On the border ring — 3 % of
        // texels — the neighbour texel belongs to the adjacent tile, which may
        // not be resident, so fall back to the analytic height at that world
        // position. Pre-erosion that is bit-identical to what the neighbour
        // tile holds, so the normals are seamless across tile edges.
        // This pass only ever produces the *un-eroded* normals a tile shows
        // while it waits its turn in the erosion queue. `Erosion.resolve`
        // recomputes normals from the eroded scratch domain, which has an apron
        // and therefore real neighbours on the border ring — so the analytic
        // fallback here is never what a finished tile ends up carrying.
        const readHeight = (dx: number, dy: number) => {
          const sx = ix.add(dx);
          const sy = iy.add(dy);
          const out = float(0).toVar();
          If(
            sx
              .greaterThanEqual(int(0))
              .and(sx.lessThan(int(TILE_TEXELS)))
              .and(sy.greaterThanEqual(int(0)))
              .and(sy.lessThan(int(TILE_TEXELS))),
            () => {
              out.assign(heightRead.depth(layer).load(uvec2(uint(sx), uint(sy))).x);
            },
          ).Else(() => {
            const world = origin.add(vec2(float(sx), float(sy)).mul(spacing));
            out.assign(fnTerrainHeight({ pos: world, seed: seedNode }));
          });
          return out;
        };

        const hL = readHeight(-1, 0);
        const hR = readHeight(1, 0);
        const hD = readHeight(0, -1);
        const hU = readHeight(0, 1);

        const inv = float(1).div(spacing.mul(2));
        const dhdx = hR.sub(hL).mul(inv);
        const dhdz = hU.sub(hD).mul(inv);

        // Surface normal of h(x,z): (-∂h/∂x, 1, -∂h/∂z), normalised.
        const n = normalize(vec3(dhdx.mul(-1), float(1), dhdz.mul(-1)));
        // Slope as |∇h| — metres of rise per metre of run, i.e. tan(angle).
        // Stored rather than the angle because material blending and the talus
        // threshold both want the tangent directly.
        const slope = length(vec2(dhdx, dhdz));

        textureStore(
          storageTexture(this.pool.surfaceArray).depth(layer),
          uvec2(gx, gy),
          vec4(n, slope),
        );
      });
    });

    this.surfaceNode = computeKernel(surface(), [WORKGROUP, WORKGROUP, 1]);
    this.surfaceNode.setName('terrainBakeSurface');
  }

  setSeed(seed: number): void {
    this.uSeed.value = seed;
  }

  /**
   * How many tiles may be baked this frame.
   *
   * Two independent limits: the quality preset's `tileBudget` (a streaming
   * smoothness knob) and a hard GPU-time ceiling derived from the 1.0 ms
   * generation slice in the frame budget. The lower wins, and it is never zero
   * — one tile per frame still makes progress, and stalling the streamer
   * entirely just moves the stutter somewhere less visible.
   */
  tilesThisFrame(tileBudget: number): number {
    const fromGpu = Math.floor(this.gpuBudgetMs / EST_MS_PER_TILE);
    return Math.max(1, Math.min(MAX_BATCH, Math.min(tileBudget, fromGpu)));
  }

  /**
   * Bake up to `maxTiles` of the pool's pending tiles, highest priority first.
   *
   * Tiles are marked ready as soon as the dispatch is submitted, not when the
   * GPU finishes: both compute passes are submitted from `update`, before the
   * frame graph records the render pass, and WebGPU executes queue submissions
   * in order. So a tile baked this frame is legitimately safe to sample in the
   * same frame's draw.
   */
  flush(maxTiles: number): number {
    // CPU submit cost only. The GPU cost of the bake lands in the profiler's
    // 'terrain-generate' series, fed from the COMPUTE timestamp query — three
    // exposes one aggregate query per frame, not one per pass, and terrain
    // baking is the engine's only compute work.
    this.profiler.begin('terrain-stream');

    // The batch is bounded by the parameter buffer, not by the caller's word.
    const limit = Math.min(maxTiles, MAX_BATCH);
    let count = 0;
    while (count < limit) {
      const slot = this.pool.takeBestPending();
      if (slot < 0) break;

      const o = count * 4;
      this.paramData[o] = this.pool.slotOriginX(slot);
      this.paramData[o + 1] = this.pool.slotOriginZ(slot);
      this.paramData[o + 2] = this.pool.slotSize(slot) / (TILE_TEXELS - 1);
      this.paramData[o + 3] = slot;

      this.pool.markReady(slot);
      count++;
    }

    if (count > 0) {
      this.paramAttr.needsUpdate = true;
      _dispatch[2] = count;
      // Separate submissions rather than `compute([bake, surface])`: the second
      // pass reads what the first wrote, and one submit per pass makes that
      // dependency a queue-order guarantee instead of a bet on intra-pass
      // hazard tracking.
      this.renderer.compute(this.bakeNode, _dispatch);
      this.renderer.compute(this.surfaceNode, _dispatch);
      this.bakedTotal += count;
    }

    this.bakedLastFrame = count;
    this.profiler.end('terrain-stream');
    return count;
  }

  dispose(): void {
    this.bakeNode.dispose();
    this.surfaceNode.dispose();
  }
}
