/**
 * Compute-shader bake of the biome mask: one dispatch per frame, batched over
 * every tile that needs classifying.
 *
 * This runs on exactly the same shape as `HeightmapGenerator` — a parameter
 * buffer indexed by the dispatch's z axis, one submit for the whole batch — and
 * for the same reason: the classification is a function of a tile's own texels
 * and nothing else, so there is no reason to pay a submit per tile.
 *
 * The ordering constraint is the interesting part. A tile becomes visible the
 * frame its height bake is submitted, so its splat has to be written *in that
 * same frame* or the material samples whatever the previous tenant of that
 * layer left behind — which is not a subtle artefact, it paints a chunk of
 * meadow with a chunk of desert. `TerrainSystem` therefore calls `flush()`
 * immediately after `HeightmapGenerator.flush()`, before the render pass is
 * recorded, and the batch ceiling here is at least the bake's so a full bake
 * batch can never outrun its classification.
 *
 * Tiles are classified a second time when their erosion job completes. That
 * pass is genuinely different, not a repeat: the first classification sees a
 * zero moisture channel, the second sees the drainage network erosion carved,
 * and the difference is wet valley floors under dry slopes.
 */

import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  computeKernel,
  clamp,
  float,
  globalId,
  int,
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
import { buildBiomeSplat } from './shaders/splat';
import { TILE_TEXELS, type TilePool } from './TilePool';

const WORKGROUP = 8;
const GROUPS = Math.ceil(TILE_TEXELS / WORKGROUP);

/**
 * Upper bound on tiles per dispatch.
 *
 * Must not be lower than `HeightmapGenerator`'s batch: a frame that bakes N
 * tiles makes N tiles visible, and every one of them needs its splat in the
 * same frame. Kept equal so the two ceilings move together.
 */
const MAX_BATCH = 8;

/**
 * Assumed cost of classifying one 129² tile, milliseconds.
 *
 * TODO(quality): this is an estimate, not a measurement, and it is the one
 * number in this file I did not verify. It is set at half the height bake's
 * 0.1 ms on the reasoning that the classifier is one dispatch against the
 * bake's two and does comparable per-texel work — but the classifier evaluates
 * four fBm chains where the bake evaluates one height function, so it could
 * plausibly be higher. It is only used to bill the erosion pass for what
 * classification already spent out of the 1.0 ms generation slice, so being
 * wrong here makes erosion slightly greedier or slightly lazier, not incorrect.
 * The frame-differencing method that measured the bake and the erosion
 * dispatches needs a way to pin the classification rate, which the scheduler
 * does not currently expose.
 */
const EST_MS_PER_TILE = 0.05;

const _dispatch: number[] = [GROUPS, GROUPS, 1];

type ComputeKernelNode = ReturnType<typeof computeKernel>;

export class SplatGenerator {
  /** Tiles classified since boot, for the overlay. */
  classifiedTotal = 0;
  /** Tiles classified in the most recent frame. */
  classifiedLastFrame = 0;

  readonly estimatedMsPerTile = EST_MS_PER_TILE;

  private uSeed = uniform(1337);

  /** xy = tile world origin, z = texel spacing in metres, w = array layer. */
  private paramData = new Float32Array(MAX_BATCH * 4);
  private paramAttr: THREE.StorageInstancedBufferAttribute;

  private kernel: ComputeKernelNode;

  constructor(
    private renderer: THREE.WebGPURenderer,
    private pool: TilePool,
    private profiler: Profiler,
    seed: number,
  ) {
    this.uSeed.value = seed;

    this.paramAttr = new THREE.StorageInstancedBufferAttribute(this.paramData, 4);
    const tileParams = storage(this.paramAttr, 'vec4', MAX_BATCH);

    // Read-write views: the classifier only reads, but a plain sampled binding
    // of a storage texture would need a sampler and interpolation it does not
    // want — every lookup here is an exact texel.
    const heightRead = storageTexture(this.pool.heightArray).toReadWrite();
    const surfaceRead = storageTexture(this.pool.surfaceArray).toReadWrite();
    const flowRead = storageTexture(this.pool.flowArray).toReadWrite();

    const seedNode = this.uSeed;

    const classify = Fn(() => {
      const gx = globalId.x;
      const gy = globalId.y;
      const p = tileParams.element(globalId.z);

      If(gx.lessThan(uint(TILE_TEXELS)).and(gy.lessThan(uint(TILE_TEXELS))), () => {
        const layer = int(p.w);
        const spacing = p.z;
        const world = p.xy.add(vec2(float(gx), float(gy)).mul(spacing));

        const h = heightRead.depth(layer).load(uvec2(gx, gy)).x;
        const surf = surfaceRead.depth(layer).load(uvec2(gx, gy));
        const normal = vec3(surf.x, surf.y, surf.z);
        const slopeTan = surf.w;
        // `b` is the moisture channel of the flow map — see TilePool's
        // documentation of the packing. Zero on a tile that has not been eroded.
        const flowMoisture = flowRead.depth(layer).load(uvec2(gx, gy)).z;

        // --- curvature ------------------------------------------------
        // Five-point Laplacian, clamped to the tile. On the border ring the
        // clamped read makes the second difference vanish, which biases the
        // outer texel toward "flat" — one texel in 129, and the alternative is
        // an analytic re-evaluation that would disagree with the *eroded*
        // height the interior uses. Flat is the honest answer at a boundary
        // where the neighbour is unknown.
        const clampCoord = (v: ReturnType<typeof int>) =>
          clamp(v, int(0), int(TILE_TEXELS - 1));
        const at = (dx: number, dy: number) =>
          heightRead
            .depth(layer)
            .load(
              uvec2(
                uint(clampCoord(int(gx).add(int(dx)))),
                uint(clampCoord(int(gy).add(int(dy)))),
              ),
            ).x;

        const lap = at(-1, 0).add(at(1, 0)).add(at(0, -1)).add(at(0, 1)).sub(h.mul(4));
        // Divide by the cell size to get a dimensionless second difference,
        // then scale so that a 0.25 rise-over-run change across one cell
        // saturates. Without the normalisation the same landform would read as
        // strongly concave on a 2 m tile and flat on a 32 m one, and the biomes
        // would visibly change under the camera as it approached.
        const concavity = clamp(lap.div(spacing).mul(4), float(-1), float(1));

        const splat = buildBiomeSplat({
          world,
          height: h,
          normal,
          slopeTan,
          flowMoisture,
          concavity,
          seed: seedNode,
        });

        textureStore(
          storageTexture(this.pool.splat0Array).depth(layer),
          uvec2(gx, gy),
          clamp(splat.w0, vec4(0, 0, 0, 0), vec4(1, 1, 1, 1)),
        );
        textureStore(
          storageTexture(this.pool.splat1Array).depth(layer),
          uvec2(gx, gy),
          clamp(splat.w1, vec4(0, 0, 0, 0), vec4(1, 1, 1, 1)),
        );
      });
    });

    this.kernel = computeKernel(classify(), [WORKGROUP, WORKGROUP, 1]);
    this.kernel.setName('terrainSplat');
  }

  setSeed(seed: number): void {
    this.uSeed.value = seed;
  }

  /**
   * Classify up to `maxTiles` layers, unclassified ones first.
   *
   * Returns the number actually dispatched. Call after the height bake in the
   * same frame — see the file header for why that ordering is not optional.
   */
  flush(maxTiles: number): number {
    this.profiler.begin('terrain-splat');

    const limit = Math.min(maxTiles, MAX_BATCH);
    let count = 0;
    while (count < limit) {
      const slot = this.pool.takeSplatPending();
      if (slot < 0) break;

      const o = count * 4;
      this.paramData[o] = this.pool.slotOriginX(slot);
      this.paramData[o + 1] = this.pool.slotOriginZ(slot);
      this.paramData[o + 2] = this.pool.slotSize(slot) / (TILE_TEXELS - 1);
      this.paramData[o + 3] = slot;
      count++;
    }

    if (count > 0) {
      this.paramAttr.needsUpdate = true;
      _dispatch[2] = count;
      this.renderer.compute(this.kernel, _dispatch);
      this.classifiedTotal += count;
    }

    this.classifiedLastFrame = count;
    this.profiler.end('terrain-splat');
    return count;
  }

  dispose(): void {
    this.kernel.dispose();
  }
}
