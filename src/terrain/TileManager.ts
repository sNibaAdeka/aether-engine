/**
 * Streaming policy for the tile pool: what to bake, in what order, and what a
 * node draws with while it waits.
 *
 * Split out from `TilePool` on purpose. The pool is an allocator — it knows
 * about layers, keys and recency and nothing else. This is where the decisions
 * live: which tiles the visible set needs, how urgent each one is, and how a
 * node that has no tile of its own still ends up drawing a surface.
 *
 * ── How the "never a hole" guarantee actually works ──────────────────────────
 * Every selected node writes two texture taps into its instance record: one for
 * its own level and one for its parent's, because the CDLOD morph blends them.
 * Both taps come from `TilePool.resolve`, which walks *up* the ancestor chain
 * until it finds a baked tile and returns a uv sub-rectangle into it. The root
 * tile is pinned and baked during `TerrainSystem.init`, before the first frame
 * is rendered, and every node in the world is a descendant of the root — so the
 * walk always terminates on real data. A node with no tile of its own therefore
 * renders a coarser version of itself, never a flat plane and never a gap.
 *
 * The visible cost of that guarantee is bounded by the priority order below:
 * near tiles first, and a tile that is somebody's parent outranks a tile that is
 * only somebody's own level, because one parent unblocks four children.
 */

import { NODE_STRIDE } from './Quadtree';
import type { TilePool } from './TilePool';

/**
 * Weight applied to a tile requested as a parent tap.
 *
 * A parent is a hard dependency of four children — until it exists, none of them
 * can draw at their own level — so it is worth baking ahead of a child at the
 * same distance.
 */
const PARENT_URGENCY = 4;

export interface TileStreamStats {
  /** Nodes drawing at their own level this frame. */
  exact: number;
  /** Nodes falling back to a coarser ancestor because their tile is not baked. */
  fallback: number;
  /** Tiles requested but refused a layer (pool full of recently-used tiles). */
  starved: number;
}

export class TileManager {
  readonly stats: TileStreamStats = { exact: 0, fallback: 0, starved: 0 };

  constructor(private pool: TilePool) {}

  /**
   * Pass 1 — ask the pool for every tile the selected set will sample.
   *
   * Runs before the bake so that tiles requested this frame can be baked this
   * frame. Priority is 1/distance, measured to the node's nearest edge exactly
   * as the quadtree measures it for LOD selection, so the two agree about what
   * "close" means.
   *
   * On-screen weighting is deliberately absent: `Quadtree.select` has already
   * frustum-culled, so every node reaching here is on screen. The one exception
   * is the "freeze culling" debug toggle, where seeing the frozen set stream in
   * at its real priorities is the point.
   */
  request(data: Float32Array, count: number, camX: number, camZ: number): void {
    const pool = this.pool;
    for (let i = 0; i < count; i++) {
      const o = i * NODE_STRIDE;
      const originX = data[o];
      const originZ = data[o + 1];
      const size = data[o + 2];
      const level = data[o + 3];

      const half = size * 0.5;
      const dx = Math.max(0, Math.abs(camX - (originX + half)) - half);
      const dz = Math.max(0, Math.abs(camZ - (originZ + half)) - half);
      const priority = 1 / (1 + Math.sqrt(dx * dx + dz * dz));

      const gx = pool.gridIndex(level, originX);
      const gz = pool.gridIndex(level, originZ);

      pool.request(level, gx, gz, priority);
      pool.request(level + 1, gx >> 1, gz >> 1, priority * PARENT_URGENCY);
    }
  }

  /**
   * Pass 2 — resolve both taps and write them into the instance buffer.
   *
   * Runs *after* the bake so a tile finished this frame is used this frame.
   *
   * The uv transform is the same for both taps: the node occupies the
   * rectangle [offset, offset + scale] of whichever ancestor tile it ends up
   * sampling, since patch-local p maps to ancestor-local
   * (nodeOrigin + p*nodeSize − ancOrigin) / ancSize. When the node's own tile
   * is missing, both taps collapse onto the same ancestor and the height mix
   * becomes a no-op — an honest loss of detail rather than a blend between two
   * unrelated surfaces.
   */
  resolve(data: Float32Array, count: number): void {
    const pool = this.pool;
    let exact = 0;
    let fallback = 0;

    for (let i = 0; i < count; i++) {
      const o = i * NODE_STRIDE;
      const originX = data[o];
      const originZ = data[o + 1];
      const size = data[o + 2];
      const level = data[o + 3];

      const gx = pool.gridIndex(level, originX);
      const gz = pool.gridIndex(level, originZ);

      const child = pool.resolve(level, gx, gz);
      const childLayer = child.layer;
      const childScale = size / child.size;
      const childOffX = (originX - child.originX) / child.size;
      const childOffZ = (originZ - child.originZ) / child.size;
      const drawnAtOwnLevel = child.level === level;

      const parent = pool.resolve(level + 1, gx >> 1, gz >> 1);
      const parentLayer = parent.layer;
      const parentScale = size / parent.size;
      const parentOffX = (originX - parent.originX) / parent.size;
      const parentOffZ = (originZ - parent.originZ) / parent.size;

      data[o + 6] = childLayer;
      data[o + 7] = parentLayer;
      data[o + 8] = childOffX;
      data[o + 9] = childOffZ;
      data[o + 10] = childScale;
      data[o + 11] = drawnAtOwnLevel ? 0 : 1;
      data[o + 12] = parentOffX;
      data[o + 13] = parentOffZ;
      data[o + 14] = parentScale;
      data[o + 15] = 0;

      if (drawnAtOwnLevel) exact++;
      else fallback++;
    }

    this.stats.exact = exact;
    this.stats.fallback = fallback;
    this.stats.starved = pool.stats.starved;
  }
}
