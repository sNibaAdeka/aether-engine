/**
 * CDLOD quadtree selection (Filip Strugar, "Continuous Distance-Dependent LOD").
 *
 * Why CDLOD rather than geometry clipmaps: it morphs between levels *in the
 * vertex shader*, so there are no T-junctions to stitch and no constraint that
 * the camera sit at the centre of a ring. The morph is not optional polish —
 * without it every level transition pops, and the traditional "fix" is to hide
 * the popping behind fog, which then has to get thicker every phase until the
 * world is invisible.
 *
 * The traversal is CPU-side and cheap: a few hundred nodes, no allocation.
 * Selected nodes are written into a flat Float32Array laid out for direct
 * upload as instance data.
 */

import type { Frustum } from './Frustum';

/** Floats per selected node in the instance buffer. Four vec4 attributes. */
export const NODE_STRIDE = 16;
/*
  layout:
   0,1  node origin XZ (world metres, corner)
   2    node size (world metres)
   3    lod level (0 = finest)
   4    morph range start (distance at which morphing begins)
   5    morph range end   (distance at which morph completes)
   6    heightmap layer for the own-level tap
   7    heightmap layer for the parent-level tap
   8,9  uv offset of this node inside the own-level tap's tile
   10   uv scale  of this node inside the own-level tap's tile
   11   1 when either tap fell back to a coarser ancestor, else 0
   12,13 uv offset inside the parent tap's tile
   14   uv scale  inside the parent tap's tile
   15   reserved

  Slots 6–15 are filled by `TileManager`, not here: the quadtree decides *what*
  is visible and the tile streamer decides *what data it can be drawn with*, and
  the second answer changes every frame while the first does not.

  Two reserved floats were not enough. The vertex shader samples two tiles — its
  own level and its parent's, so the CDLOD morph can blend heights as well as
  positions — and neither tile is guaranteed to be the node's exact extent, so
  each tap needs a layer plus a uv sub-rectangle.
*/

export interface QuadtreeConfig {
  /** Half-extent of the root node, metres. Root covers [-rootSize, +rootSize]. */
  rootSize: number;
  /** Maximum subdivision depth. Leaf size = rootSize*2 / 2^maxDepth. */
  maxDepth: number;
  /** Larger = subdivide sooner = more triangles. ~2.5 is the usual sweet spot. */
  lodFactor: number;
  /** Nodes beyond this distance are culled entirely. */
  drawDistance: number;
  /** Vertices per patch edge (33 → 32 quads). */
  patchResolution: number;
  /**
   * Fraction of a level's distance band spent morphing to the parent.
   * 0.3 means the last 30% of the band is a blend. Too small and the morph is
   * abrupt; too large and half the terrain is permanently mid-morph.
   */
  morphFraction: number;
}

export const DEFAULT_QUADTREE: QuadtreeConfig = {
  rootSize: 32768,
  maxDepth: 9,
  lodFactor: 2.5,
  drawDistance: 12000,
  patchResolution: 33,
  morphFraction: 0.3,
};

export interface SelectionStats {
  nodesVisited: number;
  nodesSelected: number;
  nodesCulled: number;
  trianglesEstimate: number;
  maxLevelUsed: number;
}

export class Quadtree {
  config: QuadtreeConfig;

  /** Flat instance data for the selected nodes. Reused across frames. */
  readonly instanceData: Float32Array;
  /** Number of nodes currently selected. */
  selectedCount = 0;

  readonly stats: SelectionStats = {
    nodesVisited: 0,
    nodesSelected: 0,
    nodesCulled: 0,
    trianglesEstimate: 0,
    maxLevelUsed: 0,
  };

  /** Per-level distance thresholds, index 0 = finest. Recomputed on config change. */
  private lodRanges: Float64Array;
  private maxNodes: number;

  constructor(config: Partial<QuadtreeConfig> = {}) {
    this.config = { ...DEFAULT_QUADTREE, ...config };
    // Generous upper bound; selection also hard-stops at this many nodes.
    this.maxNodes = 4096;
    this.instanceData = new Float32Array(this.maxNodes * NODE_STRIDE);
    this.lodRanges = new Float64Array(this.config.maxDepth + 1);
    this.computeLodRanges();
  }

  setConfig(patch: Partial<QuadtreeConfig>): void {
    Object.assign(this.config, patch);
    if (this.lodRanges.length !== this.config.maxDepth + 1) {
      this.lodRanges = new Float64Array(this.config.maxDepth + 1);
    }
    this.computeLodRanges();
  }

  /**
   * Distance at which each level stops being acceptable.
   *
   * Level 0 (leaf) covers `leafSize * lodFactor` metres; each coarser level
   * doubles. Doubling exactly matches the node size doubling, so screen-space
   * triangle density stays roughly constant with distance — which is the whole
   * point of a distance-dependent LOD.
   */
  private computeLodRanges(): void {
    const { rootSize, maxDepth, lodFactor } = this.config;
    const leafSize = (rootSize * 2) / Math.pow(2, maxDepth);
    for (let level = 0; level <= maxDepth; level++) {
      this.lodRanges[level] = leafSize * Math.pow(2, level) * lodFactor;
    }
  }

  /** Leaf node edge length in metres. */
  get leafSize(): number {
    return (this.config.rootSize * 2) / Math.pow(2, this.config.maxDepth);
  }

  /**
   * Select the visible node set for this camera position.
   *
   * `frustum` may be null to skip culling (used by the "freeze culling" debug
   * toggle, so you can fly outside the frozen frustum and see what was kept).
   */
  select(camX: number, camY: number, camZ: number, frustum: Frustum | null): number {
    this.selectedCount = 0;
    this.stats.nodesVisited = 0;
    this.stats.nodesSelected = 0;
    this.stats.nodesCulled = 0;
    this.stats.maxLevelUsed = 0;

    const { rootSize, maxDepth } = this.config;
    this.selectNode(-rootSize, -rootSize, rootSize * 2, maxDepth, camX, camY, camZ, frustum);

    const quads = this.config.patchResolution - 1;
    this.stats.nodesSelected = this.selectedCount;
    this.stats.trianglesEstimate = this.selectedCount * quads * quads * 2;
    return this.selectedCount;
  }

  /**
   * Recursive descent.
   *
   * A node is emitted when the camera is far enough away that this level's
   * triangle density is sufficient; otherwise it splits. Emitting at the
   * *coarsest acceptable* level rather than always descending is what keeps the
   * node count logarithmic in draw distance instead of quadratic.
   */
  private selectNode(
    x: number,
    z: number,
    size: number,
    level: number,
    camX: number,
    camY: number,
    camZ: number,
    frustum: Frustum | null,
  ): void {
    if (this.selectedCount >= this.maxNodes) return;
    this.stats.nodesVisited++;

    const half = size * 0.5;
    const cx = x + half;
    const cz = z + half;

    // Horizontal distance to the node's nearest edge — using the centre makes
    // large nodes subdivide far too eagerly at grazing angles.
    const dx = Math.max(0, Math.abs(camX - cx) - half);
    const dz = Math.max(0, Math.abs(camZ - cz) - half);
    const distFlat = Math.sqrt(dx * dx + dz * dz);

    if (distFlat > this.config.drawDistance) {
      this.stats.nodesCulled++;
      return;
    }

    if (frustum && !frustum.intersectsTerrainNode(x, z, size)) {
      this.stats.nodesCulled++;
      return;
    }

    // Include the vertical component so that flying high above the terrain
    // correctly coarsens it — otherwise a top-down view selects leaf nodes for
    // the entire visible area and the frame budget evaporates.
    const dist = Math.sqrt(distFlat * distFlat + camY * camY * 0.25);

    const range = this.lodRanges[level];

    if (level === 0 || dist > range) {
      this.emit(x, z, size, level, dist);
      return;
    }

    // Split.
    const childSize = half;
    const childLevel = level - 1;
    this.selectNode(x, z, childSize, childLevel, camX, camY, camZ, frustum);
    this.selectNode(x + childSize, z, childSize, childLevel, camX, camY, camZ, frustum);
    this.selectNode(x, z + childSize, childSize, childLevel, camX, camY, camZ, frustum);
    this.selectNode(x + childSize, z + childSize, childSize, childLevel, camX, camY, camZ, frustum);
  }

  private emit(x: number, z: number, size: number, level: number, _dist: number): void {
    const i = this.selectedCount * NODE_STRIDE;
    const d = this.instanceData;

    // Morph band.
    //
    // Selection emits level L when `dist > lodRanges[L]`, and the parent takes
    // over at `dist > lodRanges[L+1]`. So level L is *in use* across
    // (lodRanges[L], lodRanges[L+1]], and the morph has to run across the tail
    // of that band — not the tail of lodRanges[L], which lies entirely below
    // the distance at which this node is ever selected.
    //
    // Getting this wrong pins morphK at 1 for every patch in the world: the
    // terrain silently renders one level coarser than it selected, and the
    // morph does no work at all. It looks completely fine in a shaded render,
    // which is exactly why the mode-5 visualisation exists.
    const bandStart = level === 0 ? 0 : this.lodRanges[level];
    const bandEnd =
      level + 1 < this.lodRanges.length ? this.lodRanges[level + 1] : this.lodRanges[level] * 2;
    const morphStart = bandStart + (bandEnd - bandStart) * (1 - this.config.morphFraction);

    d[i] = x;
    d[i + 1] = z;
    d[i + 2] = size;
    d[i + 3] = level;
    d[i + 4] = morphStart;
    d[i + 5] = bandEnd;
    // Slots 6..15 belong to TileManager and are overwritten every frame after
    // selection; zeroing them here would only cost a write.

    this.selectedCount++;
    if (level > this.stats.maxLevelUsed) this.stats.maxLevelUsed = level;
  }

  /** Distance band for a level — exposed for the debug overlay and tests. */
  lodRange(level: number): number {
    return this.lodRanges[Math.max(0, Math.min(this.lodRanges.length - 1, level))];
  }
}
