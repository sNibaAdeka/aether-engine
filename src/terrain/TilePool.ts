/**
 * LRU pool of terrain tiles, stored as layers of five array textures: height,
 * surface (normal + slope), flow (erosion), and two rgba8 biome splat maps.
 *
 * A tile is the baked surface of exactly one quadtree node, identified by
 * (level, gx, gz). Because node sizes are powers of two and the world origin is
 * fixed, that triple is a complete address: two nodes with the same triple
 * always cover the same world rectangle, so a cached layer is reusable across
 * frames, across camera moves, and between a node and its own children.
 *
 * The pool answers two questions the renderer cannot proceed without:
 *   - which layer holds this tile (or, if it is not baked yet, which coarser
 *     ancestor can stand in for it — `resolve`);
 *   - is this node safe to draw at its own level (`isDrawable`), which requires
 *     the *parent* tile too, because the CDLOD morph blends both.
 *
 * The root tile is pinned and baked during `init`, and that is the whole reason
 * `resolve` can promise to always return something: every node in the world is
 * a descendant of the root, so the walk up the ancestor chain terminates on a
 * resident layer no matter how cold the pool is.
 */

import * as THREE from 'three/webgpu';
import { ResourceManager } from '@/engine/ResourceManager';

/**
 * Texels along one tile edge.
 *
 * 129 = 2^7 + 1, and the "+1" is load-bearing. Under the shared-edge convention
 * (texel i sits at patch-local i/(N-1)) a tile spans N-1 = 128 cells, so:
 *
 *  - Adjacent tiles at the same level share their edge texels *by world
 *    position*. Both evaluate the same deterministic height there, so the
 *    values agree exactly and the seam is C0 with no gutter and no neighbour
 *    lookup. A cell-centred convention (uv = local) instead leaves a
 *    size/(2N) strip at each border with nothing to interpolate toward, and
 *    clamp-to-edge then extrapolates two different values on the two sides.
 *  - 128 cells over a 32-quad patch is exactly 4 texels per quad, so every
 *    patch vertex lands on a texel centre and the finest LOD reads baked
 *    heights with zero filter error. 128 texels (127 cells) would give
 *    3.97 texels per quad and blur every vertex against the wrong lattice.
 *  - Cell spacing is size/128 = a power of two for every node size, so texel
 *    world positions are exact in f32 and tile edges round-trip exactly.
 *
 * The spec asks for 256². See `sizeLayers()` for why the memory budget and the
 * measured working set force 128 cells instead.
 */
export const TILE_TEXELS = 129;
/** Cells (quads) along one tile edge. */
export const TILE_CELLS = TILE_TEXELS - 1;

/**
 * uv mapping for the shared-edge convention.
 *
 * WebGPU resolves uv to the continuous texel index t = u*N - 0.5, and we want
 * patch-local p to land on texel p*(N-1). Solving gives uv = (p*(N-1) + 0.5)/N.
 */
export const TILE_UV_SCALE = TILE_CELLS / TILE_TEXELS;
export const TILE_UV_BIAS = 0.5 / TILE_TEXELS;

const STATE_FREE = 0;
const STATE_PENDING = 1;
const STATE_READY = 2;

/**
 * Erosion job state, tracked per layer.
 *
 * Separate from the bake state because the two are genuinely independent: a
 * tile is drawable as soon as its analytic bake is submitted, and gains its
 * channels tens of frames later. Layers in `EROSION_RUNNING` are excluded from
 * eviction — abandoning a half-finished job wastes everything already spent on
 * it, and one layer out of 256 is a cheap reservation.
 */
const EROSION_NONE = 0;
const EROSION_RUNNING = 1;
const EROSION_DONE = 2;

/**
 * Splat state, tracked per layer.
 *
 * Three states rather than a dirty bit because the two reasons a layer needs
 * classifying are not equally urgent. `NONE` means the layer holds a tile whose
 * splat has never been computed — until it is, the material would read the
 * *previous tenant's* biome weights, which paints one part of the world with
 * another part's ground. `STALE` means the weights are correct but predate
 * erosion, so the moisture channel they used was zero; that is a quality
 * refinement, not a correctness problem, and it can wait behind every `NONE`.
 */
const SPLAT_NONE = 0;
const SPLAT_FRESH = 1;
const SPLAT_STALE = 2;

/**
 * A layer whose last use is older than this may be recycled. Two seconds is
 * roughly a 180° turn plus a look back at 60 FPS — short enough to keep the
 * pool small, long enough that glancing away and back does not re-bake.
 */
const EVICT_AGE_SECONDS = 2;

/**
 * Bytes per layer: r32float height + two rgba16float auxiliaries + two rgba8
 * splat maps.
 */
const BYTES_PER_LAYER = TILE_TEXELS * TILE_TEXELS * (4 + 8 + 8 + 4 + 4);

export interface TilePoolOptions {
  /** Total GPU memory the three arrays may occupy, megabytes. */
  memoryBudgetMB: number;
  /** Hard ceiling from the device's `maxTextureArrayLayers`. */
  maxLayers: number;
}

export const DEFAULT_TILE_POOL: TilePoolOptions = {
  memoryBudgetMB: 160,
  maxLayers: 256,
};

/** Resolution of a tile request. Reused — copy the fields, do not retain it. */
export interface TileRef {
  level: number;
  gx: number;
  gz: number;
  /** Array layer holding this tile. */
  layer: number;
  /** World-space corner of the tile that `layer` actually holds. */
  originX: number;
  originZ: number;
  /** World-space edge length of that tile. */
  size: number;
}

export interface TilePoolStats {
  layers: number;
  ready: number;
  pending: number;
  /** Requests that found no free or evictable layer this frame. */
  starved: number;
  /** Layers whose erosion pass has completed. */
  eroded: number;
  /** Ready layers whose biome weights have been computed for their own tile. */
  splatted: number;
  megabytes: number;
}

/**
 * Layer count from the memory budget.
 *
 * Arithmetic, at 129² texels:
 *   height   r32float     129·129·4  =  66,564 B
 *   surface  rgba16float  129·129·8  = 133,128 B   (normal.xyz, slope)
 *   flow     rgba16float  129·129·8  = 133,128 B   (filled by the erosion pass)
 *   splat0   rgba8        129·129·4  =  66,564 B   (phase 2 biome weights)
 *   splat1   rgba8        129·129·4  =  66,564 B
 *   total per layer                  = 465,948 B  ≈ 0.444 MB
 *
 * 160 MB / 0.444 MB = 360 layers, so the budget is not the binding constraint —
 * WebGPU's default `maxTextureArrayLayers` of 256 is. 256 layers cost 113.8 MB,
 * and the measured working set is 143–215 distinct tiles
 * (own level + parent level, four camera poses at the `high` preset), so 256
 * leaves ~20 % headroom for churn while the camera moves.
 *
 * This is also why the tile is 129² and not 257². At 257² a layer costs
 * 1.77 MB, the 160 MB budget buys 90 layers, and 90 < 215: the pool would
 * evict tiles that are on screen *this frame*, and every node would render off
 * a coarse ancestor permanently. Sharing one 257² tile between a 2×2 group of
 * nodes gives each node the same 129×129 texel window for the same bytes, so
 * there is no packing that escapes the trade — at this budget and this working
 * set, 128 cells per node is what the memory buys.
 */
function sizeLayers(opts: TilePoolOptions): number {
  const fromBudget = Math.floor((opts.memoryBudgetMB * 1024 * 1024) / BYTES_PER_LAYER);
  return Math.max(32, Math.min(fromBudget, opts.maxLayers));
}

/** Reused by `resolve` — module scope because `update` must not allocate. */
const _ref: TileRef = { level: 0, gx: 0, gz: 0, layer: -1, originX: 0, originZ: 0, size: 0 };

export class TilePool {
  /** r32float. One texel = one height in metres. */
  readonly heightArray: THREE.StorageArrayTexture;
  /** rgba16float. xyz = world normal, w = slope as |∇h| (metres per metre). */
  readonly surfaceArray: THREE.StorageArrayTexture;
  /**
   * rgba16float, written by the erosion pass. This is a phase 6/7 deliverable,
   * not a debug aid, so the units are fixed and documented here:
   *
   *   r  flow accumulation  saturate(log2(1 + W) / 10), where W is water-steps
   *                         per cell per droplet round. Full scale at W = 1023.
   *                         Log because drainage spans orders of magnitude; the
   *                         normalisation constant is global, never per-tile.
   *   g  flow direction     atan2(fz, fx) / 2π + 0.5, water-weighted mean of the
   *                         directions droplets actually left the cell in. Do
   *                         not bilinear-filter it: it wraps.
   *   b  moisture           same normalisation as r, applied to flow blurred by
   *                         a σ = 4 cell Gaussian. The hydrological term only —
   *                         phase 7 combines it with slope, altitude and climate.
   *   a  cut / fill         (h_eroded − h_analytic) / 64 + 0.5, i.e. ±32 m
   *                         signed. soilDepth = max(0, (a − 0.5)·64) metres,
   *                         incision = max(0, (0.5 − a)·64).
   *
   * A layer holds zeros until its tile's erosion job finishes; `TerrainSystem`
   * gates the flow and moisture debug views on that so an unwritten layer can
   * never be mistaken for "erosion produced a flat map".
   */
  readonly flowArray: THREE.StorageArrayTexture;

  /**
   * rgba8, written by `SplatGenerator`. Biome weights 0–3, in the order the
   * material's texture array is laid out:
   *
   *   r rock · g scree · b meadow grass · a dry grass
   *
   * Weights across `splat0` and `splat1.rgb` sum to 1 by construction, so the
   * material can blend without renormalising.
   */
  readonly splat0Array: THREE.StorageArrayTexture;
  /**
   * rgba8, the second half of the biome mask.
   *
   *   r forest floor · g sand · b mud · a snow susceptibility
   *
   * `a` is deliberately not a weight. Snow is the eighth biome, but its extent
   * has to animate over minutes of game time (spec 2.1) and a baked tile cannot
   * animate. So the bake stores how well this texel *holds* snow — shallow,
   * concave, lee-side texels hold it, 60° faces do not — and the material turns
   * that into coverage every frame from the current snow line. See
   * `TerrainMaterial.snowCoverage`.
   */
  readonly splat1Array: THREE.StorageArrayTexture;

  readonly layers: number;

  readonly stats: TilePoolStats;

  /** key → slot. Keys are packed integers, see `tileKey`. */
  private keyToSlot = new Map<number, number>();

  // Slot-parallel flat arrays. One entry per layer.
  private slotKey: Int32Array;
  private slotLevel: Int32Array;
  private slotGx: Int32Array;
  private slotGz: Int32Array;
  private slotState: Uint8Array;
  private slotPinned: Uint8Array;
  private slotLastUsed: Float64Array;
  private slotPriority: Float32Array;
  private slotErosion: Uint8Array;
  private slotSplat: Uint8Array;

  /** World geometry, mirrored from the quadtree config. */
  private worldMin = 0;
  private leafSize = 1;
  private rootLevel = 0;

  private now = 0;

  constructor(
    private resources: ResourceManager,
    options: Partial<TilePoolOptions> = {},
  ) {
    const opts = { ...DEFAULT_TILE_POOL, ...options };
    this.layers = sizeLayers(opts);

    this.heightArray = this.makeArray(THREE.RedFormat, THREE.FloatType, 'terrain/heightArray', 4);
    this.surfaceArray = this.makeArray(
      THREE.RGBAFormat,
      THREE.HalfFloatType,
      'terrain/surfaceArray',
      8,
    );
    this.flowArray = this.makeArray(THREE.RGBAFormat, THREE.HalfFloatType, 'terrain/flowArray', 8);
    this.splat0Array = this.makeArray(
      THREE.RGBAFormat,
      THREE.UnsignedByteType,
      'terrain/splat0Array',
      4,
    );
    this.splat1Array = this.makeArray(
      THREE.RGBAFormat,
      THREE.UnsignedByteType,
      'terrain/splat1Array',
      4,
    );

    this.slotKey = new Int32Array(this.layers).fill(-1);
    this.slotLevel = new Int32Array(this.layers);
    this.slotGx = new Int32Array(this.layers);
    this.slotGz = new Int32Array(this.layers);
    this.slotState = new Uint8Array(this.layers);
    this.slotPinned = new Uint8Array(this.layers);
    this.slotLastUsed = new Float64Array(this.layers);
    this.slotPriority = new Float32Array(this.layers);
    this.slotErosion = new Uint8Array(this.layers);
    this.slotSplat = new Uint8Array(this.layers);

    this.stats = {
      layers: this.layers,
      ready: 0,
      pending: 0,
      starved: 0,
      eroded: 0,
      splatted: 0,
      megabytes: (this.layers * BYTES_PER_LAYER) / (1024 * 1024),
    };
  }

  private makeArray(
    format: THREE.PixelFormat,
    type: THREE.TextureDataType,
    label: string,
    bytesPerTexel: number,
  ): THREE.StorageArrayTexture {
    const tex = new THREE.StorageArrayTexture(TILE_TEXELS, TILE_TEXELS, this.layers);
    tex.format = format;
    tex.type = type;
    tex.name = label;
    // Bilinear is what makes a 129² tile carry a 33² patch: patch vertices land
    // on texel centres exactly, and the morphed in-between positions get a
    // proper blend rather than a staircase.
    tex.magFilter = THREE.LinearFilter;
    tex.minFilter = THREE.LinearFilter;
    // Tiles are addressed individually and their edges are shared with the
    // neighbour by construction; repeat wrapping would fold the far edge back
    // over the near one and put a hard crease at every tile boundary.
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    // Texture defaults this to true, but nothing ever writes a mip chain — the
    // compute pass only fills level 0 — so leaving it on costs memory for
    // undefined data.
    tex.generateMipmaps = false;
    return this.resources.track(
      tex,
      label,
      ResourceManager.textureBytes(TILE_TEXELS, TILE_TEXELS, this.layers, bytesPerTexel),
    );
  }

  /**
   * Mirror the quadtree's world geometry.
   *
   * A (level, gx, gz) triple only addresses a fixed world rectangle for a given
   * root size and depth, so changing either invalidates every cached tile. That
   * is why this flushes rather than trying to remap.
   */
  configure(rootSize: number, maxDepth: number): void {
    const worldMin = -rootSize;
    const leafSize = (rootSize * 2) / Math.pow(2, maxDepth);
    if (worldMin === this.worldMin && leafSize === this.leafSize && maxDepth === this.rootLevel) {
      return;
    }
    if (maxDepth > 10) {
      console.warn(
        `[aether] TilePool: maxDepth ${maxDepth} exceeds the 10 bits per axis in ` +
          'tileKey; tile addresses would collide. Clamp lodDepth or widen the key.',
      );
    }
    this.worldMin = worldMin;
    this.leafSize = leafSize;
    this.rootLevel = maxDepth;
    this.clear();
  }

  clear(): void {
    this.keyToSlot.clear();
    this.slotKey.fill(-1);
    this.slotState.fill(STATE_FREE);
    this.slotPinned.fill(0);
    this.slotLastUsed.fill(0);
    this.slotErosion.fill(EROSION_NONE);
    this.slotSplat.fill(SPLAT_NONE);
  }

  /** Edge length in metres of a node at `level`. */
  nodeSize(level: number): number {
    return this.leafSize * Math.pow(2, level);
  }

  /** World X of the corner of tile (level, gx). */
  originOf(level: number, g: number): number {
    return this.worldMin + g * this.nodeSize(level);
  }

  /** Grid index of the node containing world X at `level`. */
  gridIndex(level: number, world: number): number {
    return Math.floor((world - this.worldMin) / this.nodeSize(level));
  }

  /**
   * Pack (level, gx, gz) into one integer.
   *
   * 10 bits per axis covers maxDepth 10 (1024 leaves per side), which is the
   * deepest the quality presets go. Deeper worlds need a wider key, so the
   * assumption is asserted rather than left to corrupt the map silently.
   */
  static tileKey(level: number, gx: number, gz: number): number {
    return (level * 1024 + gz) * 1024 + gx;
  }

  beginFrame(nowSeconds: number): void {
    this.now = nowSeconds;
    this.stats.starved = 0;
  }

  /**
   * Ask for a tile, allocating a layer if it is not resident.
   *
   * Returns the layer index whether or not the bake has run — the caller must
   * still check `isReady` before sampling it. Returns -1 when every layer is
   * either pinned or was used too recently to recycle, which is the pool's way
   * of saying "render off an ancestor this frame".
   */
  request(level: number, gx: number, gz: number, priority: number): number {
    if (level > this.rootLevel || level < 0) return -1;
    const span = 1 << (this.rootLevel - level);
    if (gx < 0 || gz < 0 || gx >= span || gz >= span) return -1;

    const key = TilePool.tileKey(level, gx, gz);
    const existing = this.keyToSlot.get(key);
    if (existing !== undefined) {
      this.slotLastUsed[existing] = this.now;
      // Keep the highest priority seen this frame: a tile requested both as
      // somebody's parent and as its own node should bake at the parent's
      // (higher) urgency.
      if (priority > this.slotPriority[existing]) this.slotPriority[existing] = priority;
      return existing;
    }

    const slot = this.allocate();
    if (slot < 0) {
      this.stats.starved++;
      return -1;
    }

    this.keyToSlot.set(key, slot);
    this.slotKey[slot] = key;
    this.slotLevel[slot] = level;
    this.slotGx[slot] = gx;
    this.slotGz[slot] = gz;
    this.slotState[slot] = STATE_PENDING;
    this.slotLastUsed[slot] = this.now;
    this.slotPriority[slot] = priority;
    // A recycled layer holds a different tile now, so whatever erosion it had
    // is meaningless and it must queue for a fresh pass. Same for the splat —
    // and there the staleness is not merely cosmetic, so `SPLAT_NONE` outranks
    // every erosion-triggered refresh in the classification queue.
    this.slotErosion[slot] = EROSION_NONE;
    this.slotSplat[slot] = SPLAT_NONE;
    return slot;
  }

  /**
   * Free layer if there is one, otherwise the least-recently-used layer.
   *
   * Two tiers, and the second one matters more than it looks. The age gate
   * exists so a quick glance away and back does not re-bake; but if *every*
   * unpinned layer is younger than the gate, honouring it would mean refusing
   * to allocate at all — and that is exactly the state the pool lands in after
   * a teleport, when a full pool of tiles from the old position is technically
   * "recent" and none of it is on screen. Measured: 105 of 143 visible nodes
   * stuck on coarse ancestors for a full two seconds after a camera jump.
   *
   * So when nothing is stale, fall through to the plain LRU and take the layer
   * that has gone unused longest — excluding anything touched this frame, which
   * is precisely the set that is on screen right now.
   */
  private allocate(): number {
    let staleSlot = -1;
    let staleUse = Infinity;
    let lruSlot = -1;
    let lruUse = Infinity;

    for (let i = 0; i < this.layers; i++) {
      if (this.slotKey[i] === -1) return i;
      if (this.slotPinned[i] === 1) continue;
      // A pending tile has a queue position and possibly a dependent node
      // already pointing at it; recycling it would thrash forever under load.
      if (this.slotState[i] === STATE_PENDING) continue;
      // Likewise a layer mid-erosion: the job takes tens of frames and losing it
      // wastes all of them. One layer out of 256 is a cheap reservation.
      if (this.slotErosion[i] === EROSION_RUNNING) continue;
      const used = this.slotLastUsed[i];
      // Touched this frame — it is on screen, it is not a candidate.
      if (used >= this.now) continue;
      if (used < lruUse) {
        lruUse = used;
        lruSlot = i;
      }
      if (this.now - used >= EVICT_AGE_SECONDS && used < staleUse) {
        staleUse = used;
        staleSlot = i;
      }
    }

    const victim = staleSlot >= 0 ? staleSlot : lruSlot;
    if (victim >= 0) this.keyToSlot.delete(this.slotKey[victim]);
    return victim;
  }

  isReady(level: number, gx: number, gz: number): boolean {
    const slot = this.keyToSlot.get(TilePool.tileKey(level, gx, gz));
    return slot !== undefined && this.slotState[slot] === STATE_READY;
  }

  /**
   * Explicit parent-residency query.
   *
   * A node may only be drawn at its own level once its parent tile is baked
   * too, because the CDLOD morph mixes the two heights and a missing parent
   * would blend against an unwritten (zero) layer — a hole in the world in the
   * most literal sense. The root has no parent and is always drawable.
   */
  isDrawable(level: number, gx: number, gz: number): boolean {
    if (!this.isReady(level, gx, gz)) return false;
    if (level >= this.rootLevel) return true;
    return this.isReady(level + 1, gx >> 1, gz >> 1);
  }

  /**
   * Nearest baked ancestor of (level, gx, gz), inclusive.
   *
   * Terminates because the root layer is pinned and baked in `init`, so the
   * worst case is a very coarse stand-in, never a missing one. The returned
   * object is module-scope scratch — read it, do not keep it.
   */
  resolve(level: number, gx: number, gz: number): TileRef {
    let l = Math.max(0, level);
    let x = gx;
    let z = gz;
    while (l <= this.rootLevel) {
      const slot = this.keyToSlot.get(TilePool.tileKey(l, x, z));
      if (slot !== undefined && this.slotState[slot] === STATE_READY) {
        this.slotLastUsed[slot] = this.now;
        _ref.level = l;
        _ref.gx = x;
        _ref.gz = z;
        _ref.layer = slot;
        _ref.size = this.nodeSize(l);
        _ref.originX = this.originOf(l, x);
        _ref.originZ = this.originOf(l, z);
        return _ref;
      }
      l++;
      x >>= 1;
      z >>= 1;
    }
    // Only reachable before the root bake has been submitted, i.e. never after
    // init. Layer 0 is the root's, so this degrades to "flat" rather than to
    // sampling an out-of-range layer.
    _ref.level = this.rootLevel;
    _ref.gx = 0;
    _ref.gz = 0;
    _ref.layer = 0;
    _ref.size = this.nodeSize(this.rootLevel);
    _ref.originX = this.worldMin;
    _ref.originZ = this.worldMin;
    return _ref;
  }

  /** Pin a tile so LRU never reclaims it. Used for the root fallback. */
  pin(level: number, gx: number, gz: number): number {
    const slot = this.request(level, gx, gz, Number.MAX_VALUE);
    if (slot >= 0) this.slotPinned[slot] = 1;
    return slot;
  }

  /**
   * Highest-priority layer still waiting to be baked, marked as taken.
   *
   * A linear scan over ≤256 slots to pick a top-k of ≤3 is cheaper than
   * maintaining a sorted queue, and it costs nothing to keep priorities live as
   * the camera moves.
   */
  takeBestPending(): number {
    let best = -1;
    let bestPriority = -Infinity;
    for (let i = 0; i < this.layers; i++) {
      if (this.slotState[i] !== STATE_PENDING) continue;
      if (this.slotPriority[i] > bestPriority) {
        bestPriority = this.slotPriority[i];
        best = i;
      }
    }
    return best;
  }

  markReady(slot: number): void {
    this.slotState[slot] = STATE_READY;
  }

  /**
   * Claim the most urgent baked-but-un-eroded layer, or -1 if there is none.
   *
   * Same priority as the bake queue — 1/distance, parents weighted — so erosion
   * follows the streamer outward instead of picking tiles at random. That
   * matters visually: neighbouring tiles finish close together, which keeps the
   * window in which an eroded tile sits next to an un-eroded one short.
   */
  beginErosion(): number {
    let best = -1;
    let bestPriority = -Infinity;
    for (let i = 0; i < this.layers; i++) {
      if (this.slotState[i] !== STATE_READY) continue;
      if (this.slotErosion[i] !== EROSION_NONE) continue;
      if (this.slotPriority[i] > bestPriority) {
        bestPriority = this.slotPriority[i];
        best = i;
      }
    }
    if (best >= 0) this.slotErosion[best] = EROSION_RUNNING;
    return best;
  }

  finishErosion(slot: number): void {
    this.slotErosion[slot] = EROSION_DONE;
    // The tile now has a real moisture channel and a re-cut height field, and
    // both feed the biome classifier. The weights computed at bake time were
    // derived from a zero flow map, so they are correct-but-provisional; queue
    // them for a refresh behind anything still unclassified.
    if (this.slotSplat[slot] === SPLAT_FRESH) this.slotSplat[slot] = SPLAT_STALE;
  }

  /**
   * Claim the most urgent layer needing biome classification, or -1.
   *
   * Strict two-tier: every unclassified layer is served before any refresh,
   * because an unclassified layer is showing the previous tenant's biomes while
   * a stale one is merely showing pre-erosion ones. Within a tier the streamer's
   * own priority (1/distance) decides, so classification follows the camera.
   */
  takeSplatPending(): number {
    let best = -1;
    let bestTier = 3;
    let bestPriority = -Infinity;
    for (let i = 0; i < this.layers; i++) {
      if (this.slotState[i] !== STATE_READY) continue;
      const tier = this.slotSplat[i] === SPLAT_NONE ? 0 : this.slotSplat[i] === SPLAT_STALE ? 1 : 3;
      if (tier === 3) continue;
      if (tier > bestTier) continue;
      if (tier === bestTier && this.slotPriority[i] <= bestPriority) continue;
      bestTier = tier;
      bestPriority = this.slotPriority[i];
      best = i;
    }
    if (best >= 0) this.slotSplat[best] = SPLAT_FRESH;
    return best;
  }

  /** True while at least one layer is drawing another tile's biome weights. */
  hasUnclassifiedSplat(): boolean {
    for (let i = 0; i < this.layers; i++) {
      if (this.slotState[i] === STATE_READY && this.slotSplat[i] === SPLAT_NONE) return true;
    }
    return false;
  }

  /** Give a layer back to the queue after an aborted job. */
  abortErosion(slot: number): void {
    if (this.slotErosion[slot] === EROSION_RUNNING) this.slotErosion[slot] = EROSION_NONE;
  }

  /**
   * Is the layer still holding the tile an erosion job started for?
   *
   * A job spans tens of frames, and a settings change flushes the pool outright.
   * Writing the resolve pass into a layer that now holds a different tile would
   * paint one part of the world with another part's terrain, so the job checks
   * this before every dispatch rather than trusting the reservation alone.
   */
  isErosionJobValid(slot: number, key: number): boolean {
    return (
      this.slotKey[slot] === key &&
      this.slotState[slot] === STATE_READY &&
      this.slotErosion[slot] === EROSION_RUNNING
    );
  }

  keyOfSlot(slot: number): number {
    return this.slotKey[slot];
  }

  levelOfSlot(slot: number): number {
    return this.slotLevel[slot];
  }

  gxOfSlot(slot: number): number {
    return this.slotGx[slot];
  }

  gzOfSlot(slot: number): number {
    return this.slotGz[slot];
  }

  /** World extent of the tile held in `slot`, for the bake pass. */
  slotOriginX(slot: number): number {
    return this.originOf(this.slotLevel[slot], this.slotGx[slot]);
  }

  slotOriginZ(slot: number): number {
    return this.originOf(this.slotLevel[slot], this.slotGz[slot]);
  }

  slotSize(slot: number): number {
    return this.nodeSize(this.slotLevel[slot]);
  }

  /** Refresh the counters the F3 overlay reads. Call once per frame. */
  updateStats(): void {
    let ready = 0;
    let pending = 0;
    let eroded = 0;
    let splatted = 0;
    for (let i = 0; i < this.layers; i++) {
      if (this.slotState[i] === STATE_READY) ready++;
      else if (this.slotState[i] === STATE_PENDING) pending++;
      if (this.slotErosion[i] === EROSION_DONE) eroded++;
      if (this.slotState[i] === STATE_READY && this.slotSplat[i] !== SPLAT_NONE) splatted++;
    }
    this.stats.ready = ready;
    this.stats.pending = pending;
    this.stats.eroded = eroded;
    this.stats.splatted = splatted;
  }

  dispose(): void {
    this.resources.release(this.heightArray);
    this.resources.release(this.surfaceArray);
    this.resources.release(this.flowArray);
    this.resources.release(this.splat0Array);
    this.resources.release(this.splat1Array);
    this.keyToSlot.clear();
  }
}
