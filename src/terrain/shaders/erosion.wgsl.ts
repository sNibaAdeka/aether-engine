/**
 * Erosion kernels: Hans Beyer's droplet hydraulics plus Musgrave talus, in TSL.
 *
 * Everything here works on a *scratch domain* — the tile's 129² core surrounded
 * by an apron — held in storage buffers rather than in the tile's array-texture
 * layer. Two reasons, both forced:
 *
 *  - WGSL has atomics on storage buffers only, and the erosion pass needs them
 *    (see the race note below). A `texture_storage_2d_array` cannot be atomic.
 *  - Droplets run off the edge of the tile. Simulating only the 129² core makes
 *    the result depend on where the tile boundary happens to fall, and two
 *    neighbouring tiles then disagree along their shared edge. The apron is what
 *    buys agreement; the core is copied out and the apron is discarded.
 *
 * ── Heights are in CELLS, not metres ────────────────────────────────────────
 * Beyer's constants (minSlope 0.01, capacity 4, gravity 4 …) are dimensionless:
 * they assume a heightmap in [0,1] over unit cells. This world spans 1400 m of
 * relief and cell spacing runs 1 m (level 0) to 512 m (root), so `dh` in metres
 * per cell would vary by 512× across levels and `max(-dh, minSlope)` would mean
 * something different at every one. The whole simulation therefore runs on
 * h/cellSize — slope in cells per cell — and the resolve pass multiplies the
 * delta back into metres. One constant set then behaves identically at every
 * level, and `minSlope = 0.01` is 0.57° everywhere.
 *
 * ── The race, stated honestly ───────────────────────────────────────────────
 * The spec asks for "ping-pong + a 4-phase checkerboard split of the dispatch",
 * and that is exactly what runs here: four dispatches, phase p taking the world
 * cells with (x&1) | ((z&1)<<1) == p, each reading a frozen snapshot and writing
 * into a delta buffer that is applied between phases.
 *
 * The checkerboard does NOT make the pass race-free, and pretending otherwise
 * would be a lie. Droplets migrate: they take up to `MAX_LIFETIME` unit steps
 * and each step writes a 25-cell brush, so two same-phase droplets that start
 * two cells apart can be writing the same cell after a single step. At one
 * droplet per cell that is on the order of a hundred same-phase writes per cell
 * per phase — collisions are the steady state, not an edge case.
 *
 * What actually makes the result correct is the fixed-point integer accumulator:
 * every deposit and every scoop is an `atomicAdd` on an i32 in units of 2⁻¹⁶
 * cells. Integer addition is associative and commutative, so the sum is
 * bit-identical regardless of the order the GPU happens to schedule the
 * droplets. That is what satisfies the project's determinism rule — f32
 * addition is not associative, so no float-based scheme, atomic or not, could
 * reproduce the same world twice.
 *
 * So: the checkerboard survives as the *generation* structure (it partitions the
 * droplets into four Jacobi generations, which is what keeps the simulation
 * stable and tile-independent), and the atomics carry the correctness.
 */

import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  Break,
  atan,
  atomicAdd,
  atomicLoad,
  atomicStore,
  bitAnd,
  clamp,
  computeKernel,
  float,
  floor,
  globalId,
  instancedArray,
  int,
  length,
  log2,
  max,
  min,
  normalize,
  round,
  saturate,
  shiftLeft,
  sqrt,
  storageTexture,
  textureStore,
  uint,
  uniform,
  uvec2,
  vec2,
  vec3,
  vec4,
  wgslFn,
} from 'three/tsl';

import type { ShaderNode } from './debugViews';
import { fnHashU32, fnTerrainHeight } from './heightfield.wgsl';
import { TILE_TEXELS, type TilePool } from '../TilePool';

// ---------------------------------------------------------------------------
// Sizing
// ---------------------------------------------------------------------------

/**
 * Apron cells on each side of the core.
 *
 * The bound is not "how far can one droplet reach" but "how far can a boundary
 * error propagate before the per-generation clamp kills it", which is
 * ≈2·(MAX_LIFETIME + EROSION_RADIUS) = 126, rounded to 128.
 *
 * Measured on this world, comparing the shared edge column of two independently
 * eroded neighbours (max over the 129 edge texels, metres):
 *
 *   cell size    1 m     2 m     4 m     8 m     16 m    32 m
 *   apron  96    0.026   1.09    2.67    0.075   15.5    12.4
 *   apron 128    —       0.032   0.48    0.014   4.50    3.54
 *   apron 192    —       —       0.17    0.007   4.55    3.53
 *
 * 128 is the right pick, and the third row is why: going to 192 (1.8× the work)
 * buys another 3× at 4 m cells and *nothing at all* at 16 and 32 m. So the fine
 * levels are apron-limited and converge as predicted, while the coarse-level
 * residual has some other, unidentified source.
 *
 * TODO(quality): that coarse residual is the worst defect in this pass. 4.5 m at
 * 16 m cells is ≈0.28 cells against an erosion signal of ≈1 cell — the mean over
 * the edge is only 0.17 m, so it is a handful of outlier texels rather than a
 * uniform step, but at a grazing viewing angle each one opens a one-pixel crack
 * you can see the sky through. It is not apron-limited, so do not just raise
 * this number; find the actual divergence first. Note that the level-8 m pair
 * measures 0.014 m, i.e. fully converged, so the mechanism is terrain-dependent
 * rather than systematic.
 */
export const APRON = 128;

/** Scratch grid edge. 129 core + 2×128 apron. */
export const SCRATCH_TEXELS = TILE_TEXELS + 2 * APRON;
export const SCRATCH_CELLS = SCRATCH_TEXELS * SCRATCH_TEXELS;

/** Workgroup edge; 8×8 = 64 invocations. */
export const WORKGROUP = 8;
/** Workgroups needed to cover one scratch axis. */
export const SCRATCH_GROUPS = Math.ceil(SCRATCH_TEXELS / WORKGROUP);
/** Workgroups needed to cover one core axis. */
export const CORE_GROUPS = Math.ceil(TILE_TEXELS / WORKGROUP);

/**
 * Workgroup rows per droplet dispatch.
 *
 * One phase over the whole scratch domain is ~37k droplets × 60 steps × ~14.5
 * atomic writes ≈ 32M atomics — milliseconds in a single dispatch, i.e. a
 * visible hitch. Splitting the phase into horizontal bands changes nothing
 * about the result (every droplet in a phase reads the same frozen snapshot, so
 * the partition is purely a scheduling device) and turns one long dispatch into
 * several that fit the per-frame budget.
 */
export const BAND_GROUPS = 6;
export const DROPLET_BANDS = Math.ceil(SCRATCH_GROUPS / BAND_GROUPS);

/** Droplet generations per round; the spec's 4-phase checkerboard. */
export const PHASES = 4;

/**
 * Passes of the full 4-phase cycle.
 *
 * Each round spawns one droplet per scratch cell, so rounds are the droplet-count
 * knob: 1 round = SCRATCH_CELLS = 148,225 droplets, 2 rounds = 296,450. More
 * rounds also means more Jacobi generations, which is what lets coherent
 * channels emerge from the sum rather than from any single droplet's footprint.
 *
 * 2 is where the two failure modes meet. Measured going to 4 rounds (16
 * generations): channelisation barely moved (24 % → 23 % of eroded volume in the
 * most-incised 5 % of cells) but the tile seam tripled, because every extra
 * generation gives a boundary error another MAX_LIFETIME cells of reach.
 */
export const ROUNDS = 2;

/** Musgrave talus relaxation passes. 0.5^5 = 3 % residual excess slope. */
export const THERMAL_ITERATIONS = 5;

// ---------------------------------------------------------------------------
// Beyer parameters
// ---------------------------------------------------------------------------

const INERTIA = 0.05;
const CAPACITY_FACTOR = 4.0;
const MIN_SLOPE = 0.01;
/**
 * Beyer publishes 0.3 for both rates. 0.1 here, and the reason is a units
 * mismatch the published constants hide.
 *
 * They assume a heightmap in [0,1] over a few hundred cells, where a one-cell
 * drop is ~0.005 and a droplet's per-step bite is a rounding error against the
 * relief. Normalising by cell size (which the whole file depends on for
 * scale-invariance) makes a one-cell drop ≈0.5 *cells* on a 30° slope, so at
 * 0.3 the very first step of every droplet scoops out the entire drop beneath
 * it — and, because deposition lands bilinearly on four texels while erosion
 * spreads over twenty-five, each droplet leaves a single-texel bump where it
 * unloaded. Measured: a stippled, sandpaper surface with 15 % channelisation,
 * which is barely above the 5 % that uniform lowering would give.
 *
 * At 0.1 no single droplet's footprint carries the signal; the sum over the
 * generations does, which is what lets coherent channels emerge. Total erosion
 * depth is held roughly constant by raising ROUNDS to compensate.
 */
const DEPOSIT_RATE = 0.1;
const ERODE_RATE = 0.1;
const EVAPORATION = 0.01;
const GRAVITY = 4.0;
const EROSION_RADIUS = 3;
const MAX_LIFETIME = 60;

/**
 * Terminal velocity, in cells per step.
 *
 * Needed because the speed integration below accelerates downhill (as it must —
 * see there) and a 60-step run down a long slope would otherwise let speed, and
 * therefore capacity, grow without bound. Real streamflow does not: drag
 * balances gravity. 4 is the value at which capacity on a typical 30° slope
 * (−dh ≈ 0.5 cells) tops out at 0.5·4·1·4 = 8 cells of sediment, comfortably
 * more than any droplet manages to pick up under the per-step `min(−dh)` limit,
 * so the cap shapes the tail of the distribution without truncating the
 * ordinary case.
 */
const MAX_SPEED = 4.0;

/**
 * Maximum height change one generation may make to one cell, in cells.
 *
 * Not polish — without it the Jacobi formulation diverges violently, because
 * inside a generation many droplets erode the same cell while all reading the
 * same unmodified height, so each computes its `min(…, -dh)` clamp against a
 * height that is already being dug out from under it. Measured peak height
 * change without the clamp: 15 m at 1 generation, 161 m at 2, 1.2e5 m at 4,
 * 5.1e9 m at 8. With it: 1.5 m at 4, 2.3 m at 8, 3.0 m at 16 — stable.
 */
const GEN_CLAMP_CELLS = 0.125;
/** Thermal moves at most 0.25·tan(α) ≈ 0.17 cells per pass, so it never binds. */
const THERMAL_CLAMP_CELLS = 1.0;

const TAN_ROCK = 1.0; // 45°
const TAN_SOIL = 0.67451; // 34°
/** Deposited depth, in cells, at which a slope is fully "loose soil". */
const SOIL_FULL_CELLS = 0.5;
/** Musgrave's talus factor. 1.0 converges faster but oscillates. */
const TALUS_FACTOR = 0.5;

/**
 * Cell size, metres, at which erosion reaches full world-space amplitude.
 *
 * The simulation is scale-free — it always carves ≈0.4 cells — so converting
 * straight back with the tile's own spacing would give the root tile (512 m
 * cells) a 200 m deep channel network and the finest tile (1 m cells) a 0.4 m
 * one. Capping the conversion factor keeps world-space erosion depth bounded at
 * ≈0.4·16 m ≈ 6 m for every level coarser than this, which is what stops the
 * distant silhouette from being a different mountain range than the near one.
 *
 * TODO(quality): this is a bound, not a solution. The eroded surface at level
 * L+1 is genuinely not a downsample of the eroded surface at level L — coarse
 * cells carve wide shallow valleys where fine cells carve gullies — so terrain
 * still "breathes" as you fly in and channels resolve. Recon flagged this as
 * unresolved and it remains so; the CDLOD height morph keeps it from *popping*,
 * which is the part that would be unacceptable.
 */
const REFERENCE_CELL_M = 16.0;

// ---------------------------------------------------------------------------
// Fixed point
// ---------------------------------------------------------------------------

/** Height quantum, cells. i32 range is ±32768 cells — 23× the world's relief. */
const HEIGHT_QUANTUM = Math.pow(2, -16);
const HEIGHT_SCALE = 1 / HEIGHT_QUANTUM;
/** Flow quantum, in droplet-water units. u32 range 1.05e6 water-steps per cell. */
const FLOW_SCALE = 4096;
/** Direction quantum. Components are water-weighted, so |Σ| ≤ Σwater. */
const DIR_SCALE = 4096;

/**
 * Signed cut/fill range, metres, packed into flowArray.a as (dh/2R + 0.5).
 *
 * ±32 m rather than the ±8 m the sizing note first suggested: measured peak
 * |dh| is 0.76 m at 2 m cells but 27 m at 32 m cells, because the simulation is
 * scale-free and coarse tiles carve the same ≈1 cell everywhere. ±8 m would
 * have clipped every tile coarser than level 3 to a solid 0 or 1.
 * f16 resolves 2⁻¹¹ near 0.5, i.e. ~31 mm over this range.
 */
export const CUTFILL_RANGE_M = 32.0;

/**
 * Flow normalisation: `R = log2(1 + raw/ROUNDS) / FLOW_LOG_DIVISOR`.
 *
 * Two things are going on.
 *
 * `raw/ROUNDS` makes the channel expressed in *water-steps per cell per round*,
 * so the map does not change brightness when the droplet budget is retuned —
 * otherwise doubling ROUNDS would silently re-scale a deliverable that phases 6
 * and 7 read as an absolute quantity.
 *
 * The log is because drainage spans orders of magnitude and a linear ramp shows
 * only the trunk. Divisor 10 puts full scale at 1023 water-steps per round; the
 * measured distribution at 16 m cells is p05 ≈ 2.5, p50 ≈ 16, p95 ≳ 128, which
 * lands in the middle of the ramp with headroom at the top. (Divisor 8 — full
 * scale at 255 — was measured saturating: p95 clipped to 1.0 at every level
 * coarser than 2, i.e. the whole network read as solid white.)
 *
 * The constant is GLOBAL and fixed. Normalising against a per-tile maximum
 * would make the same river a different colour in adjacent tiles and put a
 * visible step in the network at every tile boundary.
 */
const FLOW_LOG_DIVISOR = 10.0;

/** Moisture blur radius in cells. Riparian vegetation reaches tens of metres. */
const BLUR_RADIUS = 8;
const BLUR_SIGMA = 4.0;

// ---------------------------------------------------------------------------
// Erosion brush
// ---------------------------------------------------------------------------

/**
 * Cells within `EROSION_RADIUS` of the droplet, weighted by (radius − distance)
 * and normalised. Radius 3 admits exactly the 5×5 block (the (±3,0) cells sit at
 * distance 3, which is not < 3), so this is 25 taps, unrolled into the shader.
 */
const BRUSH: ReadonlyArray<{ dx: number; dy: number; w: number }> = (() => {
  const raw: Array<{ dx: number; dy: number; w: number }> = [];
  let sum = 0;
  const span = EROSION_RADIUS - 1;
  for (let dy = -span; dy <= span; dy++) {
    for (let dx = -span; dx <= span; dx++) {
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d >= EROSION_RADIUS) continue;
      const w = EROSION_RADIUS - d;
      sum += w;
      raw.push({ dx, dy, w });
    }
  }
  return raw.map((c) => ({ dx: c.dx, dy: c.dy, w: c.w / sum }));
})();

/** The 8 talus neighbours, with their true centre-to-centre distance. */
const TALUS_NEIGHBOURS: ReadonlyArray<{ dx: number; dy: number; d: number }> = [
  { dx: -1, dy: 0, d: 1 },
  { dx: 1, dy: 0, d: 1 },
  { dx: 0, dy: -1, d: 1 },
  { dx: 0, dy: 1, d: 1 },
  { dx: -1, dy: -1, d: Math.SQRT2 },
  { dx: 1, dy: -1, d: Math.SQRT2 },
  { dx: -1, dy: 1, d: Math.SQRT2 },
  { dx: 1, dy: 1, d: Math.SQRT2 },
];

// ---------------------------------------------------------------------------
// Deterministic droplet spawn
// ---------------------------------------------------------------------------

/**
 * Uniform [0,1) from an ABSOLUTE world cell index.
 *
 * This is the property the whole apron argument rests on: tile A's apron must
 * simulate bit-identical droplets to tile B's core, which is only true if the
 * spawn jitter is a function of the world cell and never of the tile-local one.
 * `level` is folded in because tiles at different levels have different cell
 * grids and should not share droplet paths.
 */
const fnCellRand = wgslFn(
  /* wgsl */ `
fn aeCellRand( gx: i32, gz: i32, level: i32, seed: i32, k: i32 ) -> f32 {
  let a = bitcast<u32>( gx ) * 0x27d4eb2du;
  let b = bitcast<u32>( gz ) * 0x165667b1u;
  let c = bitcast<u32>( level ) * 0x85ebca6bu;
  let d = bitcast<u32>( k ) * 0x9e3779b9u;
  return f32( aeHashU32( a ^ b ^ c ^ d ^ bitcast<u32>( seed ) ) ) * 2.3283064365386963e-10;
}
`,
  [fnHashU32],
);

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

type ComputeKernelNode = ReturnType<typeof computeKernel>;

/**
 * Per-tile parameters.
 *
 * Uniforms rather than a storage buffer, unlike the height bake: only one tile
 * is in the erosion pipeline at a time, and every dispatch is its own
 * `renderer.compute()` call, so mutating them between calls is ordered by the
 * WebGPU queue rather than being a race. The bake batches many tiles into one
 * dispatch and therefore cannot do this.
 *
 * Declared by inference so the concrete `UniformNode<'vec2', Vector2>` /
 * `UniformNode<'int', number>` types survive to the call sites — an interface
 * written by hand here would have to name types three does not re-export.
 */
function makeUniforms() {
  return {
    /** World XZ of scratch cell (0,0). */
    origin: uniform(new THREE.Vector2()),
    /** Metres per scratch cell. */
    spacing: uniform(1),
    /** Metres per cell used to convert the delta back — min(spacing, reference). */
    outScale: uniform(1),
    /** Destination array layer. */
    layer: uniform(0, 'int'),
    /** Checkerboard phase, 0..3. */
    phase: uniform(0, 'int'),
    /** Round index, so each round gets a fresh droplet jitter. */
    round: uniform(0, 'int'),
    /** First scratch row this dispatch covers. */
    rowOffset: uniform(0, 'int'),
    /** Absolute world cell index of scratch cell (0,0). */
    cellBaseX: uniform(0, 'int'),
    cellBaseZ: uniform(0, 'int'),
    /** Quadtree level of the tile, part of the droplet hash. */
    level: uniform(0, 'int'),
    /** Per-generation delta clamp, cells. */
    genClamp: uniform(GEN_CLAMP_CELLS),
  };
}

export type ErosionUniforms = ReturnType<typeof makeUniforms>;

export interface ErosionKernels {
  seed: ComputeKernelNode;
  droplet: ComputeKernelNode;
  apply: ComputeKernelNode;
  thermal: ComputeKernelNode;
  /** 3×3 tent on the erosion delta, then the write-back of that tent. */
  smoothDelta: ComputeKernelNode;
  applySmooth: ComputeKernelNode;
  blurX: ComputeKernelNode;
  resolve: ComputeKernelNode;
}

export interface ErosionProgram {
  uniforms: ErosionUniforms;
  kernels: ErosionKernels;
  /** Every storage attribute, so the caller can register and dispose them. */
  attributes: THREE.StorageInstancedBufferAttribute[];
  /**
   * One u32: how many texels the resolve pass has written a non-zero flow
   * value into. Read back once, to prove the flow map is not empty.
   */
  verifyAttribute: THREE.StorageInstancedBufferAttribute;
  /** Total scratch bytes, for the memory readout. */
  bytes: number;
}

/**
 * Build the whole erosion program against one tile pool.
 *
 * Everything shares a single scratch domain because exactly one tile is eroded
 * at a time — the pass is amortised over frames, not over tiles, so a second
 * scratch would buy nothing but 2.5 MB.
 */
export function buildErosionProgram(pool: TilePool, worldSeed: number): ErosionProgram {
  const S = SCRATCH_TEXELS;

  const u = makeUniforms();
  const uSeed = uniform(worldSeed);
  const uSeedI = uniform(worldSeed, 'int');

  // Live heightfield, in cells. Frozen for the duration of one generation:
  // droplets only ever read it, and only the apply pass writes it.
  const hField = instancedArray(SCRATCH_CELLS, 'float');
  // The analytic surface before any erosion, kept for the cut/fill channel.
  const hBase = instancedArray(SCRATCH_CELLS, 'float');
  // Per-generation accumulator, fixed point. See the race note at the top.
  const hDelta = instancedArray(SCRATCH_CELLS, 'int').toAtomic();
  const flowAccum = instancedArray(SCRATCH_CELLS, 'uint').toAtomic();
  const flowDirX = instancedArray(SCRATCH_CELLS, 'int').toAtomic();
  const flowDirZ = instancedArray(SCRATCH_CELLS, 'int').toAtomic();
  // X half of the separable moisture blur; the Y half runs inside `resolve`,
  // where it only has to cover the 129² core rather than the whole scratch.
  const blurTmp = instancedArray(SCRATCH_CELLS, 'float');
  /**
   * Texels the resolve pass has written a NON-ZERO flow value into, ever.
   *
   * This exists so `erosionDataAvailable` can be a fact rather than a promise.
   * The flag used to be "at least one job reached its last dispatch", which is
   * a statement about the scheduler, not about the texture — it stayed true
   * while the flow map was empty, and the two debug views it gates then showed
   * a convincing nothing. Counting on the GPU, in the same kernel that does the
   * `textureStore`, is the cheapest thing that actually looks at the data.
   */
  const verifyCount = instancedArray(1, 'uint').toAtomic();

  const idx = (x: ShaderNode, y: ShaderNode): ShaderNode => y.mul(int(S)).add(x);
  const inBounds = (x: ShaderNode, y: ShaderNode): ShaderNode =>
    x
      .greaterThanEqual(int(0))
      .and(x.lessThan(int(S)))
      .and(y.greaterThanEqual(int(0)))
      .and(y.lessThan(int(S)));

  // -------------------------------------------------------------------------
  // seed — analytic height over the scratch domain, counters zeroed
  // -------------------------------------------------------------------------

  const seedKernel = Fn(() => {
    const gx = int(globalId.x);
    const gy = int(globalId.y);
    If(gx.lessThan(int(S)).and(gy.lessThan(int(S))), () => {
      const i = idx(gx, gy);
      const world = u.origin.add(vec2(float(gx), float(gy)).mul(u.spacing));
      // Heights enter the simulation already divided by cell size — see the
      // header. Everything downstream is in cells until `resolve`.
      const h = fnTerrainHeight({ pos: world, seed: uSeed }).div(u.spacing);
      hField.element(i).assign(h);
      hBase.element(i).assign(h);
      atomicStore(hDelta.element(i), int(0));
      atomicStore(flowAccum.element(i), uint(0));
      atomicStore(flowDirX.element(i), int(0));
      atomicStore(flowDirZ.element(i), int(0));
      blurTmp.element(i).assign(float(0));
    });
  });

  // -------------------------------------------------------------------------
  // droplet — one Beyer particle per cell of the current checkerboard phase
  // -------------------------------------------------------------------------

  /** Bilinear height and the analytic gradient of that same bilinear patch. */
  const sampleField = (nx: ShaderNode, ny: ShaderNode, uu: ShaderNode, vv: ShaderNode) => {
    const h00 = hField.element(idx(nx, ny));
    const h10 = hField.element(idx(nx.add(int(1)), ny));
    const h01 = hField.element(idx(nx, ny.add(int(1))));
    const h11 = hField.element(idx(nx.add(int(1)), ny.add(int(1))));
    // Differentiating the bilinear patch, not finite-differencing the lattice.
    // A finite-difference gradient here quantises the flow direction to the
    // cell grid and the channels come out stair-stepped.
    const gx = h10.sub(h00).mul(float(1).sub(vv)).add(h11.sub(h01).mul(vv));
    const gy = h01.sub(h00).mul(float(1).sub(uu)).add(h11.sub(h10).mul(uu));
    const h = h00
      .mul(float(1).sub(uu).mul(float(1).sub(vv)))
      .add(h10.mul(uu.mul(float(1).sub(vv))))
      .add(h01.mul(float(1).sub(uu).mul(vv)))
      .add(h11.mul(uu.mul(vv)));
    return { h, gx, gy, h00, h10, h01, h11 };
  };

  const addHeight = (x: ShaderNode, y: ShaderNode, amount: ShaderNode): void => {
    If(inBounds(x, y), () => {
      atomicAdd(hDelta.element(idx(x, y)), int(round(amount.mul(HEIGHT_SCALE))));
    });
  };

  const dropletKernel = Fn(() => {
    const cx = int(globalId.x);
    const cy = int(globalId.y).add(u.rowOffset);

    // World cell index — the checkerboard phase and the spawn jitter are both
    // functions of it, never of the tile-local index, so two tiles agree on
    // every droplet in their overlap.
    const wx = cx.add(u.cellBaseX);
    const wz = cy.add(u.cellBaseZ);
    const myPhase = bitAnd(wx, int(1)).add(shiftLeft(bitAnd(wz, int(1)), int(1)));

    If(cx.lessThan(int(S)).and(cy.lessThan(int(S))).and(myPhase.equal(u.phase)), () => {
      const jitterK = u.round.mul(int(2));
      const pos = vec2(
        float(cx).add(fnCellRand({ gx: wx, gz: wz, level: u.level, seed: uSeedI, k: jitterK })),
        float(cy).add(
          fnCellRand({ gx: wx, gz: wz, level: u.level, seed: uSeedI, k: jitterK.add(int(1)) }),
        ),
      ).toVar();
      const dir = vec2(0, 0).toVar();
      const speed = float(1).toVar();
      const water = float(1).toVar();
      const sediment = float(0).toVar();

      Loop({ start: int(0), end: int(MAX_LIFETIME), type: 'int', condition: '<' }, () => {
        const node = floor(pos);
        const nx = int(node.x).toVar();
        const ny = int(node.y).toVar();

        // The 2×2 patch must exist, so the last legal cell is S-2.
        If(
          nx.lessThan(int(0))
            .or(ny.lessThan(int(0)))
            .or(nx.greaterThanEqual(int(S - 1)))
            .or(ny.greaterThanEqual(int(S - 1))),
          () => {
            Break();
          },
        );

        const uu = pos.x.sub(node.x).toVar();
        const vv = pos.y.sub(node.y).toVar();
        const cur = sampleField(nx, ny, uu, vv);
        const hOld = cur.h.toVar();

        const nd = dir
          .mul(INERTIA)
          .sub(vec2(cur.gx, cur.gy).mul(1 - INERTIA))
          .toVar();
        const len = length(nd).toVar();
        If(len.lessThan(float(1e-9)), () => {
          Break();
        });
        dir.assign(nd.div(len));

        const np = pos.add(dir).toVar();
        const mx = int(floor(np.x)).toVar();
        const my = int(floor(np.y)).toVar();
        If(
          mx.lessThan(int(0))
            .or(my.lessThan(int(0)))
            .or(mx.greaterThanEqual(int(S - 1)))
            .or(my.greaterThanEqual(int(S - 1))),
          () => {
            Break();
          },
        );
        const nu = np.x.sub(float(mx));
        const nv = np.y.sub(float(my));
        const hNew = sampleField(mx, my, nu, nv).h;

        const dh = hNew.sub(hOld).toVar();
        const capacity = max(dh.negate(), float(MIN_SLOPE))
          .mul(speed)
          .mul(water)
          .mul(CAPACITY_FACTOR)
          .toVar();

        If(sediment.greaterThan(capacity).or(dh.greaterThan(float(0))), () => {
          // Deposit. Uphill steps drop exactly enough to fill the step, which is
          // what turns a droplet that ran into a wall into a sediment fan rather
          // than a droplet that climbs it.
          const amount = dh
            .greaterThan(float(0))
            .select(dh.min(sediment), sediment.sub(capacity).mul(DEPOSIT_RATE))
            .toVar();
          sediment.subAssign(amount);
          // Bilinear onto the four corners of the cell just left, with the same
          // weights the height was sampled with — otherwise deposition drifts
          // off the path the droplet actually took.
          addHeight(nx, ny, amount.mul(float(1).sub(uu).mul(float(1).sub(vv))));
          addHeight(nx.add(int(1)), ny, amount.mul(uu.mul(float(1).sub(vv))));
          addHeight(nx, ny.add(int(1)), amount.mul(float(1).sub(uu).mul(vv)));
          addHeight(nx.add(int(1)), ny.add(int(1)), amount.mul(uu.mul(vv)));

          // Stop when the droplet has reached a basin it cannot fill.
          //
          // This is the correction the Jacobi generation structure forces, and
          // without it the pass does not produce channels at all — it produces
          // craters. Beyer's droplet writes into the heightmap and re-reads it,
          // so a droplet that digs a hollow immediately sees the hollow and
          // routes out of it. Ours reads a snapshot frozen for the whole
          // generation, so a droplet that spirals into a bowl keeps stepping
          // around the same minimum for the rest of its 60-step lifetime,
          // re-reading the *original* drop under it every time and scooping it
          // out again. Worse, oscillating across the minimum alternates the
          // sign of dh, and `speed² += dh·g` on the uphill halves pumps speed
          // without bound, which inflates capacity, which erodes harder.
          //
          // Measured before this break, at 2 m cells: the flow map was a
          // diffuse haze with round red hotspots instead of a drainage network,
          // and the height delta was a field of 10–20 texel pits sitting
          // exactly on those hotspots — the deepest incision in the tile was in
          // basins rather than in channels. A droplet in a sink has finished
          // its journey; the sediment it is still carrying gets laid down at
          // its resting place by the end-of-life brush below, which is the
          // alluvial fan we actually want at the foot of a slope.
          If(dh.greaterThan(float(0)).and(sediment.lessThan(float(1e-4))), () => {
            Break();
          });
        }).Else(() => {
          // Erode. The min against -dh is load-bearing: without it the droplet
          // digs a pit deeper than the drop it just made and falls straight
          // back into it, and the result is a field of spikes.
          const amount = capacity
            .sub(sediment)
            .mul(ERODE_RATE)
            .min(dh.negate())
            .max(float(0))
            .toVar();
          sediment.addAssign(amount);
          for (const b of BRUSH) {
            addHeight(nx.add(int(b.dx)), ny.add(int(b.dy)), amount.mul(-b.w));
          }
        });

        // Downhill ACCELERATES. `dh` is negative downhill, so the term is
        // subtracted, and this is the single change that made erosion do
        // anything at all.
        //
        // The published form adds it, and a previous pass here kept that sign
        // on the grounds that flipping it made peak height change explode from
        // 0.91 m to 272 m. That measurement was real; the conclusion was not.
        // With the published sign, a droplet on any slope steeper than
        // dh < −speed²/g — i.e. a 0.25-cell drop at speed 1, which is a 14°
        // slope, which is most of this world — takes `sqrt(max(0, negative))`
        // and its speed becomes exactly 0 on its FIRST step. Capacity is
        // proportional to speed, so capacity becomes 0, so the droplet can
        // never erode again: it spends the remaining 59 steps of its lifetime
        // depositing the one scoop it took at the start. Measured: 1 eroding
        // step out of 60, and a net height change of 0.1–0.15 cells — which is
        // precisely the "erosion runs but produces no visible carving" this
        // phase started with.
        //
        // The explosion the earlier measurement saw is real too, and it is what
        // MAX_SPEED is for: unbounded acceleration over a 60-step run makes
        // capacity unbounded. Clamping to a terminal velocity keeps the
        // physical behaviour (fast water in steep channels carries more) while
        // bounding the tail.
        speed.assign(
          min(float(MAX_SPEED), sqrt(max(float(0), speed.mul(speed).sub(dh.mul(GRAVITY))))),
        );
        water.mulAssign(1 - EVAPORATION);

        // Flow accumulates *water*, not visit count: a nearly evaporated droplet
        // is not a river. Recorded at the cell the droplet is leaving, so the
        // direction stored is the one it actually took out of that cell.
        const fi = idx(nx, ny);
        atomicAdd(flowAccum.element(fi), uint(max(float(0), water.mul(FLOW_SCALE))));
        atomicAdd(flowDirX.element(fi), int(round(dir.x.mul(water).mul(DIR_SCALE))));
        atomicAdd(flowDirZ.element(fi), int(round(dir.y.mul(water).mul(DIR_SCALE))));

        pos.assign(np);
      });

      // Drop whatever is still in transit when the droplet expires.
      //
      // Beyer's published loop simply abandons it, which is a mass leak, and at
      // this droplet density it is not a rounding error: measured, it lowered a
      // 16 m-cell tile by 5.7 m on average — more than the channel incision
      // itself — and, because the leak scales with cell size, it desynchronised
      // the LOD levels (0 m at 2 m cells, 5.7 m at 16 m). Depositing it also
      // earns the alluvial fans: droplets expire where the slope runs out, so
      // this is exactly the material that should be piling up at the foot of a
      // slope rather than vanishing.
      //
      // Spread over the erosion BRUSH, not bilinearly over four texels. This
      // is the whole load a droplet carried after up to 60 steps, i.e. the
      // largest single deposit in the simulation, and dropping it on four
      // texels builds a mound rather than a fan. Measured before this change:
      // the eroded surface read as a field of round 10–15 texel blisters —
      // droplets expire in clusters where the slope runs out, so their point
      // deposits pile up in the same place — and those blisters, not the
      // channels, were what the shading actually showed. Erosion removes over
      // 25 cells; deposition has to give back over the same 25 or the
      // asymmetry alone generates the lumps.
      const endNode = floor(pos);
      const ex = int(endNode.x);
      const ey = int(endNode.y);
      for (const b of BRUSH) {
        addHeight(ex.add(int(b.dx)), ey.add(int(b.dy)), sediment.mul(b.w));
      }
    });
  });

  // -------------------------------------------------------------------------
  // apply — fold one generation's delta into the field and reset it
  // -------------------------------------------------------------------------

  const applyKernel = Fn(() => {
    const gx = int(globalId.x);
    const gy = int(globalId.y);
    If(gx.lessThan(int(S)).and(gy.lessThan(int(S))), () => {
      const i = idx(gx, gy);
      const d = float(atomicLoad(hDelta.element(i))).mul(HEIGHT_QUANTUM);
      hField.element(i).assign(hField.element(i).add(clamp(d, u.genClamp.negate(), u.genClamp)));
      atomicStore(hDelta.element(i), int(0));
    });
  });

  // -------------------------------------------------------------------------
  // thermal — Musgrave talus, scattered through the same delta buffer
  // -------------------------------------------------------------------------

  const thermalKernel = Fn(() => {
    const gx = int(globalId.x);
    const gy = int(globalId.y);
    // The stencil reaches one cell out; the outermost ring is apron we discard.
    If(
      gx.greaterThan(int(0))
        .and(gy.greaterThan(int(0)))
        .and(gx.lessThan(int(S - 1)))
        .and(gy.lessThan(int(S - 1))),
      () => {
        const i = idx(gx, gy);
        const h = hField.element(i).toVar();

        // Loose sediment slides at 34°, bedrock stands at 45°. The distinction
        // is derived, not painted: cells the droplets built up are soil, cells
        // they scoured to bedrock are rock. That is exactly what puts scree fans
        // at the foot of a cliff while the cliff itself stays steep.
        const soilDepth = max(float(0), h.sub(hBase.element(i)));
        const tanAlpha = float(TAN_ROCK)
          .add(float(TAN_SOIL - TAN_ROCK).mul(saturate(soilDepth.div(SOIL_FULL_CELLS))))
          .toVar();

        const total = float(0).toVar();
        const peak = float(0).toVar();
        // Two sweeps over the eight neighbours: the first sizes the move, the
        // second distributes it. Cheaper than caching eight vars.
        for (const n of TALUS_NEIGHBOURS) {
          const hn = hField.element(idx(gx.add(int(n.dx)), gy.add(int(n.dy))));
          // The true centre-to-centre distance, not the cell size for all eight.
          // Using cellSize for diagonals makes them read √2 too steep, so they
          // always trigger first and every peak grows a four-pointed star.
          const excess = max(float(0), h.sub(hn).sub(tanAlpha.mul(n.d)));
          total.addAssign(excess);
          peak.assign(max(peak, excess));
        }

        If(total.greaterThan(float(0)), () => {
          // The ×0.5 is what stops the slope inverting: the cell drops by at
          // most 0.25·peak while the lowest neighbour rises by at most the same,
          // so the steepest excess halves per pass and never overshoots.
          const move = peak.mul(TALUS_FACTOR * 0.5).toVar();
          atomicAdd(hDelta.element(i), int(round(move.negate().mul(HEIGHT_SCALE))));
          for (const n of TALUS_NEIGHBOURS) {
            const ni = idx(gx.add(int(n.dx)), gy.add(int(n.dy)));
            const hn = hField.element(ni);
            const excess = max(float(0), h.sub(hn).sub(tanAlpha.mul(n.d)));
            atomicAdd(
              hDelta.element(ni),
              int(round(move.mul(excess).div(total).mul(HEIGHT_SCALE))),
            );
          }
        });
      },
    );
  });

  // -------------------------------------------------------------------------
  // smoothDelta — 3×3 tent on the erosion delta only
  // -------------------------------------------------------------------------

  /**
   * Why the delta and not the surface: the analytic base carries the mountain,
   * and blurring that would undo phase 1a. Only `hField − hBase` — what erosion
   * actually moved — is smoothed, so ridges stay as sharp as the height
   * function made them while the droplet residue stops being stipple.
   *
   * A 3×3 tent is the smallest kernel that removes single-texel spikes, and
   * single-texel spikes are exactly what a particle simulation on a lattice
   * produces: erosion is spread over a 25-cell brush but per-step deposition
   * lands bilinearly on four, so the deposit side is a full lattice octave
   * sharper than the scour side. Channels are three to six texels wide and
   * survive this untouched; the noise is one to two and does not.
   *
   * It runs on the whole scratch domain rather than the core so the core's
   * border ring has real neighbours — the same reason the apron exists.
   */
  const smoothDeltaKernel = Fn(() => {
    const gx = int(globalId.x);
    const gy = int(globalId.y);
    If(
      gx.greaterThan(int(0))
        .and(gy.greaterThan(int(0)))
        .and(gx.lessThan(int(S - 1)))
        .and(gy.lessThan(int(S - 1))),
      () => {
        const acc = float(0).toVar();
        // Tent weights 1-2-1 in each axis, normalised by 16.
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const w = (dx === 0 ? 2 : 1) * (dy === 0 ? 2 : 1);
            const j = idx(gx.add(int(dx)), gy.add(int(dy)));
            acc.addAssign(hField.element(j).sub(hBase.element(j)).mul(w / 16));
          }
        }
        // Parked in `blurTmp`, which the moisture blur has not touched yet —
        // `blurX` runs after this and overwrites it. Sharing the buffer avoids
        // a seventh 0.6 MB scratch allocation for a value with a two-dispatch
        // lifetime.
        blurTmp.element(idx(gx, gy)).assign(acc);
      },
    );
  });

  /** Write the smoothed delta back over the live field. Separate pass: the
   *  tent above must read the unsmoothed neighbours of every cell. */
  const applySmoothKernel = Fn(() => {
    const gx = int(globalId.x);
    const gy = int(globalId.y);
    If(
      gx.greaterThan(int(0))
        .and(gy.greaterThan(int(0)))
        .and(gx.lessThan(int(S - 1)))
        .and(gy.lessThan(int(S - 1))),
      () => {
        const i = idx(gx, gy);
        hField.element(i).assign(hBase.element(i).add(blurTmp.element(i)));
      },
    );
  });

  // -------------------------------------------------------------------------
  // blurX — first half of the separable moisture blur
  // -------------------------------------------------------------------------

  const gaussian: number[] = [];
  {
    let sum = 0;
    for (let k = -BLUR_RADIUS; k <= BLUR_RADIUS; k++) {
      const w = Math.exp((-k * k) / (2 * BLUR_SIGMA * BLUR_SIGMA));
      gaussian.push(w);
      sum += w;
    }
    for (let k = 0; k < gaussian.length; k++) gaussian[k] /= sum;
  }

  const blurXKernel = Fn(() => {
    const gx = int(globalId.x);
    const gy = int(globalId.y);
    If(gx.lessThan(int(S)).and(gy.lessThan(int(S))), () => {
      const acc = float(0).toVar();
      for (let k = -BLUR_RADIUS; k <= BLUR_RADIUS; k++) {
        const sx = clamp(gx.add(int(k)), int(0), int(S - 1));
        acc.addAssign(
          float(atomicLoad(flowAccum.element(idx(sx, gy))))
            .div(FLOW_SCALE * ROUNDS)
            .mul(gaussian[k + BLUR_RADIUS]),
        );
      }
      blurTmp.element(idx(gx, gy)).assign(acc);
    });
  });

  // -------------------------------------------------------------------------
  // resolve — core → array layers, with normals and the flow/moisture packing
  // -------------------------------------------------------------------------

  const resolveKernel = Fn(() => {
    const tx = int(globalId.x);
    const ty = int(globalId.y);
    If(tx.lessThan(int(TILE_TEXELS)).and(ty.lessThan(int(TILE_TEXELS))), () => {
      const sx = tx.add(int(APRON));
      const sy = ty.add(int(APRON));

      // Erosion depth is converted with `outScale`, not `spacing` — see
      // REFERENCE_CELL_M for why the two differ at coarse levels.
      const height = (x: ShaderNode, y: ShaderNode): ShaderNode => {
        const i = idx(x, y);
        return hBase
          .element(i)
          .mul(u.spacing)
          .add(hField.element(i).sub(hBase.element(i)).mul(u.outScale));
      };

      const i = idx(sx, sy);
      const h = height(sx, sy).toVar();

      textureStore(
        storageTexture(pool.heightArray).depth(u.layer),
        uvec2(uint(tx), uint(ty)),
        vec4(h, 0, 0, 1),
      );

      // Normals off the *eroded* field. The apron means even the border ring has
      // real neighbours, so this has no analytic fallback — which is the whole
      // reason the un-eroded bake's one-texel border seam disappears here.
      const inv = float(1).div(u.spacing.mul(2));
      const dhdx = height(sx.add(int(1)), sy).sub(height(sx.sub(int(1)), sy)).mul(inv);
      const dhdz = height(sx, sy.add(int(1))).sub(height(sx, sy.sub(int(1)))).mul(inv);
      textureStore(
        storageTexture(pool.surfaceArray).depth(u.layer),
        uvec2(uint(tx), uint(ty)),
        vec4(normalize(vec3(dhdx.negate(), float(1), dhdz.negate())), length(vec2(dhdx, dhdz))),
      );

      // R — flow accumulation, log-normalised against a fixed global scale.
      const raw = float(atomicLoad(flowAccum.element(i))).div(FLOW_SCALE * ROUNDS);
      const flow = saturate(log2(raw.add(1)).div(FLOW_LOG_DIVISOR));

      // G — flow direction as an angle. Stored rather than derived from ∇h
      // because a deposited channel bed is locally flat and its height gradient
      // says nothing about where the water went.
      const fx = float(atomicLoad(flowDirX.element(i))).div(DIR_SCALE);
      const fz = float(atomicLoad(flowDirZ.element(i))).div(DIR_SCALE);
      const angle = atan(fz, fx).div(Math.PI * 2).add(0.5);

      // B — moisture: the Y half of the separable blur, same normalisation as R.
      const wet = float(0).toVar();
      for (let k = -BLUR_RADIUS; k <= BLUR_RADIUS; k++) {
        const yy = clamp(sy.add(int(k)), int(0), int(S - 1));
        wet.addAssign(blurTmp.element(idx(sx, yy)).mul(gaussian[k + BLUR_RADIUS]));
      }
      const moisture = saturate(log2(wet.add(1)).div(FLOW_LOG_DIVISOR));

      // A — signed cut/fill in metres, ±CUTFILL_RANGE_M mapped to [0,1]. One
      // channel, three consumers: the talus angle above, phase 2's soil/rock
      // splat mask, and phase 7's "nothing grows on bare rock".
      const cut = saturate(h.sub(hBase.element(i).mul(u.spacing)).div(2 * CUTFILL_RANGE_M).add(0.5));

      textureStore(
        storageTexture(pool.flowArray).depth(u.layer),
        uvec2(uint(tx), uint(ty)),
        vec4(flow, angle, moisture, cut),
      );

      // Tally what was actually stored, not what was meant to be. Read back
      // once by `Erosion` after the first tile completes.
      If(flow.greaterThan(float(0)), () => {
        atomicAdd(verifyCount.element(int(0)), uint(1));
      });
    });
  });

  const kernels: ErosionKernels = {
    seed: computeKernel(seedKernel(), [WORKGROUP, WORKGROUP, 1]),
    droplet: computeKernel(dropletKernel(), [WORKGROUP, WORKGROUP, 1]),
    apply: computeKernel(applyKernel(), [WORKGROUP, WORKGROUP, 1]),
    thermal: computeKernel(thermalKernel(), [WORKGROUP, WORKGROUP, 1]),
    smoothDelta: computeKernel(smoothDeltaKernel(), [WORKGROUP, WORKGROUP, 1]),
    applySmooth: computeKernel(applySmoothKernel(), [WORKGROUP, WORKGROUP, 1]),
    blurX: computeKernel(blurXKernel(), [WORKGROUP, WORKGROUP, 1]),
    resolve: computeKernel(resolveKernel(), [WORKGROUP, WORKGROUP, 1]),
  };
  kernels.seed.setName('erosionSeed');
  kernels.droplet.setName('erosionDroplet');
  kernels.apply.setName('erosionApply');
  kernels.thermal.setName('erosionThermal');
  kernels.smoothDelta.setName('erosionSmoothDelta');
  kernels.applySmooth.setName('erosionApplySmooth');
  kernels.blurX.setName('erosionBlurX');
  kernels.resolve.setName('erosionResolve');

  const attributes = [hField, hBase, hDelta, flowAccum, flowDirX, flowDirZ, blurTmp].map(
    (b) => b.value as THREE.StorageInstancedBufferAttribute,
  );

  return {
    uniforms: u,
    kernels,
    attributes,
    verifyAttribute: verifyCount.value as THREE.StorageInstancedBufferAttribute,
    bytes: SCRATCH_CELLS * 4 * attributes.length,
  };
}

export const EROSION_TUNING = {
  APRON,
  SCRATCH_TEXELS,
  PHASES,
  ROUNDS,
  THERMAL_ITERATIONS,
  MAX_LIFETIME,
  EROSION_RADIUS,
  REFERENCE_CELL_M,
  GEN_CLAMP_CELLS,
  THERMAL_CLAMP_CELLS,
  CUTFILL_RANGE_M,
  /** Droplets simulated per round: one per scratch cell. */
  DROPLETS_PER_ROUND: SCRATCH_CELLS,
} as const;
