/**
 * The terrain height function — CPU reference implementation.
 *
 * `height(x, z)` is a pure function of world coordinates and the global seed.
 * It is evaluated here for physics, height queries and worker-side tile work,
 * and in WGSL (`shaders/heightfield.wgsl.ts`) for rendering and compute tile
 * generation. **The two must agree to within 0.05 m.** If they drift, the
 * player floats above or sinks through the rendered surface — a bug that is
 * miserable to track down later, so `tests/e2e/parity.spec.ts` checks it
 * against a GPU readback on every run.
 *
 * Any edit here needs the same edit there. The WGSL file mirrors this one
 * function-for-function and line-for-line on purpose, ugly as that is; the
 * alternative is generating one from the other, which trades a mechanical
 * duplication problem for a debugging problem.
 *
 * Layer order (section 1.1 of the spec) — the order matters, each layer is
 * modulated by the ones above it:
 *   1. continentality  — where is ocean, lowland, highland
 *   2. erosion factor  — are the mountains old and rounded or young and sharp
 *   3. ridges          — the mountain chains themselves
 *   4. domain warping  — folds the ridges so they read as geology, not noise
 *   5. valleys         — cut across the ridges
 *   6. detail          — slope-modulated high frequency
 */

import { perlin2 } from '@/math/Noise';
import {
  fbm2,
  ridged2,
  billow2,
  saturate,
  smoothstep,
  mix,
  clamp,
  type FbmParams,
} from '@/math/FBM';

// ---------------------------------------------------------------------------
// World constants
// ---------------------------------------------------------------------------

/** Sea level, world Y. Everything below is ocean or lake bed. */
export const SEA_LEVEL = 0;

/** Vertical extent used to normalise heights for storage and debug colouring. */
export const MIN_HEIGHT = -420;
export const MAX_HEIGHT = 980;

/** Wavelength of the continental signal, metres. ~20 km per the spec. */
const CONTINENT_PERIOD = 20000;
/** Wavelength of the erosion-age signal. */
const EROSION_PERIOD = 9000;
/** Domain warp displacement in metres. */
const WARP_STRENGTH = 120;
const WARP_PERIOD = 3000;
/** Mountain chain wavelength. */
const RIDGE_PERIOD = 2600;
const VALLEY_PERIOD = 1700;
const DETAIL_PERIOD = 90;

// Seed offsets. Distinct large primes so the layers stay uncorrelated — reusing
// a seed between two layers produces terrain where ridges and valleys line up
// suspiciously, which reads as artificial even when you cannot say why.
const S_CONTINENT = 0x1f35;
const S_EROSION = 0x7a11;
const S_WARP_X = 0x5f3a;
const S_WARP_Y = 0x1c9d;
const S_RIDGE = 0x2b73;
const S_VALLEY = 0x63d1;
const S_DETAIL = 0x4e87;
const S_TERRACE = 0x39af;

const P_CONTINENT: FbmParams = {
  octaves: 4,
  lacunarity: 2.01734,
  gain: 0.5,
  frequency: 1 / CONTINENT_PERIOD,
  amplitude: 1,
};

const P_EROSION: FbmParams = {
  octaves: 3,
  lacunarity: 2.01734,
  gain: 0.5,
  frequency: 1 / EROSION_PERIOD,
  amplitude: 1,
};

const P_WARP: FbmParams = {
  octaves: 3,
  lacunarity: 2.01734,
  gain: 0.5,
  frequency: 1 / WARP_PERIOD,
  amplitude: 1,
};

const P_RIDGE: FbmParams = {
  octaves: 8,
  lacunarity: 2.01734,
  gain: 0.5,
  frequency: 1 / RIDGE_PERIOD,
  amplitude: 1,
};

const P_VALLEY: FbmParams = {
  octaves: 5,
  lacunarity: 2.01734,
  gain: 0.5,
  frequency: 1 / VALLEY_PERIOD,
  amplitude: 1,
};

const P_DETAIL: FbmParams = {
  octaves: 5,
  lacunarity: 2.01734,
  gain: 0.5,
  frequency: 1 / DETAIL_PERIOD,
  amplitude: 1,
};

// ---------------------------------------------------------------------------
// Continental shaping
// ---------------------------------------------------------------------------

/**
 * Map raw continental noise [-1,1] to an elevation in metres.
 *
 * The curve is deliberately not linear. It has a shelf either side of sea
 * level so the coastline produces broad beaches and a continental shelf rather
 * than a wall dropping straight into the water. Getting this wrong is the
 * single most common way procedural worlds end up looking like a bathtub.
 */
/**
 * Spread raw fBm across the usable range before the elevation curve sees it.
 *
 * fBm is a sum of independent octaves, so its output is roughly Gaussian with
 * σ ≈ 0.12 — over a 50 km transect it spans about ±0.37, never the ±1 the curve
 * below is written against. Feeding it in raw produces a world whose highest
 * point is 180 m: technically terrain, visually a car park.
 *
 * The soft saturation `x/√(1+x²)` is used instead of a hard clamp so that the
 * extremes compress into broad highlands and deep basins rather than snapping
 * to dead-flat plateaus at exactly the clamp value.
 */
export function continentShape(raw: number): number {
  const g = raw * 3.0;
  return g / Math.sqrt(1 + g * g);
}

export function continentCurve(n: number): number {
  // Ocean basin: steep down, then flattening into an abyssal plain.
  if (n < -0.12) {
    const t = saturate((-0.12 - n) / 0.88);
    return mix(-25, MIN_HEIGHT, t * t * (3 - 2 * t));
  }
  // Shelf and shoreline: nearly flat band around sea level.
  if (n < 0.08) {
    const t = saturate((n + 0.12) / 0.2);
    return mix(-25, 18, t);
  }
  // Lowland → plateau → high ground, accelerating.
  const t = saturate((n - 0.08) / 0.92);
  const eased = t * t * (0.55 + 0.45 * t);
  return mix(18, MAX_HEIGHT, eased);
}

// ---------------------------------------------------------------------------
// The height function
// ---------------------------------------------------------------------------

export interface HeightSample {
  /** World-space height, metres. */
  height: number;
  /** Continentality in [-1,1] — drives biome temperature and ocean masks. */
  continent: number;
  /** Erosion age in [0,1]: 0 = young and sharp, 1 = old and rounded. */
  erosion: number;
  /** Ridge intensity in [0,1] at this point. */
  ridge: number;
}

const _sample: HeightSample = { height: 0, continent: 0, erosion: 0, ridge: 0 };

/**
 * Full evaluation. Returns a shared object — copy what you need, do not retain.
 * `height()` below is the allocation-free scalar path used in hot loops.
 */
export function sampleHeight(x: number, z: number, seed: number): HeightSample {
  // 1. Continentality.
  const cn = continentShape(fbm2(x, z, seed + S_CONTINENT, P_CONTINENT));
  const base = continentCurve(cn);

  // 2. Erosion age. Old ranges are rounded and low-relief, young ones sharp.
  // Same Gaussian caveat: ×2.6 before the 0..1 remap, or every point in the
  // world reports "medium age" and the erosion parameter does nothing.
  const erosion = saturate(fbm2(x, z, seed + S_EROSION, P_EROSION) * 2.6 * 0.5 + 0.5);

  // Mountains only grow on land, and grow faster the higher the base already
  // is — this is what stops isolated peaks sprouting out of the ocean.
  const landMask = smoothstep(-30, 220, base);
  const reliefScale = mix(1.0, 0.34, erosion) * landMask;

  if (reliefScale <= 0.0005) {
    // Deep ocean: skip the expensive layers entirely. Worth it — most of the
    // world map is water and this is evaluated millions of times per tile.
    const oceanDetail = fbm2(x, z, seed + S_DETAIL, P_DETAIL) * 3.0;
    _sample.height = base + oceanDetail;
    _sample.continent = cn;
    _sample.erosion = erosion;
    _sample.ridge = 0;
    return _sample;
  }

  // 3+4. Domain warp, then ridges in the warped domain.
  const wx = fbm2(x + 137.31, z + 41.77, seed + S_WARP_X, P_WARP) * WARP_STRENGTH;
  const wz = fbm2(x - 91.13, z + 217.59, seed + S_WARP_Y, P_WARP) * WARP_STRENGTH;
  const rx = x + wx;
  const rz = z + wz;

  // Sharper crests on young ranges, softer on old ones.
  const sharpness = mix(2.6, 1.5, erosion);
  const ridge = ridged2(rx, rz, seed + S_RIDGE, P_RIDGE, sharpness, 0.9);
  const ridgeHeight = ridge * ridge * 620 * reliefScale;

  // 5. Valleys — inverted ridged structures running across the chains, cut in
  // rather than added on. Amplitude scales with relief so flats stay flat.
  const valley = ridged2(rz * 0.87 - 512.3, rx * 0.87 + 88.1, seed + S_VALLEY, P_VALLEY, 1.8, 0.85);
  const valleyDepth = valley * valley * 165 * reliefScale * mix(0.55, 1.0, erosion);

  let h = base + ridgeHeight - valleyDepth;

  // 6. Detail, modulated by slope: scree and broken rock on steep faces,
  // smooth alluvium in the flats. Approximated from ridge intensity, which
  // correlates with slope and costs nothing extra.
  const steep = saturate(ridge * 1.4);
  const detailAmp = mix(0.5, 4.0, steep) * landMask;
  h += fbm2(x, z, seed + S_DETAIL, P_DETAIL) * detailAmp;

  // 8. Terracing on ~15% of the land, gated by a low-frequency mask. Sedimentary
  // banding; subtle, but it makes big rock faces read as layered stone rather
  // than as an extruded noise field.
  const terraceMask = smoothstep(0.62, 0.78, saturate(perlin2(x / 5200, z / 5200, seed + S_TERRACE) * 0.5 + 0.5));
  if (terraceMask > 0.001) {
    const step = 26.0;
    const q = Math.floor(h / step);
    const f = h / step - q;
    // Smoothstep the riser so the terrace edge is a bevel, not a staircase.
    const shaped = (q + smoothstep(0.35, 0.72, f)) * step;
    // Weight 0.18, not 0.55: at higher strengths the banding reads as contour
    // lines on a topographic map rather than as sedimentary layering, and it is
    // most obvious exactly where it should be least — on broad gentle slopes.
    // Gated on steepness so it only shows on rock faces, where real bedding is.
    const bedding = smoothstep(0.25, 0.65, steep);
    h = mix(h, shaped, terraceMask * 0.18 * landMask * bedding);
  }

  _sample.height = h;
  _sample.continent = cn;
  _sample.erosion = erosion;
  _sample.ridge = ridge;
  return _sample;
}

/** Scalar height. Same function, no object churn — use this in hot loops. */
export function height(x: number, z: number, seed: number): number {
  return sampleHeight(x, z, seed).height;
}

/**
 * Surface normal from central differences.
 *
 * `eps` must match the heightmap texel spacing the GPU used, or CPU normals
 * come out subtly smoother or sharper than the shaded surface. Writes a unit
 * vector into `out`.
 */
export function normalAt(
  x: number,
  z: number,
  seed: number,
  eps: number,
  out: Float64Array,
): void {
  const hL = height(x - eps, z, seed);
  const hR = height(x + eps, z, seed);
  const hD = height(x, z - eps, seed);
  const hU = height(x, z + eps, seed);

  const nx = hL - hR;
  const nz = hD - hU;
  const ny = 2 * eps;
  const inv = 1 / Math.sqrt(nx * nx + ny * ny + nz * nz);

  out[0] = nx * inv;
  out[1] = ny * inv;
  out[2] = nz * inv;
}

/** Slope in radians from horizontal, 0 = flat. */
export function slopeAt(x: number, z: number, seed: number, eps: number): number {
  normalAt(x, z, seed, eps, _n);
  return Math.acos(clamp(_n[1], -1, 1));
}

const _n = new Float64Array(3);

/**
 * Find a good spawn point: on land, near water, with an open outlook and not
 * on bare rock. Walking a coarse grid is fine — this runs once at startup.
 */
export function findSpawn(seed: number, searchRadius = 22000): { x: number; z: number; y: number } {
  let bestScore = -Infinity;
  let bx = 0;
  let bz = 0;
  let by = 0;

  const step = searchRadius / 24;
  for (let gz = -12; gz <= 12; gz++) {
    for (let gx = -12; gx <= 12; gx++) {
      const x = gx * step;
      const z = gz * step;
      const h = height(x, z, seed);
      if (h < 6 || h > 260) continue;

      const slope = slopeAt(x, z, seed, 8);
      if (slope > 0.35) continue;

      // Prefer being close to — but not in — water, and moderately elevated so
      // the first thing the player sees is a view rather than a hillside.
      let minWaterDist = 1e9;
      for (let a = 0; a < 8; a++) {
        const ang = (a / 8) * Math.PI * 2;
        for (let d = 120; d < 1800; d += 180) {
          const sx = x + Math.cos(ang) * d;
          const sz = z + Math.sin(ang) * d;
          if (height(sx, sz, seed) < SEA_LEVEL) {
            minWaterDist = Math.min(minWaterDist, d);
            break;
          }
        }
      }

      // Relief within sight: the difference between the highest and lowest
      // ground in a 4 km ring. Without this the spawn optimiser happily picks a
      // flat coastal plain — it satisfies every other criterion and there is
      // nothing whatsoever to look at.
      let hiNear = -1e9;
      let loNear = 1e9;
      for (let a = 0; a < 8; a++) {
        const ang = (a / 8) * Math.PI * 2;
        for (const d of [900, 2200, 4000]) {
          const nh = height(x + Math.cos(ang) * d, z + Math.sin(ang) * d, seed);
          if (nh > hiNear) hiNear = nh;
          if (nh < loNear) loNear = nh;
        }
      }
      const reliefScore = smoothstep(80, 500, hiNear - loNear);

      const waterScore = minWaterDist < 1e8 ? smoothstep(1800, 300, minWaterDist) : 0;
      const heightScore = smoothstep(6, 90, h) * smoothstep(260, 120, h);
      const flatScore = 1 - saturate(slope / 0.35);
      const score = waterScore * 1.2 + heightScore + flatScore * 0.5 + reliefScore * 2.0;

      if (score > bestScore) {
        bestScore = score;
        bx = x;
        bz = z;
        by = h;
      }
    }
  }
  return { x: bx, z: bz, y: by };
}

/** Unused import guard — `billow2` is reserved for the dune biome (phase 2). */
void billow2;
