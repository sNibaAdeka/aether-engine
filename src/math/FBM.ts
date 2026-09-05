/**
 * Fractal noise machinery — the layer between raw noise and terrain.
 *
 * Mirrors `terrain/shaders/Noise.wgsl.ts`. Any change here must be reflected
 * there (and vice versa) or CPU height queries will drift from the rendered
 * surface. `tests/unit/noise.test.ts` guards determinism; the CPU/GPU parity
 * gate runs in the browser (see `scripts/parity.ts`).
 */

import { perlin2, perlin2d, worley2F1 } from './Noise';

const _d = new Float64Array(3);

/**
 * Amplitude normaliser.
 *
 * Deliberately the *limit* of the geometric series (1/(1−gain)) rather than the
 * running sum of the octaves actually evaluated. That distinction matters more
 * than it looks:
 *
 * With running-sum normalisation, a 4-octave evaluation and an 8-octave
 * evaluation of the same point return different values — the base octave gets
 * scaled down as detail is added. Terrain LOD then cannot drop octaves at
 * distance without the mountain silhouette visibly changing height as you walk
 * toward it. With limit normalisation, octave 0 always contributes exactly the
 * same amount and extra octaves only add detail on top. Octave count becomes a
 * pure quality knob.
 *
 * Cost: the output never quite reaches ±1 (6 octaves at gain 0.5 peaks at
 * ~0.984 × the noise max). Callers that need a full-range signal should remap.
 */
function fbmScale(gain: number, octaves: number): number {
  if (gain >= 1) return octaves > 0 ? 1 / octaves : 1;
  return 1 - gain;
}

export interface FbmParams {
  /** Number of octaves. */
  octaves: number;
  /** Frequency multiplier per octave. Non-integer avoids lattice alignment. */
  lacunarity: number;
  /** Amplitude multiplier per octave. */
  gain: number;
  /** Base frequency (1 / feature size in world units). */
  frequency: number;
  /** Base amplitude. */
  amplitude: number;
}

export const DEFAULT_FBM: FbmParams = {
  octaves: 6,
  lacunarity: 2.01734,
  gain: 0.5,
  frequency: 1,
  amplitude: 1,
};

/** Standard fractal Brownian motion. Output range ≈ [-1, 1] for gain 0.5. */
export function fbm2(x: number, y: number, seed: number, p: FbmParams): number {
  let sum = 0;
  let amp = 1;
  let freq = p.frequency;
  for (let i = 0; i < p.octaves; i++) {
    sum += perlin2(x * freq, y * freq, seed + i * 1013) * amp;
    amp *= p.gain;
    freq *= p.lacunarity;
  }
  return sum * fbmScale(p.gain, p.octaves) * p.amplitude;
}

/**
 * Ridged multifractal: `1 - |noise|`, sharpened by `ridgeSharpness` and with
 * each octave weighted by the previous one. The weighting is what produces
 * continuous ridge lines instead of a field of disconnected spikes.
 *
 * Output range [0, 1], 1 = ridge crest.
 */
export function ridged2(
  x: number,
  y: number,
  seed: number,
  p: FbmParams,
  ridgeSharpness = 2.0,
  weightFactor = 0.9,
): number {
  let sum = 0;
  let amp = 1;
  let freq = p.frequency;
  let weight = 1.0;

  for (let i = 0; i < p.octaves; i++) {
    let n = perlin2(x * freq, y * freq, seed + i * 1013);
    n = 1 - Math.abs(n);
    n = Math.pow(n, ridgeSharpness);
    // Musgrave's recurrence: this octave is attenuated by how ridge-like the
    // *previous* octave was, which is what makes crests continue as lines
    // instead of breaking into a field of unrelated spikes.
    const contribution = n * weight;
    weight = Math.min(1, Math.max(0, contribution * weightFactor));

    sum += contribution * amp;
    amp *= p.gain;
    freq *= p.lacunarity;
  }
  return sum * fbmScale(p.gain, p.octaves) * p.amplitude;
}

/**
 * Erosion-damped fBm (Iñigo Quílez). Each octave's contribution is attenuated
 * by the accumulated gradient so far, so detail collapses on steep faces and
 * survives on flats — the signature of a water-worn landscape.
 *
 * Value only — no analytic gradient.
 *
 * An earlier version exposed one, but the damping term depends on the
 * accumulated slope, so differentiating it exactly requires the Hessian of
 * every previous octave. Treating `damp` as locally constant was measured at
 * 6× relative error against finite differences: enough to shade a cliff as if
 * it were flat. Terrain normals come from central differences on the generated
 * heightmap anyway (both on GPU and in `gradientCentral2` below), which is
 * cheaper *and* guarantees the normals match the surface that is actually
 * rendered. Use that.
 */
export function fbmEroded2(
  x: number,
  y: number,
  seed: number,
  p: FbmParams,
  erosionStrength = 1.0,
): number {
  let sum = 0;
  let amp = 1;
  let freq = p.frequency;

  // Slope accumulator that drives the damping. Deliberately *dimensionless*:
  // the raw octave derivative scales with `freq`, so feeding it straight into
  // `1 + k·|∇|²` would make `erosionStrength` mean something different at every
  // frequency — a terrain tuned at 1/500 m would go completely flat when the
  // same parameters were reused for a 1/20 m detail layer. Normalising by the
  // octave frequency makes the control frequency-independent.
  let slopeX = 0;
  let slopeY = 0;

  for (let i = 0; i < p.octaves; i++) {
    perlin2d(x * freq, y * freq, seed + i * 1013, _d);

    slopeX += _d[1];
    slopeY += _d[2];

    const damp = 1 / (1 + erosionStrength * (slopeX * slopeX + slopeY * slopeY));
    sum += _d[0] * amp * damp;

    amp *= p.gain;
    freq *= p.lacunarity;
  }

  return sum * fbmScale(p.gain, p.octaves) * p.amplitude;
}

/**
 * Central-difference gradient of an arbitrary height function.
 *
 * The single source of truth for CPU-side slope: physics, biome
 * classification and vegetation placement all use this, so they agree with the
 * GPU normals (which are produced the same way from the heightmap) by
 * construction rather than by coincidence.
 *
 * `eps` should match the heightmap texel spacing of the tile being queried;
 * a mismatch shows up as normals that are subtly too smooth or too sharp.
 */
export function gradientCentral2(
  height: (x: number, y: number) => number,
  x: number,
  y: number,
  eps: number,
  out: Float64Array,
): void {
  const inv = 1 / (2 * eps);
  out[0] = (height(x + eps, y) - height(x - eps, y)) * inv;
  out[1] = (height(x, y + eps) - height(x, y - eps)) * inv;
}

/**
 * Domain warping — distorts the sample position with two more noise fields.
 * Without it ridged noise looks like random static; with it you get the
 * sweeping, folded appearance of real mountain chains.
 *
 * Writes the warped coordinates into `out`.
 */
const _warpParams: FbmParams = {
  octaves: 4,
  lacunarity: 2.01734,
  gain: 0.5,
  frequency: 1,
  amplitude: 1,
};

export function domainWarp2(
  x: number,
  y: number,
  seed: number,
  strength: number,
  frequency: number,
  out: Float64Array,
): void {
  // Reused module-level params: this runs once per terrain sample, i.e. tens of
  // millions of times during a tile generation pass on the CPU path.
  _warpParams.frequency = frequency;
  const wx = fbm2(x + 137.31, y + 41.77, seed ^ 0x5f3a, _warpParams);
  const wy = fbm2(x - 91.13, y + 217.59, seed ^ 0x1c9d, _warpParams);
  out[0] = x + wx * strength;
  out[1] = y + wy * strength;
}

/** Billowy noise — |fbm|, gives rounded, cloud/dune-like forms. */
export function billow2(x: number, y: number, seed: number, p: FbmParams): number {
  let sum = 0;
  let amp = 1;
  let freq = p.frequency;
  for (let i = 0; i < p.octaves; i++) {
    sum += Math.abs(perlin2(x * freq, y * freq, seed + i * 1013)) * amp;
    amp *= p.gain;
    freq *= p.lacunarity;
  }
  return (sum * fbmScale(p.gain, p.octaves) * 2 - 1) * p.amplitude;
}

/** Cellular fBm — sums Worley F1 octaves. Used for scree and forest clumping. */
export function worleyFbm2(x: number, y: number, seed: number, p: FbmParams): number {
  let sum = 0;
  let amp = 1;
  let freq = p.frequency;
  for (let i = 0; i < p.octaves; i++) {
    sum += worley2F1(x * freq, y * freq, seed + i * 7919) * amp;
    amp *= p.gain;
    freq *= p.lacunarity;
  }
  return sum * fbmScale(p.gain, p.octaves) * p.amplitude;
}

// ---------------------------------------------------------------------------
// Small helpers shared with the shaders
// ---------------------------------------------------------------------------

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function saturate(v: number): number {
  return clamp(v, 0, 1);
}

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = saturate((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

export function mix(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Remap `v` from [inMin,inMax] to [outMin,outMax] without clamping. */
export function remap(
  v: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number,
): number {
  return outMin + ((v - inMin) / (inMax - inMin)) * (outMax - outMin);
}
