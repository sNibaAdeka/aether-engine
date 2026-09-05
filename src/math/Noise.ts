/**
 * Deterministic noise primitives — CPU reference implementation.
 *
 * Every function here has a bit-for-bit algorithmic twin in `Noise.wgsl.ts`.
 * The two must stay in lockstep: the terrain height field is evaluated on the
 * GPU for rendering and on the CPU for physics/height queries, and a mismatch
 * means the player floats or sinks. See `tests/unit/noise.test.ts`.
 *
 * Parity strategy:
 *  - integer hashing only (no trig, no Math.random), so the bit patterns match
 *  - gradient vectors come from a constant table rather than sin/cos
 *  - JS accumulates in f64 where WGSL uses f32; the resulting relative error is
 *    ~1e-7, i.e. sub-millimetre on a 1 km mountain. Well inside the 0.05 m gate.
 */

// ---------------------------------------------------------------------------
// Integer hashing
// ---------------------------------------------------------------------------

/** Bit-avalanche hash (Chris Wellons' `lowbias32`, 3-round variant). */
export function hashU32(x: number): number {
  let h = x >>> 0;
  h ^= h >>> 17;
  h = Math.imul(h, 0xed5ad4bb) >>> 0;
  h ^= h >>> 11;
  h = Math.imul(h, 0xac4c1b51) >>> 0;
  h ^= h >>> 15;
  h = Math.imul(h, 0x31848bab) >>> 0;
  h ^= h >>> 14;
  return h >>> 0;
}

export function hash2i(x: number, y: number, seed: number): number {
  // Weyl-style mixing of the lattice coordinates before the avalanche, so that
  // (x,y) and (y,x) do not collide and negative coordinates behave.
  const k = (Math.imul(x | 0, 0x27d4eb2d) ^ Math.imul(y | 0, 0x165667b1) ^ (seed | 0)) >>> 0;
  return hashU32(k);
}

export function hash3i(x: number, y: number, z: number, seed: number): number {
  const k =
    (Math.imul(x | 0, 0x27d4eb2d) ^
      Math.imul(y | 0, 0x165667b1) ^
      Math.imul(z | 0, 0x9e3779b1) ^
      (seed | 0)) >>>
    0;
  return hashU32(k);
}

/** Hash → [0,1) with 24 bits of mantissa (matches WGSL's `f32(h >> 8) / 16777216`). */
export function hashToUnit(h: number): number {
  return (h >>> 8) / 16777216;
}

export function rand2(x: number, y: number, seed: number): number {
  return hashToUnit(hash2i(x, y, seed));
}

export function rand3(x: number, y: number, z: number, seed: number): number {
  return hashToUnit(hash3i(x, y, z, seed));
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

/** Quintic fade 6t⁵−15t⁴+10t³ — C² continuous, required for smooth normals. */
export function fade(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

/** Derivative of `fade`, used by the analytic-derivative noise variants. */
export function fadeDeriv(t: number): number {
  return 30 * t * t * (t * (t - 2) + 1);
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

// ---------------------------------------------------------------------------
// Gradient tables
// ---------------------------------------------------------------------------

const R2 = 0.7071067811865476;

/**
 * 8 unit vectors: the four axes and the four diagonals.
 *
 * Chosen over a 16-direction table so the GPU mirror can derive the gradient
 * from three sign bits with `select`, instead of either indexing a
 * function-scope array (which spills to memory) or calling `cos`/`sin` (eight
 * transcendentals per lattice corner — this function is evaluated ~32 times per
 * terrain sample, so that is the difference between a viable vertex shader and
 * an unusable one).
 *
 * The cost is slightly more axis-aligned structure than a 16-direction set.
 * Domain warping and eight octaves bury it; the classic reference
 * implementations use this set for the same reason.
 */
export const GRAD2: ReadonlyArray<readonly [number, number]> = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
  [R2, R2],
  [-R2, R2],
  [R2, -R2],
  [-R2, -R2],
];

/**
 * 12 gradients pointing at the midpoints of a cube's edges — Ken Perlin's
 * "improved noise" set. Not normalised (|g| = √2); the noise is rescaled below.
 */
export const GRAD3: ReadonlyArray<readonly [number, number, number]> = [
  [1, 1, 0],
  [-1, 1, 0],
  [1, -1, 0],
  [-1, -1, 0],
  [1, 0, 1],
  [-1, 0, 1],
  [1, 0, -1],
  [-1, 0, -1],
  [0, 1, 1],
  [0, -1, 1],
  [0, 1, -1],
  [0, -1, -1],
];

// ---------------------------------------------------------------------------
// Value noise
// ---------------------------------------------------------------------------

/** 2D value noise in [-1,1]. Cheap; used for masks and low-stakes modulation. */
export function valueNoise2(x: number, y: number, seed: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const ux = fade(fx);
  const uy = fade(fy);

  const v00 = rand2(ix, iy, seed);
  const v10 = rand2(ix + 1, iy, seed);
  const v01 = rand2(ix, iy + 1, seed);
  const v11 = rand2(ix + 1, iy + 1, seed);

  const a = lerp(v00, v10, ux);
  const b = lerp(v01, v11, ux);
  return lerp(a, b, uy) * 2 - 1;
}

// ---------------------------------------------------------------------------
// Perlin gradient noise
// ---------------------------------------------------------------------------

/** 2D Perlin noise, normalised to roughly [-1,1]. */
export function perlin2(x: number, y: number, seed: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;

  const g00 = GRAD2[hash2i(ix, iy, seed) & 7];
  const g10 = GRAD2[hash2i(ix + 1, iy, seed) & 7];
  const g01 = GRAD2[hash2i(ix, iy + 1, seed) & 7];
  const g11 = GRAD2[hash2i(ix + 1, iy + 1, seed) & 7];

  const n00 = g00[0] * fx + g00[1] * fy;
  const n10 = g10[0] * (fx - 1) + g10[1] * fy;
  const n01 = g01[0] * fx + g01[1] * (fy - 1);
  const n11 = g11[0] * (fx - 1) + g11[1] * (fy - 1);

  const ux = fade(fx);
  const uy = fade(fy);
  return lerp(lerp(n00, n10, ux), lerp(n01, n11, ux), uy) * 1.4142135623730951;
}

/**
 * 2D Perlin noise with analytic derivatives — returns [value, d/dx, d/dy].
 *
 * Derivatives matter more than they look: `fbmErodedDeriv` uses them to damp
 * octave amplitude on steep slopes, which is what turns generic fractal noise
 * into something that reads as eroded terrain (Iñigo Quílez's technique).
 */
export function perlin2d(x: number, y: number, seed: number, out: Float64Array): void {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;

  const g00 = GRAD2[hash2i(ix, iy, seed) & 7];
  const g10 = GRAD2[hash2i(ix + 1, iy, seed) & 7];
  const g01 = GRAD2[hash2i(ix, iy + 1, seed) & 7];
  const g11 = GRAD2[hash2i(ix + 1, iy + 1, seed) & 7];

  const fx1 = fx - 1;
  const fy1 = fy - 1;

  const n00 = g00[0] * fx + g00[1] * fy;
  const n10 = g10[0] * fx1 + g10[1] * fy;
  const n01 = g01[0] * fx + g01[1] * fy1;
  const n11 = g11[0] * fx1 + g11[1] * fy1;

  const u = fade(fx);
  const v = fade(fy);
  const du = fadeDeriv(fx);
  const dv = fadeDeriv(fy);

  const k0 = n00;
  const k1 = n10 - n00;
  const k2 = n01 - n00;
  const k3 = n00 - n10 - n01 + n11;

  const S = 1.4142135623730951;
  out[0] = (k0 + k1 * u + k2 * v + k3 * u * v) * S;

  // ∂/∂x = du·(k1 + k3·v) + gradient terms carried by the corner dot products
  const gx = lerp(lerp(g00[0], g10[0], u), lerp(g01[0], g11[0], u), v);
  const gy = lerp(lerp(g00[1], g10[1], u), lerp(g01[1], g11[1], u), v);

  out[1] = (gx + du * (k1 + k3 * v)) * S;
  out[2] = (gy + dv * (k2 + k3 * u)) * S;
}

/** 3D Perlin noise, normalised to roughly [-1,1]. */
export function perlin3(x: number, y: number, z: number, seed: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fy = y - iy;
  const fz = z - iz;
  const fx1 = fx - 1;
  const fy1 = fy - 1;
  const fz1 = fz - 1;

  const dot = (hx: number, hy: number, hz: number, dx: number, dy: number, dz: number): number => {
    const g = GRAD3[hash3i(hx, hy, hz, seed) % 12];
    return g[0] * dx + g[1] * dy + g[2] * dz;
  };

  const n000 = dot(ix, iy, iz, fx, fy, fz);
  const n100 = dot(ix + 1, iy, iz, fx1, fy, fz);
  const n010 = dot(ix, iy + 1, iz, fx, fy1, fz);
  const n110 = dot(ix + 1, iy + 1, iz, fx1, fy1, fz);
  const n001 = dot(ix, iy, iz + 1, fx, fy, fz1);
  const n101 = dot(ix + 1, iy, iz + 1, fx1, fy, fz1);
  const n011 = dot(ix, iy + 1, iz + 1, fx, fy1, fz1);
  const n111 = dot(ix + 1, iy + 1, iz + 1, fx1, fy1, fz1);

  const u = fade(fx);
  const v = fade(fy);
  const w = fade(fz);

  const x00 = lerp(n000, n100, u);
  const x10 = lerp(n010, n110, u);
  const x01 = lerp(n001, n101, u);
  const x11 = lerp(n011, n111, u);

  // Improved-Perlin 3D with |g| = √2 edge-midpoint gradients is already bounded
  // to [-1,1]; measured max over 3M samples is 0.9991. An extra 2/√3 "fix"
  // (a common copy-paste) pushes it to 1.12 and clips wherever the caller
  // assumes unit range.
  return lerp(lerp(x00, x10, v), lerp(x01, x11, v), w);
}

/** Empirically measured extreme of `perlin2` — see tests/unit/noise.test.ts. */
export const PERLIN2_MAX = 0.9823;
/** Empirically measured extreme of `perlin3`. */
export const PERLIN3_MAX = 0.9992;

// ---------------------------------------------------------------------------
// Worley / cellular noise
// ---------------------------------------------------------------------------

/**
 * 2D Worley noise. Returns F1 (distance to nearest feature point) in `out[0]`
 * and F2 in `out[1]`, both roughly in [0, 1.5].
 *
 * Used for: forest clustering (phase 7), cloud shape erosion (phase 5),
 * rock/scree variation (phase 2).
 */
export function worley2(x: number, y: number, seed: number, out: Float64Array): void {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;

  let f1 = 8.0;
  let f2 = 8.0;

  for (let oy = -1; oy <= 1; oy++) {
    for (let ox = -1; ox <= 1; ox++) {
      const h = hash2i(ix + ox, iy + oy, seed);
      // Two independent [0,1) coordinates from one hash.
      const px = ox + ((h >>> 8) & 0xfff) / 4096;
      const py = oy + ((h >>> 20) & 0xfff) / 4096;
      const dx = px - fx;
      const dy = py - fy;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < f1) {
        f2 = f1;
        f1 = d;
      } else if (d < f2) {
        f2 = d;
      }
    }
  }
  out[0] = f1;
  out[1] = f2;
}

/** Convenience: F1 only. */
export function worley2F1(x: number, y: number, seed: number): number {
  const tmp = _worleyTmp;
  worley2(x, y, seed, tmp);
  return tmp[0];
}

/** 3D Worley F1, inverted to [0,1] where 1 = cell centre. Used for cloud noise. */
export function worley3F1Inv(x: number, y: number, z: number, seed: number, cells: number): number {
  const sx = x * cells;
  const sy = y * cells;
  const sz = z * cells;
  const ix = Math.floor(sx);
  const iy = Math.floor(sy);
  const iz = Math.floor(sz);
  const fx = sx - ix;
  const fy = sy - iy;
  const fz = sz - iz;

  let f1 = 8.0;
  for (let oz = -1; oz <= 1; oz++) {
    for (let oy = -1; oy <= 1; oy++) {
      for (let ox = -1; ox <= 1; ox++) {
        // Wrap the lattice so the resulting 3D texture tiles seamlessly.
        const h = hash3i(
          wrapI(ix + ox, cells),
          wrapI(iy + oy, cells),
          wrapI(iz + oz, cells),
          seed,
        );
        const px = ox + ((h >>> 8) & 0xff) / 256;
        const py = oy + ((h >>> 16) & 0xff) / 256;
        const pz = oz + ((h >>> 24) & 0xff) / 256;
        const dx = px - fx;
        const dy = py - fy;
        const dz = pz - fz;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < f1) f1 = d2;
      }
    }
  }
  return 1 - Math.min(1, Math.sqrt(f1));
}

/** Positive modulo for lattice wrapping. */
export function wrapI(v: number, n: number): number {
  return ((v % n) + n) % n;
}

const _worleyTmp = new Float64Array(2);
