/**
 * WGSL mirror of `math/Noise.ts`, `math/FBM.ts` and `terrain/HeightField.ts`.
 *
 * ⚠ This file and its three CPU counterparts must be edited together. The
 * player's feet are placed using the CPU version and the ground is drawn using
 * this one; a divergence of more than a few centimetres is immediately visible
 * as sinking or floating. `tests/e2e/parity.spec.ts` reads heights back off the
 * GPU and compares against the CPU function on 1024 random points with a
 * 0.05 m tolerance — run it after any change here.
 *
 * Parity notes:
 *  - Integer hashing is bit-exact: JS `Math.imul` and `>>> n` on a u32 have the
 *    same semantics as WGSL `*` and `>> n` on a `u32`.
 *  - Gradients: the CPU keeps a constant table of cos/sin at multiples of π/8;
 *    here they are evaluated with `cos`/`sin` of the same angles. WGSL
 *    guarantees a few ULP, so the gradients agree to ~1e-7 — about 1e-4 m of
 *    height error, three orders of magnitude inside the tolerance.
 *  - JS accumulates in f64 and WGSL in f32. Over the full layer stack the
 *    error stays around 1e-3 m.
 *  - **World extent limit:** at |x| beyond roughly 2·10⁵ m the f32 mantissa can
 *    no longer resolve the highest-frequency lattice and detail noise starts to
 *    quantise. Fine for the intended play area; if the world ever needs to be
 *    bigger, the fix is to feed tile-relative coordinates instead of absolute.
 */

import { wgslFn } from 'three/tsl';

// ---------------------------------------------------------------------------
// Integer hashing
// ---------------------------------------------------------------------------

export const fnHashU32 = wgslFn(/* wgsl */ `
fn aeHashU32( x: u32 ) -> u32 {
  var h = x;
  h = h ^ ( h >> 17u );
  h = h * 0xed5ad4bbu;
  h = h ^ ( h >> 11u );
  h = h * 0xac4c1b51u;
  h = h ^ ( h >> 15u );
  h = h * 0x31848babu;
  h = h ^ ( h >> 14u );
  return h;
}
`);

export const fnHash2i = wgslFn(
  /* wgsl */ `
fn aeHash2i( ix: i32, iy: i32, seed: i32 ) -> u32 {
  // Multiply in u32 so the wrap-around is defined and matches JS Math.imul
  // bit for bit. Doing it in i32 relies on signed-overflow behaviour that WGSL
  // does not promise.
  let a = bitcast<u32>( ix ) * 0x27d4eb2du;
  let b = bitcast<u32>( iy ) * 0x165667b1u;
  return aeHashU32( a ^ b ^ bitcast<u32>( seed ) );
}
`,
  [fnHashU32],
);

// ---------------------------------------------------------------------------
// Perlin gradient noise
// ---------------------------------------------------------------------------

/**
 * 2D Perlin noise scaled to ≈[-1,1].
 *
 * The ×√2 at the end matches the CPU: with 16 unit gradients the raw extreme is
 * 1/√2, so this normalises to unit range (measured max 0.9823).
 */
export const fnGrad2 = wgslFn(/* wgsl */ `
fn aeGrad2( h: u32 ) -> vec2f {
  // Mirrors GRAD2 in math/Noise.ts:
  //   0:(1,0) 1:(-1,0) 2:(0,1) 3:(0,-1) 4:(r,r) 5:(-r,r) 6:(r,-r) 7:(-r,-r)
  // Derived from three bits with select() — no table indexing (which would
  // spill to memory) and no trigonometry.
  let i = h & 7u;
  let sx = select( 1.0, -1.0, ( i & 1u ) != 0u );
  let sy = select( 1.0, -1.0, ( i & 2u ) != 0u );
  let axis = select( vec2f( sx, 0.0 ), vec2f( 0.0, sx ), ( i & 2u ) != 0u );
  let diag = vec2f( sx, sy ) * 0.7071067811865476;
  return select( axis, diag, ( i & 4u ) != 0u );
}
`);

export const fnPerlin2 = wgslFn(
  /* wgsl */ `
fn aePerlin2( p: vec2f, seed: i32 ) -> f32 {
  let i = floor( p );
  let f = p - i;
  let ix = i32( i.x );
  let iy = i32( i.y );

  let g00 = aeGrad2( aeHash2i( ix,     iy,     seed ) );
  let g10 = aeGrad2( aeHash2i( ix + 1, iy,     seed ) );
  let g01 = aeGrad2( aeHash2i( ix,     iy + 1, seed ) );
  let g11 = aeGrad2( aeHash2i( ix + 1, iy + 1, seed ) );

  let n00 = dot( g00, f );
  let n10 = dot( g10, f - vec2f( 1.0, 0.0 ) );
  let n01 = dot( g01, f - vec2f( 0.0, 1.0 ) );
  let n11 = dot( g11, f - vec2f( 1.0, 1.0 ) );

  // Quintic fade — C² continuous, which the normals depend on.
  let u = f * f * f * ( f * ( f * 6.0 - 15.0 ) + 10.0 );

  let a = mix( n00, n10, u.x );
  let b = mix( n01, n11, u.x );
  return mix( a, b, u.y ) * 1.4142135623730951;
}
`,
  [fnHash2i, fnGrad2],
);

// ---------------------------------------------------------------------------
// Fractal layers
// ---------------------------------------------------------------------------

/**
 * fBm normalised by the *limit* of the amplitude series, not the running sum.
 *
 * This is what makes octave count a pure detail knob: octave 0 contributes the
 * same amount whether you evaluate 3 octaves or 10, so distant tiles can be
 * generated more cheaply without the large-scale terrain shape shifting.
 */
export const fnFbm2 = wgslFn(
  /* wgsl */ `
fn aeFbm2( p: vec2f, seed: i32, octaves: i32, frequency: f32, lacunarity: f32, gain: f32 ) -> f32 {
  var sum = 0.0;
  var amp = 1.0;
  var freq = frequency;
  for ( var i = 0; i < octaves; i = i + 1 ) {
    sum = sum + aePerlin2( p * freq, seed + i * 1013 ) * amp;
    amp = amp * gain;
    freq = freq * lacunarity;
  }
  return sum * ( 1.0 - gain );
}
`,
  [fnPerlin2],
);

/**
 * Ridged multifractal with Musgrave's octave weighting: each octave is
 * attenuated by how ridge-like the previous one was, which is what makes
 * crests continue as connected lines instead of a field of isolated spikes.
 */
export const fnRidged2 = wgslFn(
  /* wgsl */ `
fn aeRidged2( p: vec2f, seed: i32, octaves: i32, frequency: f32, lacunarity: f32, gain: f32, sharpness: f32, weightFactor: f32 ) -> f32 {
  var sum = 0.0;
  var amp = 1.0;
  var freq = frequency;
  var weight = 1.0;
  for ( var i = 0; i < octaves; i = i + 1 ) {
    var n = aePerlin2( p * freq, seed + i * 1013 );
    n = 1.0 - abs( n );
    n = pow( max( n, 0.0 ), sharpness );
    let contribution = n * weight;
    weight = clamp( contribution * weightFactor, 0.0, 1.0 );
    sum = sum + contribution * amp;
    amp = amp * gain;
    freq = freq * lacunarity;
  }
  return sum * ( 1.0 - gain );
}
`,
  [fnPerlin2],
);

// ---------------------------------------------------------------------------
// Continental shaping
// ---------------------------------------------------------------------------

/**
 * Raw continental noise → metres.
 *
 * The shelf around sea level is deliberate: without it the coastline is a wall
 * dropping into the water and the whole world reads as a bathtub.
 */
/**
 * Spread raw fBm across the usable range before the elevation curve sees it.
 * fBm is Gaussian with σ ≈ 0.12, not uniform over ±1 — see the CPU twin.
 */
export const fnContinentShape = wgslFn(/* wgsl */ `
fn aeContinentShape( raw: f32 ) -> f32 {
  let g = raw * 3.0;
  return g / sqrt( 1.0 + g * g );
}
`);

export const fnContinentCurve = wgslFn(/* wgsl */ `
fn aeContinentCurve( n: f32 ) -> f32 {
  if ( n < -0.12 ) {
    let t = clamp( ( -0.12 - n ) / 0.88, 0.0, 1.0 );
    let s = t * t * ( 3.0 - 2.0 * t );
    return mix( -25.0, -420.0, s );
  }
  if ( n < 0.08 ) {
    let t = clamp( ( n + 0.12 ) / 0.2, 0.0, 1.0 );
    return mix( -25.0, 18.0, t );
  }
  let t = clamp( ( n - 0.08 ) / 0.92, 0.0, 1.0 );
  let eased = t * t * ( 0.55 + 0.45 * t );
  return mix( 18.0, 980.0, eased );
}
`);

// ---------------------------------------------------------------------------
// The height function
// ---------------------------------------------------------------------------

/**
 * Terrain height in metres at a world XZ position.
 *
 * Mirrors `HeightField.sampleHeight`. The layer order is load-bearing — each
 * layer is modulated by the ones before it.
 */
export const fnTerrainHeight = wgslFn(
  /* wgsl */ `
fn aeTerrainHeight( pos: vec2f, seed: f32 ) -> f32 {
  let s = i32( seed );

  // 1. Continentality → base elevation.
  let cn = aeContinentShape( aeFbm2( pos, s + 0x1f35, 4, 0.00005, 2.01734, 0.5 ) );
  let base = aeContinentCurve( cn );

  // 2. Erosion age: 0 young and sharp, 1 old and rounded.
  let erosion = clamp( aeFbm2( pos, s + 0x7a11, 3, 0.000111111, 2.01734, 0.5 ) * 2.6 * 0.5 + 0.5, 0.0, 1.0 );

  // Relief only grows on land, and grows with the base elevation — this is what
  // stops isolated peaks sprouting straight out of the ocean.
  let landMask = smoothstep( -30.0, 220.0, base );
  let reliefScale = mix( 1.0, 0.34, erosion ) * landMask;

  if ( reliefScale <= 0.0005 ) {
    let oceanDetail = aeFbm2( pos, s + 0x4e87, 5, 0.011111111, 2.01734, 0.5 ) * 3.0;
    return base + oceanDetail;
  }

  // 3+4. Domain warp, then ridges evaluated in the warped domain. Without the
  // warp, ridged noise reads as random static rather than folded geology.
  let wx = aeFbm2( pos + vec2f( 137.31, 41.77 ), s + 0x5f3a, 3, 0.000333333, 2.01734, 0.5 ) * 120.0;
  let wz = aeFbm2( pos + vec2f( -91.13, 217.59 ), s + 0x1c9d, 3, 0.000333333, 2.01734, 0.5 ) * 120.0;
  let rp = pos + vec2f( wx, wz );

  let sharpness = mix( 2.6, 1.5, erosion );
  let ridge = aeRidged2( rp, s + 0x2b73, 8, 0.000384615, 2.01734, 0.5, sharpness, 0.9 );
  let ridgeHeight = ridge * ridge * 620.0 * reliefScale;

  // 5. Valleys cut across the chains — subtracted, not added.
  let vp = vec2f( rp.y * 0.87 - 512.3, rp.x * 0.87 + 88.1 );
  let valley = aeRidged2( vp, s + 0x63d1, 5, 0.000588235, 2.01734, 0.5, 1.8, 0.85 );
  let valleyDepth = valley * valley * 165.0 * reliefScale * mix( 0.55, 1.0, erosion );

  var h = base + ridgeHeight - valleyDepth;

  // 6. Detail modulated by steepness: scree on faces, smooth alluvium on flats.
  let steep = clamp( ridge * 1.4, 0.0, 1.0 );
  let detailAmp = mix( 0.5, 4.0, steep ) * landMask;
  h = h + aeFbm2( pos, s + 0x4e87, 5, 0.011111111, 2.01734, 0.5 ) * detailAmp;

  // 8. Sedimentary terracing on a masked ~15% of the land.
  let terraceRaw = clamp( aePerlin2( pos / 5200.0, s + 0x39af ) * 0.5 + 0.5, 0.0, 1.0 );
  let terraceMask = smoothstep( 0.62, 0.78, terraceRaw );
  if ( terraceMask > 0.001 ) {
    let step = 26.0;
    let q = floor( h / step );
    let f = h / step - q;
    let shaped = ( q + smoothstep( 0.35, 0.72, f ) ) * step;
    // Weak, and gated on steepness — see the CPU twin for why.
    let bedding = smoothstep( 0.25, 0.65, steep );
    h = mix( h, shaped, terraceMask * 0.18 * landMask * bedding );
  }

  return h;
}
`,
  [fnFbm2, fnRidged2, fnContinentCurve, fnContinentShape, fnPerlin2],
);
