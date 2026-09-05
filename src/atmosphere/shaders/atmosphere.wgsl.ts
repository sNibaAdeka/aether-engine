/**
 * The physics half of the atmosphere: densities, phase functions, the geometry
 * of a spherical shell, and the LUT parameterisations of Bruneton 2017 as
 * simplified by Hillaire 2020.
 *
 * Everything here is a pure function of numbers — no texture lookups, no state.
 * The ray marches that *do* sample LUTs live in `AtmosphereLUTs.ts`, because
 * `wgslFn` cannot take a texture binding and a march without LUT lookups is not
 * the algorithm.
 *
 * **Units are kilometres.** Not a style choice: the planet radius is 6360 km and
 * f32 has a 24-bit mantissa, so working in metres puts every position at
 * 6.36 × 10⁶ where consecutive representable values are half a metre apart —
 * and the altitude of a sample is `length(p) − Rg`, a subtraction of two nearly
 * equal numbers. In kilometres the same subtraction resolves to 0.5 mm. The
 * scattering coefficients below are therefore per kilometre, i.e. the values
 * tabulated in the spec (which are per metre) multiplied by 1000.
 *
 * Parameters are the spec's, which are the standard Earth ones:
 *
 *   planet radius        6360 km
 *   atmosphere radius    6420 km
 *   Rayleigh scattering  (5.802, 13.558, 33.1) × 1e-3 /km, scale height 8 km
 *   Mie scattering       3.996e-3 /km, absorption 4.4e-3 /km, scale height 1.2 km
 *   Mie asymmetry g      0.8
 *   Ozone absorption     (0.650, 1.881, 0.085) × 1e-3 /km, tent 25 ± 15 km
 *   ground albedo        0.3
 *
 * A note on why the constants are pasted *inside* every function body rather
 * than declared once at module scope: three's `wgslFn` parses the string with a
 * regexp anchored at `^fn`, so a single character before the declaration makes
 * the whole function fail to parse. WGSL allows `const` in function scope and
 * the compiler folds it away, so the duplication costs nothing but bytes.
 */

import { wgslFn } from 'three/tsl';

/** Planet radius, km. */
export const PLANET_RADIUS_KM = 6360;
/** Top of atmosphere radius, km. */
export const ATMOSPHERE_RADIUS_KM = 6420;
/** Ground albedo, Lambertian, spec 3.1. */
export const GROUND_ALBEDO = 0.3;

/** Far limit of the aerial-perspective froxel volume, km (spec 3.1: 32 km). */
export const AERIAL_MAX_KM = 32;

export const LUT_SIZES = {
  transmittanceW: 256,
  transmittanceH: 64,
  multiScatterW: 32,
  multiScatterH: 32,
  skyViewW: 200,
  skyViewH: 100,
  aerial: 32,
} as const;

/** Pasted at the top of every function body that needs the atmosphere. */
const C = /* wgsl */ `
  const AE_RG: f32 = ${PLANET_RADIUS_KM}.0;
  const AE_RT: f32 = ${ATMOSPHERE_RADIUS_KM}.0;
  const AE_RAYLEIGH_SCAT: vec3f = vec3f( 5.802e-3, 13.558e-3, 33.1e-3 );
  const AE_RAYLEIGH_H: f32 = 8.0;
  const AE_MIE_SCAT: f32 = 3.996e-3;
  const AE_MIE_ABS: f32 = 4.4e-3;
  const AE_MIE_H: f32 = 1.2;
  const AE_MIE_G: f32 = 0.8;
  const AE_OZONE_ABS: vec3f = vec3f( 0.650e-3, 1.881e-3, 0.085e-3 );
  const AE_OZONE_CENTRE: f32 = 25.0;
  const AE_OZONE_WIDTH: f32 = 15.0;
  const AE_GROUND_ALBEDO: f32 = ${GROUND_ALBEDO};
  const AE_PI: f32 = 3.14159265358979;
`;

/**
 * Turbidity → Mie density multiplier.
 *
 * The spec fixes three anchor points: 1.5 is a clear frosty day, 4 is haze, 8 is
 * the air before a thunderstorm. A linear map would make 8 only five times
 * hazier than 1.5, which is nowhere near the difference between those two skies,
 * so the curve is a power law normalised so that turbidity 2.5 — a typical
 * temperate day, and the value the Mie coefficient above is quoted for — is
 * exactly 1.
 *
 *   1.5 → 0.46    2.5 → 1.00    4 → 2.05    8 → 5.79
 */
export function turbidityToMieScale(turbidity: number): number {
  return Math.pow(Math.max(turbidity, 0.1) / 2.5, 1.5);
}

/**
 * Nearest positive intersection of a ray with a sphere centred at the origin.
 * Returns −1 when the ray misses or when both roots are behind the origin.
 */
export const fnRaySphere = wgslFn(/* wgsl */ `
fn aeRaySphere( ro: vec3f, rd: vec3f, radius: f32 ) -> f32 {
  let b = dot( ro, rd );
  let c = dot( ro, ro ) - radius * radius;
  let disc = b * b - c;
  if ( disc < 0.0 ) { return -1.0; }
  let s = sqrt( disc );
  let t0 = -b - s;
  let t1 = -b + s;
  if ( t1 < 0.0 ) { return -1.0; }
  return select( t1, t0, t0 >= 0.0 );
}
`);

/**
 * Rayleigh, Mie and ozone densities at altitude `h` km, relative to sea level.
 *
 * Ozone is a tent rather than an exponential because ozone is not in
 * hydrostatic equilibrium — it is a layer produced by photochemistry at a
 * specific altitude. It contributes no scattering at all, only absorption, and
 * it is the reason a clear zenith at twilight is blue rather than the dirty
 * orange the Rayleigh term alone would give. Leaving it out is the single most
 * common way a "physical" sky ends up looking wrong at dusk.
 */
export const fnDensities = wgslFn(/* wgsl */ `
fn aeDensities( h: f32 ) -> vec3f {
${C}
  let rayleigh = exp( -max( h, 0.0 ) / AE_RAYLEIGH_H );
  let mie = exp( -max( h, 0.0 ) / AE_MIE_H );
  let ozone = max( 0.0, 1.0 - abs( h - AE_OZONE_CENTRE ) / AE_OZONE_WIDTH );
  return vec3f( rayleigh, mie, ozone );
}
`);

/** Total extinction coefficient, per km, at altitude `h` km. */
export const fnExtinction = wgslFn(
  /* wgsl */ `
fn aeExtinction( h: f32, mieScale: f32 ) -> vec3f {
${C}
  let d = aeDensities( h );
  let mie = ( AE_MIE_SCAT + AE_MIE_ABS ) * mieScale * d.y;
  return AE_RAYLEIGH_SCAT * d.x + vec3f( mie, mie, mie ) + AE_OZONE_ABS * d.z;
}
`,
  [fnDensities],
);

/** Rayleigh scattering coefficient, per km, at altitude `h`. */
export const fnRayleighScattering = wgslFn(/* wgsl */ `
fn aeRayleighScattering( h: f32 ) -> vec3f {
${C}
  return AE_RAYLEIGH_SCAT * exp( -max( h, 0.0 ) / AE_RAYLEIGH_H );
}
`);

/** Mie scattering coefficient, per km, at altitude `h`. */
export const fnMieScattering = wgslFn(/* wgsl */ `
fn aeMieScattering( h: f32, mieScale: f32 ) -> f32 {
${C}
  return AE_MIE_SCAT * mieScale * exp( -max( h, 0.0 ) / AE_MIE_H );
}
`);

/** Rayleigh phase function. */
export const fnRayleighPhase = wgslFn(/* wgsl */ `
fn aeRayleighPhase( cosTheta: f32 ) -> f32 {
${C}
  return 3.0 / ( 16.0 * AE_PI ) * ( 1.0 + cosTheta * cosTheta );
}
`);

/**
 * Cornette–Shanks phase function, which is the Henyey–Greenstein form corrected
 * to keep the Rayleigh-like backscatter lobe. At g = 0.8 the forward peak is
 * about 90× isotropic — that peak is the white glare around a low sun and the
 * reason haze reads as haze.
 */
export const fnMiePhase = wgslFn(/* wgsl */ `
fn aeMiePhase( cosTheta: f32 ) -> f32 {
${C}
  let g = AE_MIE_G;
  let g2 = g * g;
  let k = 3.0 / ( 8.0 * AE_PI ) * ( 1.0 - g2 ) / ( 2.0 + g2 );
  let denom = 1.0 + g2 - 2.0 * g * cosTheta;
  return k * ( 1.0 + cosTheta * cosTheta ) / max( denom * sqrt( max( denom, 1e-6 ) ), 1e-6 );
}
`);

/**
 * (r, mu) → transmittance-LUT uv, Bruneton's mapping.
 *
 * The point of this parameterisation rather than a plain (altitude, cos angle)
 * grid is that it puts texture resolution where the function actually varies:
 * `d` is the distance the ray travels to the top of the atmosphere, and near the
 * horizon that distance changes by hundreds of kilometres for a thousandth of a
 * change in mu. A linear-in-mu table smears the entire horizon band into two
 * texels and produces the hard bright line at the horizon that gives away most
 * hand-rolled implementations.
 */
export const fnTransmittanceUv = wgslFn(/* wgsl */ `
fn aeTransmittanceUv( r: f32, mu: f32 ) -> vec2f {
${C}
  let H = sqrt( max( AE_RT * AE_RT - AE_RG * AE_RG, 0.0 ) );
  let rho = sqrt( max( r * r - AE_RG * AE_RG, 0.0 ) );
  let disc = r * r * ( mu * mu - 1.0 ) + AE_RT * AE_RT;
  let d = max( 0.0, -r * mu + sqrt( max( disc, 0.0 ) ) );
  let dMin = AE_RT - r;
  let dMax = rho + H;
  let xMu = ( d - dMin ) / max( dMax - dMin, 1e-6 );
  let xR = rho / max( H, 1e-6 );
  return vec2f( clamp( xMu, 0.0, 1.0 ), clamp( xR, 0.0, 1.0 ) );
}
`);

/** Transmittance-LUT uv → (r, mu). The exact inverse of `aeTransmittanceUv`. */
export const fnTransmittanceParams = wgslFn(/* wgsl */ `
fn aeTransmittanceParams( uv: vec2f ) -> vec2f {
${C}
  let H = sqrt( max( AE_RT * AE_RT - AE_RG * AE_RG, 0.0 ) );
  let rho = H * uv.y;
  let r = sqrt( rho * rho + AE_RG * AE_RG );
  let dMin = AE_RT - r;
  let dMax = rho + H;
  let d = dMin + uv.x * ( dMax - dMin );
  var mu = 1.0;
  if ( d > 0.0 ) {
    mu = ( H * H - rho * rho - d * d ) / ( 2.0 * r * d );
  }
  return vec2f( r, clamp( mu, -1.0, 1.0 ) );
}
`);

/**
 * Sky-view LUT: uv → view direction, in a frame where +Y is up.
 *
 * `uv.x` is absolute azimuth over the whole circle, deliberately *not*
 * Hillaire's half-range-with-mirror-symmetry. That symmetry only holds when
 * there is exactly one light in the sky; this atmosphere marches the sun and the
 * moon together in one pass (which is what makes twilight continuous instead of
 * switching sources at some threshold), and the two are almost never in the same
 * vertical plane. Full azimuth at 200 texels is 1.8° per texel.
 *
 * `uv.y` is split at the horizon with a squared falloff on each side, so half
 * the vertical resolution goes to the few degrees around it. That is where all
 * the interesting structure is: the horizon band goes from 8000 to 200 cd/m² in
 * under two degrees at sunset.
 */
export const fnSkyViewDir = wgslFn(/* wgsl */ `
fn aeSkyViewDir( uv: vec2f, r: f32 ) -> vec3f {
${C}
  let cosBeta = clamp( sqrt( max( r * r - AE_RG * AE_RG, 0.0 ) ) / max( r, 1e-3 ), -1.0, 1.0 );
  let beta = acos( cosBeta );
  let zenithHorizon = AE_PI - beta;

  var viewZenith: f32 = 0.0;
  if ( uv.y < 0.5 ) {
    var c = 1.0 - 2.0 * uv.y;
    c = 1.0 - c * c;
    viewZenith = zenithHorizon * c;
  } else {
    var c = uv.y * 2.0 - 1.0;
    c = c * c;
    viewZenith = zenithHorizon + beta * c;
  }

  let azimuth = uv.x * 2.0 * AE_PI;
  let s = sin( viewZenith );
  // World frame: +X east, +Y up, −Z north, azimuth measured from north
  // eastward — the same convention math/Spherical.ts declares.
  return vec3f( s * sin( azimuth ), cos( viewZenith ), -s * cos( azimuth ) );
}
`);

/** The exact inverse: a world-space view direction → sky-view LUT uv. */
export const fnSkyViewUv = wgslFn(/* wgsl */ `
fn aeSkyViewUv( dir: vec3f, r: f32 ) -> vec2f {
${C}
  let cosBeta = clamp( sqrt( max( r * r - AE_RG * AE_RG, 0.0 ) ) / max( r, 1e-3 ), -1.0, 1.0 );
  let beta = acos( cosBeta );
  let zenithHorizon = AE_PI - beta;

  let viewZenith = acos( clamp( dir.y, -1.0, 1.0 ) );

  var v: f32 = 0.0;
  if ( viewZenith < zenithHorizon ) {
    let c = viewZenith / max( zenithHorizon, 1e-6 );
    v = 0.5 * ( 1.0 - sqrt( max( 1.0 - c, 0.0 ) ) );
  } else {
    let c = ( viewZenith - zenithHorizon ) / max( beta, 1e-6 );
    v = 0.5 * ( sqrt( clamp( c, 0.0, 1.0 ) ) + 1.0 );
  }

  let azimuth = atan2( dir.x, -dir.z );
  let u = fract( azimuth / ( 2.0 * AE_PI ) + 1.0 );
  return vec2f( u, clamp( v, 0.0, 1.0 ) );
}
`);

/**
 * Half-texel inset for a LUT lookup.
 *
 * Without it the first and last texel of a table are sampled at their edge
 * rather than their centre, which for the transmittance LUT means the horizon
 * row is read as a blend with nothing. Standard, and standard to forget.
 */
export const fnUnitToTexCoord = wgslFn(/* wgsl */ `
fn aeUnitToTexCoord( x: f32, n: f32 ) -> f32 {
  return 0.5 / n + x * ( 1.0 - 1.0 / n );
}
`);

/** …and back, for a kernel that has a texel index and wants the unit range. */
export const fnTexCoordToUnit = wgslFn(/* wgsl */ `
fn aeTexCoordToUnit( u: f32, n: f32 ) -> f32 {
  return ( u - 0.5 / n ) / ( 1.0 - 1.0 / n );
}
`);

/**
 * Uniform direction on the sphere from two stratified unit numbers.
 * Used by the multiple-scattering LUT, which integrates over all directions.
 */
export const fnUniformSphere = wgslFn(/* wgsl */ `
fn aeUniformSphere( u: f32, v: f32 ) -> vec3f {
${C}
  let cosTheta = 1.0 - 2.0 * u;
  let sinTheta = sqrt( max( 1.0 - cosTheta * cosTheta, 0.0 ) );
  let phi = 2.0 * AE_PI * v;
  return vec3f( sinTheta * cos( phi ), cosTheta, sinTheta * sin( phi ) );
}
`);

/**
 * Distance from a point at radius `r` looking at `mu` to the top of the
 * atmosphere, or to the ground if the ray hits it first.
 */
export const fnRayMarchLength = wgslFn(/* wgsl */ `
fn aeRayMarchLength( r: f32, mu: f32 ) -> f32 {
${C}
  let discTop = r * r * ( mu * mu - 1.0 ) + AE_RT * AE_RT;
  let tTop = max( 0.0, -r * mu + sqrt( max( discTop, 0.0 ) ) );
  let discGround = r * r * ( mu * mu - 1.0 ) + AE_RG * AE_RG;
  if ( discGround >= 0.0 && mu < 0.0 ) {
    let tGround = -r * mu - sqrt( max( discGround, 0.0 ) );
    if ( tGround > 0.0 ) { return tGround; }
  }
  return tTop;
}
`);

/**
 * Limb darkening of the solar disc.
 *
 * `centreDistance` is the fractional radius of the sampled point, 0 at the
 * centre of the disc and 1 at the edge; `mu` below is the cosine of the angle
 * between the local photosphere normal and the line of sight. Coefficients are
 * the standard visual-band ones (Hestroffer & Magnan): the edge of the disc
 * comes out at 0.30 of the centre, which is the difference between a sun that
 * reads as a glowing sphere and one that reads as a white sticker.
 */
export const fnLimbDarkening = wgslFn(/* wgsl */ `
fn aeLimbDarkening( centreDistance: f32 ) -> f32 {
  let d = clamp( centreDistance, 0.0, 1.0 );
  let mu = sqrt( max( 1.0 - d * d, 0.0 ) );
  let u = 0.47;
  let v = 0.23;
  return max( 1.0 - u * ( 1.0 - mu ) - v * ( 1.0 - mu * mu ), 0.0 );
}
`);

/**
 * Lommel–Seeliger reflectance, used for the lunar disc.
 *
 * The moon is famously *not* Lambertian: a Lambert sphere at full phase is
 * bright in the middle and dark at the edges, but the real full moon is a flat
 * disc of near-uniform brightness. That is because lunar regolith backscatters,
 * and Lommel–Seeliger — μ₀/(μ₀+μ) — reproduces it for one divide. Using Lambert
 * here is the single most recognisable mistake in a rendered moon.
 */
export const fnLommelSeeliger = wgslFn(/* wgsl */ `
fn aeLommelSeeliger( nDotL: f32, nDotV: f32 ) -> f32 {
  let l = max( nDotL, 0.0 );
  let v = max( nDotV, 1e-3 );
  return l / ( l + v );
}
`);

/**
 * Value noise on the sphere, used for the Milky Way's dust structure.
 *
 * Deliberately not the terrain's Perlin: that one is a 2D lattice and the sky is
 * a sphere, so a 3D field evaluated on the unit sphere is the only way to avoid
 * a pinch at the poles. Three octaves is enough — the galaxy is a smooth glow
 * with dark lanes, not a detailed texture.
 */
export const fnSkyNoise = wgslFn(/* wgsl */ `
fn aeSkyNoise( p: vec3f ) -> f32 {
  let i = floor( p );
  let f = p - i;
  let u = f * f * ( 3.0 - 2.0 * f );
  var acc = 0.0;
  for ( var dz = 0; dz < 2; dz = dz + 1 ) {
    for ( var dy = 0; dy < 2; dy = dy + 1 ) {
      for ( var dx = 0; dx < 2; dx = dx + 1 ) {
        let c = i + vec3f( f32( dx ), f32( dy ), f32( dz ) );
        var h = dot( c, vec3f( 127.1, 311.7, 74.7 ) );
        h = fract( sin( h ) * 43758.5453 );
        let w = mix( 1.0 - u.x, u.x, f32( dx ) )
              * mix( 1.0 - u.y, u.y, f32( dy ) )
              * mix( 1.0 - u.z, u.z, f32( dz ) );
        acc = acc + h * w;
      }
    }
  }
  return acc;
}
`);

export const fnSkyFbm = wgslFn(
  /* wgsl */ `
fn aeSkyFbm( p: vec3f, octaves: i32 ) -> f32 {
  var sum = 0.0;
  var amp = 0.5;
  var q = p;
  for ( var o = 0; o < octaves; o = o + 1 ) {
    sum = sum + aeSkyNoise( q ) * amp;
    q = q * 2.03;
    amp = amp * 0.5;
  }
  return sum;
}
`,
  [fnSkyNoise],
);
