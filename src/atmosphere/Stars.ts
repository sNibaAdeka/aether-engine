/**
 * The night sky's own light sources: ~190 catalogued stars as instanced sprites
 * and a procedural Milky Way baked into a small equirectangular map.
 *
 * ## Why sprites rather than a baked star map
 *
 * A star is a point source. Baking one into a texture and magnifying it —
 * which is what a 2048² sky map does at a 55° field of view — turns every star
 * into a soft blob three or four pixels wide with no way to make it sharper. A
 * sprite is drawn at the resolution it is displayed at, so a first-magnitude
 * star is a tight point with a visible core, which is what a star looks like.
 * It also makes per-star twinkle and per-star atmospheric extinction free:
 * both are functions of the star's own altitude, and a sprite knows its own
 * altitude while a texel does not.
 *
 * 190 instances of a two-triangle quad is one draw call and 760 vertices.
 *
 * ## Why the Milky Way *is* baked
 *
 * The opposite argument. It is a smooth, low-frequency glow with soft dust
 * lanes; there is nothing in it above about a degree of angular frequency, so a
 * 512 × 256 equirectangular map over the whole celestial sphere (0.7° per texel)
 * is already finer than the signal. Evaluating its three octaves of 3D noise per
 * pixel instead would cost more than the entire sky pass is budgeted.
 *
 * ## Physical units
 *
 * A star of visual magnitude m delivers 2.54 × 10⁻⁶ × 10^(−0.4 m) lux at the
 * top of the atmosphere. Sirius is 9.8 × 10⁻⁶ lx; the whole sky of stars is
 * about 2 × 10⁻⁴ lx, which is a quarter of the airglow and a thousandth of a
 * full moon — exactly the ratio that makes a moonlit night wash the stars out
 * and a moonless one bring them back, with no code deciding that it should.
 */

import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  attribute,
  cameraProjectionMatrix,
  cameraViewMatrix,
  computeKernel,
  cos,
  dot,
  exp,
  float,
  globalId,
  int,
  max,
  normalize,
  positionGeometry,
  screenSize,
  sin,
  storageTexture,
  textureStore,
  uint,
  uniform,
  uvec2,
  varying,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

import type { ResourceManager } from '@/engine/ResourceManager';
import type { ShaderNode } from '@/terrain/shaders/debugViews';
import {
  GALACTIC_CENTRE_DEC_DEG,
  GALACTIC_CENTRE_RA_HOURS,
  GALACTIC_POLE_DEC_DEG,
  GALACTIC_POLE_RA_HOURS,
  STAR_COUNT,
  STAR_DATA,
} from './data/stars';
import { fnSkyFbm } from './shaders/atmosphere.wgsl';

type N = ShaderNode;

/** Illuminance of a magnitude-0 star outside the atmosphere, lux. */
const MAG0_LUX = 2.54e-6;

/** Gaussian sigma of a star sprite, in pixels, and the quad half-size in sigmas. */
const STAR_SIGMA_PX = 1.0;
const STAR_QUAD_SIGMAS = 3.2;
/** Integral of the sprite's gaussian over the pixel grid: 2πσ². */
const STAR_GAUSS_INTEGRAL = 2 * Math.PI * STAR_SIGMA_PX * STAR_SIGMA_PX;

/**
 * Procedurally placed faint stars, on top of the catalogue.
 *
 * 1 800 down to magnitude 6.4 is roughly the real naked-eye count over the
 * whole sphere (about 6 000 to mag 6.5; the shortfall is deliberate — a screen
 * cannot resolve them and they would alias into a grey wash).
 */
const PROCEDURAL_STARS = 1800;

const MILKY_W = 512;
const MILKY_H = 256;
const MILKY_WG = 8;

/**
 * B−V colour index → linear RGB, a cheap fit to the blackbody locus.
 *
 * Stars really are visibly coloured — Betelgeuse against Rigel in the same
 * constellation is the classic demonstration — and a white star field throws
 * that away for nothing. This is a two-segment approximation of the
 * Ballesteros temperature relation followed by a Planck fit, evaluated on the
 * CPU once at startup rather than per frame.
 */
function bvToRgb(bv: number, out: Float32Array, offset: number): void {
  const t = 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62));
  // Planckian locus, Krystek's rational fit, sufficient over 1500–15000 K.
  const x =
    t < 4000
      ? -0.2661239e9 / (t * t * t) - 0.234358e6 / (t * t) + 0.8776956e3 / t + 0.179910
      : -3.0258469e9 / (t * t * t) + 2.1070379e6 / (t * t) + 0.2226347e3 / t + 0.24039;
  let y: number;
  if (t < 2222) y = -1.1063814 * x * x * x - 1.34811020 * x * x + 2.18555832 * x - 0.20219683;
  else if (t < 4000) y = -0.9549476 * x * x * x - 1.37418593 * x * x + 2.09137015 * x - 0.16748867;
  else y = 3.0817580 * x * x * x - 5.87338670 * x * x + 3.75112997 * x - 0.37001483;

  const Y = 1;
  const X = (x / Math.max(y, 1e-4)) * Y;
  const Z = ((1 - x - y) / Math.max(y, 1e-4)) * Y;
  // XYZ → linear sRGB.
  let r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z;
  let g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
  let b = 0.0557 * X - 0.204 * Y + 1.057 * Z;
  r = Math.max(0, r);
  g = Math.max(0, g);
  b = Math.max(0, b);
  // Normalise to unit luminance so magnitude alone controls brightness.
  const lum = Math.max(1e-4, 0.2126 * r + 0.7152 * g + 0.0722 * b);
  out[offset] = r / lum;
  out[offset + 1] = g / lum;
  out[offset + 2] = b / lum;
}

export interface StarsOptions {
  resources: ResourceManager;
  renderer: THREE.WebGPURenderer;
}

export class Stars {
  readonly mesh: THREE.Mesh;
  /** rgba16f equirectangular map of the Milky Way, in equatorial coordinates. */
  readonly milkyWay: THREE.StorageTexture;

  /** Column-major 3×3, equatorial → world. Set every frame by the atmosphere. */
  readonly uCelestial = uniform(new THREE.Matrix3());
  /** Overall brightness, faded out by daylight so nothing draws at noon. */
  readonly uVisibility = uniform(0);
  /** Seconds, for the twinkle. */
  readonly uTime = uniform(0);
  /** Steradians subtended by one pixel — turns illuminance into radiance. */
  readonly uPixelSolidAngle = uniform(1e-7);
  /** Camera altitude expressed as a planet radius, km, for extinction. */
  readonly uCameraRadius = uniform(6360);

  private material: THREE.MeshBasicNodeMaterial;
  private geometry: THREE.InstancedBufferGeometry;
  private milkyKernel: ReturnType<typeof computeKernel>;
  private baked = false;
  private readonly milkyDispatch: number[];

  constructor(
    opts: StarsOptions,
    /** Transmittance sampler supplied by the LUT chain: (r, mu) → vec3. */
    sampleTransmittance: (r: N, mu: N) => N,
  ) {
    this.milkyWay = new THREE.StorageTexture(MILKY_W, MILKY_H);
    this.milkyWay.format = THREE.RGBAFormat;
    this.milkyWay.type = THREE.HalfFloatType;
    this.milkyWay.name = 'atmos/milkyWay';
    this.milkyWay.colorSpace = THREE.NoColorSpace;
    this.milkyWay.magFilter = THREE.LinearFilter;
    this.milkyWay.minFilter = THREE.LinearFilter;
    // Right ascension wraps at 24h; declination does not.
    this.milkyWay.wrapS = THREE.RepeatWrapping;
    this.milkyWay.wrapT = THREE.ClampToEdgeWrapping;
    this.milkyWay.generateMipmaps = false;
    this.milkyWay.mipmapsAutoUpdate = false;
    opts.resources.track(this.milkyWay, 'atmos/milkyWay', MILKY_W * MILKY_H * 8);

    this.milkyDispatch = [Math.ceil(MILKY_W / MILKY_WG), Math.ceil(MILKY_H / MILKY_WG), 1];
    this.milkyKernel = this.buildMilkyWay();

    this.geometry = this.buildGeometry();
    this.material = this.buildMaterial(sampleTransmittance);

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'stars';
    this.mesh.frustumCulled = false;
    // Drawn in the transparent pass, i.e. after everything opaque, so the depth
    // test can reject stars behind a mountain.
    this.mesh.renderOrder = 2000;
  }

  /**
   * Instance data: a unit equatorial vector plus magnitude, and a colour plus a
   * twinkle phase. Both computed once — the sky rotates by a matrix, not by
   * rewriting two thousand vectors a frame.
   *
   * Two populations, and both are needed. The catalogue gives the *shapes* —
   * Orion, the Plough, Cassiopeia — which is what makes a sky recognisable. But
   * 190 stars over 41 000 square degrees is one star per 220 deg², so a 60°
   * field holds about a dozen and reads as an almost empty sky. That was
   * measured on the first night screenshot, and it looked like a planetarium
   * with the lights half up. The other 1 800 are procedural, drawn from the real
   * magnitude distribution (N ∝ 10^0.6m, which is what a uniform density of
   * stars in a slab gives) and concentrated toward the galactic plane, where
   * they actually are.
   */
  private buildGeometry(): THREE.InstancedBufferGeometry {
    const geo = new THREE.InstancedBufferGeometry();
    // A quad in [-1,1]², expanded to pixel size in the vertex shader.
    const corners = new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]);
    geo.setAttribute('position', new THREE.BufferAttribute(corners, 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint16Array([0, 1, 2, 0, 2, 3]), 1));

    const total = STAR_COUNT + PROCEDURAL_STARS;
    const dirMag = new Float32Array(total * 4);
    const tint = new Float32Array(total * 4);

    for (let i = 0; i < STAR_COUNT; i++) {
      const ra = (STAR_DATA[i * 4] / 24) * Math.PI * 2;
      const dec = (STAR_DATA[i * 4 + 1] * Math.PI) / 180;
      const cosDec = Math.cos(dec);
      dirMag[i * 4] = cosDec * Math.cos(ra);
      dirMag[i * 4 + 1] = cosDec * Math.sin(ra);
      dirMag[i * 4 + 2] = Math.sin(dec);
      dirMag[i * 4 + 3] = STAR_DATA[i * 4 + 2];
      bvToRgb(STAR_DATA[i * 4 + 3], tint, i * 4);
      // Twinkle phase from the index — deterministic, and no two neighbours in
      // the catalogue are neighbours on the sky, so it looks uncorrelated.
      tint[i * 4 + 3] = (i * 2.39996) % (Math.PI * 2);
    }

    const poleRa = (GALACTIC_POLE_RA_HOURS / 24) * Math.PI * 2;
    const poleDec = (GALACTIC_POLE_DEC_DEG * Math.PI) / 180;
    const px = Math.cos(poleDec) * Math.cos(poleRa);
    const py = Math.cos(poleDec) * Math.sin(poleRa);
    const pz = Math.sin(poleDec);

    // Deterministic 32-bit LCG rather than Math.random: the same seed has to
    // give the same sky, and the whole project is built on that promise.
    let rng = 0x9e3779b9;
    const rand = (): number => {
      rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0;
      return rng / 4294967296;
    };

    let written = 0;
    let guard = 0;
    while (written < PROCEDURAL_STARS && guard < PROCEDURAL_STARS * 40) {
      guard++;
      // Uniform on the sphere.
      const z = rand() * 2 - 1;
      const phi = rand() * Math.PI * 2;
      const s = Math.sqrt(Math.max(0, 1 - z * z));
      const x = s * Math.cos(phi);
      const y = s * Math.sin(phi);

      // Rejection-sample toward the galactic plane: |sin b| small is where the
      // disc is. 0.35 of the stars survive anywhere, the rest only near the
      // plane, which reproduces the visible thickening of the Milky Way without
      // needing a real density model.
      const sinB = x * px + y * py + z * pz;
      const planeWeight = 0.35 + 0.65 * Math.exp(-(sinB * sinB) / (2 * 0.22 * 0.22));
      if (rand() > planeWeight) continue;

      // Magnitudes 3.2–6.4 with N ∝ 10^(0.6 m): invert the cumulative count.
      const u = rand();
      const mLo = 3.2;
      const mHi = 6.4;
      const a = Math.pow(10, 0.6 * mLo);
      const bq = Math.pow(10, 0.6 * mHi);
      const m = Math.log10(a + u * (bq - a)) / 0.6;

      const k = STAR_COUNT + written;
      dirMag[k * 4] = x;
      dirMag[k * 4 + 1] = y;
      dirMag[k * 4 + 2] = z;
      dirMag[k * 4 + 3] = m;
      // B−V from −0.2 to 1.6, skewed red: most naked-eye faint stars are late
      // K and M dwarfs and giants.
      bvToRgb(-0.2 + Math.pow(rand(), 0.6) * 1.8, tint, k * 4);
      tint[k * 4 + 3] = rand() * Math.PI * 2;
      written++;
    }

    geo.setAttribute('aStar', new THREE.InstancedBufferAttribute(dirMag, 4));
    geo.setAttribute('aStarTint', new THREE.InstancedBufferAttribute(tint, 4));
    geo.instanceCount = STAR_COUNT + written;
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e7);
    return geo;
  }

  private buildMaterial(
    sampleTransmittance: (r: N, mu: N) => N,
  ): THREE.MeshBasicNodeMaterial {
    const mat = new THREE.MeshBasicNodeMaterial();
    mat.transparent = true;
    mat.blending = THREE.AdditiveBlending;
    mat.depthWrite = false;
    mat.depthTest = true;
    // Aerial perspective must not touch a star: it is outside the atmosphere,
    // and its extinction is applied explicitly below from the transmittance LUT.
    mat.fog = false;

    const star = attribute('aStar', 'vec4');
    const tint = attribute('aStarTint', 'vec4');

    const eq = star.xyz;
    const magnitude = star.w;

    const world = normalize(this.uCelestial.mul(eq));
    const viewDir = cameraViewMatrix.mul(vec4(world, 0)).xyz;
    const clip = cameraProjectionMatrix.mul(vec4(viewDir, 1));

    // Perspective divide by hand, because the final clip position is written
    // with w = 1: the sprite has to sit at a fixed pixel size, which means its
    // corner offsets are in NDC and must not be divided again.
    const w = max(clip.w, float(1e-6));
    const ndc = vec2(clip.x.div(w), clip.y.div(w));

    const quadPx = STAR_SIGMA_PX * STAR_QUAD_SIGMAS;
    const px = vec2(
      float(quadPx).mul(2).div(screenSize.x),
      float(quadPx).mul(2).div(screenSize.y),
    );
    const offset = positionGeometry.xy.mul(px);

    // Cull: behind the camera, or below the horizon where the ground is.
    const behind = clip.w.lessThanEqual(float(0));
    const belowHorizon = world.y.lessThan(float(-0.04));
    const culled = behind.or(belowHorizon);

    mat.vertexNode = culled.select(
      vec4(0, 0, 2, 1), // outside the depth range, clipped away
      vec4(ndc.x.add(offset.x), ndc.y.add(offset.y), 1, 1),
    );

    const vQuad = varying(positionGeometry.xy);
    const vAmplitude = varying(
      Fn(() => {
        // Illuminance above the atmosphere, then extinction along the line of
        // sight from the transmittance LUT — which is why stars redden and fade
        // as they set instead of blinking out at the horizon.
        const e = float(MAG0_LUX).mul(float(10).pow(magnitude.mul(-0.4)));
        const trans = sampleTransmittance(this.uCameraRadius, world.y);
        // Scintillation. Real twinkling is an atmospheric path-length effect:
        // near the zenith it is almost absent, near the horizon it is violent.
        const airmass = float(1).div(max(world.y, float(0.05)));
        const twinkleAmount = float(0.05).mul(airmass).min(float(0.55));
        const t = this.uTime.mul(6.0).add(tint.w);
        const flicker = sin(t)
          .mul(0.6)
          .add(sin(t.mul(2.37).add(1.7)).mul(0.4))
          .mul(twinkleAmount);
        const amp = e
          .div(max(this.uPixelSolidAngle.mul(STAR_GAUSS_INTEGRAL), float(1e-12)))
          .mul(float(1).add(flicker))
          .mul(this.uVisibility);
        return vec3(amp.mul(tint.x), amp.mul(tint.y), amp.mul(tint.z)).mul(trans);
      })(),
    );

    mat.colorNode = vAmplitude;
    // Gaussian profile across the quad. `positionGeometry.xy` runs −1..1 over
    // `STAR_QUAD_SIGMAS` sigmas, so the exponent is in sigma units directly.
    const r2 = dot(vQuad, vQuad).mul(STAR_QUAD_SIGMAS * STAR_QUAD_SIGMAS);
    mat.opacityNode = exp(r2.mul(-0.5));

    return mat;
  }

  /**
   * Bake the Milky Way.
   *
   * Galactic coordinates are derived per texel from the equatorial direction:
   * latitude from the dot product with the galactic pole, and the angle to the
   * galactic centre for the brightening toward Sagittarius. The band is a
   * gaussian in latitude with two dark-lane terms — the Great Rift is a real
   * feature and it is most of what makes the band read as a galaxy seen edge-on
   * rather than as a smear of fog.
   */
  private buildMilkyWay(): ReturnType<typeof computeKernel> {
    const store = storageTexture(this.milkyWay);

    const poleRa = (GALACTIC_POLE_RA_HOURS / 24) * Math.PI * 2;
    const poleDec = (GALACTIC_POLE_DEC_DEG * Math.PI) / 180;
    const pole = new THREE.Vector3(
      Math.cos(poleDec) * Math.cos(poleRa),
      Math.cos(poleDec) * Math.sin(poleRa),
      Math.sin(poleDec),
    );
    const gcRa = (GALACTIC_CENTRE_RA_HOURS / 24) * Math.PI * 2;
    const gcDec = (GALACTIC_CENTRE_DEC_DEG * Math.PI) / 180;
    const centre = new THREE.Vector3(
      Math.cos(gcDec) * Math.cos(gcRa),
      Math.cos(gcDec) * Math.sin(gcRa),
      Math.sin(gcDec),
    );

    const poleNode = vec3(pole.x, pole.y, pole.z);
    const centreNode = vec3(centre.x, centre.y, centre.z);

    const kernel = Fn(() => {
      const gx = globalId.x;
      const gy = globalId.y;
      If(gx.lessThan(uint(MILKY_W)).and(gy.lessThan(uint(MILKY_H))), () => {
        const ra = float(gx).add(0.5).div(float(MILKY_W)).mul(Math.PI * 2);
        const dec = float(gy).add(0.5).div(float(MILKY_H)).sub(0.5).mul(Math.PI);
        const cd = cos(dec);
        const dir = vec3(cd.mul(cos(ra)), cd.mul(sin(ra)), sin(dec));

        // Galactic latitude, and how close we are to the bulge.
        const sinB = dot(dir, poleNode);
        const toCentre = dot(dir, centreNode);

        // Two-component band: a narrow bright core and a wide faint halo. A
        // single gaussian gives a fuzzy stripe with no structure; the real band
        // has a thin dense line inside a much broader glow.
        const b = sinB;
        const core = exp(b.mul(b).mul(-1 / (2 * 0.045 * 0.045)));
        const halo = exp(b.mul(b).mul(-1 / (2 * 0.16 * 0.16)));
        const band = core.mul(0.75).add(halo.mul(0.35));

        // The bulge: three times brighter toward Sagittarius, falling off over
        // about 40°, and the anticentre is genuinely dim.
        const bulge = float(1).add(
          exp(toCentre.sub(1).mul(6)).mul(2.6),
        );

        // Dust. Two scales: the Great Rift as a broad dark lane biased to one
        // side of the plane, and finer mottling.
        const lane = fnSkyFbm({ p: dir.mul(3.1), octaves: int(3) });
        const fine = fnSkyFbm({ p: dir.mul(14.0), octaves: int(3) });
        const rift = exp(b.add(0.012).mul(b.add(0.012)).mul(-1 / (2 * 0.03 * 0.03)));
        const dust = float(1)
          .sub(rift.mul(0.55).mul(lane.mul(0.8).add(0.2)))
          .mul(float(0.55).add(fine.mul(0.9)));

        // Peak surface brightness of the band near the galactic centre, cd/m².
        // The real value is around 2–3 × 10⁻⁴, i.e. a couple of times the
        // moonless zenith sky — faint, but not subtle once the eye is adapted.
        const intensity = band.mul(bulge).mul(max(dust, float(0.05))).mul(2.4e-4);
        // Slightly warm: the integrated light of the disc is dominated by K
        // giants, and the blue end is what the dust absorbs most.
        const colour = vec3(1.06, 1.0, 0.92).mul(intensity);
        textureStore(store, uvec2(gx, gy), vec4(colour.x, colour.y, colour.z, 1));
      });
    });

    const k = computeKernel(kernel(), [MILKY_WG, MILKY_WG, 1]);
    k.setName('milkyWayBake');
    return k;
  }

  bake(renderer: THREE.WebGPURenderer): void {
    if (this.baked) return;
    this.baked = true;
    renderer.compute(this.milkyKernel, this.milkyDispatch);
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
    this.milkyKernel.dispose();
    this.milkyWay.dispose();
  }
}

export { STAR_COUNT };
