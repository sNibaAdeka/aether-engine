/**
 * The Bruneton/Hillaire LUT chain: four tables, two of them computed once and
 * two of them every frame.
 *
 * | table              | size       | when        | contents                     |
 * |--------------------|------------|-------------|------------------------------|
 * | transmittance      | 256 × 64   | once        | ground→space attenuation     |
 * | multiple scattering| 32 × 32    | once        | 2nd-and-beyond order, isotropic |
 * | sky view           | 200 × 100  | every frame | sky radiance around the camera |
 * | aerial perspective | 32³ froxel | every frame | inscatter + transmittance to 32 km |
 *
 * The split is the entire point of the method. Transmittance and multiple
 * scattering depend only on the *atmosphere*, so they are constant for a given
 * turbidity and are baked at startup; sky view and aerial perspective depend on
 * where the camera and the lights are, so they are rebuilt each frame — but at
 * 20 000 and 1 024 rays respectively, not at one ray per pixel. A brute-force
 * per-pixel march of the same quality is roughly 200× the work.
 *
 * ## Two lights in one march
 *
 * Every march here integrates the sun *and* the moon in the same pass, rather
 * than switching source at some twilight threshold. That costs about 40 % more
 * per step (the density and transmittance evaluation is shared; only the phase
 * functions and the sun-transmittance lookup double) and it buys the thing the
 * spec explicitly asks for: "плавный переход день↔ночь без единого скачка".
 * With a switch there is always one frame where the sky's light source teleports
 * across the dome, and it is very visible. With both integrated, the moon's
 * contribution simply becomes the larger one somewhere around −8° solar
 * elevation, on its own, with no code involved.
 *
 * ## Units
 *
 * Kilometres and lux, throughout. Radiance comes out in cd/m² directly, which is
 * why `Exposure` can be an actual photographic calculation rather than a magic
 * multiplier. See `shaders/atmosphere.wgsl.ts` for why not metres.
 */

import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  computeKernel,
  dot,
  exp,
  float,
  globalId,
  int,
  length,
  max,
  normalize,
  smoothstep,
  storageTexture,
  texture,
  texture3D,
  textureStore,
  uint,
  uniform,
  uvec2,
  uvec3,
  vec2,
  vec3,
  vec4,
  instancedArray,
} from 'three/tsl';

import type { Profiler } from '@/engine/Profiler';
import type { ResourceManager } from '@/engine/ResourceManager';
import type { ShaderNode } from '@/terrain/shaders/debugViews';
import {
  AERIAL_MAX_KM,
  ATMOSPHERE_RADIUS_KM,
  GROUND_ALBEDO,
  LUT_SIZES,
  PLANET_RADIUS_KM,
  fnExtinction,
  fnMiePhase,
  fnMieScattering,
  fnRayMarchLength,
  fnRaySphere,
  fnRayleighPhase,
  fnRayleighScattering,
  fnSkyViewDir,
  fnSkyViewUv,
  fnTexCoordToUnit,
  fnTransmittanceParams,
  fnTransmittanceUv,
  fnUniformSphere,
  fnUnitToTexCoord,
} from './shaders/atmosphere.wgsl';

type N = ShaderNode;
type ComputeKernelNode = ReturnType<typeof computeKernel>;

/** Steps along one transmittance ray. 40 is Bruneton's own number. */
const TRANSMITTANCE_STEPS = 40;
/** Directions sampled per multiple-scattering texel, and steps along each. */
const MS_DIRECTIONS = 64;
const MS_STEPS = 20;
/** Steps along one sky-view ray. */
const SKY_STEPS = 32;
/** Sub-steps per aerial-perspective slice; 32 slices × 2 = 64 along the ray. */
const AERIAL_SUBSTEPS = 2;
/** Directions used to project the sky into SH. 512 is well past convergence for L2. */
const SH_DIRECTIONS = 512;

const WG = 8;

/**
 * Number of samples the SH projection reads per coefficient.
 *
 * Nine coefficients are computed by nine invocations in one workgroup, each
 * looping over the same 512 directions. It looks wasteful — the LUT is sampled
 * 9× more often than strictly needed — but the alternative is a workgroup
 * reduction over 9 × 3 floats with shared memory and two barriers, and this pass
 * costs microseconds either way.
 */
const SH_WORKGROUP = 16;

export interface AtmosphereLutUniforms {
  /** Unit world direction toward the sun. */
  sunDirection: ReturnType<typeof uniform>;
  /** Unit world direction toward the moon. */
  moonDirection: ReturnType<typeof uniform>;
  /** Extraterrestrial solar irradiance, lux, as an RGB triple. */
  sunIrradiance: ReturnType<typeof uniform>;
  /** Extraterrestrial lunar irradiance, lux, as an RGB triple. */
  moonIrradiance: ReturnType<typeof uniform>;
  /** Camera distance from the planet centre, km. */
  cameraRadius: ReturnType<typeof uniform>;
  /** Mie density multiplier derived from turbidity. */
  mieScale: ReturnType<typeof uniform>;
  /** Camera forward vector scaled so that adding the two basis vectors spans the frustum. */
  rayForward: ReturnType<typeof uniform>;
  rayRight: ReturnType<typeof uniform>;
  rayUp: ReturnType<typeof uniform>;
}

function makeUniforms(): AtmosphereLutUniforms {
  return {
    sunDirection: uniform(new THREE.Vector3(0, 1, 0)),
    moonDirection: uniform(new THREE.Vector3(0, -1, 0)),
    sunIrradiance: uniform(new THREE.Vector3(128000, 128000, 128000)),
    moonIrradiance: uniform(new THREE.Vector3(0, 0, 0)),
    cameraRadius: uniform(PLANET_RADIUS_KM),
    mieScale: uniform(1),
    rayForward: uniform(new THREE.Vector3(0, 0, -1)),
    rayRight: uniform(new THREE.Vector3(1, 0, 0)),
    rayUp: uniform(new THREE.Vector3(0, 1, 0)),
  };
}

export class AtmosphereLUTs {
  /** rgba16f, 256 × 64. Transmittance from a point to the top of the atmosphere. */
  readonly transmittance: THREE.StorageTexture;
  /** rgba16f, 32 × 32. Second-and-higher-order scattering, isotropic. */
  readonly multiScatter: THREE.StorageTexture;
  /** rgba16f, 200 × 100. Sky radiance in polar coordinates about the camera. */
  readonly skyView: THREE.StorageTexture;
  /** rgba16f, 32³. rgb inscatter (cd/m²), a mean transmittance, out to 32 km. */
  readonly aerial: THREE.Storage3DTexture;

  readonly u = makeUniforms();

  /**
   * CPU-readable copy of the transmittance table.
   *
   * The sun's illuminance on the ground is `E_toa × T(r, μ_sun)`, and that
   * number has to reach a `DirectionalLight`, which lives on the CPU. Rather
   * than re-implement the table in TypeScript (and then have to keep two
   * implementations in step, which is exactly the maintenance trap CLAUDE.md
   * rule 3 exists to describe), the bake kernel writes the same values into a
   * storage buffer and this class reads it back once and samples it.
   */
  private transmittanceBuffer = instancedArray(
    LUT_SIZES.transmittanceW * LUT_SIZES.transmittanceH * 4,
    'float',
  );
  private transmittanceCPU: Float32Array | null = null;
  private transmittanceRequested = false;

  /** 9 × vec4 of SH coefficients (w unused), read back for the light probe. */
  private shBuffer = instancedArray(9 * 4, 'float');
  private shCPU = new Float32Array(9 * 4);
  private shReadInFlight = false;

  private kTransmittance: ComputeKernelNode;
  private kMultiScatter: ComputeKernelNode;
  private kSkyView: ComputeKernelNode;
  private kAerial: ComputeKernelNode;
  private kSH: ComputeKernelNode;

  private staticBaked = false;

  // Reused dispatch descriptors — `renderer.compute` must not allocate.
  private readonly dTransmittance: number[];
  private readonly dMultiScatter: number[];
  private readonly dSkyView: number[];
  private readonly dAerial: number[];
  private readonly dSH: number[] = [1, 1, 1];

  constructor(
    private renderer: THREE.WebGPURenderer,
    resources: ResourceManager,
    private profiler: Profiler,
  ) {
    this.transmittance = makeStorage2D(
      LUT_SIZES.transmittanceW,
      LUT_SIZES.transmittanceH,
      'atmos/transmittance',
    );
    this.multiScatter = makeStorage2D(
      LUT_SIZES.multiScatterW,
      LUT_SIZES.multiScatterH,
      'atmos/multiScatter',
    );
    this.skyView = makeStorage2D(LUT_SIZES.skyViewW, LUT_SIZES.skyViewH, 'atmos/skyView');
    // Azimuth wraps: without repeat on U the texel at 359.1° blends with the
    // clamped edge instead of with 0.9°, and a seam runs up the sky due north.
    this.skyView.wrapS = THREE.RepeatWrapping;

    this.aerial = new THREE.Storage3DTexture(LUT_SIZES.aerial, LUT_SIZES.aerial, LUT_SIZES.aerial);
    this.aerial.format = THREE.RGBAFormat;
    this.aerial.type = THREE.HalfFloatType;
    this.aerial.name = 'atmos/aerial';
    this.aerial.colorSpace = THREE.NoColorSpace;
    this.aerial.magFilter = THREE.LinearFilter;
    this.aerial.minFilter = THREE.LinearFilter;
    this.aerial.wrapS = THREE.ClampToEdgeWrapping;
    this.aerial.wrapT = THREE.ClampToEdgeWrapping;
    this.aerial.wrapR = THREE.ClampToEdgeWrapping;

    const px = (w: number, h: number, d = 1): number => w * h * d * 8; // rgba16f
    resources.track(this.transmittance, 'atmos/transmittance', px(256, 64));
    resources.track(this.multiScatter, 'atmos/multiScatter', px(32, 32));
    resources.track(this.skyView, 'atmos/skyView', px(200, 100));
    resources.track(this.aerial, 'atmos/aerial', px(32, 32, 32));

    this.dTransmittance = [
      Math.ceil(LUT_SIZES.transmittanceW / WG),
      Math.ceil(LUT_SIZES.transmittanceH / WG),
      1,
    ];
    this.dMultiScatter = [
      Math.ceil(LUT_SIZES.multiScatterW / WG),
      Math.ceil(LUT_SIZES.multiScatterH / WG),
      1,
    ];
    this.dSkyView = [Math.ceil(LUT_SIZES.skyViewW / WG), Math.ceil(LUT_SIZES.skyViewH / WG), 1];
    this.dAerial = [Math.ceil(LUT_SIZES.aerial / WG), Math.ceil(LUT_SIZES.aerial / WG), 1];

    this.kTransmittance = this.buildTransmittance();
    this.kMultiScatter = this.buildMultiScatter();
    this.kSkyView = this.buildSkyView();
    this.kAerial = this.buildAerial();
    this.kSH = this.buildSH();
  }

  // -------------------------------------------------------------------------
  // Shared sampling helpers
  // -------------------------------------------------------------------------

  /**
   * Bilinear fetch of the transmittance table at (r, mu).
   *
   * Bruneton's table only covers rays that *escape*: its parameterisation maps
   * uv to the distance to the top of the atmosphere and says nothing about a ray
   * that meets the ground first. Sampling it below the local horizon therefore
   * returns a number that means nothing — which is exactly what drew the solar
   * disc 1.7° below the horizon in the first dusk screenshot. The guard
   * multiplies by zero there, feathered over 0.29° (about one solar diameter) so
   * a setting sun sinks rather than switches off.
   */
  sampleTransmittance = (r: N, mu: N): N => {
    const uv = fnTransmittanceUv({ r, mu });
    const u = fnUnitToTexCoord({ x: uv.x, n: float(LUT_SIZES.transmittanceW) });
    const v = fnUnitToTexCoord({ x: uv.y, n: float(LUT_SIZES.transmittanceH) });
    const t = texture(this.transmittance, vec2(u, v)).level(float(0)).xyz;
    const muHorizon = float(1)
      .sub(float(PLANET_RADIUS_KM).div(r).pow(float(2)))
      .max(float(0))
      .sqrt()
      .negate();
    return t.mul(smoothstep(muHorizon.sub(0.005), muHorizon.add(0.005), mu));
  };

  /**
   * Sky radiance in a world-space direction, cd/m². Used by `Sky.ts` and by the
   * SH projection; both want exactly the same number.
   */
  sampleSkyView = (dir: N): N => {
    const uv = fnSkyViewUv({ dir, r: this.u.cameraRadius });
    // U wraps (azimuth), so no inset there — an inset would compress the
    // 360°-wide axis by one texel and put a slow drift into the sun's azimuth.
    const v = fnUnitToTexCoord({ x: uv.y, n: float(LUT_SIZES.skyViewH) });
    return texture(this.skyView, vec2(uv.x, v)).level(float(0)).xyz;
  };

  /**
   * Aerial perspective at a screen position and a distance in kilometres.
   * Returns rgb = in-scattered radiance, a = transmittance.
   *
   * The froxel slices are quadratic in distance, so the lookup takes the square
   * root — the inverse of the distribution `buildAerial` writes.
   */
  sampleAerial = (screenUv: N, distanceKm: N): N => {
    const w = distanceKm.div(float(AERIAL_MAX_KM)).clamp(0, 1).sqrt();
    const z = fnUnitToTexCoord({ x: w, n: float(LUT_SIZES.aerial) });
    return texture3D(this.aerial, vec3(screenUv.x, screenUv.y, z)).level(float(0));
  };

  /** Bilinear fetch of the multiple-scattering table at (r, mu_light). */
  private sampleMultiScatter(r: N, mu: N): N {
    const x = mu.mul(0.5).add(0.5);
    const y = r
      .sub(float(PLANET_RADIUS_KM))
      .div(float(ATMOSPHERE_RADIUS_KM - PLANET_RADIUS_KM))
      .clamp(0, 1);
    const u = fnUnitToTexCoord({ x, n: float(LUT_SIZES.multiScatterW) });
    const v = fnUnitToTexCoord({ x: y, n: float(LUT_SIZES.multiScatterH) });
    return texture(this.multiScatter, vec2(u, v)).level(float(0)).xyz;
  }

  /**
   * One scattering march, both lights, with the analytic per-segment
   * integration from Hillaire §4.
   *
   * The naive form — accumulate `throughput * S * dt` — under-integrates badly
   * at low step counts because the source term is attenuated across the segment
   * it is emitted in. Integrating the constant-source segment exactly,
   * `(S − S·exp(−σ·dt))/σ`, is one extra divide and it is what lets 32 steps
   * look like 128.
   *
   * Returns the in-scattered radiance and the transmittance over `tMax`.
   */
  private march(
    origin: N,
    dir: N,
    tMax: N,
    steps: number,
    hitsGround: N | null,
  ): { L: N; transmittance: N } {
    const u = this.u;
    const L = vec3(0, 0, 0).toVar();
    const throughput = vec3(1, 1, 1).toVar();
    const dt = tMax.div(float(steps));

    Loop(steps, ({ i }) => {
      const tMid = float(i).add(0.5).mul(dt);
      const p = origin.add(dir.mul(tMid));
      const r = max(length(p), float(PLANET_RADIUS_KM));
      const h = r.sub(float(PLANET_RADIUS_KM));
      const up = p.div(r);

      const ext = fnExtinction({ h, mieScale: u.mieScale });
      const rayleigh = fnRayleighScattering({ h });
      const mie = fnMieScattering({ h, mieScale: u.mieScale });
      const stepT = exp(ext.mul(dt).negate());
      // Guard the divide, not the exponent: at 80 km the extinction is ~1e-9
      // and `(S − S·T)/σ` is 0/0 in f32 without it.
      const safeExt = max(ext, vec3(1e-9, 1e-9, 1e-9));

      const contribute = (lightDir: N, irradiance: N): void => {
        const muL = dot(up, lightDir);
        const lightT = this.sampleTransmittance(r, muL);
        // The planet occludes the light below the local horizon. Without this
        // the ground glows from underneath at night.
        const blocked = fnRaySphere({ ro: p, rd: lightDir, radius: float(PLANET_RADIUS_KM) });
        const visible = blocked.lessThan(float(0)).select(float(1), float(0));
        const cosT = dot(dir, lightDir);
        const single = rayleigh
          .mul(fnRayleighPhase({ cosTheta: cosT }))
          .add(vec3(mie, mie, mie).mul(fnMiePhase({ cosTheta: cosT })))
          .mul(lightT)
          .mul(visible);
        // Multiple scattering is isotropic by construction and is *not* shadowed
        // by the planet — that is what keeps the sky lit in the minutes after
        // the sun has set, which single scattering alone gets badly wrong.
        const multi = rayleigh.add(vec3(mie, mie, mie)).mul(this.sampleMultiScatter(r, muL));
        const source = single.add(multi).mul(irradiance);
        L.addAssign(throughput.mul(source.sub(source.mul(stepT)).div(safeExt)));
      };

      contribute(u.sunDirection, u.sunIrradiance);
      contribute(u.moonDirection, u.moonIrradiance);

      throughput.mulAssign(stepT);
    });

    if (hitsGround !== null) {
      If(hitsGround.greaterThan(float(0.5)), () => {
        const p = origin.add(dir.mul(tMax));
        const r = max(length(p), float(PLANET_RADIUS_KM));
        const up = p.div(r);
        const groundLight = (lightDir: N, irradiance: N): N => {
          const muL = dot(up, lightDir);
          return this.sampleTransmittance(r, muL).mul(irradiance).mul(max(muL, float(0)));
        };
        const e = groundLight(u.sunDirection, u.sunIrradiance).add(
          groundLight(u.moonDirection, u.moonIrradiance),
        );
        L.addAssign(throughput.mul(e).mul(GROUND_ALBEDO / Math.PI));
      });
    }

    return { L, transmittance: throughput };
  }

  // -------------------------------------------------------------------------
  // Kernels
  // -------------------------------------------------------------------------

  private buildTransmittance(): ComputeKernelNode {
    const W = LUT_SIZES.transmittanceW;
    const H = LUT_SIZES.transmittanceH;
    const store = storageTexture(this.transmittance);
    const buf = this.transmittanceBuffer;

    const kernel = Fn(() => {
      const gx = globalId.x;
      const gy = globalId.y;
      If(gx.lessThan(uint(W)).and(gy.lessThan(uint(H))), () => {
        const x = fnTexCoordToUnit({
          u: float(gx).add(0.5).div(float(W)),
          n: float(W),
        });
        const y = fnTexCoordToUnit({
          u: float(gy).add(0.5).div(float(H)),
          n: float(H),
        });
        const params = fnTransmittanceParams({ uv: vec2(x.clamp(0, 1), y.clamp(0, 1)) });
        const r = params.x;
        const mu = params.y;

        const tMax = fnRayMarchLength({ r, mu });
        const dt = tMax.div(float(TRANSMITTANCE_STEPS));
        const depth = vec3(0, 0, 0).toVar();

        Loop(TRANSMITTANCE_STEPS, ({ i }) => {
          const t = float(i).add(0.5).mul(dt);
          // Law of cosines rather than a vector position: the ray is defined by
          // (r, mu) alone and reconstructing a 3D point would add a cancellation
          // at 6360 km for no benefit.
          const rr = max(
            float(0).add(r.mul(r).add(t.mul(t)).add(r.mul(mu).mul(t).mul(2))).sqrt(),
            float(PLANET_RADIUS_KM),
          );
          depth.addAssign(
            fnExtinction({ h: rr.sub(float(PLANET_RADIUS_KM)), mieScale: this.u.mieScale }).mul(dt),
          );
        });

        const t = exp(depth.negate());
        textureStore(store, uvec2(gx, gy), vec4(t.x, t.y, t.z, 1));
        const base = gy.mul(uint(W)).add(gx).mul(uint(4));
        buf.element(base).assign(t.x);
        buf.element(base.add(uint(1))).assign(t.y);
        buf.element(base.add(uint(2))).assign(t.z);
        buf.element(base.add(uint(3))).assign(float(1));
      });
    });

    const k = computeKernel(kernel(), [WG, WG, 1]);
    k.setName('atmosTransmittance');
    return k;
  }

  private buildMultiScatter(): ComputeKernelNode {
    const W = LUT_SIZES.multiScatterW;
    const H = LUT_SIZES.multiScatterH;
    const store = storageTexture(this.multiScatter);
    const side = Math.round(Math.sqrt(MS_DIRECTIONS));

    const kernel = Fn(() => {
      const gx = globalId.x;
      const gy = globalId.y;
      If(gx.lessThan(uint(W)).and(gy.lessThan(uint(H))), () => {
        const muS = float(gx).add(0.5).div(float(W)).mul(2).sub(1);
        const r = float(PLANET_RADIUS_KM).add(
          float(gy)
            .add(0.5)
            .div(float(H))
            .mul(ATMOSPHERE_RADIUS_KM - PLANET_RADIUS_KM),
        );
        const p0 = vec3(0, r, 0);
        const lightDir = vec3(
          float(0).add(float(1).sub(muS.mul(muS)).max(0).sqrt()),
          muS,
          float(0),
        );

        // Hillaire §5.3: integrate over the whole sphere of view directions with
        // an isotropic phase and unit light, accumulating both the second-order
        // radiance and the fraction that goes on to scatter again. The geometric
        // series 1/(1−f) then sums every order above the second in closed form —
        // which is why this 32×32 table stands in for an infinite recursion.
        const lSum = vec3(0, 0, 0).toVar();
        const fSum = vec3(0, 0, 0).toVar();

        Loop(MS_DIRECTIONS, ({ i }) => {
          const iy = int(i).div(int(side));
          const ix = int(i).sub(iy.mul(int(side)));
          const dir = fnUniformSphere({
            u: float(ix).add(0.5).div(float(side)),
            v: float(iy).add(0.5).div(float(side)),
          });

          const mu = dir.y;
          const tMax = fnRayMarchLength({ r, mu });
          const dt = tMax.div(float(MS_STEPS));
          const throughput = vec3(1, 1, 1).toVar();
          const l = vec3(0, 0, 0).toVar();
          const f = vec3(0, 0, 0).toVar();

          Loop(MS_STEPS, ({ i: j }) => {
            const t = float(j).add(0.5).mul(dt);
            const p = p0.add(dir.mul(t));
            const rr = max(length(p), float(PLANET_RADIUS_KM));
            const h = rr.sub(float(PLANET_RADIUS_KM));
            const up = p.div(rr);

            const ext = fnExtinction({ h, mieScale: this.u.mieScale });
            const rayleigh = fnRayleighScattering({ h });
            const mie = fnMieScattering({ h, mieScale: this.u.mieScale });
            const scat = rayleigh.add(vec3(mie, mie, mie));
            const stepT = exp(ext.mul(dt).negate());
            const safeExt = max(ext, vec3(1e-9, 1e-9, 1e-9));

            const muL = dot(up, lightDir);
            const lightT = this.sampleTransmittance(rr, muL);
            const blocked = fnRaySphere({ ro: p, rd: lightDir, radius: float(PLANET_RADIUS_KM) });
            const visible = blocked.lessThan(float(0)).select(float(1), float(0));

            // Isotropic phase: 1/4π.
            const source = scat.mul(lightT).mul(visible).mul(1 / (4 * Math.PI));
            l.addAssign(throughput.mul(source.sub(source.mul(stepT)).div(safeExt)));
            f.addAssign(throughput.mul(scat.sub(scat.mul(stepT)).div(safeExt)));
            throughput.mulAssign(stepT);
          });

          // Ground bounce for the directions that hit it.
          const hitGround = fnRaySphere({
            ro: p0,
            rd: dir,
            radius: float(PLANET_RADIUS_KM),
          });
          If(hitGround.greaterThan(float(0)), () => {
            const p = p0.add(dir.mul(hitGround));
            const rr = max(length(p), float(PLANET_RADIUS_KM));
            const up = p.div(rr);
            const muL = dot(up, lightDir);
            l.addAssign(
              throughput
                .mul(this.sampleTransmittance(rr, muL))
                .mul(max(muL, float(0)))
                .mul(GROUND_ALBEDO / Math.PI),
            );
          });

          lSum.addAssign(l);
          fSum.addAssign(f);
        });

        const inv = 1 / MS_DIRECTIONS;
        const l2 = lSum.mul(inv);
        const fms = fSum.mul(inv);
        // 1/(1 − f), clamped: f approaches 1 only in a physically impossible
        // atmosphere, but a NaN here would poison every later table.
        const gain = vec3(1, 1, 1).div(max(vec3(1, 1, 1).sub(fms), vec3(1e-3, 1e-3, 1e-3)));
        const psi = l2.mul(gain);
        textureStore(store, uvec2(gx, gy), vec4(psi.x, psi.y, psi.z, 1));
      });
    });

    const k = computeKernel(kernel(), [WG, WG, 1]);
    k.setName('atmosMultiScatter');
    return k;
  }

  private buildSkyView(): ComputeKernelNode {
    const W = LUT_SIZES.skyViewW;
    const H = LUT_SIZES.skyViewH;
    const store = storageTexture(this.skyView);

    const kernel = Fn(() => {
      const gx = globalId.x;
      const gy = globalId.y;
      If(gx.lessThan(uint(W)).and(gy.lessThan(uint(H))), () => {
        const uvx = fnTexCoordToUnit({ u: float(gx).add(0.5).div(float(W)), n: float(W) });
        const uvy = fnTexCoordToUnit({ u: float(gy).add(0.5).div(float(H)), n: float(H) });
        const r = this.u.cameraRadius;
        const dir = fnSkyViewDir({ uv: vec2(uvx.clamp(0, 1), uvy.clamp(0, 1)), r });
        const p0 = vec3(0, r, 0);

        const tMax = fnRayMarchLength({ r, mu: dir.y });
        const groundT = fnRaySphere({ ro: p0, rd: dir, radius: float(PLANET_RADIUS_KM) });
        const hits = groundT.greaterThan(float(0)).select(float(1), float(0));

        const res = this.march(p0, dir, max(tMax, float(0.001)), SKY_STEPS, hits);
        // Airglow. Not scattering at all — it is oxygen and hydroxyl in the
        // mesosphere emitting at 90 km, and it is why a moonless, starless
        // patch of sky is not actually black: about 2 × 10⁻⁴ cd/m², green-white,
        // rising toward the horizon where the line of sight crosses more of the
        // emitting layer. Without it the night frame metered as pure black and
        // the auto-exposure ran to its clamp with nothing to show for it.
        // Utterly negligible by day, which is why it can simply be added.
        const airglowLimb = float(1).add(
          float(1).sub(dir.y.abs()).pow(float(6)).mul(2.5),
        );
        const airglow = vec3(0.55, 1.0, 0.78)
          .mul(1.35e-4)
          .mul(airglowLimb)
          .mul(dir.y.greaterThan(float(-0.02)).select(float(1), float(0)));
        const total = res.L.add(airglow);
        textureStore(store, uvec2(gx, gy), vec4(total.x, total.y, total.z, 1));
      });
    });

    const k = computeKernel(kernel(), [WG, WG, 1]);
    k.setName('atmosSkyView');
    return k;
  }

  /**
   * The froxel volume. One invocation per (x, y) walks all 32 slices, carrying
   * its throughput forward — so slice 31 costs the same as slice 0 and the whole
   * volume is 64 steps of march, not 32 × 32.
   *
   * Slice centres are quadratic in the slice index: `d = ((i+0.5)/32)² × 32 km`.
   * Linear slicing puts a full kilometre between the first two slices, and the
   * inscatter over the first kilometre is where all the near-field contrast loss
   * happens on a hazy day.
   */
  private buildAerial(): ComputeKernelNode {
    const S = LUT_SIZES.aerial;
    const store = storageTexture(this.aerial);
    const u = this.u;

    const kernel = Fn(() => {
      const gx = globalId.x;
      const gy = globalId.y;
      If(gx.lessThan(uint(S)).and(gy.lessThan(uint(S))), () => {
        // NDC of this froxel column's centre.
        const ndcX = float(gx).add(0.5).div(float(S)).mul(2).sub(1);
        const ndcY = float(gy).add(0.5).div(float(S)).mul(2).sub(1);
        const dir = normalize(u.rayForward.add(u.rayRight.mul(ndcX)).add(u.rayUp.mul(ndcY)));

        const r = u.cameraRadius;
        const p0 = vec3(0, r, 0);
        const atmosphereT = fnRayMarchLength({ r, mu: dir.y });

        const throughput = vec3(1, 1, 1).toVar();
        const L = vec3(0, 0, 0).toVar();
        const tPrev = float(0).toVar();

        Loop(S, ({ i: slice }) => {
          const w = float(slice).add(1).div(float(S));
          const tEnd = w.mul(w).mul(float(AERIAL_MAX_KM)).min(atmosphereT);
          const segment = max(tEnd.sub(tPrev), float(0));
          const dt = segment.div(float(AERIAL_SUBSTEPS));

          Loop(AERIAL_SUBSTEPS, ({ i: s }) => {
            const t = tPrev.add(float(s).add(0.5).mul(dt));
            const p = p0.add(dir.mul(t));
            const rr = max(length(p), float(PLANET_RADIUS_KM));
            const h = rr.sub(float(PLANET_RADIUS_KM));
            const up = p.div(rr);

            const ext = fnExtinction({ h, mieScale: u.mieScale });
            const rayleigh = fnRayleighScattering({ h });
            const mie = fnMieScattering({ h, mieScale: u.mieScale });
            const stepT = exp(ext.mul(dt).negate());
            const safeExt = max(ext, vec3(1e-9, 1e-9, 1e-9));

            const contribute = (lightDir: N, irradiance: N): void => {
              const muL = dot(up, lightDir);
              const lightT = this.sampleTransmittance(rr, muL);
              const blocked = fnRaySphere({
                ro: p,
                rd: lightDir,
                radius: float(PLANET_RADIUS_KM),
              });
              const visible = blocked.lessThan(float(0)).select(float(1), float(0));
              const cosT = dot(dir, lightDir);
              const single = rayleigh
                .mul(fnRayleighPhase({ cosTheta: cosT }))
                .add(vec3(mie, mie, mie).mul(fnMiePhase({ cosTheta: cosT })))
                .mul(lightT)
                .mul(visible);
              const multi = rayleigh
                .add(vec3(mie, mie, mie))
                .mul(this.sampleMultiScatter(rr, muL));
              const source = single.add(multi).mul(irradiance);
              L.addAssign(throughput.mul(source.sub(source.mul(stepT)).div(safeExt)));
            };

            contribute(u.sunDirection, u.sunIrradiance);
            contribute(u.moonDirection, u.moonIrradiance);
            throughput.mulAssign(stepT);
          });

          tPrev.assign(tEnd);
          // The alpha channel is the mean of the three transmittance channels.
          // Storing one number rather than three costs a little colour accuracy
          // in very hazy air and buys the whole rgb for the inscatter, which is
          // where the colour actually is. This is Hillaire's own packing.
          const meanT = throughput.x.add(throughput.y).add(throughput.z).div(3);
          textureStore(store, uvec3(gx, gy, uint(slice)), vec4(L.x, L.y, L.z, meanT));
        });
      });
    });

    const k = computeKernel(kernel(), [WG, WG, 1]);
    k.setName('atmosAerial');
    return k;
  }

  /**
   * Project the sky-view LUT onto SH L2.
   *
   * This is how CLAUDE.md rule 5 ("ambient comes from the atmosphere, never a
   * constant") is actually delivered: nine RGB coefficients go into a
   * `THREE.LightProbe` and every material's indirect diffuse comes from them.
   * The projection is over the *full* sphere, and the sky-view LUT already
   * contains ground-reflected radiance for directions below the horizon, so the
   * bounce off the terrain is in there too — which is what makes an overhang lit
   * from below by warm ground rather than by nothing.
   */
  private buildSH(): ComputeKernelNode {
    const out = this.shBuffer;
    const side = Math.round(Math.sqrt(SH_DIRECTIONS));

    const kernel = Fn(() => {
      const c = globalId.x;
      If(c.lessThan(uint(9)), () => {
        const acc = vec3(0, 0, 0).toVar();
        Loop(SH_DIRECTIONS, ({ i }) => {
          const iy = int(i).div(int(side));
          const ix = int(i).sub(iy.mul(int(side)));
          const dir = fnUniformSphere({
            u: float(ix).add(0.5).div(float(side)),
            v: float(iy).add(0.5).div(float(side)),
          });
          const radiance = this.sampleSkyView(dir);

          // Real SH basis, in three's ordering (see getShIrradianceAt.js).
          const x = dir.x;
          const y = dir.y;
          const z = dir.z;
          const basis = float(0).toVar();
          If(c.equal(uint(0)), () => {
            basis.assign(float(0.282095));
          })
            .ElseIf(c.equal(uint(1)), () => {
              basis.assign(y.mul(0.488603));
            })
            .ElseIf(c.equal(uint(2)), () => {
              basis.assign(z.mul(0.488603));
            })
            .ElseIf(c.equal(uint(3)), () => {
              basis.assign(x.mul(0.488603));
            })
            .ElseIf(c.equal(uint(4)), () => {
              basis.assign(x.mul(y).mul(1.092548));
            })
            .ElseIf(c.equal(uint(5)), () => {
              basis.assign(y.mul(z).mul(1.092548));
            })
            .ElseIf(c.equal(uint(6)), () => {
              basis.assign(z.mul(z).mul(3).sub(1).mul(0.315392));
            })
            .ElseIf(c.equal(uint(7)), () => {
              basis.assign(x.mul(z).mul(1.092548));
            })
            .Else(() => {
              basis.assign(x.mul(x).sub(y.mul(y)).mul(0.546274));
            });

          acc.addAssign(radiance.mul(basis));
        });

        // Monte-Carlo weight for uniform sphere sampling: 4π/N.
        const w = (4 * Math.PI) / SH_DIRECTIONS;
        const base = c.mul(uint(4));
        out.element(base).assign(acc.x.mul(w));
        out.element(base.add(uint(1))).assign(acc.y.mul(w));
        out.element(base.add(uint(2))).assign(acc.z.mul(w));
        out.element(base.add(uint(3))).assign(float(0));
      });
    });

    const k = computeKernel(kernel(), [SH_WORKGROUP, 1, 1]);
    k.setName('atmosSkySH');
    return k;
  }

  // -------------------------------------------------------------------------
  // Driving
  // -------------------------------------------------------------------------

  /** Bake the two atmosphere-only tables. Called once, and again if turbidity moves. */
  bakeStatic(): void {
    this.renderer.compute(this.kTransmittance, this.dTransmittance);
    this.renderer.compute(this.kMultiScatter, this.dMultiScatter);
    this.staticBaked = true;
    this.transmittanceRequested = false;
    this.transmittanceCPU = null;
  }

  get isBaked(): boolean {
    return this.staticBaked;
  }

  /** The two per-frame tables. */
  updateDynamic(frame: number): void {
    this.profiler.begin('atmos-luts');
    this.renderer.compute(this.kSkyView, this.dSkyView);
    this.renderer.compute(this.kAerial, this.dAerial);
    // The SH projection reads the sky-view LUT written a moment ago, so it must
    // follow it in the queue — which it does — but it does not have to run every
    // frame. Ambient from a 10-frame-old sky is a sixth of a second stale at
    // 60 Hz and the sun moves 0.004° in that time.
    if (frame % 10 === 0) this.renderer.compute(this.kSH, this.dSH);
    this.profiler.end('atmos-luts');

    this.requestTransmittanceReadback();
  }

  /**
   * Ask the GPU for the transmittance table, once.
   *
   * Deliberately not awaited anywhere: until it lands, `transmittanceAt`
   * returns 1 and the sun is un-attenuated, which for the two or three frames
   * involved is a slightly-too-bright first frame and nothing else.
   */
  private requestTransmittanceReadback(): void {
    if (this.transmittanceRequested || !this.staticBaked) return;
    this.transmittanceRequested = true;
    void this.renderer
      .getArrayBufferAsync(this.transmittanceBuffer.value as THREE.StorageInstancedBufferAttribute)
      .then((buffer) => {
        this.transmittanceCPU = new Float32Array(buffer);
      })
      .catch(() => {
        // Leaves `transmittanceAt` returning 1. Diagnosed by the sun being too
        // bright at sunset rather than by a silent wrong colour.
        console.warn('[aether] transmittance LUT readback failed; sun colour will not redden');
      });
  }

  /**
   * Read back the SH coefficients. Non-blocking; one request in flight.
   * Returns the last values that landed.
   */
  pollSphericalHarmonics(out: Float32Array): boolean {
    if (!this.shReadInFlight) {
      this.shReadInFlight = true;
      void this.renderer
        .getArrayBufferAsync(this.shBuffer.value as THREE.StorageInstancedBufferAttribute)
        .then((buffer) => {
          this.shCPU.set(new Float32Array(buffer));
        })
        .catch(() => {
          /* keep the previous coefficients */
        })
        .finally(() => {
          this.shReadInFlight = false;
        });
    }
    out.set(this.shCPU);
    return true;
  }

  /**
   * CPU-side bilinear fetch of the transmittance table, matching the GPU's
   * `aeTransmittanceUv` exactly. Writes into `out` (length ≥ 3).
   */
  transmittanceAt(radiusKm: number, mu: number, out: Float32Array): void {
    const data = this.transmittanceCPU;
    if (data === null) {
      out[0] = 1;
      out[1] = 1;
      out[2] = 1;
      return;
    }
    const Rg = PLANET_RADIUS_KM;
    const Rt = ATMOSPHERE_RADIUS_KM;
    const H = Math.sqrt(Rt * Rt - Rg * Rg);
    const r = Math.max(Rg, Math.min(Rt, radiusKm));
    // Same horizon guard as the GPU sampler: below it the table is undefined,
    // not small.
    const muHorizon = -Math.sqrt(Math.max(0, 1 - (Rg / r) ** 2));
    if (mu < muHorizon) {
      out[0] = 0;
      out[1] = 0;
      out[2] = 0;
      return;
    }
    const rho = Math.sqrt(Math.max(0, r * r - Rg * Rg));
    const disc = r * r * (mu * mu - 1) + Rt * Rt;
    const d = Math.max(0, -r * mu + Math.sqrt(Math.max(0, disc)));
    const dMin = Rt - r;
    const dMax = rho + H;
    const xMu = Math.min(1, Math.max(0, (d - dMin) / Math.max(1e-6, dMax - dMin)));
    const xR = Math.min(1, Math.max(0, rho / H));

    const W = LUT_SIZES.transmittanceW;
    const Hh = LUT_SIZES.transmittanceH;
    const fx = (0.5 / W + xMu * (1 - 1 / W)) * W - 0.5;
    const fy = (0.5 / Hh + xR * (1 - 1 / Hh)) * Hh - 0.5;
    const x0 = Math.min(W - 1, Math.max(0, Math.floor(fx)));
    const y0 = Math.min(Hh - 1, Math.max(0, Math.floor(fy)));
    const x1 = Math.min(W - 1, x0 + 1);
    const y1 = Math.min(Hh - 1, y0 + 1);
    const tx = Math.min(1, Math.max(0, fx - x0));
    const ty = Math.min(1, Math.max(0, fy - y0));

    for (let c = 0; c < 3; c++) {
      const a = data[(y0 * W + x0) * 4 + c];
      const b = data[(y0 * W + x1) * 4 + c];
      const cc = data[(y1 * W + x0) * 4 + c];
      const dd = data[(y1 * W + x1) * 4 + c];
      out[c] = (a * (1 - tx) + b * tx) * (1 - ty) + (cc * (1 - tx) + dd * tx) * ty;
    }
  }

  dispose(): void {
    this.kTransmittance.dispose();
    this.kMultiScatter.dispose();
    this.kSkyView.dispose();
    this.kAerial.dispose();
    this.kSH.dispose();
    this.transmittance.dispose();
    this.multiScatter.dispose();
    this.skyView.dispose();
    this.aerial.dispose();
  }
}

function makeStorage2D(w: number, h: number, name: string): THREE.StorageTexture {
  const tex = new THREE.StorageTexture(w, h);
  tex.format = THREE.RGBAFormat;
  tex.type = THREE.HalfFloatType;
  tex.name = name;
  tex.colorSpace = THREE.NoColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.mipmapsAutoUpdate = false;
  return tex;
}

export { AERIAL_MAX_KM, PLANET_RADIUS_KM, LUT_SIZES };
