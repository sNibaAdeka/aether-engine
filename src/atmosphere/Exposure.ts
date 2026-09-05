/**
 * The camera: a physical one, with an aperture, a shutter and a film speed.
 *
 * Everything upstream of this file is in absolute photometric units — the sky is
 * 8000 cd/m² at noon and 0.0002 cd/m² on a moonless night, a factor of forty
 * million. No display can show that, and no fixed multiplier can map it, so
 * something has to decide how much of that range to put on screen. That
 * something is a camera, and the reason to model a real one is that its
 * behaviour is the behaviour the eye expects: the sun blows out, the shadows
 * stay coloured, and walking out of shade takes a second to recover from.
 *
 * ## The chain
 *
 *   scene → HDR render target (rgba16f, linear, cd/m²)
 *         → histogram compute (256 log bins)
 *         → resolve compute (percentile mean → EV100 → adapted exposure)
 *         → tone-map quad (exposure · Purkinje · AgX) → canvas
 *
 * The adapted exposure never leaves the GPU. It is written into a storage
 * buffer by the resolve pass and read by the tone-map quad in the same submit,
 * so there is no readback stall and no one-frame lag. The CPU reads the same
 * buffer asynchronously every so often, purely so the F3 overlay can print it.
 *
 * ## EV100 and the 1.2 constant
 *
 *   EV100 = log2(N²/t) − log2(ISO/100)
 *   L_max = 1.2 · 2^EV100          (the "saturation based" speed constant)
 *   exposure = 1 / L_max
 *
 * and for auto-exposure, running the standard reflected-light meter equation
 * backwards with K = 12.5 (Canon/Nikon's value):
 *
 *   EV100 = log2(L · 100 / K) = log2(L · 8)
 *
 * A sunlit scene at 100 000 lx has an average luminance around 4 000 cd/m²,
 * which gives EV100 ≈ 15 — which is what a real light meter reads outdoors on a
 * bright day, and it is reassuring that it falls out of the physics rather than
 * out of a tuning session.
 */

import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  atomicAdd,
  atomicLoad,
  atomicStore,
  computeKernel,
  clamp,
  dot,
  exp2,
  float,
  globalId,
  instancedArray,
  int,
  log2,
  max,
  min,
  mix,
  screenUV,
  smoothstep,
  texture,
  toneMapping,
  uint,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

import type { Profiler } from '@/engine/Profiler';
import type { ResourceManager } from '@/engine/ResourceManager';

/** Histogram bins, spec 3.4. */
const BINS = 256;
/**
 * Luminance range the histogram covers, log2 cd/m².
 *
 * −12 is 0.00024 cd/m², a moonless zenith; +20 is 10⁶ cd/m², which is a hundred
 * times a sunlit snowfield and a thousand times below the solar disc itself. 32
 * stops over 256 bins is 0.125 stops per bin — finer than the 1/3-stop
 * granularity any real meter reports.
 */
const LOG_MIN = -12;
const LOG_MAX = 20;

/** Sampling grid for the histogram. Fixed, so its cost does not scale with resolution. */
const SAMPLES_X = 256;
const SAMPLES_Y = 144;
const WG = 8;

/** Percentile band the meter averages over (spec 3.4: 40–95). */
const PCT_LOW = 0.4;
const PCT_HIGH = 0.95;

/**
 * Stops of under-exposure applied at full darkness. See `buildResolve`.
 */
const NIGHT_STOPS = 3.0;

/** Adaptation time constants, seconds (spec 3.4). */
const TAU_BRIGHTEN = 0.8;
const TAU_DARKEN = 2.5;

export interface PhysicalCamera {
  /** f-number. */
  aperture: number;
  /** Shutter time, seconds. */
  shutter: number;
  /** Film speed. */
  iso: number;
}

export class Exposure {
  /** Linear HDR target the whole scene is drawn into, in cd/m². */
  renderTarget: THREE.RenderTarget;

  readonly camera: PhysicalCamera = { aperture: 8, shutter: 1 / 125, iso: 100 };
  /** When false, `camera` is used directly and the histogram only reports. */
  auto = true;

  /**
   * Hard rails on the adapted EV100, so a black or a blown frame cannot walk the
   * exposure off to infinity in either direction.
   *
   * These are a safety net, not the mechanism: the `NIGHT_STOPS` curve in
   * `buildResolve` is what actually keeps a night dark, and with it in place the
   * target lands around EV −6 on a moonless sky, comfortably inside the floor.
   *
   * TODO(quality): the exact numbers interact with AgX's shadow lift, which is
   * generous by design, so they would move if phase 10 changes the curve. The
   * principled version is a scotopic response applied before the tone map rather
   * than a bias on the meter.
   */
  minEV = -8;
  maxEV = 17;
  /** Metering bias in stops, the "exposure compensation" dial. */
  exposureBias = -0.4;

  /** Last values read back for the overlay. Not used for rendering. */
  readonly readout = { ev100: 15, averageLuminance: 4000, exposure: 1 / 40000 };

  private histogram = instancedArray(BINS, 'uint').toAtomic();
  /** [0] EV100, [1] average luminance, [2] exposure multiplier, [3] initialised flag. */
  private state = instancedArray(4, 'float');

  private uDt = uniform(1 / 60);
  private uManualEV = uniform(15);
  private uAuto = uniform(1);
  private uMinEV = uniform(-8);
  private uMaxEV = uniform(17);
  private uBias = uniform(0);

  private kClear: ReturnType<typeof computeKernel>;
  private kBuild: ReturnType<typeof computeKernel>;
  private kResolve: ReturnType<typeof computeKernel>;

  private quad: THREE.QuadMesh;
  private quadMaterial: THREE.NodeMaterial;

  private readonly dClear = [BINS / 64, 1, 1];
  private readonly dBuild = [SAMPLES_X / WG, SAMPLES_Y / WG, 1];
  private readonly dResolve = [1, 1, 1];

  private readInFlight = false;
  private width = 1;
  private height = 1;

  constructor(
    private renderer: THREE.WebGPURenderer,
    private resources: ResourceManager,
    private profiler: Profiler,
  ) {
    this.renderTarget = new THREE.RenderTarget(1, 1, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      depthBuffer: true,
      stencilBuffer: false,
      generateMipmaps: false,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
    });
    this.renderTarget.texture.name = 'hdr-scene';
    // Explicitly the working space: three only inserts its own colour-space
    // conversion pass when the target's space differs, and an extra full-screen
    // copy per frame for nothing is exactly the sort of thing that hides inside
    // a renderer.
    this.renderTarget.texture.colorSpace = THREE.LinearSRGBColorSpace;

    this.kClear = this.buildClear();
    this.kBuild = this.buildHistogram();
    this.kResolve = this.buildResolve();

    this.quadMaterial = this.buildToneMap();
    this.quad = new THREE.QuadMesh(this.quadMaterial);
  }

  setSize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    this.renderTarget.setSize(this.width, this.height);
    this.resources.track(
      this.renderTarget.texture,
      'post/hdr-scene',
      this.width * this.height * 8,
    );
  }

  /** EV100 of the manual camera settings. */
  manualEV100(): number {
    const { aperture, shutter, iso } = this.camera;
    return Math.log2((aperture * aperture) / shutter) - Math.log2(iso / 100);
  }

  private buildClear(): ReturnType<typeof computeKernel> {
    const hist = this.histogram;
    const k = computeKernel(
      Fn(() => {
        const i = globalId.x;
        If(i.lessThan(uint(BINS)), () => {
          atomicStore(hist.element(i), uint(0));
        });
      })(),
      [64, 1, 1],
    );
    k.setName('exposureHistClear');
    return k;
  }

  /**
   * One atomic increment per sample.
   *
   * The samples are on a fixed 256 × 144 lattice over the frame rather than one
   * per pixel: 37k samples is far past what a 256-bin histogram needs to be
   * stable (each bin holds ~140 samples in a typical scene), and it makes the
   * pass cost the same at 720p and at 4K. A per-pixel histogram at 4K is 8.3 M
   * atomics into 256 addresses, which is 32 000-way contention on every bin.
   */
  private buildHistogram(): ReturnType<typeof computeKernel> {
    const hist = this.histogram;
    const src = this.renderTarget.texture;

    const k = computeKernel(
      Fn(() => {
        const gx = globalId.x;
        const gy = globalId.y;
        If(gx.lessThan(uint(SAMPLES_X)).and(gy.lessThan(uint(SAMPLES_Y))), () => {
          const uv = vec2(
            float(gx).add(0.5).div(float(SAMPLES_X)),
            float(gy).add(0.5).div(float(SAMPLES_Y)),
          );
          const c = texture(src, uv).level(float(0)).xyz;
          const lum = max(dot(c, vec3(0.2126, 0.7152, 0.0722)), float(1e-6));
          const t = log2(lum).sub(float(LOG_MIN)).div(float(LOG_MAX - LOG_MIN));
          const bin = uint(clamp(t, float(0), float(1)).mul(float(BINS - 1)).add(0.5));
          atomicAdd(hist.element(bin), uint(1));
        });
      })(),
      [WG, WG, 1],
    );
    k.setName('exposureHistBuild');
    return k;
  }

  /**
   * One invocation walks the 256 bins, takes the log-average over the 40–95
   * percentile band, and integrates the adaptation.
   *
   * Single-threaded on purpose: 256 iterations of add-and-compare is a few
   * microseconds, and a parallel prefix sum over 256 elements would need shared
   * memory and two barriers to save nothing measurable.
   *
   * The percentile band is what makes the meter robust. The bottom 40 % is
   * usually deep shadow or, at night, empty sky; the top 5 % is the sun, a
   * specular highlight or a lamp. Averaging the whole histogram lets a single
   * bright object shut the aperture down on the entire frame — which is the
   * classic auto-exposure failure of standing in a doorway.
   */
  private buildResolve(): ReturnType<typeof computeKernel> {
    const hist = this.histogram;
    const st = this.state;

    const k = computeKernel(
      Fn(() => {
        If(globalId.x.equal(uint(0)), () => {
          const total = float(0).toVar();
          Loop(BINS, ({ i }) => {
            total.addAssign(float(atomicLoad(hist.element(int(i)))));
          });

          const loCount = total.mul(PCT_LOW);
          const hiCount = total.mul(PCT_HIGH);
          const running = float(0).toVar();
          const weight = float(0).toVar();
          const logSum = float(0).toVar();

          Loop(BINS, ({ i }) => {
            const count = float(atomicLoad(hist.element(int(i))));
            const before = running.toVar();
            running.addAssign(count);
            // Fraction of this bin that lies inside the band. Doing it
            // continuously rather than with a bin-level cutoff stops the meter
            // stepping when a bin crosses the 40 % line.
            const lo = max(before, loCount);
            const hi = min(running, hiCount);
            const inside = max(hi.sub(lo), float(0));
            const logLum = float(LOG_MIN).add(
              float(i).div(float(BINS - 1)).mul(float(LOG_MAX - LOG_MIN)),
            );
            weight.addAssign(inside);
            logSum.addAssign(logLum.mul(inside));
          });

          const meanLog = logSum.div(max(weight, float(1e-4)));
          const avgLum = exp2(meanLog);

          // Reflected-light meter, K = 12.5.
          const meteredEV = log2(max(avgLum, float(1e-6)).mul(8));

          // Night compensation — the single most important line for making a
          // night look like a night.
          //
          // A reflected-light meter renders whatever it sees as middle grey by
          // construction, so a moonlit sky at 0.003 cd/m² comes out at exactly
          // the same screen value as a noon meadow at 4000. That is what
          // "ночь как синий день" actually *is*: not a colour mistake, an
          // exposure one. The eye does the opposite — below the cone threshold
          // perceived brightness collapses — so the meter is walked back by up
          // to three stops as the metered level falls, over the range where
          // vision goes mesopic. Above EV 6 (deep dusk) the term is zero and
          // daylight metering is untouched.
          const nightBias = smoothstep(float(6), float(-4), meteredEV).mul(NIGHT_STOPS);
          const targetEV = clamp(
            this.uAuto
              .greaterThan(float(0.5))
              .select(meteredEV.add(this.uBias).add(nightBias), this.uManualEV),
            this.uMinEV,
            this.uMaxEV,
          );

          const prev = st.element(int(0));
          const initialised = st.element(int(3));
          // Different time constants each way — the whole point of spec 3.4's
          // "медленнее на потемнение". Stepping out of forest shade raises the
          // target EV, and the fast constant makes that take about a second,
          // during which the frame is genuinely over-exposed.
          const tau = targetEV
            .greaterThan(prev)
            .select(float(TAU_BRIGHTEN), float(TAU_DARKEN));
          const k2 = float(1).sub(exp2(this.uDt.div(tau).mul(-1.4427)));
          const adapted = initialised
            .lessThan(float(0.5))
            .select(targetEV, prev.add(targetEV.sub(prev).mul(k2)));

          st.element(int(0)).assign(adapted);
          st.element(int(1)).assign(avgLum);
          // L_max = 1.2 · 2^EV100; exposure is its reciprocal.
          st.element(int(2)).assign(float(1).div(exp2(adapted).mul(1.2)));
          st.element(int(3)).assign(float(1));
        });
      })(),
      [1, 1, 1],
    );
    k.setName('exposureResolve');
    return k;
  }

  /**
   * The tone-map quad: exposure, the Purkinje shift, AgX, and out.
   *
   * Output is linear; three's own output pass does the sRGB encode. That costs
   * one more full-screen copy than strictly necessary.
   *
   * TODO(quality): fold the colour-space encode into this quad and skip three's
   * output pass. It needs `renderer.outputColorSpace` set to linear so the
   * backend stops inserting its own, which is a global change worth making once
   * phase 10 owns the post chain rather than twice.
   */
  private buildToneMap(): THREE.NodeMaterial {
    const mat = new THREE.NodeMaterial();
    mat.depthTest = false;
    mat.depthWrite = false;
    const st = this.state;
    const src = this.renderTarget.texture;

    mat.fragmentNode = Fn(() => {
      const hdr = texture(src, screenUV).level(float(0));
      const exposure = st.element(int(2));
      const adaptedLum = st.element(int(1));

      // --- Purkinje shift --------------------------------------------------
      // Below about 3 cd/m² the cones stop contributing and vision moves to the
      // rods, which have no colour discrimination and peak at 507 nm — so a
      // moonlit landscape is desaturated and biased blue-green, and it is
      // *dark*, not "blue daylight". Skipping this is the single reason most
      // rendered nights look like an underexposed afternoon.
      //
      // Rod response uses the scotopic luminous efficiency fitted to linear
      // sRGB; it is negative in red, which is correct — deep red is invisible
      // to rods, which is why a red torch preserves dark adaptation.
      const c = hdr.xyz;
      // Weights are non-negative on purpose. The textbook scotopic fit has a
      // negative red coefficient — rods really are blind to deep red — but the
      // shift runs on the *composited* frame, so a red star sitting on top of a
      // faintly lit sky produced a rod response *lower* than the sky around it
      // and rendered as a black dot. Physically defensible, visibly a bug. These
      // weights keep red nearly ignored (2 %) without ever subtracting.
      const rod = dot(c, vec3(0.02, 0.72, 0.42));
      // Two gates, and they do different jobs. The first is per pixel: a bright
      // moon in a dark frame should stay a bright white moon. The second is the
      // eye's own adaptation state, so shadows on a sunny day are not treated as
      // scotopic just because they are dark.
      const pixelGate = smoothstep(float(3.0), float(0.01), dot(c, vec3(0.2126, 0.7152, 0.0722)));
      const eyeGate = smoothstep(float(1.0), float(0.02), adaptedLum);
      // 1.03 is not a taste knob: it is the factor that makes the rod response
      // come back out at the same Rec.709 luminance it went in with, so the
      // shift changes hue and saturation without also darkening the frame. At
      // the 0.42 first tried, night lost 2.5× of its brightness to what was
      // supposed to be a colour transform.
      const rodColour = vec3(0.63, 0.86, 1.28).mul(rod.mul(1.03));
      const shifted = mix(c, rodColour, pixelGate.mul(eyeGate));

      // AgX: a filmic curve with a real desaturation shoulder, so a 10⁶ cd/m²
      // sun rolls to white instead of clipping to magenta the way a naive
      // per-channel curve does.
      //
      // TODO(quality): phase 10 owns the grade. AgX with a −0.4 stop bias is a
      // neutral starting point, not a look.
      // `toneMapping()` returns a vec4 — it carries the input's alpha through —
      // so the rgb is taken explicitly rather than relying on a conversion.
      const mapped = toneMapping(THREE.AgXToneMapping, exposure, shifted);

      // Triangular-PDF dither, ±1 LSB of an 8-bit channel.
      //
      // The sky is a smooth gradient over the whole frame and the canvas is 8
      // bits, so without this every atmosphere is banded — and at night, where
      // the whole image lives in the bottom two stops, the bands are wide enough
      // to count. Two hashes subtracted give a triangular distribution, which
      // removes the quantisation *correlation* rather than just adding noise on
      // top of it.
      const seed = screenUV.mul(vec2(1731.51, 941.23));
      const n1 = seed.x.add(seed.y).sin().mul(43758.5453).fract();
      const n2 = seed.x.sub(seed.y).sin().mul(28001.8384).fract();
      const dither = n1.sub(n2).mul(1 / 255);
      return vec4(mapped.rgb.add(vec3(dither, dither, dither)), 1);
    })();

    return mat;
  }

  /**
   * Draw the scene into the HDR target, meter it, and present it.
   * Replaces the bare `renderer.render(scene, camera)` of phases 0–2.
   */
  render(scene: THREE.Scene, camera: THREE.Camera, dt: number): void {
    this.uDt.value = Math.min(dt, 0.25);
    this.uAuto.value = this.auto ? 1 : 0;
    this.uManualEV.value = this.manualEV100();
    this.uMinEV.value = this.minEV;
    this.uMaxEV.value = this.maxEV;
    this.uBias.value = this.exposureBias;

    this.profiler.begin('scene-hdr');
    this.renderer.setRenderTarget(this.renderTarget);
    this.renderer.render(scene, camera);
    this.renderer.setRenderTarget(null);
    this.profiler.end('scene-hdr');

    this.profiler.begin('exposure');
    this.renderer.compute(this.kClear, this.dClear);
    this.renderer.compute(this.kBuild, this.dBuild);
    this.renderer.compute(this.kResolve, this.dResolve);
    this.profiler.end('exposure');

    this.profiler.begin('tonemap');
    this.quad.render(this.renderer);
    this.profiler.end('tonemap');
  }

  /** Non-blocking readback, for the F3 overlay only. */
  poll(): void {
    if (this.readInFlight) return;
    this.readInFlight = true;
    void this.renderer
      .getArrayBufferAsync(this.state.value as THREE.StorageInstancedBufferAttribute)
      .then((buf) => {
        const f = new Float32Array(buf);
        this.readout.ev100 = f[0];
        this.readout.averageLuminance = f[1];
        this.readout.exposure = f[2];
      })
      .catch(() => {
        /* the overlay keeps the previous numbers */
      })
      .finally(() => {
        this.readInFlight = false;
      });
  }

  dispose(): void {
    this.kClear.dispose();
    this.kBuild.dispose();
    this.kResolve.dispose();
    this.quadMaterial.dispose();
    this.renderTarget.dispose();
  }
}
