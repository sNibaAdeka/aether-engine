/**
 * Phase 3's system: the atmosphere owns every light in the world.
 *
 * Before this, `DebugScene` held a hand-picked directional light and a tinted
 * hemisphere. Both are gone. What replaces them:
 *
 *   sun      `DirectionalLight`, colour and lux from the transmittance LUT at
 *            the camera's altitude and the real solar elevation
 *   moon     a second `DirectionalLight`, ~0.25 lx at full, phase-corrected by
 *            Meeus's photometric law
 *   ambient  a `LightProbe` whose nine SH coefficients are the sky-view LUT
 *            projected on the GPU — CLAUDE.md rule 5, discharged
 *   fog      `scene.fogNode` composites the aerial-perspective froxel volume
 *            into every material that is not the sky itself
 *
 * The consequence worth stating plainly: nothing in this file picks a colour.
 * The blue of a shadowed slope is the SH projection of a Rayleigh-scattered sky;
 * the warmth of a sunlit one is the same photons after 30 km of ozone and
 * aerosol. That is the difference the phase is for.
 */

import * as THREE from 'three/webgpu';
import {
  Fn,
  cameraPosition,
  float,
  length,
  mix,
  output,
  positionWorld,
  screenUV,
  uniform,
  vec4,
} from 'three/tsl';

import type { EngineContext, FrameContext, System } from '@/engine/types';
import { TimeOfDay, SOLAR_CONSTANT_LUX } from '@/world/TimeOfDay';
import { AtmosphereLUTs, PLANET_RADIUS_KM } from './AtmosphereLUTs';
import { turbidityToMieScale } from './shaders/atmosphere.wgsl';
import { Sky, createSkyUniforms } from './Sky';
import { Stars } from './Stars';
import { Exposure } from './Exposure';

/**
 * Solar spectrum as linear sRGB, normalised to unit luminance.
 *
 * The sun above the atmosphere is very slightly yellow of D65 (5778 K
 * blackbody against a 6500 K white point). Everything that makes a sunset
 * orange happens in the transmittance LUT, not here — if this triple were
 * tinted to taste the LUT's work would be double-counted and noon would be
 * yellow.
 */
const SUN_RGB = (() => {
  const r = 1.0;
  const g = 0.965;
  const b = 0.92;
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return new THREE.Vector3(r / lum, g / lum, b / lum);
})();

/** Lunar albedo is grey-brown; the disc reflects sunlight, so start from SUN_RGB. */
const MOON_RGB = new THREE.Vector3(SUN_RGB.x * 1.0, SUN_RGB.y * 0.98, SUN_RGB.z * 0.94);

/**
 * Solar elevation below which the star sprites are drawn.
 *
 * Not a physical fade — the sky's own radiance already buries the stars long
 * before this, and the sprites are additive so they simply stop being visible.
 * This is a fill-rate gate: 190 additive quads over a daylit sky is work for
 * nothing.
 */
const STAR_DRAW_BELOW_RAD = (-3 * Math.PI) / 180;

/** Rec.709 luminance of a linear colour. */
function luminance(c: THREE.Color): number {
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

/** Angular radius of the moon at mean distance, radians. */
const MOON_ANGULAR_RADIUS = (0.5181 * Math.PI) / 180 / 2;
const MOON_SOLID_ANGLE = 2 * Math.PI * (1 - Math.cos(MOON_ANGULAR_RADIUS));
/** Full-moon illuminance at mean distance, lux (spec 3.4 quotes 0.25). */
const FULL_MOON_LUX = 0.267;

export class AtmosphereSystem implements System {
  readonly name = 'atmosphere';

  readonly time = new TimeOfDay();
  luts?: AtmosphereLUTs;
  sky?: Sky;
  stars?: Stars;
  exposure?: Exposure;

  /**
   * Aerosol loading. 1.5 clear frosty · 2.5 typical · 4 haze · 8 pre-storm
   * (spec 3.1). Phase 8's weather drives this; until then it is a debug knob.
   */
  private turbidity = 2.4;
  private turbidityBaked = -1;

  private sun!: THREE.DirectionalLight;
  private moon!: THREE.DirectionalLight;
  private probe!: THREE.LightProbe;
  private group = new THREE.Group();

  private ctx!: EngineContext;
  private enabled = false;

  private readonly skyU = createSkyUniforms();
  /** 0 turns aerial perspective off, for A/B measurement and for screenshots. */
  readonly uAerial = uniform(1);

  // Per-frame scratch. Nothing here allocates.
  private _transmittance = new Float32Array(4);
  private _sh = new Float32Array(9 * 4);
  private _right = new THREE.Vector3();
  private _up = new THREE.Vector3();
  private _forward = new THREE.Vector3();
  private _mat3 = new THREE.Matrix3();
  private _mat3Inv = new THREE.Matrix3();

  /** Numbers for the F3 overlay. */
  readonly readout = {
    hours: 12,
    sunAltitudeDeg: 0,
    moonAltitudeDeg: 0,
    moonPhase: 0,
    sunLux: 0,
    moonLux: 0,
    ev100: 15,
    avgLuminance: 0,
    turbidity: 2.4,
  };

  async init(ctx: EngineContext): Promise<void> {
    this.ctx = ctx;
    this.group.name = 'atmosphere';
    ctx.scene.add(this.group);

    if (!ctx.hasCompute) {
      // WebGL2 fallback: no compute stage, so there are no LUTs and there is no
      // atmosphere. A single hard-coded key light and a grey fill, honestly
      // labelled.
      // TODO(quality): the WebGL2 tier gets nothing from phase 3. The LUT chain
      // could in principle be baked with render-to-texture passes, but the
      // froxel volume cannot without 3D textures, and aerial perspective is the
      // point of the phase.
      console.warn('[aether] no compute backend — atmosphere falls back to a fixed key light');
      const key = new THREE.DirectionalLight(0xfff0dc, 3.0);
      key.position.set(-2600, 2100, 1800);
      this.group.add(key, key.target);
      this.group.add(new THREE.HemisphereLight(0xa8bccc, 0x565048, 1.1));
      ctx.scene.background = new THREE.Color(0.42, 0.55, 0.72);
      return;
    }

    this.enabled = true;

    this.luts = new AtmosphereLUTs(ctx.renderer, ctx.resources, ctx.profiler);
    this.stars = new Stars(
      { resources: ctx.resources, renderer: ctx.renderer },
      this.luts.sampleTransmittance,
    );
    this.sky = new Sky(this.skyU, {
      sampleSkyView: this.luts.sampleSkyView,
      sampleTransmittance: this.luts.sampleTransmittance,
      milkyWay: this.stars.milkyWay,
    });
    this.exposure = new Exposure(ctx.renderer, ctx.resources, ctx.profiler);

    this.group.add(this.sky.mesh);
    this.group.add(this.stars.mesh);

    // Lights. Intensity is in lux and colour is a unit-luminance ratio, so the
    // product is the real irradiance on a surface facing the light.
    this.sun = new THREE.DirectionalLight(0xffffff, 0);
    this.sun.castShadow = false; // CSM is phase 4
    this.group.add(this.sun, this.sun.target);

    this.moon = new THREE.DirectionalLight(0xffffff, 0);
    this.moon.castShadow = false;
    this.group.add(this.moon, this.moon.target);

    this.probe = new THREE.LightProbe();
    this.probe.intensity = 1;
    this.group.add(this.probe);

    // Nothing clears to a colour any more: the sky pass covers whatever the
    // depth buffer leaves, and a clear colour would only show through if that
    // pass failed — in which case magenta is more useful than blue.
    ctx.scene.background = null;

    // --- aerial perspective ----------------------------------------------
    // Spec 3.2, and the reason this phase exists. Every material that is not
    // the sky or a star samples the froxel volume at its own screen position
    // and distance and composites `colour · transmittance + inscatter`.
    //
    // Delivered through `scene.fogNode` rather than by editing each material:
    // three's own `setupFog` runs it on the final lit colour of every material
    // with `fog === true`, which is exactly the hook wanted and means phases 6
    // and 7 get aerial perspective on water and foliage without knowing this
    // file exists.
    const luts = this.luts;
    const enable = this.uAerial;
    ctx.scene.fogNode = Fn(() => {
      const distKm = length(positionWorld.sub(cameraPosition)).mul(0.001);
      const ap = luts.sampleAerial(screenUV, distKm);
      // `enable` is an A/B switch, not a quality setting: it exists so the cost
      // of the effect can be measured by difference, and so a screenshot can
      // prove which artefacts belong to it. At 1 it is the identity.
      const t = mix(float(1), ap.w, enable);
      // `output` is three's accessor for the material's lit colour at this
      // point in the graph — three's own `Fog.js` reads the same property.
      return vec4(output.rgb.mul(t).add(ap.xyz.mul(enable)), output.a);
    })();

    // The two atmosphere-only tables, before the first frame.
    this.luts.bakeStatic();
    this.turbidityBaked = this.turbidity;
    this.stars.bake(ctx.renderer);

    this.applyTurbidity();
  }

  /** Turbidity, spec 3.1. Rebakes the static LUTs, which is why it is a setter. */
  setTurbidity(t: number): void {
    this.turbidity = Math.max(1, Math.min(12, t));
  }

  setTimeOfDay(hours: number): void {
    this.time.setHours(hours);
  }

  /** Named weather presets from spec 3.1; phase 8 replaces this with a real model. */
  setWeather(name: string): void {
    const table: Record<string, number> = {
      clear: 1.5,
      default: 2.4,
      haze: 4,
      storm: 8,
    };
    const t = table[name];
    if (t !== undefined) this.setTurbidity(t);
  }

  private applyTurbidity(): void {
    const scale = turbidityToMieScale(this.turbidity);
    if (this.luts) this.luts.u.mieScale.value = scale;
    this.readout.turbidity = this.turbidity;
  }

  update(dt: number, ctx: FrameContext): void {
    this.time.update(dt);
    if (!this.enabled || !this.luts || !this.exposure || !this.stars) return;

    const camera = this.ctx.camera as THREE.PerspectiveCamera;
    const u = this.luts.u;

    // Turbidity change means the atmosphere itself changed, so the two static
    // tables are stale. Rebaking is 1024 + 65k rays; it happens on a settings
    // change or a weather transition, not per frame.
    if (this.turbidity !== this.turbidityBaked) {
      this.applyTurbidity();
      this.luts.bakeStatic();
      this.turbidityBaked = this.turbidity;
    }

    // Radius of the camera in the atmosphere. Clamped just above the ground:
    // the LUT parameterisation divides by sqrt(r² − Rg²) and a camera exactly at
    // sea level makes that zero.
    const camY = Math.max(ctx.cameraPosition.y, 0.5);
    const radiusKm = PLANET_RADIUS_KM + camY * 0.001;
    u.cameraRadius.value = radiusKm;
    this.skyU.cameraRadius.value = radiusKm;
    this.stars.uCameraRadius.value = radiusKm;

    // --- sun and moon ------------------------------------------------------
    const sunDir = this.time.sun.direction;
    const moonDir = this.time.moon.direction;
    u.sunDirection.value.set(sunDir.x, sunDir.y, sunDir.z);
    u.moonDirection.value.set(moonDir.x, moonDir.y, moonDir.z);
    this.skyU.sunDirection.value.copy(u.sunDirection.value);
    this.skyU.moonDirection.value.copy(u.moonDirection.value);
    this.skyU.moonToSun.value.set(
      this.time.moonToSun.x,
      this.time.moonToSun.y,
      this.time.moonToSun.z,
    );

    u.sunIrradiance.value.copy(SUN_RGB).multiplyScalar(SOLAR_CONSTANT_LUX);
    this.skyU.sunIrradiance.value.copy(u.sunIrradiance.value);
    u.moonIrradiance.value.copy(MOON_RGB).multiplyScalar(this.time.moon.illuminanceLux);

    // Irradiance actually reaching the camera's altitude, from the same table
    // the GPU marches with. This is the only CPU consumer of the LUT and it is
    // why the table is mirrored into a storage buffer at bake time.
    this.applyLight(this.sun, sunDir, SUN_RGB, SOLAR_CONSTANT_LUX, radiusKm);
    this.applyLight(
      this.moon,
      moonDir,
      MOON_RGB,
      this.time.moon.illuminanceLux,
      radiusKm,
    );
    // Photometric illuminance, not the peak channel: `intensity` carries only
    // whichever component survived the atmosphere best, and at sunrise that is
    // red, which would read as four times the light there actually is.
    this.readout.sunLux = luminance(this.sun.color) * this.sun.intensity;
    this.readout.moonLux = luminance(this.moon.color) * this.moon.intensity;

    // Surface luminance of the lit lunar disc. Constant with phase — a crescent
    // is not dimmer per unit area than a full moon, it is simply less of it —
    // so the disc keeps one luminance and the phase shows as shape.
    this.skyU.moonLuminance.value = FULL_MOON_LUX / MOON_SOLID_ANGLE;

    // --- camera frustum basis for the froxel volume ------------------------
    camera.updateMatrixWorld();
    const m = camera.matrixWorld.elements;
    this._right.set(m[0], m[1], m[2]);
    this._up.set(m[4], m[5], m[6]);
    this._forward.set(-m[8], -m[9], -m[10]);
    const tanHalf = Math.tan((camera.fov * Math.PI) / 360);
    u.rayForward.value.copy(this._forward);
    u.rayRight.value.copy(this._right).multiplyScalar(tanHalf * camera.aspect);
    u.rayUp.value.copy(this._up).multiplyScalar(tanHalf);

    // Radians per pixel — antialiases the sun and moon discs and converts a
    // star's illuminance into the radiance of the pixels it lands on.
    const heightPx = Math.max(1, this.exposure.renderTarget.height);
    const pixelAngle = (2 * tanHalf) / heightPx;
    this.skyU.pixelAngle.value = pixelAngle;
    this.stars.uPixelSolidAngle.value = pixelAngle * pixelAngle;

    // --- celestial rotation ------------------------------------------------
    this._mat3.fromArray(this.time.celestialMatrix);
    this.stars.uCelestial.value.copy(this._mat3);
    this._mat3Inv.copy(this._mat3).transpose();
    this.skyU.worldToCelestial.value.copy(this._mat3Inv);
    this.stars.uTime.value = ctx.elapsed;
    this.stars.uVisibility.value = 1;
    this.stars.mesh.visible = this.time.sun.altitude < STAR_DRAW_BELOW_RAD;

    // --- LUTs --------------------------------------------------------------
    this.luts.updateDynamic(ctx.frame);

    // --- ambient -----------------------------------------------------------
    if (ctx.frame % 10 === 5) {
      this.luts.pollSphericalHarmonics(this._sh);
      const co = this.probe.sh.coefficients;
      for (let i = 0; i < 9; i++) {
        co[i].set(this._sh[i * 4], this._sh[i * 4 + 1], this._sh[i * 4 + 2]);
      }
    }

    if (ctx.frame % 15 === 0) this.exposure.poll();

    this.readout.hours = this.time.hours;
    this.readout.sunAltitudeDeg = (this.time.sun.altitude * 180) / Math.PI;
    this.readout.moonAltitudeDeg = (this.time.moon.altitude * 180) / Math.PI;
    this.readout.moonPhase = this.time.moonPhase;
    this.readout.ev100 = this.exposure.readout.ev100;
    this.readout.avgLuminance = this.exposure.readout.averageLuminance;
  }

  /**
   * Point a directional light along `dir` with the irradiance that survives the
   * atmosphere.
   *
   * three's directional light contributes `color × intensity` as irradiance on a
   * surface facing it, so the split between the two is arbitrary as long as the
   * product is right — colour carries the ratio, intensity carries the lux, and
   * the F3 readout is therefore in real units.
   */
  private applyLight(
    light: THREE.DirectionalLight,
    dir: { x: number; y: number; z: number },
    tint: THREE.Vector3,
    illuminanceLux: number,
    radiusKm: number,
  ): void {
    // Below the horizon the planet is in the way and the transmittance table's
    // parameterisation is not defined; both facts point the same direction.
    if (dir.y <= 0 || illuminanceLux <= 0) {
      light.intensity = 0;
      light.visible = false;
      return;
    }
    this.luts!.transmittanceAt(radiusKm, dir.y, this._transmittance);
    const r = tint.x * this._transmittance[0];
    const g = tint.y * this._transmittance[1];
    const b = tint.z * this._transmittance[2];
    const peak = Math.max(r, g, b, 1e-8);
    light.visible = true;
    light.color.setRGB(r / peak, g / peak, b / peak);
    light.intensity = peak * illuminanceLux;
    // A directional light in three points from `position` toward `target`;
    // `dir` points *at* the body, so the light sits along it.
    light.position.set(dir.x * 1000, dir.y * 1000, dir.z * 1000);
    light.target.position.set(0, 0, 0);
    light.target.updateMatrixWorld();
  }

  /** Renders the whole frame: HDR pass, metering, tone map. */
  renderFrame(dt: number): void {
    if (this.exposure) {
      this.exposure.render(this.ctx.scene, this.ctx.camera, dt);
    } else {
      this.ctx.renderer.render(this.ctx.scene, this.ctx.camera);
    }
  }

  onResize(width: number, height: number): void {
    this.exposure?.setSize(width, height);
  }

  dispose(): void {
    this.group.removeFromParent();
    this.sky?.dispose();
    this.stars?.dispose();
    this.exposure?.dispose();
    this.luts?.dispose();
  }
}
