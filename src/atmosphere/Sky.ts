/**
 * The sky pass: one full-screen triangle, depth-tested, drawn last among the
 * opaque geometry.
 *
 * Depth-tested and drawn *after* the terrain rather than before it as a
 * background, which is the point. The clip position is written with z = w, so
 * every sky fragment has depth exactly 1.0 and the LEqual test rejects it
 * wherever anything at all was drawn. On the mountain viewpoint that is 60 % of
 * the frame the sky shader never runs on. A `scene.background` would run on all
 * of it and then be painted over.
 *
 * What it composites, in order:
 *   1. the sky-view LUT — everything the atmosphere scatters, both lights
 *   2. the Milky Way, attenuated by the same atmosphere
 *   3. the moon's disc, with its phase and Lommel–Seeliger regolith
 *   4. the sun's disc, 0.53° across, with limb darkening
 *
 * The stars themselves are not here; they are 190 instanced sprites in
 * `Stars.ts`, for the reasons given in that file's header.
 *
 * Everything is emitted in cd/m². Nothing here knows about exposure — that is
 * `Exposure.ts`'s job and it happens after the whole frame is composited, which
 * is the only place a physical camera model can live.
 */

import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  acos,
  asin,
  atan,
  cameraProjectionMatrixInverse,
  cameraWorldMatrix,
  clamp,
  cross,
  dot,
  float,
  fract,
  max,
  normalize,
  positionGeometry,
  smoothstep,
  sqrt,
  texture,
  uniform,
  varying,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

import type { ShaderNode } from '@/terrain/shaders/debugViews';
import { fnLimbDarkening, fnLommelSeeliger } from './shaders/atmosphere.wgsl';

type N = ShaderNode;

/** Angular radius of the solar disc, radians (0.53° across, spec 3.3). */
const SUN_ANGULAR_RADIUS = (0.53 * Math.PI) / 180 / 2;
/** Solid angle of that disc, steradians. */
const SUN_SOLID_ANGLE = 2 * Math.PI * (1 - Math.cos(SUN_ANGULAR_RADIUS));
/**
 * Disc-average of the limb-darkening profile, 1 − u/3 − v/2.
 *
 * Dividing by it conserves flux: limb darkening redistributes the sun's light
 * from the edge toward the centre, it does not remove any, and without this
 * correction the sun would deliver 27 % less illuminance than the transmittance
 * LUT says it should.
 */
const LIMB_MEAN = 1 - 0.47 / 3 - 0.23 / 2;

/** Angular radius of the lunar disc at mean distance, radians. */
const MOON_ANGULAR_RADIUS = (0.5181 * Math.PI) / 180 / 2;

export interface SkyUniforms {
  sunDirection: ReturnType<typeof uniform>;
  moonDirection: ReturnType<typeof uniform>;
  /** Unit world direction from the moon toward the sun; orients the terminator. */
  moonToSun: ReturnType<typeof uniform>;
  /** Solar irradiance at the top of the atmosphere, lux, as RGB. */
  sunIrradiance: ReturnType<typeof uniform>;
  /** Surface luminance of the lit lunar disc, cd/m². */
  moonLuminance: ReturnType<typeof uniform>;
  cameraRadius: ReturnType<typeof uniform>;
  /** Radians subtended by one pixel — antialiases the two discs. */
  pixelAngle: ReturnType<typeof uniform>;
  /** Column-major 3×3, world → equatorial. Samples the Milky Way map. */
  worldToCelestial: ReturnType<typeof uniform>;
}

export interface SkyDeps {
  sampleSkyView: (dir: N) => N;
  sampleTransmittance: (r: N, mu: N) => N;
  milkyWay: THREE.Texture;
}

export function createSkyUniforms(): SkyUniforms {
  return {
    sunDirection: uniform(new THREE.Vector3(0, 1, 0)),
    moonDirection: uniform(new THREE.Vector3(0, -1, 0)),
    moonToSun: uniform(new THREE.Vector3(1, 0, 0)),
    sunIrradiance: uniform(new THREE.Vector3(128000, 128000, 128000)),
    moonLuminance: uniform(0),
    cameraRadius: uniform(6360),
    pixelAngle: uniform(0.0005),
    worldToCelestial: uniform(new THREE.Matrix3()),
  };
}

export class Sky {
  readonly mesh: THREE.Mesh;
  private material: THREE.MeshBasicNodeMaterial;
  private geometry: THREE.BufferGeometry;

  constructor(
    readonly u: SkyUniforms,
    deps: SkyDeps,
  ) {
    // One triangle that covers the viewport. Two triangles would put a seam
    // down the diagonal for the rasteriser to walk twice; one has no seam and
    // 33 % fewer quads on the edge.
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3),
    );
    this.geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e7);

    this.material = this.build(deps);

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'sky';
    this.mesh.frustumCulled = false;
    // Last of the opaque draws: by then the depth buffer holds the terrain and
    // the early-z test throws away most of the pass for free.
    this.mesh.renderOrder = 1000;
  }

  private build(deps: SkyDeps): THREE.MeshBasicNodeMaterial {
    const u = this.u;
    const mat = new THREE.MeshBasicNodeMaterial();
    mat.depthTest = true;
    mat.depthWrite = false;
    mat.side = THREE.DoubleSide;
    // The sky is not behind the atmosphere, it *is* the atmosphere: applying
    // the aerial-perspective fog node to it would count the same scattering
    // twice and wash the whole dome toward grey.
    mat.fog = false;
    mat.toneMapped = false;

    // z = w puts the fragment exactly on the far plane. Depth 1.0 against a
    // cleared 1.0 passes under LEqual, and anything with real depth wins.
    mat.vertexNode = vec4(positionGeometry.x, positionGeometry.y, 1, 1);

    // View ray from the inverse projection. Linear in NDC for a perspective
    // matrix, so interpolating it across the triangle is exact rather than
    // approximate — which is why this is a varying and not recomputed per pixel.
    const ndc = vec4(positionGeometry.x, positionGeometry.y, 1, 1);
    const viewH = cameraProjectionMatrixInverse.mul(ndc);
    const viewDir = viewH.xyz.div(viewH.w);
    const worldDir = cameraWorldMatrix.mul(vec4(viewDir, 0)).xyz;
    const vDir = varying(worldDir);

    mat.colorNode = Fn(() => {
      const dir = normalize(vDir);
      const radiance = deps.sampleSkyView(dir).toVar();

      // --- Milky Way ------------------------------------------------------
      // Sampled in equatorial coordinates, so it rotates with the sky rather
      // than with the camera. Attenuated by the same transmittance the stars
      // use: near the horizon the band genuinely fades out.
      const eq = normalize(u.worldToCelestial.mul(dir));
      const ra = fract(atan(eq.y, eq.x).div(Math.PI * 2).add(1));
      const dec = asin(clamp(eq.z, float(-1), float(1))).div(Math.PI).add(0.5);
      const milky = texture(deps.milkyWay, vec2(ra, dec)).level(float(0)).xyz;
      const spaceT = deps.sampleTransmittance(u.cameraRadius, dir.y);
      radiance.addAssign(milky.mul(spaceT));

      // --- moon -----------------------------------------------------------
      const cosMoon = dot(dir, u.moonDirection);
      If(cosMoon.greaterThan(float(Math.cos(MOON_ANGULAR_RADIUS * 1.6))), () => {
        const angle = acos(clamp(cosMoon, float(-1), float(1)));
        const rFrac = angle.div(float(MOON_ANGULAR_RADIUS));
        // A tangent frame on the disc. `cross` with world up is degenerate only
        // when the moon is exactly at the zenith, which the fallback covers.
        const upRef = vec3(0, 1, 0);
        const t0 = cross(upRef, u.moonDirection);
        const tangent = normalize(
          dot(t0, t0).lessThan(float(1e-6)).select(vec3(1, 0, 0), t0),
        );
        const bitangent = cross(u.moonDirection, tangent);
        // Position on the disc in units of the lunar radius.
        const px = dot(dir, tangent).div(float(MOON_ANGULAR_RADIUS));
        const py = dot(dir, bitangent).div(float(MOON_ANGULAR_RADIUS));
        const z = sqrt(max(float(1).sub(px.mul(px)).sub(py.mul(py)), float(0)));
        // Surface normal of the near hemisphere: it points back at the viewer.
        const n = normalize(
          tangent.mul(px).add(bitangent.mul(py)).sub(u.moonDirection.mul(z)),
        );
        const nDotL = dot(n, u.moonToSun);
        const nDotV = z;
        // ×2 because Lommel–Seeliger is exactly ½ everywhere at full phase, and
        // a full moon should read as a uniform disc of the stated luminance.
        const lit = fnLommelSeeliger({ nDotL, nDotV }).mul(2);
        // Earthshine: the dark limb is genuinely visible, at roughly 1 % of the
        // lit side and distinctly blue, because it is Earth's own albedo.
        const ashen = vec3(0.7, 0.82, 1.0).mul(0.012);
        const surface = vec3(lit, lit, lit).mul(vec3(1.0, 0.97, 0.91)).add(ashen);
        const edge = float(1).sub(
          smoothstep(
            float(1).sub(u.pixelAngle.div(float(MOON_ANGULAR_RADIUS))),
            float(1),
            rFrac,
          ),
        );
        radiance.addAssign(surface.mul(u.moonLuminance).mul(edge).mul(spaceT));
      });

      // --- sun ------------------------------------------------------------
      const cosSun = dot(dir, u.sunDirection);
      If(cosSun.greaterThan(float(Math.cos(SUN_ANGULAR_RADIUS * 1.5))), () => {
        const angle = acos(clamp(cosSun, float(-1), float(1)));
        const rFrac = angle.div(float(SUN_ANGULAR_RADIUS));
        const limb = fnLimbDarkening({ centreDistance: rFrac }).div(LIMB_MEAN);
        const edge = float(1).sub(
          smoothstep(
            float(1).sub(u.pixelAngle.div(float(SUN_ANGULAR_RADIUS))),
            float(1),
            rFrac,
          ),
        );
        const sunT = deps.sampleTransmittance(u.cameraRadius, u.sunDirection.y);
        radiance.addAssign(
          u.sunIrradiance.div(float(SUN_SOLID_ANGLE)).mul(sunT).mul(limb).mul(edge),
        );
      });

      return radiance;
    })();

    return mat;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}
