/**
 * TSL colour nodes for the terrain debug visualisations (keys 1–9).
 *
 * Every mode is designed to make one specific class of bug *obvious* rather
 * than merely visible:
 *  - height   banded, so a discontinuity between LOD levels shows as a broken band
 *  - normals  raw RGB, so a flipped or unnormalised normal is unmistakable
 *  - slope    with the 30° and 38° biome thresholds drawn as lines
 *  - lod      distinct hue per level, so level boundaries and their stability read at a glance
 *  - morph    black→white, so a patch stuck mid-morph or morphing the wrong way stands out
 *  - tiles    patch outline plus the heightmap texel grid, to catch seams and off-by-one sampling
 *  - checker  1 m and 10 m world-space checker, the fastest scale sanity check there is
 */

import {
  Fn,
  If,
  vec3,
  float,
  abs,
  fract,
  floor,
  step,
  clamp,
  smoothstep,
  mix,
  max,
  min,
  acos,
  normalize,
} from 'three/tsl';
import { BIOME_DEBUG_COLORS } from './splat';

/**
 * A node of any shader type.
 *
 * three r185 ships no public alias for this. `ShaderNodeObject` — which this
 * file used to import — does not exist in `@types/three@0.185`, so the import
 * was a hard compile error; and bare `Node` resolves to `Node<unknown>`, which
 * drops every swizzle and operator used below. The slots here are genuinely
 * heterogeneous (float, vec2 and vec3 all pass through the same fields), so
 * there is no one concrete node type to name. This is the untyped-external-API
 * boundary the project's no-`any` rule carves out.
 */
export type ShaderNode = any;

type N = ShaderNode;

export interface DebugViewInputs {
  /** Integer mode uniform, 0 = off. */
  mode: N;
  /** Shaded colour used when mode is 0. */
  shaded: N;
  worldPos: N;
  normal: N;
  /** CDLOD level of this patch. */
  level: N;
  /** Morph coefficient, 0..1. */
  morphK: N;
  /** Local position within the patch, 0..1 on both axes. */
  patchUV: N;
  /**
   * Position in *texels of the tile actually being sampled*, not a synthetic
   * grid. Integer values are real texel boundaries, so mode 6 shows where the
   * baked lattice sits relative to the patch — the fastest way to catch an
   * off-by-one in the uv mapping or a node sampling the wrong sub-rectangle.
   */
  texelCoord: N;
  /** 1 when this node is drawing off a coarser ancestor's tile. */
  tileFallback: N;
  /** Accumulated water flow, 0..1. Zero until the erosion pass runs. */
  flow: N;
  /** Accumulated moisture, 0..1. */
  moisture: N;
  /**
   * The eight biome weights after snow has taken its share. Undefined on the
   * WebGL2 fallback, where nothing classifies anything — modes 10 and 11 then
   * fall through to the shaded view rather than reading a null node.
   */
  weights?: N[];
  /** Index of the dominant biome, 0..7. */
  dominant: N;
  /** Snow coverage, 0..1. */
  snow: N;
}

/** Perceptually ordered ramp: dark blue → cyan → green → yellow → white. */
const ramp = Fn(([t]: [N]) => {
  const x = clamp(t, float(0), float(1));
  const a = vec3(0.05, 0.05, 0.25);
  const b = vec3(0.1, 0.55, 0.75);
  const c = vec3(0.25, 0.7, 0.3);
  const d = vec3(0.95, 0.85, 0.3);
  const e = vec3(1.0, 1.0, 1.0);
  const s1 = smoothstep(0.0, 0.25, x);
  const s2 = smoothstep(0.25, 0.5, x);
  const s3 = smoothstep(0.5, 0.75, x);
  const s4 = smoothstep(0.75, 1.0, x);
  return mix(mix(mix(mix(a, b, s1), c, s2), d, s3), e, s4);
});

/** Thin antialiased line wherever `v` crosses a multiple of `period`. */
const gridLine = Fn(([v, period, width]: [N, N, N]) => {
  const f = abs(fract(v.div(period).add(0.5)).sub(0.5)).mul(period);
  return float(1).sub(smoothstep(float(0), width, f));
});

/** Weighted sum of the biome key colours. Grey when there is nothing to show. */
function biomeMix(weights: N[] | undefined): N {
  if (!weights) return vec3(0.35, 0.35, 0.38);
  let out = vec3(0, 0, 0);
  for (let k = 0; k < BIOME_DEBUG_COLORS.length && k < weights.length; k++) {
    const c = BIOME_DEBUG_COLORS[k];
    out = out.add(vec3(c[0], c[1], c[2]).mul(weights[k]));
  }
  return out;
}

/** Key colour of one biome index, selected without a branch. */
function biomeKey(index: N): N {
  let out = vec3(0, 0, 0);
  for (let k = 0; k < BIOME_DEBUG_COLORS.length; k++) {
    const c = BIOME_DEBUG_COLORS[k];
    // A ±0.5 window around the integer: exact equality on a float that came
    // through a varying is a coin flip.
    const hit = step(float(k - 0.5), index).mul(step(index, float(k + 0.5)));
    out = out.add(vec3(c[0], c[1], c[2]).mul(hit));
  }
  return out;
}

export function buildDebugColor(i: DebugViewInputs): N {
  return Fn(() => {
    const out = vec3(0).toVar();
    const n = normalize(i.normal);

    // Slope in radians from horizontal.
    const slope = acos(clamp(n.y, float(-1), float(1)));

    If(i.mode.equal(1), () => {
      // Height: continuous ramp over the world's vertical range, overlaid with
      // 50 m contour bands. A LOD seam shows up as a kinked contour long before
      // it is visible in shading.
      const t = clamp(i.worldPos.y.add(420).div(1400), float(0), float(1));
      const band = gridLine(i.worldPos.y, float(50), float(1.2));
      const sea = smoothstep(float(1.5), float(0), abs(i.worldPos.y));
      out.assign(mix(ramp(t), vec3(0.1, 0.1, 0.12), band.mul(0.5)));
      out.assign(mix(out, vec3(0.9, 0.2, 0.2), sea));
    })
      .ElseIf(i.mode.equal(2), () => {
        // Normals: raw, remapped to 0..1. Any component pinned at 0.5 across a
        // whole patch means the normal is being lost somewhere.
        out.assign(n.mul(0.5).add(0.5));
      })
      .ElseIf(i.mode.equal(3), () => {
        // Slope with the biome thresholds drawn in: 30° (scree) and 38° (rock).
        const deg = slope.mul(57.29577951308232);
        out.assign(ramp(clamp(deg.div(70), float(0), float(1))));
        const l30 = gridLine(deg.sub(30), float(1000), float(0.6));
        const l38 = gridLine(deg.sub(38), float(1000), float(0.6));
        out.assign(mix(out, vec3(1, 1, 1), max(l30, l38).mul(0.85)));
      })
      .ElseIf(i.mode.equal(4), () => {
        // LOD level: a distinct hue per level, stable across frames so a
        // flickering patch is immediately obvious.
        const h = fract(i.level.mul(0.2137).add(0.05));
        out.assign(
          vec3(
            abs(h.mul(6).sub(3)).sub(1),
            float(2).sub(abs(h.mul(6).sub(2))),
            float(2).sub(abs(h.mul(6).sub(4))),
          ).clamp(0, 1),
        );
        // Darken toward patch edges so individual patches stay countable.
        const edge = min(
          min(i.patchUV.x, float(1).sub(i.patchUV.x)),
          min(i.patchUV.y, float(1).sub(i.patchUV.y)),
        );
        out.assign(out.mul(smoothstep(float(0), float(0.03), edge).mul(0.4).add(0.6)));
      })
      .ElseIf(i.mode.equal(5), () => {
        // Morph coefficient. Should be 0 near the camera, ramp smoothly to 1 at
        // the far edge of each level's band, and never jump.
        out.assign(vec3(i.morphK));
        out.assign(mix(out, vec3(0.9, 0.3, 0.1), step(float(0.999), i.morphK).mul(0.6)));
      })
      .ElseIf(i.mode.equal(6), () => {
        // Tile borders: patch outline in red, real heightmap texel grid in grey.
        // A correct mapping puts a texel line exactly on the patch outline and
        // an integer number of texels across the patch; sampling one texel off
        // shows immediately as the grid sliding against the outline.
        const gx = gridLine(i.texelCoord.x, float(1), float(0.06));
        const gy = gridLine(i.texelCoord.y, float(1), float(0.06));
        const grid = max(gx, gy);

        const edge = min(
          min(i.patchUV.x, float(1).sub(i.patchUV.x)),
          min(i.patchUV.y, float(1).sub(i.patchUV.y)),
        );
        const outline = float(1).sub(smoothstep(float(0), float(0.006), edge));

        const base = mix(vec3(0.12, 0.13, 0.15), vec3(0.4, 0.42, 0.45), grid.mul(0.6));
        // Amber wash on nodes that are still waiting for their own tile and are
        // borrowing an ancestor's — that is streaming state made visible rather
        // than left to be guessed at from a slightly soft silhouette.
        out.assign(mix(base, vec3(0.75, 0.45, 0.1), i.tileFallback.mul(0.55)));
        out.assign(mix(out, vec3(1.0, 0.25, 0.2), outline));
      })
      .ElseIf(i.mode.equal(7), () => {
        // Flow accumulation, straight ramp. The erosion pass already stores
        // log2(1+raw)/8, so a second log here would flatten the whole drainage
        // network into a uniform mid-tone and read as "erosion did nothing".
        out.assign(ramp(clamp(i.flow, float(0), float(1))));
      })
      .ElseIf(i.mode.equal(8), () => {
        out.assign(mix(vec3(0.45, 0.36, 0.24), vec3(0.15, 0.35, 0.7), clamp(i.moisture, float(0), float(1))));
      })
      .ElseIf(i.mode.equal(10), () => {
        // Biome mix: every weight painted in its key colour and summed. A
        // healthy classification reads as broad regions with ragged, mottled
        // borders. Two failure modes are instantly visible here and nowhere
        // else: flat mid-grey means the weights are not normalised, and a
        // razor-straight edge means a condition is a comparison rather than a
        // band.
        out.assign(biomeMix(i.weights));
      })
      .ElseIf(i.mode.equal(11), () => {
        // Dominant biome only, no blending — the classification the material's
        // top-two selection actually acts on. Compare against mode 10: where
        // this one shows a hard edge and mode 10 shows a gradient, the two
        // leading weights are close and the height blend is doing the work.
        out.assign(biomeKey(i.dominant));
        // Snow is coverage rather than a baked weight, so it is drawn as a
        // brightening on top: that makes the animated snow line visible against
        // a static classification.
        out.assign(mix(out, vec3(1, 1, 1), i.snow.mul(0.85)));
      })
      .ElseIf(i.mode.equal(9), () => {
        // World-space checker: 1 m squares tinted by 10 m squares. Verifies
        // world scale and that UVs are not stretched across LOD levels.
        const c1 = fract(floor(i.worldPos.x).add(floor(i.worldPos.z)).mul(0.5)).mul(2);
        const c10 = fract(floor(i.worldPos.x.div(10)).add(floor(i.worldPos.z.div(10))).mul(0.5)).mul(2);
        const fine = mix(vec3(0.35), vec3(0.55), c1);
        out.assign(mix(fine, fine.mul(vec3(1.0, 0.85, 0.7)), c10));
      })
      .Else(() => {
        out.assign(i.shaded);
      });

    return out;
  })();
}
