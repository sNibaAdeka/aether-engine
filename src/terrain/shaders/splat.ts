/**
 * Biome classification — the table in spec 2.1, as a shader graph.
 *
 * This is a *node builder*, not a `Fn`: it is called once, from the splat bake
 * kernel, and returns two vec4 nodes. Keeping it out of a function node means
 * the eight biome scores share their sub-expressions (temperature is read by
 * five of them) instead of being recomputed behind a call boundary.
 *
 * The classifier's whole job is to turn six scalar fields — height, slope,
 * moisture, temperature, aspect, curvature — into eight weights that sum to one.
 * Two decisions in here are worth stating up front because they are not obvious
 * from the spec text:
 *
 * 1. **Scores are soft, then normalised.** Every condition in the table is a
 *    `smoothstep` band rather than a comparison, and the result is divided by
 *    the total. Hard `if` boundaries would give exactly the "резкая линия" the
 *    acceptance criteria forbid; a plain weighted average of colours would give
 *    the "мыльный градиент" they also forbid. The soft-score/normalise split
 *    puts the transition width under explicit control here, and lets the
 *    material sharpen it back up with height-based blending at shading time.
 *
 * 2. **Snow is not a baked weight.** Spec 2.1 requires the snow line to advance
 *    and retreat over minutes of game time. A tile is baked once and may sit
 *    resident for the whole session, so anything baked cannot animate. What is
 *    baked instead is *susceptibility*: how well this texel holds snow if snow
 *    reaches it. The material combines that with the live snow line every frame.
 *    See `TerrainMaterial`.
 */

import {
  wgslFn,
  float,
  vec2,
  vec3,
  vec4,
  atan,
  clamp,
  smoothstep,
  mix,
  max,
  length,
} from 'three/tsl';
import { fnFbm2, fnPerlin2 } from './heightfield.wgsl';
import type { ShaderNode } from './debugViews';

type N = ShaderNode;

/**
 * f32-only wrappers around the terrain noise.
 *
 * `aeFbm2` takes `i32` for its seed and octave count. Feeding those from TSL
 * means every call site has to produce integer nodes of exactly the right type
 * or the generated WGSL fails to compile with an argument-type error that names
 * neither the call site nor the parameter. Wrapping once, here, means the
 * classifier can pass plain floats and the conversion lives in one place.
 */
const fnSplatFbm = wgslFn(
  /* wgsl */ `
fn aeSplatFbm( p: vec2f, seed: f32, octaves: f32, frequency: f32 ) -> f32 {
  return aeFbm2( p, i32( seed ), i32( octaves ), frequency, 2.01734, 0.5 );
}
`,
  [fnFbm2],
);

const fnSplatPerlin = wgslFn(
  /* wgsl */ `
fn aeSplatPerlin( p: vec2f, seed: f32 ) -> f32 {
  return aePerlin2( p, i32( seed ) );
}
`,
  [fnPerlin2],
);

/**
 * Low-frequency noise for the material's macro variation, ≈[-1,1].
 *
 * Shared with the classifier deliberately: the same lattice drives the biome
 * mottling and the albedo modulation, so a patch that classifies slightly
 * differently also *looks* slightly different, instead of the two effects
 * beating against each other at their own unrelated periods.
 */
export const fnMacroNoise = fnSplatPerlin;

/**
 * Climate constants. The world is one continent a few hundred kilometres
 * across, so "latitude" is a linear ramp in Z rather than a spherical model —
 * the point is that the north end is colder than the south end, not that it is
 * geodetically right. +Z is south.
 */
export const CLIMATE = {
  /**
   * Sea-level temperature at the world origin, °C.
   *
   * 6 °C, not the 15 °C of a real temperate sea level, and the reason is the
   * terrain rather than the meteorology. This world's peaks were *measured*,
   * not assumed: a 1 km sweep of ±60 km around the origin tops out at 902 m,
   * and the mountain-noon viewpoint's own massif at 788 m. A realistic 15 °C
   * sea level puts the 0 °C isotherm at 2300 m, i.e. no snow anywhere, ever —
   * which is exactly what the first draft rendered.
   *
   * 6 °C puts the mean line at 920 m and the spring line at about 645 m, so the
   * summits carry snow and their shoulders do not. The consequence is that this
   * is a cool maritime world — Iceland, not Provence — and the biome table is
   * calibrated for it: meadow, moss, scree, bare rock and bog, with the desert
   * condition (T > 22 °C) genuinely unreachable. That is a coherent landscape,
   * not a compromise, but it is a *choice* forced by the height field.
   */
  BASE_C: 6.0,
  /** Warming per metre travelled south (+Z). 12 °C over 120 km. */
  LATITUDE_C_PER_M: 12.0 / 120000,
  /** Environmental lapse rate, °C per metre. The spec fixes this at −6.5 °C/km. */
  LAPSE_C_PER_M: 6.5 / 1000,
  /** Regional warm/cold anomaly amplitude, °C. Breaks the latitude banding. */
  ANOMALY_C: 3.5,
  /**
   * A south-facing slope gets more sun. Not decoration: it is what makes one
   * side of a valley forested and the other side dry grass, which is the single
   * most recognisable "this is a real landscape" cue in a mountain shot.
   */
  ASPECT_C: 2.6,
  /**
   * Reference snow line for the *rock* condition, metres.
   *
   * Static on purpose, and not the same number as the live snow line. "Bare
   * rock above the snow line" is a statement about where soil cannot persist
   * over decades, which is a property of the climate's mean — not of where the
   * snow happens to be this afternoon. Tying it to the animated line would make
   * bedrock appear and disappear with the season.
   */
  SNOWLINE_REF_M: 970,
} as const;

/**
  * Height at which trees stop, metres.
  *
  * 520 m, which is low, because the treeline tracks the snow line and this
  * world's is at 645–920 m. Sub-arctic treelines really are this low.
  */
const TREELINE_M = 520;

export interface SplatInputs {
  /** World XZ of this texel (`.x` is world X, `.y` is world Z). */
  world: N;
  /** Height above sea level, metres. */
  height: N;
  /** Surface normal, world space, unit length. */
  normal: N;
  /** |∇h| — metres of rise per metre of run, i.e. tan(slope). */
  slopeTan: N;
  /**
   * Hydrological moisture from the erosion flow map, 0..1. Zero on a tile whose
   * erosion job has not run yet, which is why the classifier adds its own
   * climatic base rather than trusting this alone.
   */
  flowMoisture: N;
  /**
   * Laplacian of the height field, non-dimensionalised by the cell size and
   * clamped to ±1. Positive in hollows (neighbours are higher), negative on
   * crests.
   */
  concavity: N;
  /** Seed, as a float. */
  seed: N;
}

export interface SplatOutputs {
  /** rock · scree · meadow grass · dry grass */
  w0: N;
  /** forest floor · sand · mud · snow susceptibility */
  w1: N;
  /** Temperature in °C at this texel. */
  temperature: N;
  /** Final moisture, 0..1, after climate and aspect. */
  moisture: N;
}

/** A soft band: 1 inside [a,b], falling to 0 over `fade` on each side. */
function band(x: N, a: number, b: number, fade: number): N {
  return smoothstep(float(a - fade), float(a), x).mul(
    smoothstep(float(b), float(b + fade), x).oneMinus(),
  );
}

export function buildBiomeSplat(i: SplatInputs): SplatOutputs {
  const h = i.height;
  const n = i.normal;
  const seed = i.seed;

  // --- slope ------------------------------------------------------------
  // The table is written in degrees, so convert once and use degrees
  // throughout: a threshold you can compare against the spec by eye is worth
  // more than the two ALU ops an atan costs in a pass that runs once per tile.
  const slopeDeg = atan(i.slopeTan).mul(57.29577951308232);

  // --- aspect -----------------------------------------------------------
  // On a flat texel the horizontal normal vanishes and the aspect is
  // meaningless, so it is faded in with steepness rather than normalised
  // blindly — which would amplify numerical noise into stripes on flat ground.
  const horiz = length(vec2(n.x, n.z));
  const southness = n.z
    .div(max(horiz, float(1e-4)))
    .mul(smoothstep(float(0.04), float(0.3), i.slopeTan));

  // --- temperature ------------------------------------------------------
  // Latitude analogue + lapse rate + a regional anomaly + aspect. The anomaly
  // is what stops the biomes from reading as horizontal stripes across the map.
  const anomaly = fnSplatFbm({
    p: i.world,
    seed: seed.add(float(15895)),
    octaves: float(3),
    frequency: float(0.000045),
  }).mul(CLIMATE.ANOMALY_C * 2.4);

  const temperature = float(CLIMATE.BASE_C)
    .add(i.world.y.mul(float(CLIMATE.LATITUDE_C_PER_M)))
    .sub(max(h, float(0)).mul(float(CLIMATE.LAPSE_C_PER_M)))
    .add(anomaly)
    .add(southness.mul(float(CLIMATE.ASPECT_C)));

  // --- moisture ---------------------------------------------------------
  // Four contributions, because the erosion flow map alone is both too sparse
  // (it only knows about channels) and absent on freshly baked tiles:
  //   · a climatic base from low-frequency noise,
  //   · proximity to sea level as a stand-in for "near water",
  //   · hollows collect and crests shed,
  //   · the hydrological term from the flow map, which is what puts wet ground
  //     in the valley floors erosion actually carved.
  const climaticWet = fnSplatFbm({
    p: i.world.add(vec2(4211.7, -1877.3)),
    seed: seed.add(float(23441)),
    octaves: float(4),
    frequency: float(0.00009),
  })
    .mul(1.55)
    .add(0.34);

  const nearWater = smoothstep(float(90), float(0), h).mul(0.3);
  const hollowWet = clamp(i.concavity, float(0), float(1)).mul(0.22);
  // Warm air holds its water badly: hot ground dries out.
  const heatDry = smoothstep(float(16), float(30), temperature).mul(0.3);
  // A south-facing slope is drier than the north-facing one across the valley.
  const aspectDry = max(southness, float(0)).mul(0.14);

  const moisture = clamp(
    climaticWet
      .add(nearWater)
      .add(hollowWet)
      // Weighted at 0.35 rather than the 0.55 this started at. Erosion completes
      // one tile at a time over tens of seconds, so during streaming an eroded
      // tile sits next to an un-eroded one and this term is the *whole*
      // difference between them. At 0.55 the boundary was a visible square of
      // differently-classified ground; at 0.35 it is a tonal shift.
      // TODO(quality): the honest fix is to classify from a moisture field that
      // does not depend on erosion progress, or to defer classification until a
      // tile is eroded — the latter costs a hole in the biome map for seconds.
      .add(i.flowMoisture.mul(0.35))
      .sub(heatDry)
      .sub(aspectDry),
    float(0),
    float(1),
  );

  // Metre-scale mottling, added to each score so that the *boundaries* between
  // biomes are ragged instead of being smooth level sets of a smooth field.
  // This is what makes grass reach into cracks in the rock rather than stopping
  // along a contour line, and it costs one noise lookup.
  const mottle = fnSplatPerlin({ p: i.world.mul(0.19), seed: seed.add(float(30627)) }).mul(0.5);
  const mottleCoarse = fnSplatPerlin({
    p: i.world.mul(0.021),
    seed: seed.add(float(4597)),
  }).mul(0.5);

  // --- the table --------------------------------------------------------

  // Скала: уклон > 38° или высота > snowline − 200.
  // Both are real rock-forming mechanisms — too steep to hold soil, or too cold
  // and wind-scoured for it — so they combine with max, not sum.
  const rockSteep = smoothstep(float(33), float(41), slopeDeg);
  const rockHigh = smoothstep(
    float(CLIMATE.SNOWLINE_REF_M - 280),
    float(CLIMATE.SNOWLINE_REF_M - 120),
    h,
  ).mul(mix(float(0.55), float(1.0), smoothstep(float(6), float(14), slopeDeg)));
  // Convex ground (ridges, arêtes) sheds its cover; concave ground buries the
  // bedrock under whatever fell off the slope above.
  const rockConvex = clamp(i.concavity.mul(-0.55), float(0), float(0.35));
  const rock = max(rockSteep, rockHigh).add(rockConvex).add(mottle.mul(0.1)).max(float(0.02));

  // Осыпь: уклон 30–40°, ниже скалы. Talus is a *deposit*, so it wants both the
  // angle of repose and a place to sit: concave, and not on a face so steep
  // that nothing stays.
  const scree = band(slopeDeg, 28, 39, 7)
    .mul(smoothstep(float(40), float(48), slopeDeg).oneMinus())
    .mul(clamp(i.concavity.mul(0.5).add(0.75), float(0.3), float(1.25)))
    .add(mottle.mul(0.12))
    .max(float(0));

  // Луговая трава: уклон < 25°, влажность 0.3–0.7.
  const meadow = smoothstep(float(19), float(31), slopeDeg)
    .oneMinus()
    .mul(band(moisture, 0.3, 0.72, 0.16))
    .mul(smoothstep(float(-1), float(6), temperature))
    .mul(smoothstep(float(TREELINE_M - 60), float(TREELINE_M + 220), h).oneMinus())
    .add(mottle.mul(0.14))
    .max(float(0));

  // Сухая трава: влажность < 0.3, тепло.
  const dryGrass = smoothstep(float(0.2), float(0.44), moisture)
    .oneMinus()
    // Warmth threshold set for an 8 °C sea level: at the 15 °C the first draft
    // used, dry grass could not appear anywhere in this world and the whole
    // map classified as meadow.
    .mul(smoothstep(float(0), float(8), temperature))
    .mul(smoothstep(float(22), float(35), slopeDeg).oneMinus())
    .add(mottle.mul(0.1))
    .max(float(0.015));

  // Лесная подстилка: «под пологом леса». There is no forest until phase 7, so
  // this is the canopy *mask* the tree placement will later read — same inputs,
  // same thresholds — rather than a guess at what litter looks like under trees
  // that do not exist yet.
  // TODO(quality): phase 7 must consume this mask instead of deriving its own,
  // or the litter will not line up with the trees standing on it.
  const forestNoise = fnSplatFbm({
    p: i.world.add(vec2(-8123.4, 3319.8)),
    seed: seed.add(float(11595)),
    octaves: float(4),
    frequency: float(0.00028),
  })
    .mul(2.1)
    .add(0.5);
  const forest = smoothstep(float(0.42), float(0.62), forestNoise)
    .mul(band(temperature, 2.0, 19.0, 4.0))
    .mul(smoothstep(float(0.34), float(0.5), moisture))
    .mul(smoothstep(float(26), float(38), slopeDeg).oneMinus())
    .mul(smoothstep(float(TREELINE_M - 180), float(TREELINE_M), h).oneMinus())
    .add(mottleCoarse.mul(0.1))
    .max(float(0));

  // Песок: у воды, высота < 4 м, малый уклон. Plus hot, bone-dry lowland — the
  // spec lists only the beach case, but a world that puts dry grass on 30 °C
  // ground at 5 % moisture and never any sand reads as a single climate zone.
  const beach = smoothstep(float(7), float(1), h)
    .mul(smoothstep(float(-4), float(0.2), h))
    .mul(smoothstep(float(9), float(20), slopeDeg).oneMinus())
    .mul(1.6);
  const desert = smoothstep(float(22), float(31), temperature)
    .mul(smoothstep(float(0.1), float(0.26), moisture).oneMinus())
    .mul(smoothstep(float(14), float(26), slopeDeg).oneMinus());
  const sand = max(beach, desert).add(mottle.mul(0.08)).max(float(0));

  // Грязь/суглинок: влажность > 0.75, низины.
  // The concavity factor is a gate here, not a bias: mud is standing water and
  // sediment, and both need somewhere to stand. Multiplying by a term that is
  // 0.25 even on a convex slope — which is what the first draft did — spread
  // loam across every damp hillside in the world.
  const mud = smoothstep(float(0.68), float(0.9), moisture)
    .mul(smoothstep(float(7), float(15), slopeDeg).oneMinus())
    .mul(clamp(i.concavity.mul(1.4).add(0.25), float(0), float(1.3)))
    .add(mottle.mul(0.05))
    .max(float(0));

  // --- normalise --------------------------------------------------------
  // The floors on `rock` and `dryGrass` guarantee a non-zero sum, so this
  // cannot divide by zero even where every condition fails.
  const total = rock.add(scree).add(meadow).add(dryGrass).add(forest).add(sand).add(mud);
  const inv = float(1).div(max(total, float(1e-4)));

  // --- snow susceptibility ---------------------------------------------
  // Not a weight. How well the texel *holds* snow, if snow reaches it:
  //   · shallow ground holds it, a 55° face does not,
  //   · hollows and lee slopes drift up, crests and windward faces scour bare,
  //   · a decametre noise breaks the edge so the snow line is not a contour.
  // Wind is taken as coming from the west (+X normals face into it).
  // TODO(quality): that wind direction is a constant; when phase 8 lands a real
  // wind field, drift orientation should follow it instead.
  const holdSlope = smoothstep(float(52), float(24), slopeDeg);
  const drift = clamp(i.concavity.mul(0.45).add(0.78), float(0.35), float(1.25));
  const lee = clamp(n.x, float(0), float(1)).mul(0.35).oneMinus();
  const driftNoise = fnSplatPerlin({ p: i.world.mul(0.045), seed: seed.add(float(27183)) })
    .mul(0.22)
    .add(1.0);
  const snowHold = clamp(holdSlope.mul(drift).mul(lee).mul(driftNoise), float(0), float(1));

  return {
    w0: vec4(rock.mul(inv), scree.mul(inv), meadow.mul(inv), dryGrass.mul(inv)),
    w1: vec4(forest.mul(inv), sand.mul(inv), mud.mul(inv), snowHold),
    temperature,
    moisture,
  };
}

/** Names in the order the splat channels and the material's array layers use. */
export const BIOME_NAMES = [
  'rock',
  'scree',
  'meadow',
  'dryGrass',
  'forestFloor',
  'sand',
  'mud',
  'snow',
] as const;

export const BIOME_COUNT = BIOME_NAMES.length;

/**
 * Colour key for the biome debug view, in the same order.
 * Deliberately garish and maximally separated in hue — the point is to tell
 * eight classes apart at a glance, not to look like ground.
 */
export const BIOME_DEBUG_COLORS: readonly (readonly [number, number, number])[] = [
  [0.62, 0.62, 0.66], // rock — grey
  [0.85, 0.55, 0.2], // scree — orange
  [0.15, 0.72, 0.2], // meadow — green
  [0.86, 0.82, 0.2], // dry grass — yellow
  [0.06, 0.3, 0.12], // forest floor — dark green
  [0.95, 0.86, 0.55], // sand — pale
  [0.4, 0.24, 0.12], // mud — brown
  [1.0, 1.0, 1.0], // snow — white
];

/** Turn eight weights into the debug key colour. */
export function biomeDebugColor(w: readonly N[]): N {
  let out = vec3(0, 0, 0);
  for (let k = 0; k < BIOME_DEBUG_COLORS.length; k++) {
    const c = BIOME_DEBUG_COLORS[k];
    out = out.add(vec3(c[0], c[1], c[2]).mul(w[k]));
  }
  return out;
}
