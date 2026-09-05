/**
 * The terrain surface shader (spec 2.2), as a TSL graph.
 *
 * Six techniques, each solving one specific way that a large procedural
 * landscape betrays itself:
 *
 *  1. **Triplanar, gated on slope.** A single top-down projection smears into
 *     vertical stripes on anything steeper than about 45°, and cliffs are where
 *     the eye goes first. Triplanar costs three sets of samples, so it is faded
 *     in only above 30° — below that the top-down projection is already correct
 *     and paying 3× for it is pure waste.
 *
 *  2. **Stochastic tiling on grass and sand.** A 2 m texture repeated across a
 *     kilometre is a visible grid from any altitude, and no amount of macro
 *     variation hides it because the *structure* repeats, not just the tone.
 *     Heitz & Neyret's simplex-grid blend with randomised offsets breaks the
 *     lattice; the variance-preserving recombination stops the three-way blend
 *     from washing out toward the texture's mean, which is what a naive
 *     weighted average does and why it looks soapy.
 *
 *  3. **Multi-scale detail.** Each texture is sampled at 1× and at 8×, blended
 *     by distance. The 8× sample supplies the millimetre grain the 1× sample
 *     mips away up close; fading it out with distance keeps it from turning
 *     into noise at range.
 *
 *  4. **Height-based layer blending.** Two splat layers are combined by their
 *     displacement channels, not by a linear lerp: whichever layer stands
 *     higher at a texel wins there. This is the single largest visual
 *     difference in the file — grass grows *between* the stones of a scree
 *     slope instead of being airbrushed over them.
 *
 *  5. **Parallax occlusion mapping under 15 m on rock.** Rock is where the
 *     height field has real relief (joints, bedding) and where the camera gets
 *     close enough for a flat normal map to read as a decal. Step count is
 *     adaptive: a grazing view needs more steps than a head-on one.
 *
 *  6. **Macro variation.** A 70 m and a 190 m noise modulate albedo by ±12 %
 *     and roughness alongside it. Without it every large flat area reads as a
 *     single sheet of plastic, no matter how good the detail is.
 *
 * Everything samples with an explicit mip level. That is not a micro-
 * optimisation: several of the techniques above sit inside `If` blocks, and
 * implicit-derivative sampling under non-uniform control flow is undefined in
 * WGSL. The level is computed once, outside every branch, from the screen-space
 * derivative of the world position.
 */

import {
  Fn,
  If,
  Loop,
  Break,
  struct,
  float,
  int,
  vec2,
  vec3,
  vec4,
  abs,
  dot,
  clamp,
  smoothstep,
  step,
  mix,
  max,
  sign,
  floor,
  fract,
  sin,
  log2,
  length,
  pow,
  normalize,
  texture,
  uniform,
  dFdx,
  dFdy,
  cameraPosition,
} from 'three/tsl';

import type { ShaderNode } from './shaders/debugViews';
import { fnMacroNoise } from './shaders/splat';
import { MATERIAL_TEX_METRES, MATERIAL_TEX_SIZE, type MaterialTextures } from './MaterialTextures';

type N = ShaderNode;

/** Ratio between the coarse and the fine detail sample. */
const DETAIL_SCALE = 8;
/** Distance over which the 8× detail fades out, metres. */
const DETAIL_NEAR = 6;
const DETAIL_FAR = 55;

/**
 * Distance band in which the stochastic blend is worth its three taps.
 *
 * Below `STOCH_NEAR` a single 2 m tile fills most of the screen and there is no
 * repeat to see. Above `STOCH_FAR` the 1× sample is mipped past the point where
 * its structure survives, so the repeat is invisible again — and the blend
 * would still cost three taps per texture. The band between is exactly where
 * the acceptance criterion ("from 800 m: no visible repeating pattern") lives.
 */
const STOCH_NEAR = 9;
const STOCH_FADE = 14;
const STOCH_FAR = 1400;

/** Parallax occlusion is applied under this distance only. */
const POM_MAX_M = 15;
/** Depth of the parallax effect, in texture periods. */
const POM_DEPTH = 0.05;
const POM_STEPS_MIN = 8;
const POM_STEPS_MAX = 24;

/**
 * How sharply two layers interpenetrate at their height-blend boundary.
 *
 * Small values give a hard mosaic edge, large values approach a linear lerp.
 * 0.14 is about a seventh of the 1× height range, which reads as grass finding
 * the gaps between stones rather than as either a cut-out or a fade.
 */
const HEIGHT_BLEND_WIDTH = 0.14;

/** Distance over which the detail normal is faded out, metres. */
const NORMAL_FADE_NEAR = 140;
const NORMAL_FADE_FAR = 800;

export interface TerrainMaterialUniforms {
  /**
   * Altitude of the current snow line, metres. Driven toward its target over
   * minutes of game time by `SnowLine` — see spec 2.1, "не мгновенно".
   */
  snowLine: ReturnType<typeof uniform>;
  /** Vertical softness of the snow edge, metres. */
  snowFade: ReturnType<typeof uniform>;
  /** 0 disables the stochastic blend (for A/B measurement). */
  stochastic: ReturnType<typeof uniform>;
  /** 0 disables parallax occlusion. */
  parallax: ReturnType<typeof uniform>;
  /** 0 disables triplanar projection. */
  triplanar: ReturnType<typeof uniform>;
  /** Macro-variation amount, 0..1. */
  macro: ReturnType<typeof uniform>;
  /** World seed, as a float. */
  seed: ReturnType<typeof uniform>;
}

export function createTerrainMaterialUniforms(seed: number): TerrainMaterialUniforms {
  return {
    snowLine: uniform(1150),
    snowFade: uniform(80),
    stochastic: uniform(1),
    parallax: uniform(1),
    triplanar: uniform(1),
    macro: uniform(1),
    seed: uniform(seed),
  };
}

export interface TerrainMaterialInputs {
  /** Fragment world position. */
  worldPos: N;
  /** Baked surface normal, world space, unit length. */
  normal: N;
  /** Biome weights 0–3: rock, scree, meadow, dry grass. */
  splat0: N;
  /** Biome weights 4–6 plus snow susceptibility in `.w`. */
  splat1: N;
  tex: MaterialTextures;
  u: TerrainMaterialUniforms;
}

export interface TerrainMaterialOutputs {
  albedo: N;
  roughness: N;
  /** Ambient occlusion. Applies to indirect light only — see CLAUDE.md. */
  ao: N;
  /** Perturbed world-space normal. */
  normal: N;
  /** Snow coverage 0..1, for the debug views. */
  snow: N;
  /** All eight weights after snow has taken its share, for the debug views. */
  weights: N[];
  /** Index of the dominant biome, 0..7, for the debug views. */
  dominant: N;
}

/**
 * Simplex-grid tiling weights and cell ids (Heitz & Neyret 2018, §3.3).
 *
 * The uv plane is skewed onto an equilateral triangle lattice; every point then
 * lies inside exactly one triangle and has barycentric weights against its
 * three corners. Each corner contributes a copy of the texture at a random
 * offset, so the visible period is the *lattice's*, not the texture's — and the
 * lattice sits at an irrational angle to the texture axes, so there is no
 * repeat to find.
 */
function triangleGrid(uv: N): { w: N; v0: N; v1: N; v2: N } {
  const p = uv.mul(3.4641016151); // 2·√3 — one lattice cell per texture period
  // Skew onto the triangle lattice. Written out rather than as a mat2 multiply
  // so the column-major convention cannot be got wrong silently.
  const skewed = vec2(p.x.sub(p.y.mul(0.57735027)), p.y.mul(1.1547005));
  const base = floor(skewed);
  const f = fract(skewed);
  const third = float(1).sub(f.x).sub(f.y);

  // Upper or lower triangle of the rhombus, by the sign of the third
  // barycentric. Branchless: this feeds three texture fetches and a divergent
  // branch here costs more than the arithmetic it saves.
  const upper = step(float(0), third);
  const w = vec3(
    mix(third.negate(), third, upper),
    mix(float(1).sub(f.y), f.y, upper),
    mix(float(1).sub(f.x), f.x, upper),
  );

  return {
    w,
    v0: base.add(mix(vec2(1, 1), vec2(0, 0), upper)),
    v1: base.add(mix(vec2(1, 0), vec2(0, 1), upper)),
    v2: base.add(mix(vec2(0, 1), vec2(1, 0), upper)),
  };
}

/** Cheap 2D→2D hash for the per-cell texture offsets. */
function hash22(p: N): N {
  const s = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
  return fract(sin(s).mul(43758.5453));
}

/**
 * Struct returned by the surface function.
 *
 * The whole shader has to live inside one TSL `Fn`, because `If` and `Loop`
 * need a shader stack to push onto and there is none at module scope. One `Fn`
 * can only return one value, so the eight outputs travel in a struct — the two
 * weight vectors packed exactly as the splat textures store them, so the debug
 * views unpack them the same way everything else does.
 */
const TerrainSurface = struct(
  {
    albedo: 'vec3',
    roughness: 'float',
    ao: 'float',
    normal: 'vec3',
    snow: 'float',
    dominant: 'float',
    w0: 'vec4',
    w1: 'vec4',
  },
  'AeTerrainSurface',
);

export function buildTerrainMaterial(i: TerrainMaterialInputs): TerrainMaterialOutputs {
  const surface = terrainSurface(i);
  return {
    albedo: surface.get('albedo'),
    roughness: surface.get('roughness'),
    ao: surface.get('ao'),
    normal: surface.get('normal'),
    snow: surface.get('snow'),
    dominant: surface.get('dominant'),
    weights: [
      surface.get('w0').x,
      surface.get('w0').y,
      surface.get('w0').z,
      surface.get('w0').w,
      surface.get('w1').x,
      surface.get('w1').y,
      surface.get('w1').z,
      surface.get('w1').w,
    ],
  };
}

function terrainSurface(i: TerrainMaterialInputs): N {
  return Fn(() => buildSurfaceGraph(i))();
}

function buildSurfaceGraph(i: TerrainMaterialInputs): N {
  const u = i.u;
  const world = i.worldPos;
  const n = normalize(i.normal);

  const toCam = cameraPosition.sub(world);
  const dist = length(toCam);
  const viewDir = toCam.div(max(dist, float(1e-4)));

  // --- mip level --------------------------------------------------------
  // World metres covered by one pixel, from the screen-space derivative of the
  // world position. Computed here, at the top of the shader, because every
  // sample below may sit inside a branch and WGSL forbids implicit derivatives
  // there. It also gives the multi-scale blend an exact three-level offset for
  // the 8× sample instead of letting the hardware guess.
  const footprint = max(length(dFdx(world)), length(dFdy(world)));
  const texelsPerMetre = MATERIAL_TEX_SIZE / MATERIAL_TEX_METRES;
  const lod1 = max(log2(max(footprint.mul(texelsPerMetre), float(1e-5))), float(0));
  const lod8 = max(lod1.sub(3), float(0));
  // The 1×1 mip *is* the texture's mean. Asking for a level past the end of the
  // chain clamps to it, which makes the exact level count a non-issue.
  const lodMean = float(20);

  // --- snow -------------------------------------------------------------
  // Coverage, not a baked weight. `splat1.w` is how well this texel *holds*
  // snow (shallow, concave, lee); the snow line says whether snow reaches it at
  // all. Multiplying the two is what makes the line ragged and terrain-aware
  // instead of a horizontal cut across the mountain.
  const snowHold = i.splat1.w;
  const altitude = smoothstep(u.snowLine.sub(u.snowFade), u.snowLine.add(u.snowFade), world.y);
  const snow = clamp(altitude.mul(snowHold).mul(1.35), float(0), float(1));

  // Everything else shares what is left. The splat channels already sum to 1,
  // so scaling them by (1 − snow) and adding snow keeps the sum at 1.
  const keep = float(1).sub(snow);
  const w: N[] = [
    i.splat0.x.mul(keep),
    i.splat0.y.mul(keep),
    i.splat0.z.mul(keep),
    i.splat0.w.mul(keep),
    i.splat1.x.mul(keep),
    i.splat1.y.mul(keep),
    i.splat1.z.mul(keep),
    snow,
  ];

  // --- pick the two dominant layers ------------------------------------
  // Eight materials at two scales with a three-tap stochastic blend is over a
  // hundred texture fetches per pixel; nothing shipping does that. The weights
  // are sparse in practice — the classifier's soft bands overlap two or three
  // biomes, never eight — so the top two carry the surface and the rest are
  // rounding. The height blend below then makes the boundary between them read
  // as interpenetrating materials rather than as a two-way crossfade.
  //
  // TODO(quality): where three biomes genuinely meet the third is dropped, and
  // the junction can show a faint straight edge along the line where the second
  // and third swap rank. Visible in the biome debug view at meadow/forest/scree
  // triple points; not visible in the shaded render at this weighting.
  const idx0 = float(0).toVar();
  const idx1 = float(0).toVar();
  const wgt0 = float(-1).toVar();
  const wgt1 = float(-1).toVar();

  for (let k = 0; k < w.length; k++) {
    const wk = w[k];
    If(wk.greaterThan(wgt0), () => {
      idx1.assign(idx0);
      wgt1.assign(wgt0);
      idx0.assign(float(k));
      wgt0.assign(wk);
    }).ElseIf(wk.greaterThan(wgt1), () => {
      idx1.assign(float(k));
      wgt1.assign(wk);
    });
  }

  const wSum = max(wgt0.add(wgt1), float(1e-4));
  const a0 = wgt0.div(wSum);
  const a1 = float(1).sub(a0);

  // --- projections ------------------------------------------------------
  const invM = 1 / MATERIAL_TEX_METRES;
  const uvY = vec2(world.x, world.z).mul(invM);
  // The side projections are mirrored by the sign of the normal so the texture
  // does not run backwards on one side of a ridge — invisible on isotropic
  // noise, glaring on anything with a direction, and granite bedding has one.
  const sgnX = sign(n.x);
  const sgnZ = sign(n.z);
  const uvX = vec2(world.z.mul(sgnX), world.y.negate()).mul(invM);
  const uvZ = vec2(world.x.mul(sgnZ.negate()), world.y.negate()).mul(invM);

  // Spec 2.2: blendWeight = pow(abs(normal), 8), normalised.
  const bwRaw = pow(abs(n), vec3(8, 8, 8));
  const bwSum = max(bwRaw.x.add(bwRaw.y).add(bwRaw.z), float(1e-5));
  const bw = bwRaw.div(bwSum);

  // cos 30° = 0.866, cos 42° = 0.743. Fading between them turns triplanar fully
  // on by the time the surface is steep enough for the top-down projection to
  // stretch visibly, and leaves it fully off where it is not needed.
  const triK = smoothstep(float(0.866), float(0.743), n.y).mul(u.triplanar);

  // --- detail and stochastic gates -------------------------------------
  const detailK = float(1).sub(smoothstep(float(DETAIL_NEAR), float(DETAIL_FAR), dist));
  const stochK = smoothstep(float(STOCH_NEAR), float(STOCH_NEAR + STOCH_FADE), dist)
    .mul(float(1).sub(smoothstep(float(STOCH_FAR * 0.55), float(STOCH_FAR), dist)))
    .mul(u.stochastic);

  const tapAlbedo = (idx: N, uv: N, lod: N): N =>
    texture(i.tex.albedoAO, uv).depth(int(idx)).level(lod);
  const tapNRH = (idx: N, uv: N, lod: N): N =>
    texture(i.tex.normalRH, uv).depth(int(idx)).level(lod);

  /**
   * Three-tap stochastic sample with variance-preserving recombination.
   *
   * This is Heitz & Neyret's tiling and their variance-preserving blend, but
   * *not* their full histogram-preserving transform: that needs a per-texture
   * T(input)→Gaussian LUT and its inverse, precomputed and stored alongside
   * every material. The variance term alone recovers most of the contrast a
   * naive blend loses and costs one extra fetch of the 1×1 mip, rather than two
   * more textures per biome.
   *
   * TODO(quality): the full histogram transform would hold contrast exactly
   * where this holds it approximately. Not visible in the acceptance shots;
   * would matter for a close-up of sand.
   */
  // NOTE: no `toVar()` in this file takes a name. Parts of it are inlined
  // several times, and three itself compiles the material more than once; a
  // named declaration emitted twice makes it rename the second and warn. That
  // was 148 warnings per run in the debug bridge's error list, which is exactly
  // how a real error gets missed. Anonymous vars are numbered silently.
  const stochastic = (sample: (uv: N, lod: N) => N, uv: N, lod: N, mean: () => N, k: N): N => {
    // Two branches, and both of them matter to the frame budget.
    //
    // The plain tap is only needed where the blend is not fully on, because
    // `mix(plain, preserved, 1)` is just `preserved`. The three-tap blend plus
    // its mean is only needed where `k` is not zero. In the near field only the
    // first runs, in the far field only the first runs, and in the band between
    // only the second — so the common cost is one sample or four, never five.
    // Measured: leaving both unconditional cost 1.1 ms of the terrain pass.
    const out = mean().mul(0).toVar();
    If(k.lessThan(float(0.99)), () => {
      out.assign(sample(uv, lod));
    });
    If(k.greaterThan(float(0.01)), () => {
      const g = triangleGrid(uv);
      const s0 = sample(uv.add(hash22(g.v0)), lod);
      const s1 = sample(uv.add(hash22(g.v1)), lod);
      const s2 = sample(uv.add(hash22(g.v2)), lod);
      const blended = s0.mul(g.w.x).add(s1.mul(g.w.y)).add(s2.mul(g.w.z));
      // √(w0² + w1² + w2²): 1 when one corner owns the pixel (so the blend is
      // exactly a single tap there) and 1/√3 at an equal split, where a plain
      // average would have lost √3 of its standard deviation.
      const m = mean();
      const norm = max(length(vec3(g.w.x, g.w.y, g.w.z)), float(1e-3));
      const preserved = m.add(blended.sub(m).div(norm));
      // `k` fades to the *plain* sample rather than to the mean, so the
      // transition out of the band does not go flat.
      out.assign(mix(out, preserved, k));
    });
    return out;
  };

  /**
   * Sample one biome layer: triplanar where needed, stochastic where useful,
   * multi-scale, with parallax on rock.
   *
   * The normal comes back as a world-space *perturbation vector*, not as a
   * normal. That is what makes the triplanar combination tractable: each
   * projection's tangent offset maps onto two world axes directly, the three
   * are summed by the blend weights, and the caller adds the result to the
   * baked surface normal. Reconstructing a full normal per projection and then
   * blending normals is where triplanar shaders usually acquire their lean.
   */
  const sampleLayer = (idx: N): { albedo: N; pert: N; rough: N; ao: N; height: N } => {
    // Which techniques this biome gets. Rock (0) and scree (1) take parallax —
    // spec 2.2, "только на камне и гальке". Meadow (2), dry grass (3), sand (5)
    // and snow (7) take the stochastic blend: the spec names grass and sand,
    // and a snowfield has exactly the same large-uniform-area problem.
    const isRockish = step(idx, float(1.5));
    // The stochastic blend runs on *every* biome, not only the two the spec
    // names. Grass and sand are where it is mandatory, but the rock close-up
    // showed the reason to go further: granite's joint network is a Worley
    // pattern with a 40 cm cell inside a 2 m texture, and from 40 m away that
    // repeat reads as a regular hexagonal lattice across the whole face — the
    // most obviously artificial thing in any of the acceptance shots. Since the
    // blend is already gated to the 9–1400 m band and skipped outside it, the
    // marginal cost of extending it is confined to exactly the pixels where the
    // repeat would otherwise be visible.
    const kStoch = stochK;

    // --- parallax -----------------------------------------------------
    // Offsets the *top* projection's uv only. On a steep face the top
    // projection is faded out by `triK` anyway, and a full triplanar parallax
    // would triple an already expensive loop for an effect that only reads at
    // grazing angles on near-flat rock.
    const pomK = isRockish
      .mul(float(1).sub(smoothstep(float(POM_MAX_M * 0.6), float(POM_MAX_M), dist)))
      .mul(float(1).sub(triK))
      .mul(u.parallax);

    const uvYp = uvY.toVar();

    If(pomK.greaterThan(float(0.01)), () => {
      // Tangent frame of the top-down projection is the world frame: T = +X,
      // B = +Z, N = +Y, so the view vector's tangent-space form is a reshuffle.
      const vt = vec2(viewDir.x, viewDir.z);
      const vn = max(abs(viewDir.y), float(0.15));
      // Grazing views need more steps: the ray crosses more of the height field
      // per unit of depth, so a coarse march steps straight over ridges.
      const steps = mix(float(POM_STEPS_MAX), float(POM_STEPS_MIN), abs(viewDir.y));
      const layerStep = float(1).div(steps);
      const delta = vt.div(vn).mul(float(POM_DEPTH).mul(pomK)).mul(layerStep).negate();

      const curLayer = float(1).toVar();
      const curUV = uvY.toVar();
      const curH = tapNRH(idx, uvY, lod1).w.toVar();
      const prevUV = uvY.toVar();
      const prevH = curH.toVar();

      Loop(POM_STEPS_MAX, ({ i: stepIndex }) => {
        If(float(stepIndex).greaterThanEqual(steps), () => {
          Break();
        });
        If(curH.greaterThanEqual(curLayer), () => {
          Break();
        });
        prevUV.assign(curUV);
        prevH.assign(curH);
        curLayer.subAssign(layerStep);
        curUV.addAssign(delta);
        curH.assign(tapNRH(idx, curUV, lod1).w);
      });

      // Interpolate between the last step above the surface and the first
      // below it. Without it the offset quantises to the step size and the
      // surface visibly terraces at eight steps.
      const after = curH.sub(curLayer);
      const before = prevH.sub(curLayer.add(layerStep));
      const t = clamp(after.div(max(after.sub(before), float(1e-4))), float(0), float(1));
      uvYp.assign(mix(curUV, prevUV, t));
    });

    /** One projection, resolved at both scales. */
    const projection = (uv: N): { alb: N; nrh: N } => {
      // The means are passed as thunks so the 1×1-mip fetches are only emitted
      // inside the branch that needs them.
      const alb = stochastic(
        (q, l) => tapAlbedo(idx, q, l),
        uv,
        lod1,
        () => tapAlbedo(idx, uv, lodMean),
        kStoch,
      ).toVar();
      const nrh = stochastic(
        (q, l) => tapNRH(idx, q, l),
        uv,
        lod1,
        () => tapNRH(idx, uv, lodMean),
        kStoch,
      ).toVar();

      // The 8× sample is only fetched where it can be seen. Explicit level, so
      // the branch is legal.
      If(detailK.greaterThan(float(0.01)), () => {
        const uv8 = uv.mul(DETAIL_SCALE);
        const alb8 = tapAlbedo(idx, uv8, lod8);
        const nrh8 = tapNRH(idx, uv8, lod8);
        // Albedo: an overlay against the fine sample's own mid-grey, so the
        // tint comes from the 1× layer and only the *modulation* comes from the
        // 8×. A straight mix would double the colour variation and read as dirt.
        const mod8 = alb8.xyz.mul(2.6).add(0.35);
        // AO composes multiplicatively: a pit inside a pit is darker.
        const ao8 = mix(float(1), alb8.w, detailK.mul(0.6));
        alb.assign(
          vec4(mix(alb.xyz, alb.xyz.mul(mod8), detailK.mul(0.55)), alb.w.mul(ao8)),
        );
        // Normal: add the fine deviation rather than blending toward it, so the
        // coarse shape survives instead of being averaged away.
        nrh.assign(
          vec4(
            nrh.x.add(nrh8.x.sub(0.5).mul(detailK.mul(0.85))),
            nrh.y.add(nrh8.y.sub(0.5).mul(detailK.mul(0.85))),
            mix(nrh.z, nrh8.z, detailK.mul(0.4)),
            nrh.w.add(nrh8.w.sub(0.5).mul(detailK.mul(0.35))),
          ),
        );
      });

      return { alb, nrh };
    };

    const top = projection(uvYp);

    const albedo = top.alb.xyz.toVar();
    const ao = top.alb.w.toVar();
    const rough = top.nrh.z.toVar();
    const height = top.nrh.w.toVar();
    // Top projection: u → world +X, v → world +Z.
    const pert = vec3(
      top.nrh.x.sub(0.5).mul(2),
      float(0),
      top.nrh.y.sub(0.5).mul(2),
    ).toVar();

    If(triK.greaterThan(float(0.02)), () => {
      const sideX = projection(uvX);
      const sideZ = projection(uvZ);

      const wX = bw.x;
      const wY = bw.y;
      const wZ = bw.z;
      const inv = float(1).div(max(wX.add(wY).add(wZ), float(1e-5)));

      const blend3 = (y: N, x: N, z: N): N =>
        y.mul(wY).add(x.mul(wX)).add(z.mul(wZ)).mul(inv);

      // X projection: u → world Z (mirrored), v → world −Y.
      const pertX = vec3(
        float(0),
        sideX.nrh.y.sub(0.5).mul(-2),
        sideX.nrh.x.sub(0.5).mul(2).mul(sgnX),
      );
      // Z projection: u → world X (mirrored), v → world −Y.
      const pertZ = vec3(
        sideZ.nrh.x.sub(0.5).mul(2).mul(sgnZ.negate()),
        sideZ.nrh.y.sub(0.5).mul(-2),
        float(0),
      );

      albedo.assign(mix(albedo, blend3(top.alb.xyz, sideX.alb.xyz, sideZ.alb.xyz), triK));
      ao.assign(mix(ao, blend3(top.alb.w, sideX.alb.w, sideZ.alb.w), triK));
      rough.assign(mix(rough, blend3(top.nrh.z, sideX.nrh.z, sideZ.nrh.z), triK));
      height.assign(mix(height, blend3(top.nrh.w, sideX.nrh.w, sideZ.nrh.w), triK));
      pert.assign(mix(pert, blend3(pert, pertX, pertZ), triK));
    });

    return { albedo, pert, rough, ao, height };
  };

  const l0 = sampleLayer(idx0);

  const albedoMix = l0.albedo.toVar();
  const roughMix = l0.rough.toVar();
  const aoMix = l0.ao.toVar();
  const pertMix = l0.pert.toVar();

  // --- height-based blend ----------------------------------------------
  // Spec 2.2: `h1 + w1 > h2 + w2`, not a lerp. Each layer's displacement is
  // biased by its splat weight, and whichever ends up higher owns the texel.
  // The `HEIGHT_BLEND_WIDTH` window softens the winner-takes-all edge just
  // enough to antialias it — any wider and it degenerates back into a lerp.
  //
  // The second layer is only sampled where it is actually present. That is not
  // a micro-optimisation: it is half the fragment cost of this shader. Most of
  // a landscape is one biome — a meadow slope is meadow all the way across —
  // and 2 % is well below what the height blend could show even at its
  // sharpest, so nothing is lost where the branch is taken.
  //
  // A world-space perturbation of the boundary, at 3.6 m and 14 m.
  //
  // This exists because of a resolution mismatch that is easy to miss. The
  // splat mask's own metre-scale mottling is baked into a tile, and a tile at
  // LOD 3 covers eight times the ground of one at LOD 0 with the same 129²
  // texels — so the mottling is filtered away with distance and everything from
  // about 50 m out flattens into a single wash of one biome. Measured on the
  // close-up shot: distinct patchiness to 25 m, featureless green beyond.
  //
  // Perturbing the height blend in *world* space instead is resolution-
  // independent: the same 3.6 m fingering appears at 20 m and at 400 m, because
  // it is evaluated per pixel from the world position rather than sampled from
  // a tile. It also does the job the height blend exists for — it is what turns
  // a smooth boundary into grass reaching into the cracks in the rock.
  const fingerA = fnMacroNoise({ p: vec2(world.x, world.z).mul(1 / 3.6), seed: u.seed });
  const fingerB = fnMacroNoise({
    p: vec2(world.x, world.z).mul(1 / 14),
    seed: u.seed.add(float(911)),
  });
  // Both terms are faded out once their period drops below a handful of pixels.
  // Without this the 3.6 m term aliases into a field of confetti across the
  // whole middle distance — measured on the mountain shot, where at 2 km one
  // pixel covers 1.2 m of ground and the noise gets three pixels per period.
  // This is the same trade the detail normal makes: per-pixel structure that
  // the screen cannot resolve is not detail, it is noise.
  const fingerFadeA = float(1).sub(smoothstep(float(70), float(260), dist));
  const fingerFadeB = float(1).sub(smoothstep(float(280), float(1000), dist));
  const finger = fingerA
    .mul(fingerFadeA.mul(0.45))
    .add(fingerB.mul(fingerFadeB.mul(0.55)))
    .mul(0.22);

  // The blend window widens with distance, and that is not a performance
  // dodge — it is the correct filter. Height blending is winner-takes-all by
  // construction, so it turns a smooth weight field into a binary mosaic at
  // whatever frequency the height maps carry. Up close that mosaic is the whole
  // point (grass between stones). Past a few hundred metres one pixel spans
  // several mosaic cells and the correct answer is their *average* — which is
  // exactly what a wide window degenerates to. Leaving the window narrow at
  // range gave a field of green confetti over grey, visible across the entire
  // middle distance of the mountain shot, and no mip chain can fix it because
  // the mosaic is generated per pixel rather than sampled.
  const blendWidth = mix(
    float(HEIGHT_BLEND_WIDTH),
    float(1.6),
    smoothstep(float(120), float(600), dist),
  );

  If(a1.greaterThan(float(0.02)), () => {
    const l1 = sampleLayer(idx1);
    const d0 = l0.height.add(a0).add(finger);
    const d1 = l1.height.add(a1).sub(finger);
    const cut = max(d0, d1).sub(blendWidth);
    const b0 = max(d0.sub(cut), float(0));
    const b1 = max(d1.sub(cut), float(0));
    const k1 = b1.div(max(b0.add(b1), float(1e-4)));

    albedoMix.assign(mix(l0.albedo, l1.albedo, k1));
    roughMix.assign(mix(l0.rough, l1.rough, k1));
    aoMix.assign(mix(l0.ao, l1.ao, k1));
    pertMix.assign(mix(l0.pert, l1.pert, k1));
  });

  // --- macro variation ---------------------------------------------------
  // Two periods rather than one: a single octave at 70 m gives a regular
  // blotching that reads as a low-resolution lightmap. Adding a 190 m term
  // breaks its rhythm for the cost of one more noise lookup.
  const macroA = fnMacroNoise({ p: vec2(world.x, world.z).mul(1 / 70), seed: u.seed });
  const macroB = fnMacroNoise({
    p: vec2(world.x, world.z).mul(1 / 190),
    seed: u.seed.add(float(5501)),
  });
  const macro = clamp(macroA.mul(0.6).add(macroB.mul(0.4)), float(-1), float(1)).mul(u.macro);
  const albedoOut = albedoMix.mul(float(1).add(macro.mul(0.12)));
  const roughOut = clamp(roughMix.add(macro.mul(0.1)), float(0.04), float(1));

  // --- detail normal into world space -----------------------------------
  // The perturbation is expressed on world axes by construction, so it is added
  // to the baked normal directly. Fading it with distance is not cosmetic: a
  // per-pixel normal at one pixel per 30 m of ground is pure aliasing, and the
  // shimmer it produces is exactly what TAA cannot fix in phase 10.
  const detailStrength = float(1).sub(
    smoothstep(float(NORMAL_FADE_NEAR), float(NORMAL_FADE_FAR), dist),
  );
  const worldNormal = normalize(n.add(pertMix.mul(detailStrength.mul(0.55))));

  return TerrainSurface({
    albedo: albedoOut,
    roughness: roughOut,
    ao: clamp(aoMix, float(0), float(1)),
    normal: worldNormal,
    snow,
    dominant: idx0,
    w0: vec4(w[0], w[1], w[2], w[3]),
    w1: vec4(w[4], w[5], w[6], w[7]),
  });
}

export const TERRAIN_MATERIAL_CONSTANTS = {
  DETAIL_SCALE,
  DETAIL_NEAR,
  DETAIL_FAR,
  STOCH_NEAR,
  STOCH_FAR,
  POM_MAX_M,
  POM_STEPS_MIN,
  POM_STEPS_MAX,
  HEIGHT_BLEND_WIDTH,
  MATERIAL_TEX_METRES,
} as const;
