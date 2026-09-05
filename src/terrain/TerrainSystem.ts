/**
 * CDLOD terrain: one grid patch, instanced across every selected quadtree node,
 * displaced in the vertex shader by a *baked* heightmap tile.
 *
 * One geometry, one material, one draw call for the entire visible world. The
 * quadtree runs on the CPU (a few hundred nodes, no allocation) and uploads a
 * flat instance buffer; everything else happens on the GPU.
 *
 * The morph is the part that earns its keep. Each patch vertex slides toward
 * the position it would occupy at the parent level as the camera retreats, so
 * by the time a node actually switches level it has already converged to its
 * parent's silhouette — no popping, and therefore no need to hide popping
 * behind fog. Since phase 1b the *height* is morphed too, by blending the
 * node's own baked tile against its parent's: with erosion coming, those two
 * tiles are genuinely different surfaces and blending only the XZ positions
 * would leave a step at every level switch.
 *
 * What changed in 1b, and why: the vertex shader used to call the analytic
 * height function five times per vertex (once for the position, four more for
 * central-difference normals), every frame. That cost 10.6 ms against a 2.0 ms
 * budget. It now reads four texture samples instead — own-level height, parent
 * height, own-level normal, parent normal — from tiles baked once by
 * `HeightmapGenerator` and kept resident by `TilePool`/`TileManager`.
 */

import * as THREE from 'three/webgpu';
import {
  Fn,
  attribute,
  positionGeometry,
  cameraPosition,
  uniform,
  varying,
  int,
  texture,
  vec2,
  vec3,
  float,
  fract,
  clamp,
  smoothstep,
  mix,
  normalize,
  length,
  round,
  sqrt,
} from 'three/tsl';

import type { EngineContext, FrameContext, System } from '@/engine/types';
import { Quadtree, NODE_STRIDE, type QuadtreeConfig } from './Quadtree';
import { Frustum } from './Frustum';
import { fnTerrainHeight } from './shaders/heightfield.wgsl';
import { buildDebugColor, type ShaderNode } from './shaders/debugViews';
import { TilePool, TILE_CELLS, TILE_UV_BIAS, TILE_UV_SCALE } from './TilePool';
import { TileManager } from './TileManager';
import { HeightmapGenerator } from './HeightmapGenerator';
import { Erosion } from './Erosion';
import { SplatGenerator } from './SplatGenerator';
import { MaterialTextures } from './MaterialTextures';
import { SnowLine } from './SnowLine';
import {
  buildTerrainMaterial,
  createTerrainMaterialUniforms,
  type TerrainMaterialUniforms,
} from './TerrainMaterial';

/** Vertices along one patch edge. 33 → 32×32 quads → 2048 triangles. */
const PATCH_RES = 33;

/**
 * The frame budget's "generation" slice, milliseconds (CLAUDE.md).
 *
 * Since phase 1b task 2 this line covers two consumers: the analytic height
 * bake and the erosion pass. They are budgeted together rather than given a row
 * each because they compete for the same compute queue and the same visual
 * goal — getting a correct surface in front of the camera.
 */
const GENERATION_BUDGET_MS = 1.0;

export interface TerrainDebugState {
  nodes: number;
  triangles: number;
  maxLevel: number;
  visited: number;
  culled: number;
  /** Tiles baked and resident. */
  tilesReady: number;
  tilesPending: number;
  /** Nodes drawing off a coarser ancestor because their own tile is not baked. */
  tilesFallback: number;
  tilesBaked: number;
  /** Tiles whose erosion pass has completed. */
  tilesEroded: number;
  /** Progress of the tile currently being eroded, 0..1; negative when idle. */
  erosionProgress: number;
  /** Ready tiles whose biome mask has been computed for their own extent. */
  tilesSplatted: number;
  /** Current snow-line altitude, metres. */
  snowLine: number;
  /** Where the snow line is heading, metres. */
  snowLineTarget: number;
}

export class TerrainSystem implements System {
  readonly name = 'terrain';

  readonly quadtree: Quadtree;
  readonly frustum = new Frustum();

  pool!: TilePool;
  tiles!: TileManager;
  generator!: HeightmapGenerator;
  erosion?: Erosion;
  splat?: SplatGenerator;
  materialTextures?: MaterialTextures;
  readonly snowLine = new SnowLine();
  /**
   * The world clock, set by `main.ts` once the atmosphere exists. The snow line
   * follows it rather than integrating its own season and hour — see
   * `SnowLine.update`.
   */
  clock?: { season: number; hours: number; timeScale: number };

  /**
   * True once the erosion pass has written real data into `flowArray`.
   *
   * The flow and moisture debug views refuse to switch on until this is set, so
   * nobody mistakes an all-zero texture for "erosion produced a flat map".
   */
  erosionDataAvailable = false;

  private mesh!: THREE.Mesh;
  private geometry!: THREE.InstancedBufferGeometry;
  private material!: THREE.MeshStandardNodeMaterial;
  private interleaved!: THREE.InstancedInterleavedBuffer;

  private uSeed = uniform(1337);
  private uMorphEnabled = uniform(1);
  private uLodColors = uniform(0);
  /** Debug visualisation mode, driven by the number row. */
  readonly uDebugView = uniform(0);
  /** Material knobs, exposed so the A/B measurement scripts can toggle them. */
  readonly materialUniforms: TerrainMaterialUniforms = createTerrainMaterialUniforms(1337);

  private ctx!: EngineContext;
  private disposables: Array<{ dispose(): void }> = [];
  /** False on the WebGL2 fallback, where there is no compute stage to bake with. */
  private baked = true;

  readonly debugState: TerrainDebugState = {
    nodes: 0,
    triangles: 0,
    maxLevel: 0,
    visited: 0,
    culled: 0,
    tilesReady: 0,
    tilesPending: 0,
    tilesFallback: 0,
    tilesBaked: 0,
    tilesEroded: 0,
    erosionProgress: -1,
    tilesSplatted: 0,
    snowLine: 0,
    snowLineTarget: 0,
  };

  private _vpMatrix = new THREE.Matrix4();

  constructor(config: Partial<QuadtreeConfig> = {}) {
    this.quadtree = new Quadtree({ patchResolution: PATCH_RES, ...config });
  }

  async init(ctx: EngineContext): Promise<void> {
    this.ctx = ctx;
    this.uSeed.value = ctx.seed;
    this.materialUniforms.seed.value = ctx.seed;
    this.baked = ctx.hasCompute;
    this.snowLine.snap();
    this.materialUniforms.snowLine.value = this.snowLine.state.current;

    this.pool = new TilePool(ctx.resources);
    this.tiles = new TileManager(this.pool);

    if (this.baked) {
      // r32float is only hardware-filterable behind an optional WebGPU feature.
      // Without it three routes the sample through its own biquadratic fallback,
      // which hard-codes *repeat* wrapping — tiles would wrap their far edge
      // back over the near one and crease at every boundary.
      // TODO(quality): no fallback tier is implemented for that case; the check
      // exists so the failure is diagnosable rather than mysterious.
      if (!ctx.renderer.hasFeature('float32-filterable')) {
        console.warn(
          '[aether] float32-filterable missing — heightmap tiles will be sampled ' +
            'through a repeat-wrapped fallback and tile borders will crease',
        );
      }
      this.generator = new HeightmapGenerator(ctx.renderer, this.pool, ctx.profiler, ctx.seed);
      this.erosion = new Erosion(
        ctx.renderer,
        this.pool,
        ctx.profiler,
        ctx.resources,
        ctx.seed,
      );
      this.splat = new SplatGenerator(ctx.renderer, this.pool, ctx.profiler, ctx.seed);
      this.materialTextures = new MaterialTextures(ctx.renderer, ctx.resources, ctx.seed);
    } else {
      console.warn('[aether] no compute backend — terrain falls back to analytic vertex heights');
    }

    this.geometry = this.buildPatchGeometry();
    this.material = this.buildMaterial();

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'terrain';
    // Patches are repositioned per instance in the vertex shader, so three's
    // own frustum test would evaluate a meaningless bounding box and drop the
    // world at random angles. Culling is the quadtree's job.
    this.mesh.frustumCulled = false;
    this.mesh.receiveShadow = true;
    this.mesh.castShadow = true;
    ctx.scene.add(this.mesh);

    this.applySettings();
    ctx.settings.onChange(() => this.applySettings());

    // Bake the root tile before the first frame. Every node in the world is a
    // descendant of it, so once it exists `TilePool.resolve` can never fail to
    // return a surface — which is the entire basis of the no-holes guarantee.
    if (this.baked) {
      // Synthesise the biome PBR set before anything samples it. One submit per
      // biome, all of it before the first frame — the alternative is a material
      // that reads an uninitialised array on frame 1 and shows eight layers of
      // whatever the driver left in that memory.
      this.materialTextures?.generate();

      this.pool.beginFrame(0);
      this.pool.pin(this.quadtree.config.maxDepth, 0, 0);
      this.generator.flush(1);
      // The root tile is now drawable, so it needs its biome mask in the same
      // breath — see SplatGenerator's header for why "in the same frame" is a
      // correctness requirement and not a nicety.
      this.splat?.flush(1);
    }
  }

  private applySettings(): void {
    const s = this.ctx.settings;
    this.quadtree.setConfig({
      drawDistance: s.quality.drawDistance,
      maxDepth: s.quality.lodDepth,
    });
    // A (level, gx, gz) triple only names a fixed world rectangle for a given
    // root size and depth, so this flushes the pool when lodDepth changes.
    this.pool.configure(this.quadtree.config.rootSize, this.quadtree.config.maxDepth);
    this.uLodColors.value = s.debug.showLodColors ? 1 : 0;
    this.material.wireframe = s.debug.wireframe;
  }

  /**
   * The single reusable patch: a unit grid in XZ, with grid indices carried as
   * an attribute because the morph needs to know which vertices are the "odd"
   * ones that collapse onto their even neighbours.
   */
  private buildPatchGeometry(): THREE.InstancedBufferGeometry {
    const n = PATCH_RES;
    const verts = n * n;
    const positions = new Float32Array(verts * 3);
    const gridIdx = new Float32Array(verts * 2);

    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const k = j * n + i;
        positions[k * 3] = i / (n - 1);
        positions[k * 3 + 1] = 0;
        positions[k * 3 + 2] = j / (n - 1);
        gridIdx[k * 2] = i;
        gridIdx[k * 2 + 1] = j;
      }
    }

    const quads = (n - 1) * (n - 1);
    const indices = new Uint32Array(quads * 6);
    let o = 0;
    for (let j = 0; j < n - 1; j++) {
      for (let i = 0; i < n - 1; i++) {
        const a = j * n + i;
        const b = a + 1;
        const c = a + n;
        const d = c + 1;
        indices[o++] = a;
        indices[o++] = c;
        indices[o++] = b;
        indices[o++] = b;
        indices[o++] = c;
        indices[o++] = d;
      }
    }

    const geo = new THREE.InstancedBufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('aGrid', new THREE.BufferAttribute(gridIdx, 2));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));

    // A vertex attribute is at most 4 components, and a node descriptor is 16
    // floats, so the quadtree's flat buffer is bound as one interleaved buffer
    // with four vec4 views. No copy, no per-frame repacking.
    this.interleaved = new THREE.InstancedInterleavedBuffer(
      this.quadtree.instanceData,
      NODE_STRIDE,
      1,
    );
    this.interleaved.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('aNode', new THREE.InterleavedBufferAttribute(this.interleaved, 4, 0));
    geo.setAttribute('aMorph', new THREE.InterleavedBufferAttribute(this.interleaved, 4, 4));
    geo.setAttribute('aTileChild', new THREE.InterleavedBufferAttribute(this.interleaved, 4, 8));
    geo.setAttribute('aTileParent', new THREE.InterleavedBufferAttribute(this.interleaved, 4, 12));
    geo.instanceCount = 0;

    // Unused for culling, but three still wants a sphere for sorting; make it
    // enormous rather than let it be derived from the unit patch.
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 1e7);

    this.disposables.push(geo);
    return geo;
  }

  private buildMaterial(): THREE.MeshStandardNodeMaterial {
    const mat = new THREE.MeshStandardNodeMaterial();
    mat.roughness = 0.94;
    mat.metalness = 0.0;

    const seed = this.uSeed;
    const gridDim = float(PATCH_RES - 1);

    // aNode       = [originX, originZ, size, level]
    // aMorph      = [morphStart, morphEnd, layerChild, layerParent]
    // aTileChild  = [uvOffsetX, uvOffsetZ, uvScale, fallbackFlag]
    // aTileParent = [uvOffsetX, uvOffsetZ, uvScale, reserved]
    const node = attribute('aNode', 'vec4');
    const morphAttr = attribute('aMorph', 'vec4');
    const tileC = attribute('aTileChild', 'vec4');
    const tileP = attribute('aTileParent', 'vec4');
    const grid = attribute('aGrid', 'vec2');

    const origin = node.xy;
    const size = node.z;
    const level = node.w;

    const local = positionGeometry.xz; // [0,1]²
    const worldFlat = origin.add(local.mul(size));

    // --- CDLOD morph ---------------------------------------------------
    // Distance is measured to the *unmorphed* position; using the morphed one
    // would make the blend depend on its own output.
    //
    // The vertical term is not decoration: `Quadtree.selectNode` decides which
    // level a node is emitted at using sqrt(distFlat² + camY²/4), and the morph
    // has to complete by the distance at which the parent takes over. Measuring
    // the morph on the horizontal distance alone (as phase 1a did) puts the
    // level switch *ahead* of the morph whenever the camera is high — the patch
    // hands over to its coarser neighbour before it has converged on the
    // parent's lattice. That was invisible while both levels sampled the same
    // analytic surface; once erosion makes them genuinely different surfaces it
    // is a hairline crack you can see the sky through, on every LOD ring.
    const camXZ = vec2(cameraPosition.x, cameraPosition.z);
    const distFlat = length(worldFlat.sub(camXZ));
    const dist = sqrt(
      distFlat.mul(distFlat).add(cameraPosition.y.mul(cameraPosition.y).mul(0.25)),
    );
    const morphK = smoothstep(morphAttr.x, morphAttr.y, dist).mul(this.uMorphEnabled);

    // Odd grid vertices slide onto their even neighbours — which is exactly the
    // parent level's vertex set, so the two levels agree at the moment of the
    // switch and the transition is invisible.
    const fracPart = fract(grid.mul(0.5)).mul(2).div(gridDim);
    const morphedLocal = local.sub(fracPart.mul(morphK));
    const world = origin.add(morphedLocal.mul(size));

    // --- Tile sampling --------------------------------------------------
    // A tap's tile is not necessarily this node's own extent: when the node's
    // tile is not baked yet it borrows an ancestor's, and the offset/scale pair
    // selects the right sub-rectangle. Both taps are fed the *morphed* local
    // coordinate, the same one the vertex position uses — feeding one of them
    // the unmorphed coordinate would reintroduce exactly the pop the morph
    // exists to remove.
    const localC = tileC.xy.add(morphedLocal.mul(tileC.z));
    const localP = tileP.xy.add(morphedLocal.mul(tileP.z));
    // Shared-edge texel mapping: patch-local p sits on texel p*(N-1), and
    // bilinear resolves uv to texel index u*N - 0.5, hence (p*(N-1) + 0.5)/N.
    const uvC = localC.mul(TILE_UV_SCALE).add(TILE_UV_BIAS);
    const uvP = localP.mul(TILE_UV_SCALE).add(TILE_UV_BIAS);
    const layerC = int(morphAttr.z);
    const layerP = int(morphAttr.w);

    // Named so the baked and the analytic paths converge on one type instead of
    // a union of two shader-graph shapes.
    let h: ShaderNode;
    let normalVS: ShaderNode;
    let flow: ShaderNode;
    let moisture: ShaderNode;
    let splat0: ShaderNode = null;
    let splat1: ShaderNode = null;

    if (this.baked) {
      const heightArray = this.pool.heightArray;
      const surfaceArray = this.pool.surfaceArray;
      const flowArray = this.pool.flowArray;

      // --- vertex stage: height only -----------------------------------
      // `.level(0)` is mandatory, not an optimisation: a plain sample would
      // need screen-space derivatives, which do not exist in a vertex shader.
      const hC = texture(heightArray, uvC).depth(layerC).level(float(0)).x;
      const hP = texture(heightArray, uvP).depth(layerP).level(float(0)).x;
      h = mix(hC, hP, morphK);

      // --- fragment stage: normal, flow, moisture -----------------------
      //
      // These are sampled PER PIXEL, and that is the whole reason erosion is
      // visible at all. A tile is 129² texels; a patch is 33² vertices, i.e.
      // one vertex per four texels. Erosion carves channels three to six texels
      // wide, so sampling the normal and flow maps in the vertex shader and
      // interpolating — which is what this did until now — throws away 15 of
      // every 16 texels and bilinearly smears the rest. Measured on a level-4
      // tile: the drainage network is unmistakably dendritic at texel
      // resolution and reduces to a featureless mid-grey wash on the vertex
      // lattice. That single mistake produced both "erosion does nothing" (the
      // shading normal never saw the channels) and "mode 7 is a flat wash".
      //
      // Only the *geometry* is limited to the vertex lattice. Displacement
      // stays per-vertex — it has to — so the silhouette still carries erosion
      // only down to the 4-texel vertex spacing, while the shading carries all
      // of it. That is the standard CDLOD split and it is why the patch does
      // not need to be subdivided further.
      const vUvC = varying(uvC);
      const vUvP = varying(uvP);
      const vMorphK = varying(morphK);
      // Layer indices are per-instance constants, so interpolating them across
      // a triangle is exact — but the barycentric sum is only exact to f32, so
      // round before truncating to int rather than letting 42.99999 become 42.
      const layerCf = int(round(varying(morphAttr.z)));
      const layerPf = int(round(varying(morphAttr.w)));

      // Normals come from the same pair of tiles as the height, so they morph
      // in lockstep with it. They were baked from central differences of the
      // baked heightmap, which means they describe the surface actually being
      // drawn — including whatever erosion carved into it.
      const nC = texture(surfaceArray, vUvC).depth(layerCf).level(float(0)).xyz;
      const nP = texture(surfaceArray, vUvP).depth(layerPf).level(float(0)).xyz;
      normalVS = normalize(mix(nC, nP, vMorphK));

      // flowArray packing (see Erosion's resolve pass):
      //   r flow accumulation · g flow direction angle · b moisture · a cut/fill
      // `g` is deliberately skipped here — it is an angle that wraps at ±π, so
      // bilinear filtering across the wrap gives a meaningless average. Phase 6
      // point-samples it on the CPU when it traces river courses.
      const flowSample = texture(flowArray, vUvC).depth(layerCf).level(float(0));
      flow = flowSample.x;
      moisture = flowSample.z;

      // Biome weights, sampled per pixel from the node's own tile only.
      //
      // Deliberately NOT morphed against the parent tile the way height and
      // normal are. Morphing would blend two classifications sampled at
      // different resolutions, and because the weights are then run through a
      // top-two selection and a height blend, a 50/50 mix of two different
      // rankings does not produce anything between them — it produces a third
      // material. A hard switch at the LOD boundary is invisible (the biome
      // fields are continuous in world space, so both tiles classify the same
      // ground almost identically); a blend of rankings is not.
      splat0 = texture(this.pool.splat0Array, vUvC).depth(layerCf).level(float(0));
      splat1 = texture(this.pool.splat1Array, vUvC).depth(layerCf).level(float(0));
    } else {
      // WebGL2 fallback: no compute stage, so nothing can bake. This is the
      // pre-1b path, kept only so the degraded backend still shows a world.
      // TODO(quality): it costs five height evaluations per vertex and misses
      // erosion entirely — the WebGL2 tier is a courtesy, not a supported mode.
      h = fnTerrainHeight({ pos: world, seed });
      const eps = size.div(gridDim);
      const hL = fnTerrainHeight({ pos: world.sub(vec2(eps, 0)), seed });
      const hR = fnTerrainHeight({ pos: world.add(vec2(eps, 0)), seed });
      const hD = fnTerrainHeight({ pos: world.sub(vec2(0, eps)), seed });
      const hU = fnTerrainHeight({ pos: world.add(vec2(0, eps)), seed });
      // Analytic normals are a function of the vertex position, so on this path
      // they genuinely belong in the vertex stage — five height evaluations per
      // *pixel* is not a trade anyone wants. `varying()` here keeps the two
      // branches interface-compatible: both hand a fragment-stage node onward.
      normalVS = varying(normalize(vec3(hL.sub(hR), eps.mul(2), hD.sub(hU))));
      flow = float(0);
      moisture = float(0);
    }

    const worldPos = vec3(world.x, h, world.y);

    mat.positionNode = worldPos;

    // `varying()` forces evaluation into the vertex stage and interpolates the
    // result, so these are computed once per vertex rather than per fragment.
    // Note what is NOT in this list: the normal and the flow/moisture pair.
    // Those are sampled per pixel above — see the comment there.
    const vHeight = varying(h);
    const vLevel = varying(level);
    const vMorph = varying(morphK);
    const vWorld = varying(worldPos);
    const vPatchUV = varying(morphedLocal);
    // Position in texels of the tile actually sampled — mode 6 draws this, so
    // the grid it shows is the real lattice and not a decorative one.
    const vTexel = varying(localC.mul(TILE_CELLS));
    const vFallback = varying(tileC.w);

    // Legacy F4 "LOD colours" toggle, kept because it composes with the shaded
    // view; the number-row modes replace the shading entirely.
    const lodTint = vec3(
      fract(vLevel.mul(0.37)),
      fract(vLevel.mul(0.61).add(0.33)),
      fract(vLevel.mul(0.83).add(0.66)),
    ).mul(vMorph.mul(0.5).add(0.5));

    let shaded: ShaderNode;
    let snowCoverage: ShaderNode = float(0);
    let weights: ShaderNode[] | undefined;
    let dominant: ShaderNode = float(0);

    if (this.baked && this.materialTextures) {
      const surface = buildTerrainMaterial({
        worldPos: vWorld,
        normal: normalVS,
        splat0,
        splat1,
        tex: this.materialTextures,
        u: this.materialUniforms,
      });

      // `colorNode` on a standard material is the *albedo*, not the final
      // colour: three runs it through the lighting model. Which is the point —
      // phase 3 replaces the provisional sun and hemisphere with a physical
      // atmosphere and this material picks that up for free.
      shaded = mix(surface.albedo, lodTint, this.uLodColors);
      mat.roughnessNode = surface.roughness;
      // Ambient occlusion goes in the dedicated slot rather than being
      // multiplied into the albedo, because three applies `aoNode` to indirect
      // light only. Multiplying it into the colour would darken the direct sun
      // as well, which is the classic way to make an AO pass look like dirt.
      mat.aoNode = surface.ao;
      mat.normalNode = surface.normal;

      snowCoverage = surface.snow;
      weights = surface.weights;
      dominant = surface.dominant;
    } else {
      // WebGL2 fallback: no compute stage, so there are no baked splat maps and
      // no synthesised PBR set. Flat slope-based tinting, exactly as phase 1
      // had it.
      // TODO(quality): the WebGL2 tier gets none of phase 2. It is a courtesy
      // path, not a supported mode.
      mat.normalNode = normalize(normalVS);
      shaded = Fn(() => {
        const nrm = normalize(normalVS);
        const slope = float(1).sub(clamp(nrm.y, float(0), float(1)));
        const rock = vec3(0.3, 0.29, 0.28);
        const soil = vec3(0.44, 0.42, 0.37);
        const base = mix(soil, rock, smoothstep(0.22, 0.6, slope));
        const snowAmount = smoothstep(520, 780, vHeight).mul(smoothstep(0.5, 0.22, slope));
        return mix(mix(base, vec3(0.88, 0.9, 0.94), snowAmount), lodTint, this.uLodColors);
      })();
    }

    mat.colorNode = buildDebugColor({
      mode: this.uDebugView,
      shaded,
      worldPos: vWorld,
      normal: normalVS,
      level: vLevel,
      morphK: vMorph,
      patchUV: vPatchUV,
      texelCoord: vTexel,
      tileFallback: vFallback,
      // Per-pixel taps of the flow array, so mode 7 draws the drainage network
      // at the resolution it was simulated at. They read zero until a tile's
      // erosion job finishes, which is why `erosionDataAvailable` still gates
      // the two views that consume them.
      flow,
      moisture,
      weights,
      dominant,
      snow: snowCoverage,
    });

    this.disposables.push(mat);
    return mat;
  }

  update(dt: number, ctx: FrameContext): void {
    const cam = this.ctx.camera as THREE.PerspectiveCamera;
    const settings = this.ctx.settings;

    // The snow line moves whether or not anything is being baked: it is a
    // property of the climate, not of the streamer.
    this.snowLine.update(dt, this.clock);
    this.materialUniforms.snowLine.value = this.snowLine.state.current;
    this.debugState.snowLine = this.snowLine.state.current;
    this.debugState.snowLineTarget = this.snowLine.state.target;

    this.frustum.frozen = settings.debug.freezeCulling;
    if (!this.frustum.frozen) {
      cam.updateMatrixWorld();
      this._vpMatrix.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
      this.frustum.setFromProjectionMatrix(this._vpMatrix);
    }

    const count = this.quadtree.select(
      ctx.cameraPosition.x,
      ctx.cameraPosition.y,
      ctx.cameraPosition.z,
      this.frustum,
    );

    if (this.baked) {
      // Wall clock, not `ctx.elapsed`: the shader clock wraps every ~3600 s and
      // an LRU that goes backwards in time would evict the entire pool at once.
      this.pool.beginFrame(performance.now() * 0.001);
      // Re-pin every frame so the root survives a settings-driven pool flush.
      this.pool.pin(this.quadtree.config.maxDepth, 0, 0);

      const data = this.quadtree.instanceData;
      this.tiles.request(data, count, ctx.cameraPosition.x, ctx.cameraPosition.z);
      // Bake before resolving: both compute passes are submitted here, ahead of
      // the frame graph's render pass, so a tile finished this frame is legal
      // to sample in this frame's draw.
      const baked = this.generator.flush(
        this.generator.tilesThisFrame(settings.quality.tileBudget),
      );
      // Classify immediately after the bake and before `resolve` hands layer
      // indices to the draw. A tile that becomes visible this frame without its
      // biome mask samples the previous tenant's weights, which is not a subtle
      // artefact — it paints a piece of meadow onto a piece of desert. Two
      // extra tiles of headroom on top of `baked` let the erosion-triggered
      // refresh queue drain without ever delaying a fresh one.
      const classified = this.splat ? this.splat.flush(baked + 2) : 0;
      this.tiles.resolve(data, count);

      // Erosion and the height bake share the frame budget's 1.0 ms generation
      // slice, and the bake gets first claim: an un-eroded tile is a surface a
      // node can draw, while erosion only improves one that already exists.
      // Erosion therefore runs on the remainder and idles entirely during a
      // streaming burst.
      if (this.erosion) {
        this.erosion.gpuBudgetMs = Math.max(
          0,
          GENERATION_BUDGET_MS -
            baked * this.generator.estimatedMsPerTile -
            classified * (this.splat?.estimatedMsPerTile ?? 0),
        );
        this.erosion.update();
        this.erosionDataAvailable = this.erosion.hasResults;
        this.debugState.erosionProgress = this.erosion.stats.progress;
      }

      this.pool.updateStats();

      this.debugState.tilesReady = this.pool.stats.ready;
      this.debugState.tilesPending = this.pool.stats.pending;
      this.debugState.tilesFallback = this.tiles.stats.fallback;
      this.debugState.tilesBaked = this.generator.bakedTotal;
      this.debugState.tilesEroded = this.pool.stats.eroded;
      this.debugState.tilesSplatted = this.pool.stats.splatted;
    }

    this.geometry.instanceCount = count;
    this.interleaved.needsUpdate = true;

    const s = this.quadtree.stats;
    this.debugState.nodes = s.nodesSelected;
    this.debugState.triangles = s.trianglesEstimate;
    this.debugState.maxLevel = s.maxLevelUsed;
    this.debugState.visited = s.nodesVisited;
    this.debugState.culled = s.nodesCulled;
  }

  dispose(): void {
    this.mesh?.removeFromParent();
    this.erosion?.dispose();
    this.splat?.dispose();
    this.materialTextures?.dispose();
    this.generator?.dispose();
    this.pool?.dispose();
    for (const d of this.disposables) d.dispose();
    this.disposables.length = 0;
  }
}
