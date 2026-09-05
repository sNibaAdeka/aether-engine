/**
 * WebGPURenderer setup, backend detection and resize handling.
 *
 * WebGPU is the primary target — the whole engine assumes compute shaders,
 * storage buffers and indirect draw. The WebGL2 path exists so the page does
 * not show a black screen on unsupported hardware, but it is a degraded mode,
 * not a peer: no volumetric clouds, no FFT ocean, no GPU-driven grass.
 */

import { WebGPURenderer } from 'three/webgpu';
import * as THREE from 'three/webgpu';

export type Backend = 'webgpu' | 'webgl2';

export interface RendererInfo {
  backend: Backend;
  adapterName: string;
  architecture: string;
  vendor: string;
  maxStorageBufferSize: number;
  maxComputeWorkgroupsPerDimension: number;
  hasTimestampQuery: boolean;
  hasFloat32Filterable: boolean;
  /** WebGPU "compatibility" feature level — reduced limits, no MSAA. */
  compatibilityMode: boolean;
}

export interface CreateRendererResult {
  renderer: WebGPURenderer;
  info: RendererInfo;
  canvas: HTMLCanvasElement;
}

/** Probe support without creating a renderer. Used by the boot screen. */
export async function probeWebGPU(): Promise<{
  supported: boolean;
  reason?: string;
  adapterInfo?: GPUAdapterInfo;
}> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    return { supported: false, reason: 'navigator.gpu is not available in this browser' };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      return { supported: false, reason: 'No suitable GPU adapter was returned' };
    }
    return { supported: true, adapterInfo: adapter.info };
  } catch (err) {
    return { supported: false, reason: `Adapter request failed: ${String(err)}` };
  }
}

/**
 * Device limits we ask for above the WebGPU defaults, clamped to the adapter.
 *
 * Only one entry so far, and it is not speculative. The phase-2 splat kernel
 * binds five storage textures in one compute stage — height, surface and flow
 * to read, two splat maps to write — against a default
 * `maxStorageTexturesPerShaderStage` of 4. Every M-series and desktop adapter
 * reports 8; the default is a floor set for the weakest conformant hardware.
 *
 * Clamping matters more than raising: `requestDevice` fails outright, with no
 * fallback, if any requested limit exceeds what the adapter reports. So this
 * probes first and asks for the smaller of what we want and what exists, and a
 * machine that genuinely only has 4 simply keeps 4 — and then fails in the
 * splat kernel with a clear validation error rather than at boot with none.
 */
async function requestedLimits(): Promise<Record<string, number>> {
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return {};
    const want = { maxStorageTexturesPerShaderStage: 8 };
    const out: Record<string, number> = {};
    for (const [key, value] of Object.entries(want)) {
      const available = (adapter.limits as unknown as Record<string, number>)[key];
      if (typeof available === 'number' && available >= value) out[key] = value;
      else if (typeof available === 'number') out[key] = available;
    }
    return out;
  } catch {
    return {};
  }
}

export interface CreateRendererOptions {
  canvas?: HTMLCanvasElement;
  forceWebGL?: boolean;
  /** Disable the WebGL2 fallback and fail loudly instead. */
  strictWebGPU?: boolean;
}

export async function createRenderer(
  opts: CreateRendererOptions = {},
): Promise<CreateRendererResult> {
  const canvas = opts.canvas ?? document.createElement('canvas');
  canvas.id = 'aether-canvas';

  const probe = opts.forceWebGL ? { supported: false } : await probeWebGPU();
  const wantWebGPU = probe.supported;

  if (!wantWebGPU && opts.strictWebGPU) {
    throw new Error('WebGPU is required but unavailable');
  }

  const requiredLimits = wantWebGPU ? await requestedLimits() : {};

  const renderer = new WebGPURenderer({
    canvas,
    // Raised above the WebGPU defaults where the adapter allows it — see
    // `requestedLimits`. Asking for a limit the adapter cannot meet fails the
    // device request outright, so every entry is clamped to what was probed.
    requiredLimits,
    antialias: false, // we do our own AA (TAA/FXAA) in the post stack
    alpha: false,
    forceWebGL: !wantWebGPU,
    powerPreference: 'high-performance',
    // The swap chain needs its own depth attachment. From phase 10 the post
    // stack renders into an offscreen target and the default framebuffer only
    // receives a tonemapped fullscreen triangle — but until then we draw
    // geometry directly here, and turning depth off silently degrades the
    // scene to painter's-algorithm draw order.
    depth: true,
    stencil: false,
    // Enables `resolveTimestampsAsync` — real GPU-side pass durations rather
    // than CPU submit times, which on a deep pipeline differ by an order of
    // magnitude and would make every budget number in PROGRESS.md a lie.
    trackTimestamp: true,
  });

  // Physically-based pipeline: everything is linear until the tonemap node at
  // the very end of the post stack, which writes sRGB.
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping; // handled in PostStack
  renderer.setClearColor(0x000000, 1);

  await renderer.init();

  // `isWebGPUBackend` is declared on WebGPUBackend but `renderer.backend` is
  // typed as the abstract base, so the discriminant needs a narrowing cast.
  const isWebGPU = (renderer.backend as unknown as { isWebGPUBackend?: boolean }).isWebGPUBackend;
  const backend: Backend = isWebGPU === true ? 'webgpu' : 'webgl2';

  const info = await collectInfo(renderer, backend, probe.adapterInfo);

  console.info(
    `[aether] backend=${info.backend} adapter="${info.adapterName}" vendor=${info.vendor} ` +
      `arch=${info.architecture} timestampQuery=${info.hasTimestampQuery} ` +
      `compat=${info.compatibilityMode}`,
  );

  // three routes both device loss and uncaptured GPU errors through these hooks.
  // The DOM `webglcontextlost` event only fires on the WebGL2 fallback path, so
  // relying on it alone would silently miss every WebGPU device loss.
  renderer.onDeviceLost = (loss: { message?: string; reason?: string | null }): void => {
    console.error(`[aether] GPU device lost: ${loss.reason ?? 'unknown'} — ${loss.message ?? ''}`);
    window.dispatchEvent(new CustomEvent('aether:context-lost', { detail: loss }));
  };
  renderer.domElement.addEventListener('webglcontextlost', onContextLost, false);

  return { renderer, info, canvas: renderer.domElement as HTMLCanvasElement };
}

function onContextLost(e: Event): void {
  e.preventDefault();
  console.error('[aether] graphics context lost');
  window.dispatchEvent(new CustomEvent('aether:context-lost'));
}

async function collectInfo(
  renderer: WebGPURenderer,
  backend: Backend,
  probedAdapterInfo?: GPUAdapterInfo,
): Promise<RendererInfo> {
  const info: RendererInfo = {
    backend,
    adapterName: 'unknown',
    architecture: 'unknown',
    vendor: 'unknown',
    maxStorageBufferSize: 0,
    maxComputeWorkgroupsPerDimension: 0,
    hasTimestampQuery: false,
    hasFloat32Filterable: false,
    compatibilityMode: false,
  };

  if (backend === 'webgpu') {
    // three does not retain the GPUAdapter, only the GPUDevice. Prefer the
    // device's own `adapterInfo` and fall back to the adapter we probed before
    // constructing the renderer.
    const be = renderer.backend as unknown as {
      device?: GPUDevice;
      compatibilityMode?: boolean;
    };
    const device = be.device;
    const ai =
      (device as unknown as { adapterInfo?: GPUAdapterInfo })?.adapterInfo ?? probedAdapterInfo;

    if (ai) {
      info.adapterName = ai.device || ai.description || ai.architecture || 'unnamed';
      info.architecture = ai.architecture || 'unknown';
      info.vendor = ai.vendor || 'unknown';
    }
    if (device) {
      info.maxStorageBufferSize = device.limits.maxStorageBufferBindingSize;
      info.maxComputeWorkgroupsPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
      info.hasTimestampQuery = device.features.has('timestamp-query');
      info.hasFloat32Filterable = device.features.has('float32-filterable');
      // three requests `featureLevel: 'compatibility'`. When the device comes
      // back without `core-features-and-limits` we are in WebGPU compatibility
      // mode, which caps MSAA and tightens several binding limits — worth
      // knowing before blaming a later phase for a mysterious validation error.
      info.compatibilityMode = !device.features.has('core-features-and-limits');
    }
  } else {
    const gl = renderer.getContext() as WebGL2RenderingContext | null;
    if (gl) {
      const dbg = gl.getExtension('WEBGL_debug_renderer_info');
      if (dbg) {
        info.adapterName = String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL));
        info.vendor = String(gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL));
      }
    }
  }
  return info;
}

/**
 * Owns canvas sizing. Separated from the renderer so the render-scale
 * controller and the settings panel have one place to poke.
 */
export class ResizeController {
  /** CSS pixel size of the canvas. */
  width = 1;
  height = 1;
  /** Backing-store size actually rendered. */
  renderWidth = 1;
  renderHeight = 1;

  private pixelRatioCap = 2;
  private observer: ResizeObserver | null = null;
  private onResizeCb: ((w: number, h: number) => void) | null = null;

  constructor(
    private renderer: WebGPURenderer,
    private container: HTMLElement,
  ) {}

  start(onResize: (w: number, h: number) => void): void {
    this.onResizeCb = onResize;
    this.observer = new ResizeObserver(() => this.apply());
    this.observer.observe(this.container);
    window.addEventListener('resize', this.apply);
    this.apply();
  }

  /** Render scale from Settings; re-applied whenever it changes. */
  renderScale = 1.0;

  setRenderScale(scale: number): void {
    if (Math.abs(scale - this.renderScale) < 1e-4) return;
    this.renderScale = scale;
    this.apply();
  }

  private apply = (): void => {
    const rect = this.container.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    const dpr = Math.min(window.devicePixelRatio || 1, this.pixelRatioCap);

    this.width = w;
    this.height = h;
    this.renderWidth = Math.max(1, Math.floor(w * dpr * this.renderScale));
    this.renderHeight = Math.max(1, Math.floor(h * dpr * this.renderScale));

    // setSize with updateStyle=false, then drive the backing store via
    // setPixelRatio so CSS size and render size stay decoupled.
    this.renderer.setSize(w, h, false);
    this.renderer.setPixelRatio(dpr * this.renderScale);

    const canvas = this.renderer.domElement as HTMLCanvasElement;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    this.onResizeCb?.(this.renderWidth, this.renderHeight);
  };

  dispose(): void {
    this.observer?.disconnect();
    window.removeEventListener('resize', this.apply);
  }
}
