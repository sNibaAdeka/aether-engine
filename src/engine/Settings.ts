/**
 * Quality presets and live settings.
 *
 * Every renderer knob lives here so the settings panel, the auto-benchmark and
 * the dynamic resolution controller all mutate one authoritative object.
 * Systems subscribe to changes rather than polling.
 */

export type QualityPreset = 'potato' | 'low' | 'medium' | 'high' | 'ultra';
export type CloudMode = 'flat' | 'quarter' | 'half' | 'half128';
export type WaterMode = 'gerstner4' | 'gerstner8' | 'fft256' | 'fft512';
export type AAMode = 'none' | 'fxaa' | 'taa' | 'taa2x';
export type SsrMode = 'off' | 'water' | 'waterWet' | 'all';
export type AoMode = 'off' | 'half' | 'full';
export type TonemapMode = 'agx' | 'aces' | 'reinhard' | 'none';

export interface QualitySettings {
  /** Internal render resolution multiplier. */
  renderScale: number;
  /** Terrain draw distance, metres. */
  drawDistance: number;
  shadowCascades: number;
  shadowResolution: number;
  clouds: CloudMode;
  grassRadius: number;
  grassDensity: number;
  water: WaterMode;
  gtao: AoMode;
  ssr: SsrMode;
  /** 0 = off, 1 = low, 2 = normal, 3 = high. */
  volumetricLight: number;
  antialias: AAMode;
  /** Max terrain tiles generated per frame. */
  tileBudget: number;
  /** CDLOD max depth. Lower = coarser world. */
  lodDepth: number;
}

export const PRESETS: Record<QualityPreset, QualitySettings> = {
  potato: {
    renderScale: 0.5,
    drawDistance: 1200,
    shadowCascades: 2,
    shadowResolution: 1024,
    clouds: 'flat',
    grassRadius: 12,
    grassDensity: 0.15,
    water: 'gerstner4',
    gtao: 'off',
    ssr: 'off',
    volumetricLight: 0,
    antialias: 'none',
    tileBudget: 1,
    lodDepth: 6,
  },
  low: {
    renderScale: 0.65,
    drawDistance: 2000,
    shadowCascades: 2,
    shadowResolution: 1024,
    clouds: 'flat',
    grassRadius: 25,
    grassDensity: 0.3,
    water: 'gerstner4',
    gtao: 'off',
    ssr: 'off',
    volumetricLight: 0,
    antialias: 'fxaa',
    tileBudget: 1,
    lodDepth: 7,
  },
  medium: {
    renderScale: 0.8,
    drawDistance: 5000,
    shadowCascades: 3,
    shadowResolution: 2048,
    clouds: 'quarter',
    grassRadius: 50,
    grassDensity: 0.6,
    water: 'gerstner8',
    gtao: 'half',
    ssr: 'water',
    volumetricLight: 1,
    antialias: 'taa',
    tileBudget: 2,
    lodDepth: 8,
  },
  high: {
    renderScale: 1.0,
    drawDistance: 12000,
    shadowCascades: 4,
    shadowResolution: 2048,
    clouds: 'half',
    grassRadius: 90,
    grassDensity: 1.0,
    water: 'fft256',
    gtao: 'half',
    ssr: 'waterWet',
    volumetricLight: 2,
    antialias: 'taa',
    tileBudget: 2,
    lodDepth: 9,
  },
  ultra: {
    renderScale: 1.0,
    drawDistance: 20000,
    shadowCascades: 4,
    shadowResolution: 4096,
    clouds: 'half128',
    grassRadius: 140,
    grassDensity: 1.6,
    water: 'fft512',
    gtao: 'full',
    ssr: 'all',
    volumetricLight: 3,
    antialias: 'taa2x',
    tileBudget: 3,
    lodDepth: 10,
  },
};

export interface PostSettings {
  bloom: boolean;
  bloomStrength: number;
  dof: boolean;
  motionBlur: boolean;
  tonemap: TonemapMode;
  exposureCompensation: number;
  filmGrain: number;
  chromaticAberration: number;
  vignette: number;
  sharpen: number;
  autoExposure: boolean;
}

export const DEFAULT_POST: PostSettings = {
  bloom: true,
  bloomStrength: 0.06,
  dof: false,
  motionBlur: false,
  tonemap: 'agx',
  exposureCompensation: 0,
  filmGrain: 0.02,
  chromaticAberration: 0.4,
  vignette: 0.25,
  sharpen: 0.4,
  autoExposure: true,
};

export interface DebugSettings {
  showOverlay: boolean;
  wireframe: boolean;
  freezeCulling: boolean;
  showLodColors: boolean;
  showTileGrid: boolean;
  disableFog: boolean;
}

type Listener = (s: Settings) => void;

export class Settings {
  preset: QualityPreset = 'high';
  quality: QualitySettings = { ...PRESETS.high };
  post: PostSettings = { ...DEFAULT_POST };
  debug: DebugSettings = {
    showOverlay: true,
    wireframe: false,
    freezeCulling: false,
    showLodColors: false,
    showTileGrid: false,
    disableFog: false,
  };

  /** Target frame time in ms for the dynamic resolution controller. */
  targetFrameMs = 16.6;
  /** When true, render scale is adjusted automatically to hit the target. */
  dynamicResolution = true;
  /** Render scale actually in use — quality.renderScale after dynamic scaling. */
  effectiveRenderScale = 1.0;

  /** Field of view, vertical degrees. */
  fov = 60;
  /** Master audio volume. */
  volume = 0.8;
  /** Head bob amplitude multiplier, 0 disables. */
  headBob = 1.0;

  /** Honoured by every animated system; set from the media query on boot. */
  reducedMotion = false;

  private listeners = new Set<Listener>();

  constructor() {
    if (typeof window !== 'undefined' && window.matchMedia) {
      const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
      this.reducedMotion = mq.matches;
      mq.addEventListener('change', (e) => {
        this.reducedMotion = e.matches;
        this.emit();
      });
    }
  }

  /**
   * Swap in a preset.
   *
   * Mutates `this.quality` in place rather than replacing the object. lil-gui
   * captures the target object by reference when a controller is created and
   * never re-resolves it, so reassigning here would silently orphan every
   * quality slider in the F4 panel: the widgets would keep writing to the old
   * object while the engine read the new one.
   */
  applyPreset(preset: QualityPreset): void {
    const source = PRESETS[preset];
    if (!source) {
      console.warn(`[aether] unknown quality preset "${preset}", ignoring`);
      return;
    }
    this.preset = preset;
    Object.assign(this.quality, source);
    this.effectiveRenderScale = this.quality.renderScale;
    this.emit();
  }

  /** Called by the dynamic resolution controller. Clamped to preset ±. */
  setEffectiveRenderScale(scale: number): void {
    const base = this.quality.renderScale;
    const clamped = Math.max(base * 0.6, Math.min(base, scale));
    if (Math.abs(clamped - this.effectiveRenderScale) < 1e-4) return;
    this.effectiveRenderScale = clamped;
    this.emit();
  }

  onChange(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  emit(): void {
    for (const l of this.listeners) l(this);
  }

  toJSON(): string {
    return JSON.stringify({
      preset: this.preset,
      quality: this.quality,
      post: this.post,
      fov: this.fov,
      volume: this.volume,
      headBob: this.headBob,
      dynamicResolution: this.dynamicResolution,
    });
  }

  loadFromJSON(json: string): boolean {
    try {
      const data = JSON.parse(json) as Partial<{
        preset: QualityPreset;
        quality: QualitySettings;
        post: PostSettings;
        fov: number;
        volume: number;
        headBob: number;
        dynamicResolution: boolean;
      }>;
      if (data.preset && data.preset in PRESETS) {
        this.preset = data.preset;
        // In-place for the same reason as applyPreset.
        Object.assign(this.quality, PRESETS[data.preset], data.quality ?? {});
      }
      if (data.post) Object.assign(this.post, DEFAULT_POST, data.post);
      if (typeof data.fov === 'number') this.fov = data.fov;
      if (typeof data.volume === 'number') this.volume = data.volume;
      if (typeof data.headBob === 'number') this.headBob = data.headBob;
      if (typeof data.dynamicResolution === 'boolean') {
        this.dynamicResolution = data.dynamicResolution;
      }
      this.effectiveRenderScale = this.quality.renderScale;
      this.emit();
      return true;
    } catch {
      return false;
    }
  }

  save(): void {
    try {
      localStorage.setItem('aether.settings', this.toJSON());
    } catch {
      /* private browsing / quota — non-fatal */
    }
  }

  restore(): void {
    try {
      const s = localStorage.getItem('aether.settings');
      if (s) this.loadFromJSON(s);
    } catch {
      /* ignore */
    }
  }
}
