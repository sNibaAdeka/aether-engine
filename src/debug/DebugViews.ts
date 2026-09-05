/**
 * Debug visualisation modes, bound to the number row.
 *
 * These exist because the alternative is guessing. A morph coefficient that is
 * subtly wrong looks identical to a correct one in a shaded render — until you
 * fly and see the surface swim. Painting it directly turns a two-hour hunt into
 * a glance.
 *
 *   0  off — normal shading
 *   1  height          absolute elevation, banded
 *   2  normals         world-space normal as RGB
 *   3  slope           flat → steep heat ramp, with the 30°/38° biome cuts marked
 *   4  lod level       hue per CDLOD level
 *   5  morph K         black = no morph, white = fully collapsed to parent
 *   6  tile borders    patch outline + heightmap texel grid
 *   7  flow            accumulated water flow from the erosion pass
 *   8  moisture        accumulated moisture from the erosion pass
 *   9  checker         world-space 1 m / 10 m checker for scale and UV sanity
 *  10  biome mix       all eight splat weights in their key colours (Shift+0)
 *  11  biome id        the dominant biome only, plus snow coverage (Shift+1)
 *
 * The number row ran out at 9, so the phase-2 views are on Shift+digit. That is
 * not elegant, but a second modifier is cheaper than renumbering views that are
 * referenced from PROGRESS.md, from three screenshot scripts and from the
 * comments that explain what each one caught.
 */

export const DEBUG_VIEWS = [
  'off',
  'height',
  'normals',
  'slope',
  'lod',
  'morph',
  'tiles',
  'flow',
  'moisture',
  'checker',
  'biomeMix',
  'biomeId',
] as const;

export type DebugViewName = (typeof DEBUG_VIEWS)[number];

export interface DebugViewsOptions {
  /** Called whenever the mode changes, with the numeric index. */
  onChange: (mode: number, name: DebugViewName) => void;
  /** Optional transient message sink (the HUD). */
  notify?: (text: string) => void;
}

const LABELS: Record<DebugViewName, string> = {
  off: 'обычный рендер',
  height: 'высота',
  normals: 'нормали',
  slope: 'уклон',
  lod: 'уровень LOD',
  morph: 'коэффициент морфа',
  tiles: 'границы тайлов',
  flow: 'карта стока',
  moisture: 'влажность',
  checker: 'шахматка 1/10 м',
  biomeMix: 'смесь биомов',
  biomeId: 'доминирующий биом',
};

/** Modes that depend on data the erosion pass produces. */
const NEEDS_EROSION: ReadonlySet<DebugViewName> = new Set(['flow', 'moisture']);

/** Modes that depend on the splat bake, which the WebGL2 fallback never runs. */
const NEEDS_SPLAT: ReadonlySet<DebugViewName> = new Set(['biomeMix', 'biomeId']);

/** First mode reached with Shift held rather than with a bare digit. */
const SHIFT_BASE = 10;

export class DebugViews {
  mode = 0;

  /** Set by the terrain system once the erosion pass has produced its maps. */
  erosionDataAvailable = false;

  /** Set by the terrain system once at least one tile has been classified. */
  splatDataAvailable = false;

  constructor(private opts: DebugViewsOptions) {
    window.addEventListener('keydown', this.onKey);
  }

  get name(): DebugViewName {
    return DEBUG_VIEWS[this.mode] ?? 'off';
  }

  private onKey = (e: KeyboardEvent): void => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const target = e.target as HTMLElement | null;
    if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return;

    // Digit0..Digit9 — use `code` so it works regardless of keyboard layout,
    // which matters because the UI is Russian and the dev keyboard may not be.
    // Shift shifts the whole row up by ten, which is how the phase-2 biome
    // views are reached.
    if (!/^Digit[0-9]$/.test(e.code)) return;
    const n = Number(e.code.slice(5)) + (e.shiftKey ? SHIFT_BASE : 0);
    if (n >= DEBUG_VIEWS.length) return;

    e.preventDefault();
    this.set(n);
  };

  set(mode: number): void {
    const clamped = Math.max(0, Math.min(DEBUG_VIEWS.length - 1, mode));
    const name = DEBUG_VIEWS[clamped];

    if (NEEDS_EROSION.has(name) && !this.erosionDataAvailable) {
      this.opts.notify?.(`${LABELS[name]} — данных ещё нет (эрозия не посчитана)`);
      return;
    }
    if (NEEDS_SPLAT.has(name) && !this.splatDataAvailable) {
      this.opts.notify?.(`${LABELS[name]} — данных ещё нет (splat не посчитан)`);
      return;
    }

    this.mode = clamped;
    this.opts.onChange(clamped, name);
    this.opts.notify?.(clamped === 0 ? LABELS.off : `${clamped} · ${LABELS[name]}`);
  }

  cycle(): void {
    this.set((this.mode + 1) % DEBUG_VIEWS.length);
  }

  dispose(): void {
    window.removeEventListener('keydown', this.onKey);
  }
}
