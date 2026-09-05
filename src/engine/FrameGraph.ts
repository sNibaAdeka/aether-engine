/**
 * The render pass order, written down once, explicitly.
 *
 * This list is the spine of the renderer. Passes register themselves against a
 * declared `PassId`; the graph runs them in the fixed order below regardless of
 * registration order, so a system added late cannot silently reorder the frame.
 *
 * Passes may be absent (a preset with clouds off simply never registers
 * `volumetric-clouds`) and may declare an update cadence, so amortised passes
 * — distant shadow cascades, cloud shadows, atmosphere LUTs — are skipped on
 * frames where they would be wasted work.
 */

import type { FrameContext } from './types';
import type { Profiler } from './Profiler';

export const PASS_ORDER = [
  'atmosphere-luts',
  'cloud-noise',
  'ocean-fft',
  'wind-field',
  'terrain-generate',
  'gpu-culling',
  'shadow-csm',
  'cloud-shadows',
  'depth-prepass',
  'hiz-build',
  'main-opaque',
  'sky',
  'volumetric-clouds',
  'volumetric-light',
  'water',
  'transparent',
  'gtao',
  'ssr',
  'lighting-composite',
  'taa',
  'auto-exposure',
  'bloom',
  'dof',
  'motion-blur',
  'tonemap',
  'sharpen',
  'ui',
] as const;

export type PassId = (typeof PASS_ORDER)[number];

const PASS_INDEX: ReadonlyMap<PassId, number> = new Map(
  PASS_ORDER.map((id, i) => [id, i] as const),
);

export interface PassOptions {
  /** Run every N frames. 1 = every frame. */
  cadence?: number;
  /** Phase offset so amortised passes do not all land on the same frame. */
  phase?: number;
  /** When false the pass is skipped entirely (quality preset gating). */
  enabled?: () => boolean;
}

interface RegisteredPass {
  id: PassId;
  order: number;
  execute: (ctx: FrameContext) => void;
  cadence: number;
  phase: number;
  enabled: () => boolean;
}

const ALWAYS = (): boolean => true;

export class FrameGraph {
  private passes: RegisteredPass[] = [];
  private dirty = false;

  constructor(private profiler: Profiler) {}

  register(id: PassId, execute: (ctx: FrameContext) => void, opts: PassOptions = {}): () => void {
    const order = PASS_INDEX.get(id);
    if (order === undefined) {
      throw new Error(`FrameGraph: unknown pass id "${id}". Add it to PASS_ORDER.`);
    }
    const pass: RegisteredPass = {
      id,
      order,
      execute,
      cadence: Math.max(1, opts.cadence ?? 1),
      phase: opts.phase ?? 0,
      enabled: opts.enabled ?? ALWAYS,
    };
    this.passes.push(pass);
    this.dirty = true;
    return () => {
      const i = this.passes.indexOf(pass);
      if (i >= 0) this.passes.splice(i, 1);
    };
  }

  private sortIfNeeded(): void {
    if (!this.dirty) return;
    // Stable sort keeps registration order within a pass id, which matters when
    // two systems both contribute to e.g. `transparent`.
    this.passes.sort((a, b) => a.order - b.order);
    this.dirty = false;
  }

  execute(ctx: FrameContext): void {
    this.sortIfNeeded();
    const passes = this.passes;
    for (let i = 0; i < passes.length; i++) {
      const p = passes[i];
      if (p.cadence > 1 && (ctx.frame + p.phase) % p.cadence !== 0) continue;
      if (!p.enabled()) continue;
      this.profiler.begin(p.id);
      p.execute(ctx);
      this.profiler.end(p.id);
    }
  }

  /** Human-readable dump of the active graph — surfaced in the debug overlay. */
  describe(): string {
    this.sortIfNeeded();
    return this.passes
      .map((p) => `${p.id}${p.cadence > 1 ? ` (1/${p.cadence})` : ''}${p.enabled() ? '' : ' [off]'}`)
      .join('\n');
  }

  get passCount(): number {
    return this.passes.length;
  }

  clear(): void {
    this.passes.length = 0;
    this.dirty = false;
  }
}
