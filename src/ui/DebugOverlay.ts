/**
 * F3 overlay: frame time graph, per-section breakdown, draw counters, world
 * state. The point is to make a regression obvious the moment it lands rather
 * than three phases later.
 *
 * Updates at 10 Hz — a per-frame DOM write would itself distort the numbers.
 */

import type { Profiler } from '@/engine/Profiler';
import type { Engine } from '@/engine/Engine';

const GRAPH_W = 260;
const GRAPH_H = 48;
const HISTORY = 120;

export interface WorldReadout {
  timeOfDay?: string;
  weather?: string;
  biome?: string;
  tiles?: string;
  extra?: Record<string, string | number>;
}

export class DebugOverlay {
  private root: HTMLDivElement;
  private textEl: HTMLPreElement;
  private canvas: HTMLCanvasElement;
  private gctx: CanvasRenderingContext2D;
  private history = new Float32Array(HISTORY);
  private lastUpdate = 0;
  private visible = true;

  /** Systems push their status here; the overlay just renders it. */
  readout: WorldReadout = {};

  constructor(
    private engine: Engine,
    private profiler: Profiler,
  ) {
    this.root = document.createElement('div');
    this.root.className = 'aether-overlay';
    this.root.setAttribute('role', 'status');
    this.root.setAttribute('aria-live', 'off');

    this.canvas = document.createElement('canvas');
    this.canvas.width = GRAPH_W * 2;
    this.canvas.height = GRAPH_H * 2;
    this.canvas.style.width = `${GRAPH_W}px`;
    this.canvas.style.height = `${GRAPH_H}px`;
    this.canvas.style.display = 'block';
    this.canvas.style.borderRadius = '3px';
    this.canvas.style.background = 'rgba(0,0,0,0.35)';

    const g = this.canvas.getContext('2d');
    if (!g) throw new Error('DebugOverlay: 2D context unavailable');
    this.gctx = g;
    this.gctx.scale(2, 2);

    this.textEl = document.createElement('pre');
    this.textEl.className = 'aether-overlay-text';

    this.root.appendChild(this.canvas);
    this.root.appendChild(this.textEl);
    document.body.appendChild(this.root);

    window.addEventListener('keydown', this.onKey);
  }

  private onKey = (e: KeyboardEvent): void => {
    if (e.code === 'F3') {
      e.preventDefault();
      this.toggle();
    }
  };

  toggle(): void {
    this.visible = !this.visible;
    this.root.style.display = this.visible ? 'block' : 'none';
  }

  setVisible(v: boolean): void {
    this.visible = v;
    this.root.style.display = v ? 'block' : 'none';
  }

  update(now: number): void {
    if (!this.visible) return;
    if (now - this.lastUpdate < 100) return;
    this.lastUpdate = now;

    this.drawGraph();
    this.drawText();
  }

  private drawGraph(): void {
    const g = this.gctx;
    const n = this.profiler.frameHistory(this.history);
    g.clearRect(0, 0, GRAPH_W, GRAPH_H);

    // Full-scale adapts to the workload: a fixed 40 ms axis renders a healthy
    // 1.5 ms frame as a 2-pixel stub, which hides exactly the variance the
    // graph exists to show. Never below 20 ms so the 16.6 ms budget line stays
    // on screen, never above 60 ms so one catastrophic frame does not flatten
    // everything else.
    const fullScale = Math.min(60, Math.max(20, this.profiler.frameMsP95 * 1.6));
    const scale = GRAPH_H / fullScale;
    g.strokeStyle = 'rgba(120,220,140,0.45)';
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(0, GRAPH_H - 16.6 * scale);
    g.lineTo(GRAPH_W, GRAPH_H - 16.6 * scale);
    g.stroke();

    g.strokeStyle = 'rgba(240,180,90,0.35)';
    g.beginPath();
    g.moveTo(0, GRAPH_H - 33.3 * scale);
    g.lineTo(GRAPH_W, GRAPH_H - 33.3 * scale);
    g.stroke();

    if (n === 0) return;

    const step = GRAPH_W / HISTORY;
    for (let i = 0; i < n; i++) {
      const ms = this.history[i];
      const h = Math.min(GRAPH_H, ms * scale);
      // Green under budget, amber approaching, red over.
      g.fillStyle = ms < 16.6 ? '#6ee7a0' : ms < 25 ? '#ffcc66' : '#ff6b6b';
      g.fillRect(i * step, GRAPH_H - h, Math.max(1, step - 0.4), h);
    }
  }

  private drawText(): void {
    const p = this.profiler;
    const e = this.engine;
    const cam = e.camera.position;
    const r = this.readout;

    const lines: string[] = [];
    lines.push(
      `${p.fps.toFixed(0)} FPS   ${p.frameMs.toFixed(2)} ms  p95 ${p.frameMsP95.toFixed(2)} ms`,
    );
    lines.push(`cpu ${p.cpuFrameMs.toFixed(2)} ms  p95 ${p.cpuFrameMsP95.toFixed(2)} ms`);
    lines.push(
      `${e.rendererInfo.backend.toUpperCase()}  scale ${e.settings.effectiveRenderScale.toFixed(2)}  ` +
        `${e.resize.renderWidth}×${e.resize.renderHeight}`,
    );
    lines.push(
      `draws ${p.drawCalls}  tris ${fmtK(p.triangles)}  compute ${p.computeCalls}  ` +
        `mem ~${p.gpuMemoryMB.toFixed(0)} MB`,
    );
    lines.push(
      `pos ${cam.x.toFixed(1)} ${cam.y.toFixed(1)} ${cam.z.toFixed(1)}   preset ${e.settings.preset}`,
    );

    if (r.timeOfDay || r.weather || r.biome) {
      lines.push(
        `${r.timeOfDay ?? '—'}  ${r.weather ?? '—'}  ${r.biome ?? '—'}${r.tiles ? `  tiles ${r.tiles}` : ''}`,
      );
    }
    if (r.extra) {
      for (const [k, v] of Object.entries(r.extra)) lines.push(`${k}: ${v}`);
    }

    lines.push('');
    const stats = p.stats();
    // Only show sections that cost something; a wall of 0.00 lines is noise.
    for (const s of stats) {
      if (s.cpuP50 < 0.02 && s.gpuP50 < 0.02) continue;
      const gpu = s.gpuP50 > 0 ? ` gpu ${s.gpuP50.toFixed(2)}` : '';
      lines.push(`${pad(s.name, 20)} ${s.cpuP50.toFixed(2)} (p95 ${s.cpuP95.toFixed(2)})${gpu}`);
    }
    lines.push('');
    lines.push('F3 stats · F4 settings · F1 fly/player · F2 screenshot · H help');

    this.textEl.textContent = lines.join('\n');
  }

  dispose(): void {
    window.removeEventListener('keydown', this.onKey);
    this.root.remove();
  }
}

function pad(s: string, n: number): string {
  return s.length >= n ? s.slice(0, n) : s + ' '.repeat(n - s.length);
}

function fmtK(n: number): string {
  if (n > 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n > 1e3) return `${(n / 1e3).toFixed(0)}k`;
  return String(n);
}
