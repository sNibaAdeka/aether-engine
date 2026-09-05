/**
 * Section profiler with CPU wall-clock timings and — where the backend exposes
 * them — GPU timestamps.
 *
 * Keeps a 120-frame ring buffer per section and reports p50/p95, because the
 * mean hides exactly the hitches we care about. A 60 FPS average with a 90 ms
 * p95 is a stuttering mess; a 62 FPS average with a 17 ms p95 is smooth.
 */

const RING = 120;

interface Section {
  name: string;
  /** CPU milliseconds, ring buffer. */
  cpu: Float32Array;
  /** GPU milliseconds, ring buffer. Zero when unavailable. */
  gpu: Float32Array;
  cursor: number;
  count: number;
  startMark: number;
  /** True for series fed only by GPU timestamps, with no CPU begin/end pair. */
  gpuOnly: boolean;
}

export interface SectionStats {
  name: string;
  cpuLast: number;
  cpuP50: number;
  cpuP95: number;
  gpuLast: number;
  gpuP50: number;
}

export class Profiler {
  private sections = new Map<string, Section>();
  private order: string[] = [];

  /**
   * Wall-clock interval between presented frames, from the rAF timestamp.
   *
   * This — not the CPU span of the tick — is what the user experiences.
   * Measuring `performance.now()` around the tick body reports how long it took
   * to *encode* the frame; three's `render()` returns as soon as the command
   * buffer is submitted, so on a GPU-bound scene that number can say 1.1 ms
   * while the display is actually managing 45 FPS.
   */
  private frameTimes = new Float32Array(RING);
  private frameCursor = 0;
  private frameCount = 0;
  private lastPresent = 0;

  /** CPU span of the tick body — how much main-thread time the frame cost. */
  private cpuFrameTimes = new Float32Array(RING);
  private cpuFrameCursor = 0;
  private cpuFrameCount = 0;
  private frameStart = 0;

  /** Renderer-reported counters, refreshed by `Engine` each frame. */
  drawCalls = 0;
  triangles = 0;
  computeCalls = 0;
  /** Rough GPU memory estimate in MB, maintained by ResourceManager. */
  gpuMemoryMB = 0;

  /** Set true once the backend confirms timestamp queries work. */
  gpuTimingAvailable = false;

  private statsCache: SectionStats[] = [];
  private scratch = new Float32Array(RING);

  private section(name: string): Section {
    let s = this.sections.get(name);
    if (!s) {
      s = {
        name,
        cpu: new Float32Array(RING),
        gpu: new Float32Array(RING),
        cursor: 0,
        count: 0,
        startMark: 0,
        gpuOnly: false,
      };
      this.sections.set(name, s);
      this.order.push(name);
    }
    return s;
  }

  /** @param now the rAF timestamp, so the interval reflects actual presents. */
  beginFrame(now: number): void {
    if (this.lastPresent > 0) {
      const interval = now - this.lastPresent;
      // Ignore the resume spike after a hidden tab — it is not a dropped frame.
      if (interval < 1000) {
        this.frameTimes[this.frameCursor] = interval;
        this.frameCursor = (this.frameCursor + 1) % RING;
        if (this.frameCount < RING) this.frameCount++;
      }
    }
    this.lastPresent = now;
    this.frameStart = performance.now();
  }

  endFrame(): void {
    const dt = performance.now() - this.frameStart;
    this.cpuFrameTimes[this.cpuFrameCursor] = dt;
    this.cpuFrameCursor = (this.cpuFrameCursor + 1) % RING;
    if (this.cpuFrameCount < RING) this.cpuFrameCount++;
  }

  /** Reset the present clock so a pause does not register as a 3-second frame. */
  resumeFromPause(): void {
    this.lastPresent = 0;
  }

  /** CPU main-thread cost of the tick body, ms (p50). */
  get cpuFrameMs(): number {
    return this.percentile(this.cpuFrameTimes, this.cpuFrameCount, 0.5);
  }

  get cpuFrameMsP95(): number {
    return this.percentile(this.cpuFrameTimes, this.cpuFrameCount, 0.95);
  }

  begin(name: string): void {
    this.section(name).startMark = performance.now();
  }

  end(name: string): void {
    const s = this.section(name);
    const dt = performance.now() - s.startMark;
    s.cpu[s.cursor] = dt;
    // GPU value is written separately by `recordGpu`; clear it so a stale value
    // from 120 frames ago is not reported as current.
    s.gpu[s.cursor] = 0;
    s.cursor = (s.cursor + 1) % RING;
    if (s.count < RING) s.count++;
  }

  /**
   * Attach an async GPU duration to the most recent slot of a section that is
   * already being timed on the CPU.
   */
  recordGpu(name: string, ms: number): void {
    const s = this.sections.get(name);
    // A GPU-only series must keep appending, not overwrite the previous slot —
    // otherwise its ring holds exactly one sample forever and the reported
    // "p50" is really the latest reading, which is precisely the hitch-hiding
    // behaviour the percentiles exist to prevent.
    if (!s || s.gpuOnly || s.count === 0) {
      this.pushGpu(name, ms);
      return;
    }
    const idx = (s.cursor - 1 + RING) % RING;
    s.gpu[idx] = ms;
    this.gpuTimingAvailable = true;
  }

  /**
   * Append a GPU-only sample. Used for whole-pipeline timestamps that have no
   * matching CPU begin/end pair.
   */
  pushGpu(name: string, ms: number): void {
    const s = this.section(name);
    s.gpuOnly = true;
    s.cpu[s.cursor] = 0;
    s.gpu[s.cursor] = ms;
    s.cursor = (s.cursor + 1) % RING;
    if (s.count < RING) s.count++;
    this.gpuTimingAvailable = true;
  }

  /** Time a synchronous function and attribute it to `name`. */
  measure<T>(name: string, fn: () => T): T {
    this.begin(name);
    try {
      return fn();
    } finally {
      this.end(name);
    }
  }

  private percentile(buf: Float32Array, count: number, p: number): number {
    if (count === 0) return 0;
    const scratch = this.scratch;
    for (let i = 0; i < count; i++) scratch[i] = buf[i];
    // Insertion sort: count ≤ 120 and the array is nearly sorted frame to
    // frame, so this beats allocating a subarray and calling .sort().
    for (let i = 1; i < count; i++) {
      const v = scratch[i];
      let j = i - 1;
      while (j >= 0 && scratch[j] > v) {
        scratch[j + 1] = scratch[j];
        j--;
      }
      scratch[j + 1] = v;
    }
    const idx = Math.min(count - 1, Math.max(0, Math.round(p * (count - 1))));
    return scratch[idx];
  }

  get fps(): number {
    const p50 = this.percentile(this.frameTimes, this.frameCount, 0.5);
    return p50 > 0 ? 1000 / p50 : 0;
  }

  get frameMs(): number {
    return this.percentile(this.frameTimes, this.frameCount, 0.5);
  }

  get frameMsP95(): number {
    return this.percentile(this.frameTimes, this.frameCount, 0.95);
  }

  get frameMsLast(): number {
    return this.frameTimes[(this.frameCursor - 1 + RING) % RING];
  }

  /** Copy of the frame-time ring in chronological order, for the graph. */
  frameHistory(out: Float32Array): number {
    const n = this.frameCount;
    for (let i = 0; i < n; i++) {
      out[i] = this.frameTimes[(this.frameCursor - n + i + RING) % RING];
    }
    return n;
  }

  /** Reused array — do not retain across frames. */
  stats(): SectionStats[] {
    this.statsCache.length = 0;
    for (const name of this.order) {
      const s = this.sections.get(name);
      if (!s || s.count === 0) continue;
      const last = (s.cursor - 1 + RING) % RING;
      this.statsCache.push({
        name,
        cpuLast: s.cpu[last],
        cpuP50: this.percentile(s.cpu, s.count, 0.5),
        cpuP95: this.percentile(s.cpu, s.count, 0.95),
        gpuLast: s.gpu[last],
        gpuP50: this.percentile(s.gpu, s.count, 0.5),
      });
    }
    return this.statsCache;
  }

  /** Markdown table of current timings — pasted straight into PROGRESS.md. */
  report(): string {
    const rows = this.stats()
      .map(
        (s) =>
          `| ${s.name} | ${s.cpuP50.toFixed(2)} | ${s.cpuP95.toFixed(2)} | ${
            s.gpuP50 > 0 ? s.gpuP50.toFixed(2) : '—'
          } |`,
      )
      .join('\n');
    return [
      `frame p50 ${this.frameMs.toFixed(2)} ms · p95 ${this.frameMsP95.toFixed(2)} ms · ${this.fps.toFixed(0)} FPS`,
      `cpu   p50 ${this.cpuFrameMs.toFixed(2)} ms · p95 ${this.cpuFrameMsP95.toFixed(2)} ms`,
      `draws ${this.drawCalls} · tris ${(this.triangles / 1000).toFixed(0)}k`,
      '',
      '| section | cpu p50 | cpu p95 | gpu p50 |',
      '|---|---|---|---|',
      rows,
    ].join('\n');
  }

  reset(): void {
    this.sections.clear();
    this.order.length = 0;
    this.frameTimes.fill(0);
    this.frameCount = 0;
    this.frameCursor = 0;
    this.cpuFrameTimes.fill(0);
    this.cpuFrameCount = 0;
    this.cpuFrameCursor = 0;
    this.lastPresent = 0;
  }
}
