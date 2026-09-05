/**
 * Typed worker pool.
 *
 * Jobs are dispatched to the least-loaded worker; results come back as
 * transferables so large Float32Arrays never get structured-cloned. A worker
 * that throws fails only its own job — the pool keeps running, because losing
 * the whole terrain pipeline to one bad tile is not an acceptable failure mode.
 */

export interface WorkerJob<Req, Res> {
  kind: string;
  payload: Req;
  /** Buffers to transfer (not copy) into the worker. */
  transfer?: Transferable[];
  resolve: (value: Res) => void;
  reject: (err: Error) => void;
}

interface PendingJob {
  id: number;
  kind: string;
  /** Which worker owns this job. Required to fail only that worker's work. */
  owner: PoolWorker;
  resolve: (value: unknown) => void;
  reject: (err: Error) => void;
  startedAt: number;
}

interface PoolWorker {
  worker: Worker;
  inFlight: number;
  index: number;
}

export interface WorkerPoolOptions {
  /** Number of workers. Defaults to hardwareConcurrency-2, clamped to [1,6]. */
  size?: number;
  /** Factory so callers own the `new Worker(new URL(...))` call Vite needs. */
  create: () => Worker;
  name?: string;
}

export class WorkerPool {
  private workers: PoolWorker[] = [];
  private pending = new Map<number, PendingJob>();
  private nextId = 1;
  private disposed = false;
  readonly name: string;

  constructor(opts: WorkerPoolOptions) {
    this.name = opts.name ?? 'pool';
    const hw = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4;
    const size = Math.max(1, Math.min(6, opts.size ?? hw - 2));

    for (let i = 0; i < size; i++) {
      const worker = opts.create();
      const pw: PoolWorker = { worker, inFlight: 0, index: i };
      worker.onmessage = (e: MessageEvent) => this.onMessage(pw, e);
      worker.onerror = (e: ErrorEvent) => this.onError(pw, e);
      this.workers.push(pw);
    }
  }

  get size(): number {
    return this.workers.length;
  }

  get queueDepth(): number {
    return this.pending.size;
  }

  private onMessage(pw: PoolWorker, e: MessageEvent): void {
    const data = e.data as { id: number; ok: boolean; result?: unknown; error?: string };
    // Decrement before the lookup: a reply for an already-failed job still
    // means this worker finished a unit of work, and skipping the decrement
    // leaves `inFlight` permanently inflated, which quietly biases
    // `leastLoaded` for the rest of the session.
    pw.inFlight = Math.max(0, pw.inFlight - 1);
    const job = this.pending.get(data.id);
    if (!job) return;
    this.pending.delete(data.id);
    if (data.ok) {
      job.resolve(data.result);
    } else {
      job.reject(new Error(data.error ?? 'worker job failed'));
    }
  }

  private onError(pw: PoolWorker, e: ErrorEvent): void {
    console.error(`[aether] worker ${this.name}#${pw.index} error:`, e.message);
    // `onerror` does not say which job died, so fail everything this worker
    // owned — and nothing else. Jobs on healthy workers must survive; losing
    // the whole terrain queue to one bad tile is not an acceptable failure.
    for (const [id, job] of this.pending) {
      if (job.owner !== pw) continue;
      job.reject(new Error(`worker ${this.name}#${pw.index} crashed: ${e.message}`));
      this.pending.delete(id);
    }
    pw.inFlight = 0;
  }

  private leastLoaded(): PoolWorker {
    let best = this.workers[0];
    for (let i = 1; i < this.workers.length; i++) {
      if (this.workers[i].inFlight < best.inFlight) best = this.workers[i];
    }
    return best;
  }

  run<Req, Res>(kind: string, payload: Req, transfer: Transferable[] = []): Promise<Res> {
    if (this.disposed) return Promise.reject(new Error('WorkerPool disposed'));
    const id = this.nextId++;
    const pw = this.leastLoaded();
    pw.inFlight++;

    return new Promise<Res>((resolve, reject) => {
      this.pending.set(id, {
        id,
        kind,
        owner: pw,
        resolve: resolve as (v: unknown) => void,
        reject,
        startedAt: performance.now(),
      });
      pw.worker.postMessage({ id, kind, payload }, transfer);
    });
  }

  /** Jobs older than `ms`, for the debug overlay's stall detection. */
  stalled(ms: number): number {
    const now = performance.now();
    let n = 0;
    for (const job of this.pending.values()) {
      if (now - job.startedAt > ms) n++;
    }
    return n;
  }

  dispose(): void {
    this.disposed = true;
    for (const pw of this.workers) pw.worker.terminate();
    this.workers.length = 0;
    for (const job of this.pending.values()) {
      job.reject(new Error('WorkerPool disposed'));
    }
    this.pending.clear();
  }
}

/**
 * Boilerplate for the worker side: register handlers by kind and this wires up
 * the message protocol, error reporting and transferables.
 */
export function serveWorker(
  handlers: Record<string, (payload: never) => { result: unknown; transfer?: Transferable[] }>,
): void {
  self.onmessage = (e: MessageEvent): void => {
    const { id, kind, payload } = e.data as { id: number; kind: string; payload: never };
    const handler = handlers[kind];
    if (!handler) {
      (self as unknown as Worker).postMessage({ id, ok: false, error: `unknown kind "${kind}"` });
      return;
    }
    try {
      const { result, transfer } = handler(payload);
      (self as unknown as Worker).postMessage({ id, ok: true, result }, transfer ?? []);
    } catch (err) {
      (self as unknown as Worker).postMessage({
        id,
        ok: false,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  };
}
