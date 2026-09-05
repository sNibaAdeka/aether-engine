/**
 * GPU resource ownership: creation, reference counting, budgeted release.
 *
 * Two jobs:
 *  1. Nothing leaks. Every texture/buffer/material created through here is
 *     tracked and disposed on teardown.
 *  2. Nothing is created inside the hot loop. Allocations queued during a frame
 *     are drained against a per-frame time budget, because creating a material
 *     mid-frame triggers a shader compile and a visible hitch.
 */

import type { Texture, Material, BufferGeometry } from 'three/webgpu';

type Disposable = { dispose: () => void };

interface Tracked {
  resource: Disposable;
  label: string;
  bytes: number;
  refs: number;
}

export class ResourceManager {
  private tracked = new Map<Disposable, Tracked>();
  private byLabel = new Map<string, Disposable>();
  private pendingWork: Array<{ label: string; run: () => void }> = [];

  /** Rough total of tracked GPU bytes. */
  totalBytes = 0;

  /** Milliseconds of resource work allowed per frame. */
  frameBudgetMs = 2.0;

  track<T extends Disposable>(resource: T, label: string, bytes = 0): T {
    const existing = this.tracked.get(resource);
    if (existing) {
      existing.refs++;
      return resource;
    }
    this.tracked.set(resource, { resource, label, bytes, refs: 1 });
    this.byLabel.set(label, resource);
    this.totalBytes += bytes;
    return resource;
  }

  get<T extends Disposable>(label: string): T | undefined {
    return this.byLabel.get(label) as T | undefined;
  }

  release(resource: Disposable): void {
    const t = this.tracked.get(resource);
    if (!t) return;
    t.refs--;
    if (t.refs > 0) return;
    this.tracked.delete(resource);
    this.byLabel.delete(t.label);
    this.totalBytes -= t.bytes;
    try {
      resource.dispose();
    } catch (err) {
      console.warn(`[aether] dispose failed for "${t.label}":`, err);
    }
  }

  /**
   * Queue expensive resource creation. Drained by `drain()` inside the frame's
   * budget, so a burst of tile loads spreads over several frames instead of
   * spiking one.
   */
  enqueue(label: string, run: () => void): void {
    this.pendingWork.push({ label, run });
  }

  /** Returns the number of items completed this frame. */
  drain(budgetMs = this.frameBudgetMs): number {
    if (this.pendingWork.length === 0) return 0;
    const start = performance.now();
    let done = 0;
    while (this.pendingWork.length > 0) {
      const item = this.pendingWork.shift();
      if (!item) break;
      try {
        item.run();
      } catch (err) {
        console.error(`[aether] resource work "${item.label}" failed:`, err);
      }
      done++;
      if (performance.now() - start >= budgetMs) break;
    }
    return done;
  }

  get pendingCount(): number {
    return this.pendingWork.length;
  }

  get megabytes(): number {
    return this.totalBytes / (1024 * 1024);
  }

  /** Byte size estimate for a texture; used to keep the memory readout honest. */
  static textureBytes(
    width: number,
    height: number,
    depth: number,
    bytesPerTexel: number,
    mipped = false,
  ): number {
    const base = width * height * depth * bytesPerTexel;
    return mipped ? Math.floor(base * 1.334) : base;
  }

  disposeAll(): void {
    for (const [resource, t] of this.tracked) {
      try {
        resource.dispose();
      } catch (err) {
        console.warn(`[aether] dispose failed for "${t.label}":`, err);
      }
    }
    this.tracked.clear();
    this.byLabel.clear();
    this.pendingWork.length = 0;
    this.totalBytes = 0;
  }

  /** Debug listing, largest first. */
  inventory(limit = 20): Array<{ label: string; mb: number; refs: number }> {
    return [...this.tracked.values()]
      .sort((a, b) => b.bytes - a.bytes)
      .slice(0, limit)
      .map((t) => ({ label: t.label, mb: t.bytes / (1024 * 1024), refs: t.refs }));
  }
}

export type TrackableTexture = Texture;
export type TrackableMaterial = Material;
export type TrackableGeometry = BufferGeometry;
