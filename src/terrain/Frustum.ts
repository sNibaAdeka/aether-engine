/**
 * Frustum culling for terrain nodes.
 *
 * Kept separate from three's `Frustum` because terrain nodes are 2D footprints
 * with a known vertical span, not arbitrary bounding boxes — testing them as an
 * AABB with the world's full height range is both simpler and, because the
 * height range is a compile-time constant, faster than building a real box per
 * node.
 *
 * No allocation: plane coefficients live in one Float64Array.
 */

import type { Matrix4 } from 'three/webgpu';
import { MAX_HEIGHT, MIN_HEIGHT } from './HeightField';

/** left, right, bottom, top, near, far — 6 planes × 4 coefficients. */
const PLANE_COUNT = 6;

export class Frustum {
  private planes = new Float64Array(PLANE_COUNT * 4);
  /** Set false by the "freeze culling" debug toggle. */
  frozen = false;

  /**
   * Extract planes from a view-projection matrix (Gribb & Hartmann).
   * `m` must be `projection * view`, in three's column-major element order.
   */
  setFromProjectionMatrix(m: Matrix4): void {
    if (this.frozen) return;
    const e = m.elements;
    const p = this.planes;

    // Row-major access into three's column-major array: row r, column c is
    // e[c * 4 + r].
    const m00 = e[0];
    const m10 = e[1];
    const m20 = e[2];
    const m30 = e[3];
    const m01 = e[4];
    const m11 = e[5];
    const m21 = e[6];
    const m31 = e[7];
    const m02 = e[8];
    const m12 = e[9];
    const m22 = e[10];
    const m32 = e[11];
    const m03 = e[12];
    const m13 = e[13];
    const m23 = e[14];
    const m33 = e[15];

    // left = row3 + row0
    this.setPlane(p, 0, m30 + m00, m31 + m01, m32 + m02, m33 + m03);
    // right = row3 - row0
    this.setPlane(p, 1, m30 - m00, m31 - m01, m32 - m02, m33 - m03);
    // bottom = row3 + row1
    this.setPlane(p, 2, m30 + m10, m31 + m11, m32 + m12, m33 + m13);
    // top = row3 - row1
    this.setPlane(p, 3, m30 - m10, m31 - m11, m32 - m12, m33 - m13);
    // near = row3 + row2  (WebGPU clip space is z ∈ [0,1], so this is row2)
    this.setPlane(p, 4, m20, m21, m22, m23);
    // far = row3 - row2
    this.setPlane(p, 5, m30 - m20, m31 - m21, m32 - m22, m33 - m23);
  }

  private setPlane(p: Float64Array, i: number, a: number, b: number, c: number, d: number): void {
    const len = Math.sqrt(a * a + b * b + c * c);
    const inv = len > 0 ? 1 / len : 0;
    const o = i * 4;
    p[o] = a * inv;
    p[o + 1] = b * inv;
    p[o + 2] = c * inv;
    p[o + 3] = d * inv;
  }

  /**
   * Test a terrain node footprint against the frustum.
   *
   * Conservative: uses the world's full vertical extent rather than the node's
   * actual height range. A per-node min/max would cull better but requires the
   * heightmap to already exist, which is a chicken-and-egg problem during
   * streaming — and being conservative here only costs a handful of nodes.
   */
  intersectsTerrainNode(x: number, z: number, size: number): boolean {
    const x1 = x + size;
    const z1 = z + size;
    const p = this.planes;

    for (let i = 0; i < PLANE_COUNT; i++) {
      const o = i * 4;
      const a = p[o];
      const b = p[o + 1];
      const c = p[o + 2];
      const d = p[o + 3];

      // Positive vertex: the AABB corner furthest along the plane normal. If
      // even that corner is behind the plane, the whole box is outside.
      const px = a >= 0 ? x1 : x;
      const py = b >= 0 ? MAX_HEIGHT : MIN_HEIGHT;
      const pz = c >= 0 ? z1 : z;

      if (a * px + b * py + c * pz + d < 0) return false;
    }
    return true;
  }

  /** Sphere test, for vegetation and object culling in later phases. */
  intersectsSphere(x: number, y: number, z: number, radius: number): boolean {
    const p = this.planes;
    for (let i = 0; i < PLANE_COUNT; i++) {
      const o = i * 4;
      if (p[o] * x + p[o + 1] * y + p[o + 2] * z + p[o + 3] < -radius) return false;
    }
    return true;
  }
}
