/**
 * What is left of the phase-0 placeholder scene: nothing.
 *
 * It began as a calibration rig — a ground grid and 1/2/10 m reference solids,
 * so that "is that mountain 400 m or 40 m?" had an answer. Phase 1 removed the
 * geometry, because a 2 km slab at y=0 punched straight through the real
 * heightfield. Phase 3 removes the rest: the provisional `DirectionalLight` and
 * `HemisphereLight` that stood in for the sky, and the flat blue
 * `scene.background` behind them.
 *
 * Every light in the world now comes from `AtmosphereSystem` — a sun and a moon
 * whose colour is the transmittance LUT evaluated at their real elevation, and
 * an ambient term that is the sky-view LUT projected into spherical harmonics.
 * CLAUDE.md rule 5 says ambient is never a constant; this file is where the
 * constant used to be.
 *
 * The class is kept rather than deleted so the system list still reads as a
 * history of what the frame is made of, and so that anything a future phase
 * needs to drop into the scene for debugging has an obvious home.
 */

import * as THREE from 'three/webgpu';
import type { EngineContext, FrameContext, System } from '@/engine/types';

export class DebugScene implements System {
  readonly name = 'debug-scene';

  private group = new THREE.Group();

  async init(ctx: EngineContext): Promise<void> {
    this.group.name = 'debug-scene';
    ctx.scene.add(this.group);
  }

  update(_dt: number, _ctx: FrameContext): void {
    // Nothing to do. See the header.
  }

  dispose(): void {
    this.group.removeFromParent();
  }
}
