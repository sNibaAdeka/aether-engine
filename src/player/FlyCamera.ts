/**
 * Debug free camera. Exists so the terrain, atmosphere and LOD work can be
 * inspected long before there is a player controller — and so the phase-1
 * acceptance test ("fly at 200 m/s, look for cracks") is actually runnable.
 */

import { Euler, Vector3, type PerspectiveCamera } from 'three/webgpu';
import type { Input } from './Input';
import type { FrameContext, System, EngineContext } from '@/engine/types';

const _forward = new Vector3();
const _right = new Vector3();
const _move = new Vector3();
const _look = { x: 0, y: 0 };
const _pad = { x: 0, y: 0 };

const MAX_PITCH = Math.PI / 2 - 0.001;

export class FlyCamera implements System {
  readonly name = 'fly-camera';

  enabled = true;
  /** Base speed, m/s. Shift multiplies, Alt divides, wheel scales. */
  speed = 40;
  /** Exponential smoothing time constant for velocity, seconds. */
  smoothing = 0.12;

  private euler = new Euler(0, 0, 0, 'YXZ');
  private velocity = new Vector3();
  private camera: PerspectiveCamera;

  constructor(
    camera: PerspectiveCamera,
    private input: Input,
  ) {
    this.camera = camera;
    this.euler.setFromQuaternion(camera.quaternion);
  }

  async init(_ctx: EngineContext): Promise<void> {
    // Nothing to allocate.
  }

  /** Place the camera and aim it at a target — used by the screenshot harness. */
  setPose(position: Vector3, target: Vector3): void {
    this.camera.position.copy(position);
    _forward.copy(target).sub(position).normalize();
    this.euler.y = Math.atan2(-_forward.x, -_forward.z);
    this.euler.x = Math.asin(Math.max(-1, Math.min(1, _forward.y)));
    this.euler.z = 0;
    this.camera.quaternion.setFromEuler(this.euler);
    this.velocity.set(0, 0, 0);
  }

  update(dt: number, _ctx: FrameContext): void {
    if (!this.enabled) return;

    // Look.
    this.input.consumeMouseDelta(_look);
    if (this.input.readGamepadLook(_pad, dt)) {
      _look.x += _pad.x;
      _look.y += _pad.y;
    }
    if (_look.x !== 0 || _look.y !== 0) {
      this.euler.y -= _look.x;
      this.euler.x -= _look.y;
      this.euler.x = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, this.euler.x));
      this.camera.quaternion.setFromEuler(this.euler);
    }

    // Speed control.
    const wheel = this.input.consumeWheel();
    if (wheel !== 0) {
      this.speed *= Math.pow(1.0015, -wheel);
      this.speed = Math.max(0.5, Math.min(4000, this.speed));
    }

    // Movement basis.
    this.camera.getWorldDirection(_forward);
    _right.crossVectors(_forward, this.camera.up).normalize();

    _move.set(0, 0, 0);
    if (this.input.isDown('forward')) _move.add(_forward);
    if (this.input.isDown('back')) _move.sub(_forward);
    if (this.input.isDown('right')) _move.add(_right);
    if (this.input.isDown('left')) _move.sub(_right);
    if (this.input.isDown('jump') || this.input.isDown('up')) _move.y += 1;
    if (this.input.isDown('crouch') || this.input.isDown('down')) _move.y -= 1;

    if (this.input.readGamepadMove(_pad)) {
      _move.addScaledVector(_right, _pad.x);
      _move.addScaledVector(_forward, -_pad.y);
    }

    if (_move.lengthSq() > 0) _move.normalize();

    let speed = this.speed;
    if (this.input.isDown('sprint')) speed *= 6;
    if (this.input.isDown('walk')) speed *= 0.2;
    _move.multiplyScalar(speed);

    // Critically-damped-ish exponential approach; frame-rate independent.
    const k = 1 - Math.exp(-dt / Math.max(1e-4, this.smoothing));
    this.velocity.lerp(_move, k);

    this.camera.position.addScaledVector(this.velocity, dt);
  }

  dispose(): void {
    // Nothing owned.
  }
}
