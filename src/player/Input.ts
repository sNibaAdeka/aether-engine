/**
 * Keyboard / mouse / gamepad input with pointer lock.
 *
 * State is polled, not event-driven, so the simulation reads a consistent
 * snapshot per frame. Mouse deltas accumulate between polls and are consumed by
 * `consumeMouseDelta`, which prevents the camera from double-applying a delta
 * when a frame renders twice.
 */

export type ActionName =
  | 'forward'
  | 'back'
  | 'left'
  | 'right'
  | 'jump'
  | 'sprint'
  | 'crouch'
  | 'walk'
  | 'up'
  | 'down';

const KEY_MAP: Record<string, ActionName> = {
  KeyW: 'forward',
  ArrowUp: 'forward',
  KeyS: 'back',
  ArrowDown: 'back',
  KeyA: 'left',
  ArrowLeft: 'left',
  KeyD: 'right',
  ArrowRight: 'right',
  Space: 'jump',
  ShiftLeft: 'sprint',
  ShiftRight: 'sprint',
  ControlLeft: 'crouch',
  KeyC: 'crouch',
  AltLeft: 'walk',
  KeyE: 'up',
  KeyQ: 'down',
};

export class Input {
  private actions = new Set<ActionName>();
  private keysDown = new Set<string>();
  private justPressed = new Set<string>();

  mouseDX = 0;
  mouseDY = 0;
  wheelDelta = 0;
  pointerLocked = false;

  /** Radians per pixel of mouse movement. */
  sensitivity = 0.0022;
  invertY = false;

  private element: HTMLElement;
  private gamepadIndex = -1;
  private enabled = true;

  constructor(element: HTMLElement) {
    this.element = element;
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
    window.addEventListener('blur', this.onBlur);
    element.addEventListener('mousedown', this.onMouseDown);
    document.addEventListener('mousemove', this.onMouseMove);
    document.addEventListener('pointerlockchange', this.onPointerLockChange);
    element.addEventListener('wheel', this.onWheel, { passive: true });
    window.addEventListener('gamepadconnected', this.onGamepadConnected);
    window.addEventListener('gamepaddisconnected', this.onGamepadDisconnected);
  }

  /** Disable movement input (photo mode UI, settings panel focus). */
  setEnabled(v: boolean): void {
    this.enabled = v;
    if (!v) this.actions.clear();
  }

  private onKeyDown = (e: KeyboardEvent): void => {
    // Never swallow browser shortcuts or typing in a field.
    if (e.metaKey || e.ctrlKey) return;
    const target = e.target as HTMLElement | null;
    if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return;

    if (!this.keysDown.has(e.code)) this.justPressed.add(e.code);
    this.keysDown.add(e.code);

    const action = KEY_MAP[e.code];
    if (action && this.enabled) {
      this.actions.add(action);
      if (e.code === 'Space') e.preventDefault();
    }
  };

  private onKeyUp = (e: KeyboardEvent): void => {
    this.keysDown.delete(e.code);
    const action = KEY_MAP[e.code];
    if (action) this.actions.delete(action);
  };

  private onBlur = (): void => {
    this.actions.clear();
    this.keysDown.clear();
  };

  private onMouseDown = (): void => {
    if (!this.pointerLocked && this.enabled) {
      void this.element.requestPointerLock();
    }
  };

  private onMouseMove = (e: MouseEvent): void => {
    if (!this.pointerLocked) return;
    this.mouseDX += e.movementX;
    this.mouseDY += e.movementY;
  };

  private onPointerLockChange = (): void => {
    this.pointerLocked = document.pointerLockElement === this.element;
    if (!this.pointerLocked) {
      this.mouseDX = 0;
      this.mouseDY = 0;
    }
  };

  private onWheel = (e: WheelEvent): void => {
    this.wheelDelta += e.deltaY;
  };

  private onGamepadConnected = (e: GamepadEvent): void => {
    this.gamepadIndex = e.gamepad.index;
  };

  private onGamepadDisconnected = (): void => {
    this.gamepadIndex = -1;
  };

  isDown(action: ActionName): boolean {
    return this.actions.has(action);
  }

  isKeyDown(code: string): boolean {
    return this.keysDown.has(code);
  }

  /** True exactly once per physical key press. */
  wasPressed(code: string): boolean {
    if (this.justPressed.has(code)) {
      this.justPressed.delete(code);
      return true;
    }
    return false;
  }

  /** Reads and zeroes the accumulated look delta, in radians. */
  consumeMouseDelta(out: { x: number; y: number }): void {
    out.x = this.mouseDX * this.sensitivity;
    out.y = this.mouseDY * this.sensitivity * (this.invertY ? -1 : 1);
    this.mouseDX = 0;
    this.mouseDY = 0;
  }

  consumeWheel(): number {
    const d = this.wheelDelta;
    this.wheelDelta = 0;
    return d;
  }

  /** Left stick as [-1,1]², deadzoned. Writes into `out`; returns true if any. */
  readGamepadMove(out: { x: number; y: number }): boolean {
    if (this.gamepadIndex < 0) return false;
    const pad = navigator.getGamepads?.()[this.gamepadIndex];
    if (!pad) return false;
    const dz = 0.15;
    const ax = pad.axes[0] ?? 0;
    const ay = pad.axes[1] ?? 0;
    out.x = Math.abs(ax) > dz ? ax : 0;
    out.y = Math.abs(ay) > dz ? ay : 0;
    return out.x !== 0 || out.y !== 0;
  }

  /** Right stick, for look. Returns radians for this frame given `dt`. */
  readGamepadLook(out: { x: number; y: number }, dt: number): boolean {
    if (this.gamepadIndex < 0) return false;
    const pad = navigator.getGamepads?.()[this.gamepadIndex];
    if (!pad) return false;
    const dz = 0.15;
    const ax = pad.axes[2] ?? 0;
    const ay = pad.axes[3] ?? 0;
    const speed = 2.5 * dt;
    out.x = (Math.abs(ax) > dz ? ax : 0) * speed;
    out.y = (Math.abs(ay) > dz ? ay : 0) * speed * (this.invertY ? -1 : 1);
    return out.x !== 0 || out.y !== 0;
  }

  /** Call once per frame after all consumers have read state. */
  endFrame(): void {
    this.justPressed.clear();
  }

  dispose(): void {
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    window.removeEventListener('blur', this.onBlur);
    this.element.removeEventListener('mousedown', this.onMouseDown);
    document.removeEventListener('mousemove', this.onMouseMove);
    document.removeEventListener('pointerlockchange', this.onPointerLockChange);
    this.element.removeEventListener('wheel', this.onWheel);
    window.removeEventListener('gamepadconnected', this.onGamepadConnected);
    window.removeEventListener('gamepaddisconnected', this.onGamepadDisconnected);
  }
}
