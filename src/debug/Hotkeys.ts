/**
 * Global hotkeys that are not owned by a single system.
 *
 * F1 fly/player, F2 clean screenshot, P photo mode. F3/F4/H are owned by the
 * overlay, settings panel and HUD respectively.
 */

import type { Engine } from '@/engine/Engine';
import type { FlyCamera } from '@/player/FlyCamera';
import type { HUD } from '@/ui/HUD';
import type { DebugOverlay } from '@/ui/DebugOverlay';

export type CameraMode = 'fly' | 'player';

export interface HotkeyDeps {
  engine: Engine;
  flyCamera: FlyCamera;
  hud: HUD;
  overlay: DebugOverlay;
  /** Present from phase 11 onward. Until then F1 reports that honestly. */
  playerController?: { setEnabled(v: boolean): void };
  onPhotoMode?: (enabled: boolean) => void;
}

export class Hotkeys {
  mode: CameraMode = 'fly';
  photoMode = false;

  constructor(private deps: HotkeyDeps) {
    window.addEventListener('keydown', this.onKey);
  }

  private onKey = (e: KeyboardEvent): void => {
    const target = e.target as HTMLElement | null;
    if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return;

    switch (e.code) {
      case 'F1':
        e.preventDefault();
        this.toggleMode();
        break;
      case 'F2':
        e.preventDefault();
        void this.captureScreenshot();
        break;
      case 'KeyP':
        if (e.metaKey || e.ctrlKey) return;
        this.togglePhotoMode();
        break;
      default:
        break;
    }
  };

  toggleMode(): void {
    const { flyCamera, playerController, hud } = this.deps;
    if (!playerController) {
      hud.showHint('Контроллер игрока появится в фазе 11 — пока только fly-камера');
      return;
    }
    this.mode = this.mode === 'fly' ? 'player' : 'fly';
    flyCamera.enabled = this.mode === 'fly';
    playerController.setEnabled(this.mode === 'player');
    hud.showHint(this.mode === 'fly' ? 'Свободная камера' : 'Игрок');
  }

  togglePhotoMode(): void {
    this.photoMode = !this.photoMode;
    const { hud, overlay, onPhotoMode } = this.deps;
    hud.setVisible(!this.photoMode);
    if (this.photoMode) overlay.setVisible(false);
    onPhotoMode?.(this.photoMode);
    if (!this.photoMode) hud.showHint('Фоторежим выключен');
  }

  /**
   * Full-resolution capture with the UI hidden. The renderer draws to the
   * canvas with `preserveDrawingBuffer: false`, so the read has to happen in
   * the same task as a fresh render — hence the explicit renderOnce().
   */
  private async captureScreenshot(): Promise<void> {
    const { engine, hud, overlay } = this.deps;
    const hudWasVisible = !this.photoMode;

    hud.setVisible(false);
    overlay.setVisible(false);

    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    engine.renderOnce();

    const canvas = engine.renderer.domElement as HTMLCanvasElement;
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));

    if (hudWasVisible) hud.setVisible(true);
    overlay.setVisible(engine.settings.debug.showOverlay);

    if (!blob) {
      hud.showHint('Не удалось сохранить скриншот');
      return;
    }

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    a.href = url;
    a.download = `aether-${stamp}.png`;
    a.click();
    URL.revokeObjectURL(url);

    hud.showHint(`Сохранено · ${canvas.width}×${canvas.height}`);
  }

  dispose(): void {
    window.removeEventListener('keydown', this.onKey);
  }
}
