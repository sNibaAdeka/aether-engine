/**
 * F4 settings panel (lil-gui).
 *
 * Every knob from the quality table plus per-effect toggles. Changing a preset
 * rewrites the individual controls so the UI never lies about what is active.
 */

import GUI from 'lil-gui';
import { PRESETS, type QualityPreset, type Settings } from '@/engine/Settings';

export interface SettingsPanelHooks {
  /** Called when the user asks for a benchmark-driven preset. */
  onAutoDetect?: () => void;
  /** Called when the world seed changes; the world must regenerate. */
  onSeedChange?: (seed: number) => void;
  /** Exposed so later phases can add their own folders. */
  onReady?: (gui: GUI) => void;
}

export class SettingsPanel {
  private gui: GUI;
  private visible = false;
  private qualityControllers: Array<{ updateDisplay: () => void }> = [];

  private unsubscribe: () => void;
  private presetController: { updateDisplay: () => void } | null = null;
  private presetProxy = { preset: 'high' as QualityPreset | 'auto', seed: 1337 };

  constructor(
    private settings: Settings,
    private hooks: SettingsPanelHooks = {},
  ) {
    this.gui = new GUI({ title: 'AETHER', width: 300 });
    this.gui.domElement.classList.add('aether-gui');
    this.build();
    this.setVisible(false);
    window.addEventListener('keydown', this.onKey);
    // Resync on *any* settings change, not just changes made through this
    // panel: the debug bridge and the auto-quality controller both mutate
    // settings, and a panel showing stale values is worse than no panel.
    this.unsubscribe = settings.onChange(() => this.refreshAll());
    this.hooks.onReady?.(this.gui);
  }

  private build(): void {
    const s = this.settings;

    this.presetProxy.preset = s.preset;

    this.presetController = this.gui
      .add(this.presetProxy, 'preset', ['auto', ...Object.keys(PRESETS)])
      .name('Пресет качества')
      .onChange((v: string) => {
        if (v === 'auto') {
          if (this.hooks.onAutoDetect) {
            this.hooks.onAutoDetect();
          } else {
            // Not implemented until phase 12. Say so instead of silently
            // leaving the dropdown reading "auto" while nothing happened.
            console.warn('[aether] auto quality detection is not implemented yet (phase 12)');
            this.presetProxy.preset = s.preset;
            this.presetController?.updateDisplay();
          }
          return;
        }
        s.applyPreset(v as QualityPreset);
        s.save();
      });

    this.gui
      .add(this.presetProxy, 'seed')
      .name('Seed мира')
      .onFinishChange((v: number) => this.hooks.onSeedChange?.(Math.floor(v)));

    // --- Render ---------------------------------------------------------
    const fRender = this.gui.addFolder('Рендер');
    this.qualityControllers.push(
      fRender
        .add(s.quality, 'renderScale', 0.4, 1.0, 0.05)
        .name('Render scale')
        .onChange(() => {
          s.setEffectiveRenderScale(s.quality.renderScale);
          s.emit();
        }),
      fRender
        .add(s.quality, 'drawDistance', 500, 20000, 100)
        .name('Дальность, м')
        .onChange(() => s.emit()),
      fRender.add(s, 'fov', 50, 90, 1).name('FOV').onChange(() => s.emit()),
      fRender.add(s, 'dynamicResolution').name('Динамическое разрешение'),
      fRender.add(s, 'targetFrameMs', 8, 33, 0.1).name('Целевой кадр, мс'),
    );

    // --- Terrain --------------------------------------------------------
    const fTerrain = this.gui.addFolder('Террейн');
    this.qualityControllers.push(
      fTerrain.add(s.quality, 'lodDepth', 4, 10, 1).name('Глубина LOD').onChange(() => s.emit()),
      fTerrain.add(s.quality, 'tileBudget', 1, 6, 1).name('Тайлов/кадр').onChange(() => s.emit()),
    );

    // --- Shadows --------------------------------------------------------
    const fShadow = this.gui.addFolder('Тени');
    this.qualityControllers.push(
      fShadow
        .add(s.quality, 'shadowCascades', 1, 4, 1)
        .name('Каскадов')
        .onChange(() => s.emit()),
      fShadow
        .add(s.quality, 'shadowResolution', [512, 1024, 2048, 4096])
        .name('Разрешение')
        .onChange(() => s.emit()),
    );

    // --- Sky & clouds ---------------------------------------------------
    const fSky = this.gui.addFolder('Небо и облака');
    this.qualityControllers.push(
      fSky
        .add(s.quality, 'clouds', ['flat', 'quarter', 'half', 'half128'])
        .name('Облака')
        .onChange(() => s.emit()),
      fSky
        .add(s.quality, 'volumetricLight', 0, 3, 1)
        .name('Объёмный свет')
        .onChange(() => s.emit()),
    );

    // --- Vegetation -----------------------------------------------------
    const fVeg = this.gui.addFolder('Растительность');
    this.qualityControllers.push(
      fVeg.add(s.quality, 'grassRadius', 0, 200, 5).name('Радиус травы, м').onChange(() => s.emit()),
      fVeg.add(s.quality, 'grassDensity', 0, 2, 0.05).name('Плотность').onChange(() => s.emit()),
    );

    // --- Water ----------------------------------------------------------
    const fWater = this.gui.addFolder('Вода');
    this.qualityControllers.push(
      fWater
        .add(s.quality, 'water', ['gerstner4', 'gerstner8', 'fft256', 'fft512'])
        .name('Симуляция')
        .onChange(() => s.emit()),
      fWater
        .add(s.quality, 'ssr', ['off', 'water', 'waterWet', 'all'])
        .name('SSR')
        .onChange(() => s.emit()),
    );

    // --- Post -----------------------------------------------------------
    const fPost = this.gui.addFolder('Пост-обработка');
    fPost
      .add(s.quality, 'antialias', ['none', 'fxaa', 'taa', 'taa2x'])
      .name('Сглаживание')
      .onChange(() => s.emit());
    fPost.add(s.quality, 'gtao', ['off', 'half', 'full']).name('GTAO').onChange(() => s.emit());
    fPost.add(s.post, 'bloom').name('Bloom').onChange(() => s.emit());
    fPost.add(s.post, 'bloomStrength', 0, 0.3, 0.005).name('Bloom сила').onChange(() => s.emit());
    fPost.add(s.post, 'autoExposure').name('Авто-экспозиция').onChange(() => s.emit());
    fPost
      .add(s.post, 'exposureCompensation', -4, 4, 0.1)
      .name('Экспокоррекция, EV')
      .onChange(() => s.emit());
    fPost
      .add(s.post, 'tonemap', ['agx', 'aces', 'reinhard', 'none'])
      .name('Тонемап')
      .onChange(() => s.emit());
    fPost.add(s.post, 'dof').name('DOF').onChange(() => s.emit());
    fPost.add(s.post, 'motionBlur').name('Motion blur').onChange(() => s.emit());
    fPost.add(s.post, 'filmGrain', 0, 0.08, 0.002).name('Зерно').onChange(() => s.emit());
    fPost
      .add(s.post, 'chromaticAberration', 0, 2, 0.05)
      .name('Хром. аберрация')
      .onChange(() => s.emit());
    fPost.add(s.post, 'vignette', 0, 1, 0.02).name('Виньетка').onChange(() => s.emit());
    fPost.add(s.post, 'sharpen', 0, 1, 0.02).name('Резкость (CAS)').onChange(() => s.emit());
    fPost.close();

    // --- Debug ----------------------------------------------------------
    const fDebug = this.gui.addFolder('Отладка');
    fDebug.add(s.debug, 'wireframe').name('Wireframe').onChange(() => s.emit());
    fDebug.add(s.debug, 'showLodColors').name('Цвета LOD').onChange(() => s.emit());
    fDebug.add(s.debug, 'showTileGrid').name('Сетка тайлов').onChange(() => s.emit());
    fDebug.add(s.debug, 'freezeCulling').name('Заморозить culling').onChange(() => s.emit());
    fDebug.add(s.debug, 'disableFog').name('Без тумана').onChange(() => s.emit());
    fDebug.close();

    // --- Comfort --------------------------------------------------------
    const fComfort = this.gui.addFolder('Комфорт');
    fComfort.add(s, 'headBob', 0, 1.5, 0.05).name('Покачивание камеры');
    fComfort.add(s, 'volume', 0, 1, 0.02).name('Громкость').onChange(() => s.emit());
    fComfort.add(s, 'reducedMotion').name('Меньше движения').onChange(() => s.emit());
  }

  /**
   * Re-sync every displayed value from the settings object.
   *
   * `Settings` mutates `quality`/`post` in place precisely so the controllers
   * keep pointing at live data — this only refreshes what the widgets show.
   */
  private refreshAll(): void {
    this.presetProxy.preset = this.settings.preset;
    this.presetController?.updateDisplay();
    for (const c of this.qualityControllers) c.updateDisplay();
    this.gui.controllersRecursive().forEach((c) => c.updateDisplay());
  }

  /** Later phases attach their own folders through this. */
  get root(): GUI {
    return this.gui;
  }

  private onKey = (e: KeyboardEvent): void => {
    if (e.code === 'F4') {
      e.preventDefault();
      this.setVisible(!this.visible);
    }
  };

  setVisible(v: boolean): void {
    this.visible = v;
    this.gui.domElement.style.display = v ? '' : 'none';
  }

  dispose(): void {
    window.removeEventListener('keydown', this.onKey);
    this.unsubscribe();
    this.gui.destroy();
  }
}
