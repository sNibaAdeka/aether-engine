/**
 * Bootstrap only. Everything of substance lives in a system.
 */

import { Vector3 } from 'three/webgpu';
import { Engine } from '@/engine/Engine';
import { probeWebGPU } from '@/engine/Renderer';
import { Input } from '@/player/Input';
import { FlyCamera } from '@/player/FlyCamera';
import { DebugScene } from '@/debug/DebugScene';
import { TerrainSystem } from '@/terrain/TerrainSystem';
import { AtmosphereSystem } from '@/atmosphere/AtmosphereSystem';
import { findSpawn, height as terrainHeight } from '@/terrain/HeightField';
import { DebugOverlay } from '@/ui/DebugOverlay';
import { SettingsPanel } from '@/ui/SettingsPanel';
import { LoadingScreen } from '@/ui/LoadingScreen';
import { HUD } from '@/ui/HUD';
import { Hotkeys } from '@/debug/Hotkeys';
import { DebugViews } from '@/debug/DebugViews';
import { installDebugBridge, installErrorCollector } from '@/debug/DebugBridge';
import '@/ui/styles.css';

installErrorCollector();

const params = new URLSearchParams(location.search);
const seedParam = params.get('seed');
const forceWebGL = params.get('webgl') === '1';

async function boot(useFallback: boolean): Promise<void> {
  const container = document.getElementById('app');
  if (!container) throw new Error('#app container missing');

  const loading = new LoadingScreen();
  loading.plan([
    { label: 'Проверка WebGPU', weight: 1 },
    { label: 'Инициализация рендерера', weight: 3 },
    { label: 'Создание сцены', weight: 2 },
    { label: 'Интерфейс', weight: 1 },
    { label: 'Первый кадр', weight: 2 },
  ]);

  try {
    loading.step('Проверка WebGPU');
    const probe = await probeWebGPU();
    if (!probe.supported && !useFallback && !forceWebGL) {
      loading.showUnsupported(probe.reason ?? 'WebGPU не поддерживается', () => {
        void boot(true);
      });
      return;
    }
    if (probe.adapterInfo) {
      loading.setDetail(
        `${probe.adapterInfo.vendor || '?'} · ${probe.adapterInfo.architecture || '?'}`,
      );
    }

    loading.step('Инициализация рендерера');
    const engine = new Engine({
      container,
      seed: seedParam ? Number(seedParam) : 1337,
      forceWebGL: useFallback || forceWebGL,
    });
    await engine.init();
    engine.settings.restore();

    loading.step('Создание сцены');
    const input = new Input(engine.renderer.domElement as HTMLCanvasElement);
    const flyCamera = new FlyCamera(engine.camera, input);
    // Framed so the 1/2/10 m reference solids and the 1.75 m figure all read at
    // once — the point of the placeholder scene is judging scale.
    flyCamera.setPose(new Vector3(-1, 7, 30), new Vector3(-1, 2.5, 0));

    const terrain = new TerrainSystem();
    const atmosphere = new AtmosphereSystem();
    await engine.addSystem(new DebugScene());
    // Atmosphere before terrain: the snow line reads the world clock, and a
    // clock that updates after its consumers is a frame of skew in something
    // that is supposed to be the single source of truth for "when".
    await engine.addSystem(atmosphere);
    await engine.addSystem(terrain);
    await engine.addSystem(flyCamera);
    terrain.clock = atmosphere.time;
    // `ResizeController.start()` fires before any system exists, so the first
    // size has to be handed over by hand; every later one arrives by callback.
    atmosphere.onResize(engine.resize.renderWidth, engine.resize.renderHeight);

    // Start somewhere worth looking at: on land, near water, with an outlook.
    const spawn = findSpawn(engine.seed);
    flyCamera.setPose(
      new Vector3(spawn.x, spawn.y + 60, spawn.z + 180),
      new Vector3(spawn.x + 200, spawn.y + 30, spawn.z - 600),
    );

    // Since phase 3 the frame is no longer drawn straight to the swap chain:
    // the scene goes into a linear HDR target in cd/m², a compute histogram
    // meters it, and a tone-map quad presents it. A physical camera cannot be
    // applied any other way — you cannot expose a frame that has already been
    // clamped to [0,1]. Phase 10 extends this into the full post stack.
    engine.frameGraph.register('main-opaque', (ctx) => {
      atmosphere.renderFrame(ctx.dt);
    });

    loading.step('Интерфейс');
    const hud = new HUD();
    const overlay = new DebugOverlay(engine, engine.profiler);
    const settingsPanel = new SettingsPanel(engine.settings, {
      onSeedChange: (seed) => {
        const url = new URL(location.href);
        url.searchParams.set('seed', String(seed));
        location.href = url.toString();
      },
    });

    engine.frameGraph.register('ui', () => {
      overlay.update(performance.now());
      input.endFrame();
    });

    const hotkeys = new Hotkeys({ engine, flyCamera, hud, overlay });
    const debugViews = new DebugViews({
      onChange: (mode) => {
        terrain.uDebugView.value = mode;
      },
      notify: (text) => hud.showHint(text),
    });
    installDebugBridge({
      engine,
      flyCamera,
      hud,
      overlay,
      setTimeOfDay: (hours) => atmosphere.setTimeOfDay(hours),
      setWeather: (name) => atmosphere.setWeather(name),
      sampleHeight: (x, z) => terrainHeight(x, z, engine.seed),
    });

    engine.frameGraph.register('ui', () => {
      const t = terrain.debugState;
      const a = atmosphere.readout;
      overlay.readout.tiles = `${t.nodes} nodes · ${t.tilesReady} tiles`;
      overlay.readout.extra = {
        tris: `${(t.triangles / 1000).toFixed(0)}k (lvl≤${t.maxLevel})`,
        tiles: `+${t.tilesPending} queued · ${t.tilesFallback} coarse · ${t.tilesBaked} baked`,
        erosion:
          t.erosionProgress >= 0
            ? `${t.tilesEroded} eroded · ${(t.erosionProgress * 100).toFixed(0)}% current`
            : `${t.tilesEroded} eroded · idle`,
        biomes: `${t.tilesSplatted} splatted · снег ${t.snowLine.toFixed(0)}→${t.snowLineTarget.toFixed(0)} м`,
        sky:
          `${a.hours.toFixed(2)} ч · солнце ${a.sunAltitudeDeg.toFixed(1)}° · ` +
          `луна ${a.moonAltitudeDeg.toFixed(1)}° ф${a.moonPhase.toFixed(2)}`,
        light:
          `${a.sunLux.toFixed(0)} лк солнце · ${a.moonLux.toFixed(3)} лк луна · ` +
          `EV100 ${a.ev100.toFixed(2)} · L̄ ${a.avgLuminance.toPrecision(3)} кд/м²`,
      };
      // The flow and moisture views stay locked until the erosion pass has
      // actually written something, so an all-zero texture can never be
      // mistaken for a result. Terrain owns the flag; the switcher reads it.
      debugViews.erosionDataAvailable = terrain.erosionDataAvailable;
      debugViews.splatDataAvailable = t.tilesSplatted > 0;
    });

    loading.step('Первый кадр');
    engine.start();

    await new Promise<void>((resolve) => {
      if (engine.ready) return resolve();
      window.addEventListener('aether:ready', () => resolve(), { once: true });
    });

    await loading.hide();
    hud.showHint('Клик — захват курсора · F3 статистика · F4 настройки · H справка');

    // Keep a handle for console poking during development.
    Object.assign(window as unknown as Record<string, unknown>, {
      __AETHER__: {
        engine,
        input,
        flyCamera,
        settingsPanel,
        overlay,
        hud,
        hotkeys,
        debugViews,
        terrain,
        atmosphere,
      },
    });
  } catch (err) {
    console.error('[aether] boot failed', err);
    loading.showFatal(err instanceof Error ? `${err.message}\n\n${err.stack ?? ''}` : String(err));
  }
}

void boot(false);
