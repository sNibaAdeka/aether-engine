/**
 * The animated snow line (spec 2.1: "не мгновенно — с плавным переходом за
 * минуты игрового времени").
 *
 * Two altitudes, not one. The *target* is where the 0 °C isotherm is right now,
 * which follows the season and the daily cycle and can move hundreds of metres
 * in an hour of game time. The *current* line chases it with a first-order lag
 * whose time constant is minutes, because snow is a mass of water that has to
 * physically fall or melt: an isotherm can jump, a snowpack cannot.
 *
 * Getting that distinction wrong is what makes weather in games read as a
 * switch being flipped. With the lag, a cold front arrives as snow creeping
 * down the mountain over several minutes, and a thaw as it retreating — which
 * is a thing the player can watch happen.
 *
 * The altitude itself comes from the same climate model the biome classifier
 * uses (`CLIMATE` in `shaders/splat.ts`): sea-level temperature divided by the
 * lapse rate is the height at which the air reaches freezing. Sharing that
 * constant is what keeps the snow the material draws in the same place as the
 * rock the classifier put under it.
 */

import { CLIMATE } from './shaders/splat';

/**
 * Seconds for the line to cover 63 % of the distance to its target.
 *
 * 150 s at the default 60× time compression is 2.5 minutes of real time for a
 * visible move and about 10 minutes to settle — slow enough to read as weather
 * rather than as a slider, fast enough that a player who stops to watch sees
 * something happen.
 */
const TIME_CONSTANT_S = 150;

/**
 * Amplitude of the seasonal swing in sea-level temperature, °C.
 *
 * ±6 °C around a 6 °C mean puts the winter line below sea level (snow to the
 * shore) and the summer one at about 1850 m (no snow at all, since nothing in
 * this world is that tall). The whole range is used, which is the point.
 */
const SEASON_AMPLITUDE_C = 6;

/**
 * Amplitude of the day/night swing, °C.
 *
 * 1.2 °C is only ±185 m of snow line, and that is the point: a real snowpack
 * does not retreat 500 m up the mountain at noon and come back down at
 * midnight. The daily term exists so the line is never quite still, not so it
 * dominates the seasonal one.
 */
const DIURNAL_AMPLITUDE_C = 1.2;

export interface SnowLineState {
  /** Altitude of the line being rendered, metres. */
  current: number;
  /** Altitude the line is heading for, metres. */
  target: number;
  /** Sea-level temperature driving the target, °C. */
  seaLevelC: number;
}

export class SnowLine {
  readonly state: SnowLineState = {
    current: CLIMATE.SNOWLINE_REF_M,
    target: CLIMATE.SNOWLINE_REF_M,
    seaLevelC: CLIMATE.BASE_C,
  };

  /**
   * Year fraction 0..1. Phase 8 owns this; until then it advances with time.
   *
   * 0.167 is mid-spring, which puts the snow line at about 645 m at noon — high
   * enough that the valleys are green and low enough that every peak over
   * 1000 m carries snow. That is the state the `mountain-noon` acceptance shot
   * is judged in, so it is the state the world starts in.
   */
  season = 0.167;
  /** Day fraction 0..1, midnight at 0. */
  timeOfDay = 0.5;

  /**
   * Game seconds per real second. Only used when no world clock is supplied;
   * 60 makes a full day pass in 24 minutes, which is what makes the lag
   * observable in a test run rather than only in a long session.
   */
  timeScale = 60;

  /**
   * Advance the model. `dt` is real seconds.
   *
   * The target is recomputed every frame and the current value integrated
   * toward it with an exponential that is *frame-rate independent* — a plain
   * `current += (target - current) * k` would make the snow line move faster on
   * a fast machine, which is the kind of bug that only shows up when someone
   * profiles on different hardware.
   *
   * Since phase 3 the season and the hour come from `world/TimeOfDay`, which is
   * the same clock the sun's position is computed from. They used to be
   * integrated here independently, and two clocks that can disagree is exactly
   * how a world ends up with the snow melting at a different hour from the one
   * the sun is overhead at.
   */
  update(dt: number, clock?: { season: number; hours: number; timeScale: number }): void {
    const gameDt = dt * (clock ? clock.timeScale : this.timeScale);
    if (clock) {
      this.season = clock.season;
      this.timeOfDay = clock.hours / 24;
    } else {
      this.season = (this.season + gameDt / (365 * 24 * 3600)) % 1;
      this.timeOfDay = (this.timeOfDay + gameDt / (24 * 3600)) % 1;
    }

    const seasonal = -Math.cos(this.season * Math.PI * 2) * SEASON_AMPLITUDE_C;
    const diurnal = -Math.cos(this.timeOfDay * Math.PI * 2) * DIURNAL_AMPLITUDE_C;
    this.state.seaLevelC = CLIMATE.BASE_C + seasonal + diurnal;

    // Height of the 0 °C isotherm. Clamped below at −200 m so a hard winter
    // parks the line under the terrain (snow everywhere) rather than at a
    // negative altitude the shader would have to special-case.
    this.state.target = Math.max(-200, this.state.seaLevelC / CLIMATE.LAPSE_C_PER_M);

    const k = 1 - Math.exp(-gameDt / TIME_CONSTANT_S);
    this.state.current += (this.state.target - this.state.current) * k;
  }

  /** Jump straight to the target. For tests and for the initial frame. */
  snap(clock?: { season: number; hours: number; timeScale: number }): void {
    this.update(0, clock);
    this.state.current = this.state.target;
  }
}
