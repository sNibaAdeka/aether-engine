/**
 * The world clock: date, time, latitude, longitude — and, derived from them,
 * where the sun and the moon are.
 *
 * This is the single authority for "when". The atmosphere reads it, the snow
 * line reads it, and phase 8's weather will read it. Nothing else is allowed to
 * keep its own idea of the time of day, because two clocks that disagree produce
 * a sunset happening at a different hour from the snow melting.
 *
 * The site is 62° N, 19° W — the North Atlantic, which is where the terrain's
 * measured relief (902 m maximum) and the 6 °C sea-level calibration of phase 2
 * put this world. At that latitude the solar arc is genuinely dramatic across
 * the year: 4.5 hours of daylight at the winter solstice and 20 at the summer
 * one, with hour-long golden light at either end of a summer day because the sun
 * approaches the horizon at a shallow angle. That shallow approach is worth more
 * to the look of the phase-3 screenshots than anything in the scattering code.
 */

import {
  celestialToWorldMatrix,
  daysSinceJ2000,
  equatorialToHorizontal,
  horizontalToWorld,
  julianDay,
  localSiderealTime,
  moonBrightnessFactor,
  moonState,
  sunEquatorial,
  type HorizontalCoords,
  type Vec3Like,
} from '@/math/Spherical';

/**
 * Game seconds per real second.
 *
 * 60 makes a full day take 24 real minutes. That is fast enough for the
 * screenshot sweep and for a player to watch a sunset happen, and slow enough
 * that the sun does not visibly slide while standing still.
 */
const DEFAULT_TIME_SCALE = 60;

/** Illuminance of a full moon at the zenith, mean distance, lux (spec 3.4). */
const FULL_MOON_LUX = 0.267;

/** Extraterrestrial solar illuminance normal to the beam, lux. */
export const SOLAR_CONSTANT_LUX = 128000;

export interface CelestialBody {
  /** Unit world-space direction toward the body. */
  direction: Vec3Like;
  /** Altitude above the horizon, radians. */
  altitude: number;
  /** Azimuth from North toward East, radians. */
  azimuth: number;
  /**
   * Illuminance this body would deliver on a surface facing it, *outside* the
   * atmosphere, lux. The atmosphere's transmittance is applied by the caller,
   * because it depends on the observer's altitude and on the turbidity.
   */
  illuminanceLux: number;
}

export class TimeOfDay {
  /** Degrees, North positive. */
  latitude = 62.0;
  /** Degrees, East positive. */
  longitude = -19.0;

  /** Calendar date, UTC. */
  year = 2026;
  month = 4;
  day = 14;

  /** Hours since local midnight, [0, 24). */
  hours = 12;

  /** Game seconds per real second. Set to 0 to freeze — the screenshot harness does. */
  timeScale = DEFAULT_TIME_SCALE;

  readonly sun: CelestialBody = {
    direction: { x: 0, y: 1, z: 0 },
    altitude: Math.PI / 2,
    azimuth: 0,
    illuminanceLux: SOLAR_CONSTANT_LUX,
  };

  readonly moon: CelestialBody = {
    direction: { x: 0, y: -1, z: 0 },
    altitude: -Math.PI / 2,
    azimuth: 0,
    illuminanceLux: 0,
  };

  /** Illuminated fraction of the lunar disc, 0..1. */
  moonIlluminated = 0;
  /** 0 new · 0.25 first quarter · 0.5 full · 0.75 last quarter. */
  moonPhase = 0;
  /**
   * World-space direction from the moon toward the sun, used to orient the
   * terminator on the drawn disc.
   */
  readonly moonToSun: Vec3Like = { x: 1, y: 0, z: 0 };

  /** Column-major 3×3, equatorial J2000 → world. Feeds the star field. */
  readonly celestialMatrix = new Float32Array(9);

  /** Year fraction 0..1, for the seasonal terms the snow line uses. */
  season = 0;

  // Module-level scratch would be shared between two TimeOfDay instances; these
  // are per-instance and reused every frame, which is the same guarantee.
  private _horizontal: HorizontalCoords = { altitude: 0, azimuth: 0 };
  private _sunEquatorial = { ra: 0, dec: 0 };

  constructor() {
    this.recompute();
  }

  /** Advance by `dt` real seconds. */
  update(dt: number): void {
    if (this.timeScale !== 0) {
      const gameHours = (dt * this.timeScale) / 3600;
      this.hours += gameHours;
      while (this.hours >= 24) {
        this.hours -= 24;
        this.advanceDay();
      }
    }
    this.recompute();
  }

  /** Jump to an hour of the current day. Drives `__AETHER_DEBUG__.setTimeOfDay`. */
  setHours(h: number): void {
    this.hours = ((h % 24) + 24) % 24;
    this.recompute();
  }

  private advanceDay(): void {
    const daysInMonth = new Date(Date.UTC(this.year, this.month, 0)).getUTCDate();
    this.day += 1;
    if (this.day > daysInMonth) {
      this.day = 1;
      this.month += 1;
      if (this.month > 12) {
        this.month = 1;
        this.year += 1;
      }
    }
  }

  /**
   * Recompute both bodies. Called every frame; allocates nothing.
   *
   * `hours` is local *mean solar* time, not UTC: the longitude offset is
   * subtracted before the almanac formulae see it. Without that the sun would
   * peak at 13:16 at this longitude and every screenshot label would be a lie.
   */
  private recompute(): void {
    const utcHours = this.hours - this.longitude / 15;
    const jd = julianDay(this.year, this.month, this.day, utcHours);
    const d = daysSinceJ2000(jd);
    const lst = localSiderealTime(d, this.longitude);

    const sunEq = sunEquatorial(d);
    this._sunEquatorial.ra = sunEq.ra;
    this._sunEquatorial.dec = sunEq.dec;
    equatorialToHorizontal(sunEq.ra, sunEq.dec, lst, this.latitude, this._horizontal);
    this.sun.altitude = this._horizontal.altitude;
    this.sun.azimuth = this._horizontal.azimuth;
    horizontalToWorld(this._horizontal, this.sun.direction);
    this.sun.illuminanceLux = SOLAR_CONSTANT_LUX;

    const moon = moonState(d);
    equatorialToHorizontal(moon.ra, moon.dec, lst, this.latitude, this._horizontal);
    this.moon.altitude = this._horizontal.altitude;
    this.moon.azimuth = this._horizontal.azimuth;
    horizontalToWorld(this._horizontal, this.moon.direction);

    this.moonIlluminated = moon.illuminatedFraction;
    this.moonPhase = moon.phase;
    // Inverse-square on the distance: perigee full moons are 30 % brighter than
    // apogee ones, which is a real and visible thing.
    const distanceFactor = (385000.56 / moon.distanceKm) ** 2;
    this.moon.illuminanceLux =
      FULL_MOON_LUX * moonBrightnessFactor(Math.PI - moon.elongation) * distanceFactor;

    // Terminator orientation: the bright limb faces the sun, so the direction
    // from moon to sun projected onto the disc is all the shader needs.
    this.moonToSun.x = this.sun.direction.x - this.moon.direction.x;
    this.moonToSun.y = this.sun.direction.y - this.moon.direction.y;
    this.moonToSun.z = this.sun.direction.z - this.moon.direction.z;
    const len =
      Math.hypot(this.moonToSun.x, this.moonToSun.y, this.moonToSun.z) || 1;
    this.moonToSun.x /= len;
    this.moonToSun.y /= len;
    this.moonToSun.z /= len;

    celestialToWorldMatrix(lst, this.latitude, this.celestialMatrix);

    // Day of year as a fraction, for the snow line's seasonal term. Cheap and
    // exact enough: the 0.25 day offset of a leap year is 0.07 % of a season.
    const startOfYear = julianDay(this.year, 1, 1, 0);
    this.season = ((jd - startOfYear) / 365.2422) % 1;
  }
}
