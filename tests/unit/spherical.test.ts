/**
 * Astronomy sanity checks.
 *
 * These are not almanac-accuracy tests — the low-precision series in
 * `math/Spherical.ts` is only good to a hundredth of a degree for the sun and a
 * third of a degree for the moon, and comparing against a published ephemeris to
 * that tolerance needs the ephemeris. What they check is the class of mistake
 * that actually happens and is invisible on screen for weeks: a sign flip in the
 * hour angle, a degrees/radians mix, an azimuth measured from the wrong axis, a
 * frame convention that disagrees with the shader's.
 */

import { describe, expect, it } from 'vitest';
import {
  celestialToWorldMatrix,
  daysSinceJ2000,
  equatorialToHorizontal,
  greenwichMeanSiderealTime,
  horizontalToWorld,
  julianDay,
  moonBrightnessFactor,
  moonState,
  sunEquatorial,
  type HorizontalCoords,
} from '@/math/Spherical';

const RAD = 180 / Math.PI;
const h: HorizontalCoords = { altitude: 0, azimuth: 0 };

/** Solar altitude at local solar noon, degrees, for a latitude and a date. */
function noonAltitude(year: number, month: number, day: number, latitude: number): number {
  const jd = julianDay(year, month, day, 12);
  const d = daysSinceJ2000(jd);
  const sun = sunEquatorial(d);
  // At solar noon the hour angle is zero by definition, so feed the sun's own
  // right ascension as the sidereal time. That isolates the declination and the
  // latitude, which is what this is testing.
  equatorialToHorizontal(sun.ra, sun.dec, sun.ra, latitude, h);
  return h.altitude * RAD;
}

describe('julianDay', () => {
  it('matches the standard epoch', () => {
    // J2000.0 is 2000-01-01 12:00 UTC = JD 2451545.0.
    expect(julianDay(2000, 1, 1, 12)).toBeCloseTo(2451545.0, 6);
  });

  it('advances by one per day', () => {
    expect(julianDay(2026, 3, 1, 0) - julianDay(2026, 2, 28, 0)).toBeCloseTo(1, 9);
  });
});

describe('sun', () => {
  it('is overhead at the equator at an equinox', () => {
    // 20 March 2026 is the March equinox; declination is within 0.4° of zero.
    expect(Math.abs(noonAltitude(2026, 3, 20, 0) - 90)).toBeLessThan(1.0);
  });

  it('reaches 90 − φ + 23.44 at the June solstice', () => {
    const lat = 62;
    const expected = 90 - lat + 23.44;
    expect(noonAltitude(2026, 6, 21, lat)).toBeCloseTo(expected, 0);
  });

  it('reaches 90 − φ − 23.44 at the December solstice', () => {
    const lat = 62;
    const expected = 90 - lat - 23.44;
    expect(noonAltitude(2026, 12, 21, lat)).toBeCloseTo(expected, 0);
  });

  it('declination swings the full ±23.44° over a year', () => {
    let min = 90;
    let max = -90;
    for (let day = 0; day < 366; day++) {
      const dec = sunEquatorial(daysSinceJ2000(julianDay(2026, 1, 1, 12) + day)).dec * RAD;
      min = Math.min(min, dec);
      max = Math.max(max, dec);
    }
    expect(max).toBeGreaterThan(23.0);
    expect(min).toBeLessThan(-23.0);
  });
});

describe('sidereal time', () => {
  it('gains about four minutes a day on solar time', () => {
    const a = greenwichMeanSiderealTime(0);
    const b = greenwichMeanSiderealTime(1);
    // 360.98564736629 − 360 = 0.98564736629° per day = 3m 56s of sidereal gain.
    let delta = ((b - a) * RAD + 360) % 360;
    if (delta > 180) delta -= 360;
    expect(delta).toBeCloseTo(0.98564736629, 4);
  });
});

describe('horizontal frame', () => {
  it('puts the zenith at +Y and the north point at −Z', () => {
    const up = horizontalToWorld({ altitude: Math.PI / 2, azimuth: 0 }, { x: 0, y: 0, z: 0 });
    expect(up.y).toBeCloseTo(1, 6);

    const north = horizontalToWorld({ altitude: 0, azimuth: 0 }, { x: 0, y: 0, z: 0 });
    expect(north.z).toBeCloseTo(-1, 6);

    // Azimuth increases toward the east, which is +X.
    const east = horizontalToWorld(
      { altitude: 0, azimuth: Math.PI / 2 },
      { x: 0, y: 0, z: 0 },
    );
    expect(east.x).toBeCloseTo(1, 6);
  });

  it('puts the north celestial pole at altitude φ due north', () => {
    // Declination +90° is the pole; its hour angle is undefined, so any sidereal
    // time will do.
    equatorialToHorizontal(0, Math.PI / 2, 1.234, 62, h);
    expect(h.altitude * RAD).toBeCloseTo(62, 6);
    // Azimuth is wrapped to [0, 2π), so due north is 0 or 360 depending on
    // which side of the branch cut the rounding lands.
    const azNorth = Math.min(h.azimuth * RAD, 360 - h.azimuth * RAD);
    expect(azNorth).toBeCloseTo(0, 4);
  });
});

describe('celestialToWorldMatrix', () => {
  const m = new Float32Array(9);

  it('is orthonormal', () => {
    celestialToWorldMatrix(2.1, 62, m);
    // Columns are unit length and mutually perpendicular.
    for (let c = 0; c < 3; c++) {
      const x = m[c * 3];
      const y = m[c * 3 + 1];
      const z = m[c * 3 + 2];
      expect(Math.hypot(x, y, z)).toBeCloseTo(1, 5);
    }
    const dot01 = m[0] * m[3] + m[1] * m[4] + m[2] * m[5];
    const dot02 = m[0] * m[6] + m[1] * m[7] + m[2] * m[8];
    const dot12 = m[3] * m[6] + m[4] * m[7] + m[5] * m[8];
    expect(Math.abs(dot01)).toBeLessThan(1e-5);
    expect(Math.abs(dot02)).toBeLessThan(1e-5);
    expect(Math.abs(dot12)).toBeLessThan(1e-5);
  });

  it('agrees with equatorialToHorizontal for a sample of directions', () => {
    const lst = 2.1;
    const lat = 62;
    celestialToWorldMatrix(lst, lat, m);
    for (const [ra, dec] of [
      [0.4, 0.9],
      [3.2, -0.5],
      [5.9, 0.2],
      [1.1, -1.2],
    ]) {
      // Equatorial unit vector, the same convention the matrix is built for.
      const ex = Math.cos(dec) * Math.cos(ra);
      const ey = Math.cos(dec) * Math.sin(ra);
      const ez = Math.sin(dec);
      // Column-major multiply.
      const wx = m[0] * ex + m[3] * ey + m[6] * ez;
      const wy = m[1] * ex + m[4] * ey + m[7] * ez;
      const wz = m[2] * ex + m[5] * ey + m[8] * ez;

      equatorialToHorizontal(ra, dec, lst, lat, h);
      const ref = horizontalToWorld(h, { x: 0, y: 0, z: 0 });
      expect(wx).toBeCloseTo(ref.x, 5);
      expect(wy).toBeCloseTo(ref.y, 5);
      expect(wz).toBeCloseTo(ref.z, 5);
    }
  });
});

describe('moon', () => {
  it('cycles through a full set of phases in a synodic month', () => {
    let sawNew = false;
    let sawFull = false;
    const start = daysSinceJ2000(julianDay(2026, 1, 1, 0));
    for (let i = 0; i < 30; i++) {
      const s = moonState(start + i);
      if (s.illuminatedFraction < 0.05) sawNew = true;
      if (s.illuminatedFraction > 0.95) sawFull = true;
      expect(s.illuminatedFraction).toBeGreaterThanOrEqual(0);
      expect(s.illuminatedFraction).toBeLessThanOrEqual(1);
      expect(s.phase).toBeGreaterThanOrEqual(0);
      expect(s.phase).toBeLessThanOrEqual(1);
    }
    expect(sawNew).toBe(true);
    expect(sawFull).toBe(true);
  });

  it('moves about 13° a day against the stars', () => {
    const a = moonState(0).eclipticLongitude;
    const b = moonState(1).eclipticLongitude;
    let delta = ((b - a) * RAD + 360) % 360;
    if (delta > 180) delta -= 360;
    expect(delta).toBeGreaterThan(11);
    expect(delta).toBeLessThan(16);
  });

  it('stays within a plausible distance range', () => {
    for (let i = 0; i < 60; i++) {
      const d = moonState(i).distanceKm;
      expect(d).toBeGreaterThan(355000);
      expect(d).toBeLessThan(410000);
    }
  });

  it('falls off far faster than the illuminated fraction', () => {
    // The whole point of using Meeus's photometric law rather than the lit area:
    // a half moon is about a tenth of a full moon, not a half.
    expect(moonBrightnessFactor(0)).toBeCloseTo(1, 6);
    const half = moonBrightnessFactor(Math.PI / 2);
    expect(half).toBeLessThan(0.15);
    expect(half).toBeGreaterThan(0.03);
  });
});
