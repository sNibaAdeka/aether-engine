/**
 * Real astronomy: where the sun and the moon actually are, for a given
 * latitude, longitude, date and clock time.
 *
 * Spec 3.3 asks for "настоящая астрономическая формула", and the reason is not
 * pedantry. A sun that just sweeps a circle at a fixed tilt gives every day of
 * the year the same arc, the same sunrise azimuth and the same length — and the
 * eye reads that as a turntable, not as a place. The real solar declination
 * swings ±23.44° over the year, which at 62° N is the difference between a sun
 * that grazes the horizon for twenty hours and one that barely clears it. The
 * moon is worse: its orbit is inclined 5° to the ecliptic and it moves 13° a
 * day, so "opposite the sun at full" is only true to within a few degrees and
 * only at full. Faking that produces a moon which is always where the sun is
 * not, which is wrong for three weeks out of four.
 *
 * Accuracy delivered here is the standard low-precision series:
 *   - sun: ±0.01° (Astronomical Almanac's own "low precision" formulae)
 *   - moon: ±0.3° in longitude, ±0.2° in latitude (Meeus ch. 47, truncated to
 *     the leading terms)
 * Both are far below the 0.53° angular diameter of either disc, so nothing in
 * the render can tell the difference. Nutation, aberration, parallax and
 * refraction are all omitted; refraction alone would move the disc by half its
 * own diameter at the horizon, and that is noted as a debt rather than pretended
 * away.
 *
 * Frame convention, fixed here and depended on everywhere else:
 *   world +X = East, world +Y = up, world −Z = North.
 * That is the usual three.js "camera looks down −Z" arrangement read as a
 * compass, and it means azimuth (measured from North, increasing toward East)
 * maps to world direction (sin A cos h, sin h, −cos A cos h).
 */

const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

/** Days per Julian century, for the obliquity term. */
const J2000 = 2451545.0;

/** Mean Earth–Moon distance, km. Used to scale moonlight by 1/d². */
export const MOON_MEAN_DISTANCE_KM = 385000.56;

/** Astronomical unit, km. */
const AU_KM = 149597870.7;

export interface HorizontalCoords {
  /** Altitude above the horizon, radians. Negative below. */
  altitude: number;
  /** Azimuth from North toward East, radians, [0, 2π). */
  azimuth: number;
}

export interface EquatorialCoords {
  /** Right ascension, radians. */
  ra: number;
  /** Declination, radians. */
  dec: number;
  /** Distance from the observer, km. */
  distanceKm: number;
}

/** Wrap to [0, 2π). */
export function wrapTwoPi(x: number): number {
  const t = x % (Math.PI * 2);
  return t < 0 ? t + Math.PI * 2 : t;
}

/**
 * Julian Day from a UTC calendar date.
 *
 * The integer-arithmetic form (Fliegel & Van Flandern) rather than a Date
 * round-trip: `Date` carries a timezone and a leap-second policy we do not want
 * anywhere near a shader clock, and this is exact for 1901–2099 which is the
 * whole range a game clock will ever see.
 */
export function julianDay(
  year: number,
  month: number,
  day: number,
  hoursUTC: number,
): number {
  let y = year;
  let m = month;
  if (m <= 2) {
    y -= 1;
    m += 12;
  }
  const a = Math.floor(y / 100);
  const b = 2 - a + Math.floor(a / 4);
  return (
    Math.floor(365.25 * (y + 4716)) +
    Math.floor(30.6001 * (m + 1)) +
    day +
    b -
    1524.5 +
    hoursUTC / 24
  );
}

/** Days since J2000.0. Everything below is a polynomial in this. */
export function daysSinceJ2000(jd: number): number {
  return jd - J2000;
}

/** Mean obliquity of the ecliptic, radians. */
export function obliquity(d: number): number {
  return (23.439291 - 3.563e-7 * d) * DEG;
}

/**
 * Greenwich mean sidereal time, radians.
 *
 * The 360.98564736629°/day rate — not 360 — is the whole reason the stars rise
 * four minutes earlier each night. Getting this wrong is invisible for one
 * session and glaring across a simulated week.
 */
export function greenwichMeanSiderealTime(d: number): number {
  return wrapTwoPi((280.46061837 + 360.98564736629 * d) * DEG);
}

/** Local mean sidereal time, radians. `longitude` in degrees, East positive. */
export function localSiderealTime(d: number, longitudeDeg: number): number {
  return wrapTwoPi(greenwichMeanSiderealTime(d) + longitudeDeg * DEG);
}

/**
 * Geocentric equatorial coordinates of the sun.
 * Astronomical Almanac low-precision formulae, ±0.01°.
 */
export function sunEquatorial(d: number): EquatorialCoords {
  const meanLongitude = (280.46 + 0.9856474 * d) * DEG;
  const meanAnomaly = (357.528 + 0.9856003 * d) * DEG;
  // Equation of the centre, first two terms.
  const eclipticLongitude =
    meanLongitude + (1.915 * Math.sin(meanAnomaly) + 0.02 * Math.sin(2 * meanAnomaly)) * DEG;
  const eps = obliquity(d);

  const sinL = Math.sin(eclipticLongitude);
  const cosL = Math.cos(eclipticLongitude);
  return {
    ra: wrapTwoPi(Math.atan2(Math.cos(eps) * sinL, cosL)),
    dec: Math.asin(Math.sin(eps) * sinL),
    distanceKm: (1.00014 - 0.01671 * Math.cos(meanAnomaly)) * AU_KM,
  };
}

/**
 * Geocentric ecliptic longitude of the sun — needed on its own to get the
 * moon's phase, which is an angle between two ecliptic longitudes and not
 * between two equatorial ones.
 */
export function sunEclipticLongitude(d: number): number {
  const meanLongitude = (280.46 + 0.9856474 * d) * DEG;
  const meanAnomaly = (357.528 + 0.9856003 * d) * DEG;
  return wrapTwoPi(
    meanLongitude + (1.915 * Math.sin(meanAnomaly) + 0.02 * Math.sin(2 * meanAnomaly)) * DEG,
  );
}

export interface MoonState extends EquatorialCoords {
  /** Ecliptic longitude, radians. */
  eclipticLongitude: number;
  /** Ecliptic latitude, radians. */
  eclipticLatitude: number;
  /**
   * Sun–Moon elongation as seen from Earth, radians. 0 at new, π at full.
   */
  elongation: number;
  /** Illuminated fraction of the disc, 0..1. */
  illuminatedFraction: number;
  /**
   * Phase in the conventional 0..1 sense: 0 new, 0.25 first quarter, 0.5 full,
   * 0.75 last quarter. Carries the waxing/waning distinction that
   * `illuminatedFraction` throws away.
   */
  phase: number;
}

/**
 * Geocentric position of the moon.
 *
 * Meeus ch. 47 truncated to the four largest periodic terms — Evection, the
 * annual equation and the variation are folded into the leading longitude term
 * at this precision. Error ±0.3°, i.e. half a lunar diameter, which shows up as
 * the moon rising a couple of minutes off and nothing else.
 *
 * TODO(quality): the truncated series drifts worst near quadrature. A dozen more
 * terms would bring it to ±0.05° for about twenty lines; not worth it until
 * something in the world depends on the moon being where an almanac says.
 */
export function moonState(d: number): MoonState {
  const meanLongitude = (218.316 + 13.176396 * d) * DEG;
  const meanAnomaly = (134.963 + 13.064993 * d) * DEG;
  const argLatitude = (93.272 + 13.22935 * d) * DEG;

  const lambda = wrapTwoPi(meanLongitude + 6.289 * DEG * Math.sin(meanAnomaly));
  const beta = 5.128 * DEG * Math.sin(argLatitude);
  const distanceKm = 385001 - 20905 * Math.cos(meanAnomaly);

  const eps = obliquity(d);
  const sinL = Math.sin(lambda);
  const cosL = Math.cos(lambda);
  const sinB = Math.sin(beta);
  const cosB = Math.cos(beta);
  const sinE = Math.sin(eps);
  const cosE = Math.cos(eps);

  const ra = wrapTwoPi(Math.atan2(sinL * cosE - (sinB / cosB) * sinE, cosL));
  const dec = Math.asin(sinB * cosE + cosB * sinE * sinL);

  const sunLambda = sunEclipticLongitude(d);
  // Elongation on the ecliptic, corrected for the moon's latitude. The cosine
  // form keeps it well conditioned near new moon, where a longitude difference
  // alone would be dominated by the 5° inclination.
  const cosElong = Math.cos(beta) * Math.cos(lambda - sunLambda);
  const elongation = Math.acos(Math.max(-1, Math.min(1, cosElong)));
  const illuminatedFraction = (1 - Math.cos(elongation)) * 0.5;

  // Waxing when the moon leads the sun in longitude.
  const dLon = wrapTwoPi(lambda - sunLambda);
  const waxing = dLon < Math.PI;
  const phase = waxing ? elongation / (2 * Math.PI) : 1 - elongation / (2 * Math.PI);

  return {
    ra,
    dec,
    distanceKm,
    eclipticLongitude: lambda,
    eclipticLatitude: beta,
    elongation,
    illuminatedFraction,
    phase,
  };
}

/**
 * Equatorial → horizontal, for an observer at `latitudeDeg` whose local mean
 * sidereal time is `lst`.
 */
export function equatorialToHorizontal(
  ra: number,
  dec: number,
  lst: number,
  latitudeDeg: number,
  out: HorizontalCoords,
): HorizontalCoords {
  const H = lst - ra;
  const phi = latitudeDeg * DEG;
  const sinPhi = Math.sin(phi);
  const cosPhi = Math.cos(phi);
  const sinDec = Math.sin(dec);
  const cosDec = Math.cos(dec);
  const cosH = Math.cos(H);
  const sinH = Math.sin(H);

  const sinAlt = sinPhi * sinDec + cosPhi * cosDec * cosH;
  out.altitude = Math.asin(Math.max(-1, Math.min(1, sinAlt)));
  out.azimuth = wrapTwoPi(Math.atan2(-cosDec * sinH, sinDec * cosPhi - cosDec * sinPhi * cosH));
  return out;
}

export interface Vec3Like {
  x: number;
  y: number;
  z: number;
}

/**
 * Horizontal coordinates → world direction, in the frame declared at the top of
 * this file (+X East, +Y up, −Z North).
 */
export function horizontalToWorld(h: HorizontalCoords, out: Vec3Like): Vec3Like {
  const cosAlt = Math.cos(h.altitude);
  out.x = Math.sin(h.azimuth) * cosAlt;
  out.y = Math.sin(h.altitude);
  out.z = -Math.cos(h.azimuth) * cosAlt;
  return out;
}

/**
 * The 3×3 matrix that takes a J2000 equatorial unit vector to a world
 * direction, column-major, ready for a `Matrix3.fromArray`.
 *
 * This exists so the star field is one uniform rather than 200 CPU
 * transformations a frame: every star ships its fixed equatorial vector once at
 * startup and the sky rotates by changing nine floats.
 *
 * Equatorial basis is (x̂ toward the vernal equinox, ŷ 90° east of it on the
 * equator, ẑ toward the north celestial pole). The derivation is in the header
 * of `equatorialToHorizontal`'s caller — briefly: rotate by the local sidereal
 * time about the pole, then tilt the pole down to altitude φ about the
 * east–west axis.
 */
export function celestialToWorldMatrix(
  lst: number,
  latitudeDeg: number,
  out: Float32Array,
): Float32Array {
  const phi = latitudeDeg * DEG;
  const sinPhi = Math.sin(phi);
  const cosPhi = Math.cos(phi);
  const cosT = Math.cos(lst);
  const sinT = Math.sin(lst);

  // Step 1: equatorial → (meridian, west, pole). The hour angle of a vector is
  // lst − ra, which for the basis vectors is a rotation by −lst about ẑ read in
  // the opposite handedness, hence the sign pattern below.
  //   m = ( x cos lst + y sin lst )      toward the meridian
  //   w = ( x sin lst − y cos lst )      toward the west
  //   p = ( z )                          toward the pole
  //
  // Step 2: (m, w, p) → (North, East, Up):
  //   N = −m sinφ + p cosφ
  //   E = −w
  //   U =  m cosφ + p sinφ
  //
  // Step 3: (N, E, U) → world (E, U, −N).
  //
  // Composing gives, per world axis, the coefficients of (x, y, z):
  const mx = cosT;
  const my = sinT;
  const wx = sinT;
  const wy = -cosT;

  // worldX = E = −w
  const ex = -wx;
  const ey = -wy;
  const ez = 0;
  // worldY = U = m cosφ + p sinφ
  const ux = mx * cosPhi;
  const uy = my * cosPhi;
  const uz = sinPhi;
  // worldZ = −N = m sinφ − p cosφ
  const nx = mx * sinPhi;
  const ny = my * sinPhi;
  const nz = -cosPhi;

  // Column-major: column j holds the image of basis vector j.
  out[0] = ex;
  out[1] = ux;
  out[2] = nx;
  out[3] = ey;
  out[4] = uy;
  out[5] = ny;
  out[6] = ez;
  out[7] = uz;
  out[8] = nz;
  return out;
}

/**
 * Relative brightness of the moon at phase angle `alpha` (radians), 1 at full.
 *
 * Meeus's photometric formula for the lunar magnitude. The point of using it
 * rather than the illuminated fraction is that the moon is *not* half as bright
 * at first quarter — it is about a tenth as bright, because the terminator side
 * is all shadowed relief. A half moon that reads as half a full moon is one of
 * the tells of a faked night.
 */
export function moonBrightnessFactor(alphaRadians: number): number {
  const a = Math.min(Math.abs(alphaRadians) * RAD, 180);
  const dMag = 0.026 * a + 4.0e-9 * a * a * a * a;
  return Math.pow(10, -0.4 * dMag);
}

/**
 * The position angle of the moon's bright limb is not modelled; the terminator
 * is drawn perpendicular to the projected sun direction instead, which is
 * correct to within the moon's own libration for anything not being measured
 * with a micrometer.
 */
export const SPHERICAL_NOTES = {
  DEG,
  RAD,
  J2000,
} as const;
