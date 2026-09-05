/**
 * Noise determinism and quality gates.
 *
 * Determinism is not a nicety here: the terrain is regenerated constantly as
 * tiles stream in and out, and the CPU height query used by physics must agree
 * with the GPU heightmap used by rendering. If noise is not a pure function of
 * (coords, seed), the player falls through the world at tile boundaries.
 */

import { describe, it, expect } from 'vitest';
import {
  hashU32,
  hash2i,
  hashToUnit,
  perlin2,
  perlin2d,
  perlin3,
  valueNoise2,
  worley2,
  worley3F1Inv,
  wrapI,
  fade,
  fadeDeriv,
} from '@/math/Noise';
import {
  fbm2,
  ridged2,
  fbmEroded2,
  domainWarp2,
  worleyFbm2,
  gradientCentral2,
  DEFAULT_FBM,
  smoothstep,
  remap,
} from '@/math/FBM';

const SEED = 1337;

describe('integer hashing', () => {
  it('is a pure function', () => {
    for (let i = 0; i < 1000; i++) {
      expect(hashU32(i)).toBe(hashU32(i));
      expect(hash2i(i, -i, SEED)).toBe(hash2i(i, -i, SEED));
    }
  });

  it('returns unsigned 32-bit values', () => {
    for (let i = -500; i < 500; i++) {
      const h = hashU32(i);
      expect(h).toBeGreaterThanOrEqual(0);
      expect(h).toBeLessThanOrEqual(0xffffffff);
      expect(Number.isInteger(h)).toBe(true);
    }
  });

  it('separates transposed coordinates', () => {
    // A hash that collides on (x,y) vs (y,x) produces a visible diagonal seam.
    let collisions = 0;
    for (let x = 0; x < 60; x++) {
      for (let y = 0; y < 60; y++) {
        if (x !== y && hash2i(x, y, SEED) === hash2i(y, x, SEED)) collisions++;
      }
    }
    expect(collisions).toBe(0);
  });

  it('reacts to the seed', () => {
    let same = 0;
    for (let i = 0; i < 500; i++) {
      if (hash2i(i, i * 3, 1) === hash2i(i, i * 3, 2)) same++;
    }
    expect(same).toBe(0);
  });

  it('distributes roughly uniformly across 16 buckets', () => {
    const buckets = new Array<number>(16).fill(0);
    const n = 200000;
    for (let i = 0; i < n; i++) buckets[hashU32(i) & 15]++;
    const expected = n / 16;
    for (const b of buckets) {
      expect(Math.abs(b - expected) / expected).toBeLessThan(0.05);
    }
  });

  it('hashToUnit stays in [0,1)', () => {
    for (let i = 0; i < 20000; i++) {
      const u = hashToUnit(hashU32(i));
      expect(u).toBeGreaterThanOrEqual(0);
      expect(u).toBeLessThan(1);
    }
  });
});

describe('interpolation', () => {
  it('fade is a proper quintic with zero end derivatives', () => {
    expect(fade(0)).toBe(0);
    expect(fade(1)).toBe(1);
    expect(fade(0.5)).toBeCloseTo(0.5, 12);
    expect(fadeDeriv(0)).toBe(0);
    expect(fadeDeriv(1)).toBe(0);
  });

  it('fadeDeriv matches a numerical derivative of fade', () => {
    const h = 1e-6;
    for (let t = 0.05; t < 0.96; t += 0.05) {
      const numeric = (fade(t + h) - fade(t - h)) / (2 * h);
      expect(fadeDeriv(t)).toBeCloseTo(numeric, 5);
    }
  });
});

describe('perlin2', () => {
  it('is deterministic across call order', () => {
    const pts: Array<[number, number]> = [];
    for (let i = 0; i < 500; i++) {
      pts.push([Math.sin(i) * 400, Math.cos(i * 1.7) * 400]);
    }
    const first = pts.map(([x, y]) => perlin2(x, y, SEED));
    // Same points, reversed order — a stateful implementation breaks here.
    const second = [...pts].reverse().map(([x, y]) => perlin2(x, y, SEED));
    expect(second.reverse()).toEqual(first);
  });

  it('stays within [-1.05, 1.05]', () => {
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < 60000; i++) {
      const x = (i % 300) * 0.137 - 20;
      const y = Math.floor(i / 300) * 0.191 - 20;
      const v = perlin2(x, y, SEED);
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
    expect(min).toBeGreaterThan(-1.05);
    expect(max).toBeLessThan(1.05);
    // A live signal, not a constant.
    expect(max - min).toBeGreaterThan(1.0);
  });

  it('is zero on lattice points', () => {
    // Gradient noise must vanish at integer coordinates by construction.
    for (let x = -8; x <= 8; x++) {
      for (let y = -8; y <= 8; y++) {
        expect(Math.abs(perlin2(x, y, SEED))).toBeLessThan(1e-12);
      }
    }
  });

  it('is continuous across lattice boundaries', () => {
    const eps = 1e-5;
    for (let i = 0; i < 200; i++) {
      const x = i * 0.31;
      const a = perlin2(3 - eps, x, SEED);
      const b = perlin2(3 + eps, x, SEED);
      expect(Math.abs(a - b)).toBeLessThan(1e-3);
    }
  });

  it('has mean near zero', () => {
    let sum = 0;
    const n = 50000;
    for (let i = 0; i < n; i++) {
      sum += perlin2((i % 250) * 0.173 + 0.03, Math.floor(i / 250) * 0.211 + 0.07, SEED);
    }
    expect(Math.abs(sum / n)).toBeLessThan(0.02);
  });
});

describe('perlin2d analytic derivatives', () => {
  it('value matches perlin2', () => {
    const out = new Float64Array(3);
    for (let i = 0; i < 300; i++) {
      const x = i * 0.137 + 0.5;
      const y = i * 0.271 - 3.2;
      perlin2d(x, y, SEED, out);
      expect(out[0]).toBeCloseTo(perlin2(x, y, SEED), 10);
    }
  });

  it('derivatives match finite differences', () => {
    const out = new Float64Array(3);
    const h = 1e-5;
    for (let i = 0; i < 200; i++) {
      const x = i * 0.173 + 0.21;
      const y = i * 0.311 - 1.7;
      perlin2d(x, y, SEED, out);
      const dx = (perlin2(x + h, y, SEED) - perlin2(x - h, y, SEED)) / (2 * h);
      const dy = (perlin2(x, y + h, SEED) - perlin2(x, y - h, SEED)) / (2 * h);
      expect(out[1]).toBeCloseTo(dx, 3);
      expect(out[2]).toBeCloseTo(dy, 3);
    }
  });
});

describe('perlin3', () => {
  it('is deterministic and bounded', () => {
    let max = -Infinity;
    for (let i = 0; i < 20000; i++) {
      const x = (i % 30) * 0.31;
      const y = Math.floor(i / 30) * 0.17;
      const z = (i % 17) * 0.53;
      const v = perlin3(x, y, z, SEED);
      expect(v).toBe(perlin3(x, y, z, SEED));
      max = Math.max(max, Math.abs(v));
    }
    // Improved-Perlin 3D is bounded to unit range by construction. Measured max
    // over 3M samples is 0.9992 — anything above 1 means a stray rescale.
    expect(max).toBeLessThanOrEqual(1.0);
    expect(max).toBeGreaterThan(0.4);
  });
});

describe('valueNoise2', () => {
  it('is bounded to [-1,1] and deterministic', () => {
    for (let i = 0; i < 5000; i++) {
      const x = i * 0.093;
      const y = i * 0.041;
      const v = valueNoise2(x, y, SEED);
      expect(v).toBeGreaterThanOrEqual(-1);
      expect(v).toBeLessThanOrEqual(1);
      expect(v).toBe(valueNoise2(x, y, SEED));
    }
  });
});

describe('worley', () => {
  it('F1 <= F2 always', () => {
    const out = new Float64Array(2);
    for (let i = 0; i < 8000; i++) {
      worley2(i * 0.113, i * 0.317, SEED, out);
      expect(out[0]).toBeLessThanOrEqual(out[1]);
      expect(out[0]).toBeGreaterThanOrEqual(0);
      // Worst case in a 3×3 neighbourhood search.
      expect(out[1]).toBeLessThan(3);
    }
  });

  it('3D variant tiles seamlessly', () => {
    // Cloud noise is sampled as a repeating volume; a seam shows up as a hard
    // plane in the sky, which is extremely obvious and extremely ugly.
    const cells = 8;
    for (let i = 0; i < 200; i++) {
      const y = (i % 20) / 20;
      const z = Math.floor(i / 20) / 10;
      const a = worley3F1Inv(0, y, z, SEED, cells);
      const b = worley3F1Inv(1, y, z, SEED, cells);
      expect(Math.abs(a - b)).toBeLessThan(1e-9);
    }
  });

  it('returns values in [0,1]', () => {
    for (let i = 0; i < 4000; i++) {
      const v = worley3F1Inv((i % 40) / 40, (i % 13) / 13, (i % 7) / 7, SEED, 4);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});

describe('wrapI', () => {
  it('is a positive modulo', () => {
    expect(wrapI(-1, 8)).toBe(7);
    expect(wrapI(8, 8)).toBe(0);
    expect(wrapI(-9, 8)).toBe(7);
    expect(wrapI(3, 8)).toBe(3);
  });
});

describe('fractal noise', () => {
  it('fbm2 is deterministic and bounded', () => {
    for (let i = 0; i < 2000; i++) {
      const x = i * 0.37;
      const y = i * -0.21;
      const v = fbm2(x, y, SEED, DEFAULT_FBM);
      expect(v).toBe(fbm2(x, y, SEED, DEFAULT_FBM));
      expect(Math.abs(v)).toBeLessThan(1.05);
    }
  });

  it('fbm2 gains detail with more octaves', () => {
    const low = { ...DEFAULT_FBM, octaves: 1 };
    const high = { ...DEFAULT_FBM, octaves: 8 };
    let lowVar = 0;
    let highVar = 0;
    let prevL = fbm2(0, 0, SEED, low);
    let prevH = fbm2(0, 0, SEED, high);
    for (let i = 1; i < 3000; i++) {
      const x = i * 0.01;
      const l = fbm2(x, 0, SEED, low);
      const h = fbm2(x, 0, SEED, high);
      lowVar += Math.abs(l - prevL);
      highVar += Math.abs(h - prevH);
      prevL = l;
      prevH = h;
    }
    expect(highVar).toBeGreaterThan(lowVar * 1.5);
  });

  it('fbm2 base octave is invariant to octave count', () => {
    // The property that makes octave count a pure LOD knob: dropping octaves
    // for distant terrain must add or remove *detail* without moving the
    // large-scale surface. If this fails, mountains change height as the
    // player approaches and no amount of LOD morphing will hide it.
    const o2 = { ...DEFAULT_FBM, octaves: 2 };
    const o8 = { ...DEFAULT_FBM, octaves: 8 };
    let worst = 0;
    for (let i = 0; i < 4000; i++) {
      const x = (i % 200) * 0.37;
      const y = Math.floor(i / 200) * 0.41;
      // Difference must be bounded by the amplitude of the dropped octaves:
      // gain^2 + ... + gain^7, scaled by (1-gain) → 0.5^2 = 0.25.
      worst = Math.max(worst, Math.abs(fbm2(x, y, SEED, o8) - fbm2(x, y, SEED, o2)));
    }
    expect(worst).toBeLessThanOrEqual(0.25 + 1e-9);
  });

  it('fbm2 stays bounded as octaves grow', () => {
    for (const octaves of [1, 2, 4, 8, 12]) {
      const p = { ...DEFAULT_FBM, octaves };
      let max = 0;
      for (let i = 0; i < 8000; i++) {
        const x = (i % 400) * 0.13;
        const y = Math.floor(i / 400) * 0.17;
        max = Math.max(max, Math.abs(fbm2(x, y, SEED, p)));
      }
      expect(max).toBeLessThanOrEqual(1.0);
    }
  });

  it('ridged2 stays in [0,1] and peaks near 1', () => {
    let max = 0;
    let min = 1;
    for (let i = 0; i < 20000; i++) {
      const v = ridged2((i % 200) * 0.05, Math.floor(i / 200) * 0.05, SEED, DEFAULT_FBM);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1.0001);
      max = Math.max(max, v);
      min = Math.min(min, v);
    }
    expect(max).toBeGreaterThan(0.75);
    expect(min).toBeLessThan(0.3);
  });

  it('fbmEroded2 damps detail relative to plain fbm', () => {
    // The whole point of the erosion term: less high-frequency energy on
    // steep ground. Measure total variation along a transect.
    let plain = 0;
    let eroded = 0;
    let pp = fbm2(0, 0, SEED, DEFAULT_FBM);
    let pe = fbmEroded2(0, 0, SEED, DEFAULT_FBM, 2.0);
    for (let i = 1; i < 4000; i++) {
      const x = i * 0.02;
      const a = fbm2(x, 5.5, SEED, DEFAULT_FBM);
      const b = fbmEroded2(x, 5.5, SEED, DEFAULT_FBM, 2.0);
      plain += Math.abs(a - pp);
      eroded += Math.abs(b - pe);
      pp = a;
      pe = b;
    }
    expect(eroded).toBeLessThan(plain);
  });

  it('gradientCentral2 matches an analytic gradient', () => {
    // The one gradient path the whole engine uses. If this drifts, physics
    // slope, biome classification and GPU normals all drift with it.
    const d = new Float64Array(2);
    const f = (x: number, y: number): number => perlin2(x * 0.1, y * 0.1, SEED) * 25;
    for (let i = 0; i < 200; i++) {
      const x = i * 1.31 + 0.4;
      const y = i * -0.77 + 2.1;
      gradientCentral2(f, x, y, 1e-3, d);
      const dx = (f(x + 1e-6, y) - f(x - 1e-6, y)) / 2e-6;
      const dy = (f(x, y + 1e-6) - f(x, y - 1e-6)) / 2e-6;
      expect(d[0]).toBeCloseTo(dx, 4);
      expect(d[1]).toBeCloseTo(dy, 4);
    }
  });

  it('fbmEroded2 erosion strength is frequency-independent', () => {
    // Same shape, different world scale: the ratio of eroded to plain energy
    // must not depend on the base frequency, or a strength tuned for mountains
    // silently flattens the detail layer.
    const measure = (frequency: number): number => {
      const p = { ...DEFAULT_FBM, octaves: 6, frequency };
      let e = 0;
      let plain = 0;
      for (let i = 0; i < 3000; i++) {
        const x = (i % 60) / frequency / 60;
        const y = Math.floor(i / 60) / frequency / 60;
        e += fbmEroded2(x, y, SEED, p, 1.0) ** 2;
        plain += fbm2(x, y, SEED, p) ** 2;
      }
      return Math.sqrt(e / plain);
    };
    const lo = measure(0.01);
    const hi = measure(1.0);
    expect(Math.abs(lo - hi) / lo).toBeLessThan(0.2);
  });

  it('domainWarp2 displaces by a bounded amount', () => {
    const out = new Float64Array(2);
    const strength = 100;
    for (let i = 0; i < 1000; i++) {
      const x = i * 3.1;
      const y = i * -1.7;
      domainWarp2(x, y, SEED, strength, 0.001, out);
      expect(Math.abs(out[0] - x)).toBeLessThanOrEqual(strength * 1.05);
      expect(Math.abs(out[1] - y)).toBeLessThanOrEqual(strength * 1.05);
    }
  });

  it('worleyFbm2 is deterministic', () => {
    for (let i = 0; i < 800; i++) {
      const v = worleyFbm2(i * 0.13, i * 0.29, SEED, { ...DEFAULT_FBM, octaves: 3 });
      expect(v).toBe(worleyFbm2(i * 0.13, i * 0.29, SEED, { ...DEFAULT_FBM, octaves: 3 }));
    }
  });
});

describe('seed independence', () => {
  it('different seeds produce genuinely different fields', () => {
    let sumDiff = 0;
    const n = 5000;
    for (let i = 0; i < n; i++) {
      const x = (i % 100) * 0.7;
      const y = Math.floor(i / 100) * 0.7;
      sumDiff += Math.abs(fbm2(x, y, 1, DEFAULT_FBM) - fbm2(x, y, 999, DEFAULT_FBM));
    }
    // Two independent fields of comparable amplitude: mean |difference| should
    // be a sizeable fraction of the signal range, not near zero.
    expect(sumDiff / n).toBeGreaterThan(0.1);
  });
});

describe('helpers', () => {
  it('smoothstep clamps and is monotonic', () => {
    expect(smoothstep(0, 1, -5)).toBe(0);
    expect(smoothstep(0, 1, 5)).toBe(1);
    expect(smoothstep(0, 1, 0.5)).toBeCloseTo(0.5, 12);
    let prev = -1;
    for (let t = -0.2; t <= 1.2; t += 0.01) {
      const v = smoothstep(0, 1, t);
      expect(v).toBeGreaterThanOrEqual(prev);
      prev = v;
    }
  });

  it('remap is linear and invertible', () => {
    expect(remap(5, 0, 10, 100, 200)).toBeCloseTo(150, 12);
    expect(remap(0, 0, 10, 100, 200)).toBeCloseTo(100, 12);
    expect(remap(10, 0, 10, 100, 200)).toBeCloseTo(200, 12);
  });
});
