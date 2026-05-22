// Port of thurstone/inference.py
// AbilityCalibrator.solve_from_dividends / solve_from_prices via per-iteration global curve.
// The 2D (loc, scale) path is included; use set_scales() before calling solve.

import { Density } from "./density.js";
import { StatePricer } from "./pricing.js";
import { winnerOfMany, expectedPayoffWithMultiplicity } from "./order_stats.js";
import { ClusterSplitter } from "./clustering.js";

export function densitiesFromOffsets(base, offsets) {
  return offsets.map(o => base.shiftFractional(o));
}

// numpy.interp equivalent. xp must be strictly ascending.
export function interp(x, xp, fp) {
  if (xp.length === 0) return 0;
  if (x <= xp[0]) return fp[0];
  if (x >= xp[xp.length - 1]) return fp[fp.length - 1];
  // binary search
  let lo = 0, hi = xp.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >>> 1;
    if (xp[mid] <= x) lo = mid; else hi = mid;
  }
  const t = (x - xp[lo]) / (xp[hi] - xp[lo]);
  return fp[lo] + t * (fp[hi] - fp[lo]);
}

export function sortAscByFirst(pairs) {
  pairs.sort((a, b) => a[0] - b[0]);
  return pairs;
}

export function uniqueByFirst(xp, fp) {
  // np.unique semantics: keep first occurrence
  const outX = [], outF = [];
  let prev = NaN;
  for (let i = 0; i < xp.length; i++) {
    if (xp[i] !== prev) {
      outX.push(xp[i]);
      outF.push(fp[i]);
      prev = xp[i];
    }
  }
  return [Float64Array.from(outX), Float64Array.from(outF)];
}

export function median(arr) {
  const a = [...arr].sort((x, y) => x - y);
  const n = a.length;
  if (n === 0) return 0;
  return n % 2 ? a[(n - 1) >>> 1] : 0.5 * (a[n / 2 - 1] + a[n / 2]);
}

export class AbilityCalibrator {
  constructor(base, {
    offsetGrid = null,
    nIter = 3,
    scales = null,
    scaleSpan = 0.5,
    scaleSteps = 3,
    locSpan = 5.0,
    locStep = 0.25,
    skewA = 0.0,
  } = {}) {
    this.base = base;
    this.nIter = nIter;
    this.scales = scales == null ? null : Float64Array.from(scales);
    this.scaleSpan = scaleSpan;
    this.scaleSteps = scaleSteps;
    this.locSpan = locSpan;
    this.locStep = locStep;
    this.skewA = skewA;
    this.lookupCurve1dPrices = null;
    this.lookupCurve1dInverse = null;
    this.lookupCurves2dPrices = new Map();
    this.lookupCurves2dInverse = new Map();

    if (offsetGrid == null) {
      const L = this.base.lattice.L;
      const lo = -Math.trunc(L / 2);
      const hi = Math.trunc(L / 2);
      const grid = [];
      for (let g = lo; g < hi; g++) grid.push(g);
      grid.reverse();    // match Python [::-1] on range(int(-L/2), int(L/2))
      this.offsetGrid = grid;
    } else {
      this.offsetGrid = [...offsetGrid];
    }
  }

  setScales(scales) { this.scales = Float64Array.from(scales); }

  densityFor(loc, scale) {
    return Density.skewNormal(this.base.lattice, { loc, scale, a: this.skewA });
  }

  rebuildCurves1d(locs) {
    const unit = this.base.lattice.unit;
    const offsets = locs.map(l => l / unit);
    const grid = this.offsetGrid;
    const densField = densitiesFromOffsets(this.base, offsets);
    const { density: dAll, mult: multAll } = winnerOfMany(densField);
    const cdfAll = dAll.cdf();
    const implied = new Float64Array(grid.length);
    for (let i = 0; i < grid.length; i++) {
      const dg = this.base.shiftFractional(grid[i]);
      const ep = expectedPayoffWithMultiplicity(dg, dAll, multAll, null, cdfAll);
      let s = 0; for (let k = 0; k < ep.length; k++) s += ep[k];
      implied[i] = s;
    }
    // Cache loc -> price curve (sorted ascending in loc)
    const locsPhys = grid.map(g => unit * g);
    const ord = locsPhys.map((_, k) => k).sort((a, b) => locsPhys[a] - locsPhys[b]);
    const locsSorted = ord.map(i => locsPhys[i]);
    const pricesSorted = ord.map(i => implied[i]);
    this.lookupCurve1dPrices = { locs: Float64Array.from(locsSorted), prices: Float64Array.from(pricesSorted) };
    // Cache inverse curve (sorted ascending in price, unique)
    const pairs = grid.map((g, i) => [implied[i], g]);
    sortAscByFirst(pairs);
    const xp = pairs.map(p => p[0]);
    const fp = pairs.map(p => p[1]);
    const [xu, fu] = uniqueByFirst(xp, fp);
    this.lookupCurve1dInverse = { prices: xu, offsets: fu };
  }

  rebuildCurves2d(locs, scales) {
    const n = locs.length;
    const densField = [];
    for (let j = 0; j < n; j++) densField.push(this.densityFor(locs[j], scales[j]));
    const { density: dAll, mult: multAll } = winnerOfMany(densField);
    const cdfAll = dAll.cdf();
    const locGrid = [];
    for (let v = -this.locSpan; v <= this.locSpan + 1e-9; v += this.locStep) locGrid.push(+v);
    const uniqueScales = [...new Set(scales.map(s => +s))].sort((a, b) => a - b);
    this.lookupCurves2dPrices.clear();
    this.lookupCurves2dInverse.clear();
    for (const s of uniqueScales) {
      const pCurve = new Float64Array(locGrid.length);
      for (let i = 0; i < locGrid.length; i++) {
        const d = this.densityFor(locGrid[i], s);
        const ep = expectedPayoffWithMultiplicity(d, dAll, multAll, null, cdfAll);
        let sum = 0; for (let k = 0; k < ep.length; k++) sum += ep[k];
        pCurve[i] = sum;
      }
      this.lookupCurves2dPrices.set(s, { locs: Float64Array.from(locGrid), prices: pCurve });
      const pairs = locGrid.map((l, i) => [pCurve[i], l]);
      sortAscByFirst(pairs);
      const [xu, fu] = uniqueByFirst(pairs.map(p => p[0]), pairs.map(p => p[1]));
      this.lookupCurves2dInverse.set(s, { prices: xu, offsets: fu });
    }
  }

  solveFromPrices(prices, { initialOffsets = null } = {}) {
    const pricesArr = Float64Array.from(prices);
    const n = pricesArr.length;

    if (this.scales == null || this.scales.length !== n) {
      // 1D path
      const unit = this.base.lattice.unit;
      let offsets = (initialOffsets == null ? Array(n).fill(0) : [...initialOffsets]).map(o => o / unit);
      const grid = this.offsetGrid;
      for (let iter = 0; iter < this.nIter; iter++) {
        const densField = densitiesFromOffsets(this.base, offsets);
        const { density: dAll, mult: multAll } = winnerOfMany(densField);
        const cdfAll = dAll.cdf();
        const implied = new Float64Array(grid.length);
        for (let i = 0; i < grid.length; i++) {
          const dg = this.base.shiftFractional(grid[i]);
          const ep = expectedPayoffWithMultiplicity(dg, dAll, multAll, null, cdfAll);
          let s = 0; for (let k = 0; k < ep.length; k++) s += ep[k];
          implied[i] = s;
        }
        // Cache (loc -> price) and (price -> offset)
        const locsPhys = grid.map(g => unit * g);
        const ord = locsPhys.map((_, k) => k).sort((a, b) => locsPhys[a] - locsPhys[b]);
        this.lookupCurve1dPrices = {
          locs: Float64Array.from(ord.map(i => locsPhys[i])),
          prices: Float64Array.from(ord.map(i => implied[i])),
        };
        const pairs = grid.map((g, i) => [implied[i], g]);
        sortAscByFirst(pairs);
        const [xu, fu] = uniqueByFirst(pairs.map(p => p[0]), pairs.map(p => p[1]));
        this.lookupCurve1dInverse = { prices: xu, offsets: fu };
        // Invert all prices at once
        const lo = xu[0], hi = xu[xu.length - 1];
        for (let i = 0; i < n; i++) {
          let p = pricesArr[i];
          if (p < lo) p = lo; if (p > hi) p = hi;
          offsets[i] = interp(p, xu, fu);
        }
      }
      return offsets.map(o => o * unit);
    }

    // 2D (loc, scale) path
    const scales = this.scales;
    const unit = this.base.lattice.unit;
    let locs = initialOffsets == null ? new Float64Array(n) : Float64Array.from(initialOffsets.map(o => o * unit));
    const locGrid = [];
    for (let v = -this.locSpan; v <= this.locSpan + 1e-9; v += this.locStep) locGrid.push(+v);

    const scaleGridFor = (si) => {
      if (this.scaleSteps <= 1) return [Math.max(1e-6, si)];
      const out = new Float64Array(this.scaleSteps);
      const step = (2 * this.scaleSpan) / (this.scaleSteps - 1);
      for (let k = 0; k < this.scaleSteps; k++) out[k] = Math.max(1e-6, si + (-this.scaleSpan + k * step));
      return out;
    };

    for (let iter = 0; iter < this.nIter; iter++) {
      const densField = [];
      for (let j = 0; j < n; j++) densField.push(this.densityFor(locs[j], scales[j]));
      const { density: dAll, mult: multAll } = winnerOfMany(densField);
      const cdfAll = dAll.cdf();
      const perRunnerSg = []; const allS = new Set();
      for (let i = 0; i < n; i++) {
        const sg = scaleGridFor(scales[i]);
        perRunnerSg.push(sg);
        for (const s of sg) allS.add(+s);
      }
      const cache = new Map();
      for (const s of allS) {
        const pCurve = new Float64Array(locGrid.length);
        for (let k = 0; k < locGrid.length; k++) {
          const d = this.densityFor(locGrid[k], s);
          const ep = expectedPayoffWithMultiplicity(d, dAll, multAll, null, cdfAll);
          let sum = 0; for (let j = 0; j < ep.length; j++) sum += ep[j];
          pCurve[k] = sum;
        }
        this.lookupCurves2dPrices.set(s, { locs: Float64Array.from(locGrid), prices: pCurve });
        const pairs = locGrid.map((l, i) => [pCurve[i], l]);
        sortAscByFirst(pairs);
        const [xu, fu] = uniqueByFirst(pairs.map(p => p[0]), pairs.map(p => p[1]));
        cache.set(s, { xu, fu });
        this.lookupCurves2dInverse.set(s, { prices: xu, offsets: fu });
      }
      for (let i = 0; i < n; i++) {
        const si = scales[i], pi = pricesArr[i], sg = perRunnerSg[i];
        const locEsts = new Float64Array(sg.length);
        for (let k = 0; k < sg.length; k++) {
          const { xu, fu } = cache.get(+sg[k]);
          let p = pi; if (p < xu[0]) p = xu[0]; if (p > xu[xu.length - 1]) p = xu[xu.length - 1];
          locEsts[k] = interp(p, xu, fu);
        }
        locs[i] = sg.length === 1 ? locEsts[0] : interp(si, Float64Array.from(sg), locEsts);
      }
    }
    const m = median(Array.from(locs));
    return Array.from(locs, x => x - m);
  }

  solveFromDividends(dividends, { nanValue } = {}) {
    const prices = StatePricer.pricesFromDividends(dividends, nanValue);
    return this.solveFromPrices(prices);
  }

  statePricesFromAbility(ability) {
    const offsets = ability.map(a => a / this.base.lattice.unit);
    return new ClusterSplitter().extendedStatePrices(this.base, offsets);
  }

  dividendsFromAbility(ability, multiplicity = 1.0) {
    const p = this.statePricesFromAbility(ability);
    return p.map(pi => 1.0 / (multiplicity * pi));
  }
}
