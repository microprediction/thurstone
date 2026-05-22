// Port of thurstone/global_ls.py
// Relative-then-LS: invert each race independently, center, slope-weight, average per item.

import { AbilityCalibrator, interp, median } from "./inference.js";

function gradient(y, x) {
  const n = y.length;
  const out = new Float64Array(n);
  if (n === 1) { out[0] = 0; return out; }
  out[0] = (y[1] - y[0]) / (x[1] - x[0]);
  out[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
  for (let i = 1; i < n - 1; i++) {
    out[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
  }
  return out;
}

export class RaceLS {
  constructor({ calibrator, horseIds, prices, scales = null, localLocs = null }) {
    this.calibrator = calibrator;
    this.horseIds = [...horseIds];
    this.prices = Float64Array.from(prices);
    this.scales = scales == null ? null : Float64Array.from(scales);
    this.localLocs = localLocs == null ? null : Float64Array.from(localLocs);
  }
}

export class GlobalLSCalibrator {
  constructor(horseIds) {
    this.horseIds = [...horseIds];
    this.races = [];
    this.theta = Object.fromEntries(this.horseIds.map(h => [h, 0.0]));
  }

  addRace(calibrator, horseIds, prices, scales = null) {
    if (calibrator.lookupCurve1dPrices == null && calibrator.lookupCurves2dPrices.size === 0) {
      calibrator.solveFromPrices(prices);
    }
    const localLocs = calibrator.solveFromPrices(prices);
    this.races.push(new RaceLS({ calibrator, horseIds, prices, scales, localLocs }));
  }

  _invertAndCenter(race) {
    if (race.localLocs == null) {
      race.localLocs = Float64Array.from(race.calibrator.solveFromPrices(race.prices));
    }
    const m = median(Array.from(race.localLocs));
    return race.localLocs.map(x => x - m);
  }

  _slopeWeight(cal, loc, scale = null) {
    try {
      if (scale != null && cal.lookupCurves2dPrices.size > 0) {
        const scales = [...cal.lookupCurves2dPrices.keys()].sort((a, b) => a - b);
        // pick nearest
        let sSel = scales[0], best = Math.abs(scales[0] - scale);
        for (const s of scales) { const d = Math.abs(s - scale); if (d < best) { best = d; sSel = s; } }
        const curve = cal.lookupCurves2dPrices.get(sSel);
        const dp = gradient(curve.prices, curve.locs);
        let muC = loc;
        if (muC < curve.locs[0]) muC = curve.locs[0];
        if (muC > curve.locs[curve.locs.length - 1]) muC = curve.locs[curve.locs.length - 1];
        return Math.abs(interp(muC, curve.locs, dp)) + 1e-12;
      }
      if (cal.lookupCurve1dPrices) {
        const c = cal.lookupCurve1dPrices;
        const dp = gradient(c.prices, c.locs);
        let muC = loc;
        if (muC < c.locs[0]) muC = c.locs[0];
        if (muC > c.locs[c.locs.length - 1]) muC = c.locs[c.locs.length - 1];
        return Math.abs(interp(muC, c.locs, dp)) + 1e-12;
      }
    } catch (_) {}
    return 1.0;
  }

  fit({ useSlopeWeights = true, ridge = 0.0, weightCap = null } = {}) {
    const sumWY = Object.fromEntries(this.horseIds.map(h => [h, 0.0]));
    const sumW = Object.fromEntries(this.horseIds.map(h => [h, ridge]));
    for (const race of this.races) {
      const centered = this._invertAndCenter(race);
      for (let j = 0; j < race.horseIds.length; j++) {
        const hid = race.horseIds[j];
        let w = 1.0;
        if (useSlopeWeights) {
          const sc = race.scales == null ? null : race.scales[j];
          const locForSlope = race.localLocs ? race.localLocs[j] : centered[j];
          w = this._slopeWeight(race.calibrator, locForSlope, sc);
          if (weightCap != null) w = Math.min(w, weightCap);
        }
        if (sumWY[hid] === undefined) continue;  // unknown horse
        sumWY[hid] += w * centered[j];
        sumW[hid] += w;
      }
    }
    for (const h of this.horseIds) {
      const d = sumW[h];
      this.theta[h] = d > 0 ? sumWY[h] / d : 0.0;
    }
    const med = median(this.horseIds.map(h => this.theta[h]));
    for (const h of this.horseIds) this.theta[h] -= med;
  }
}
