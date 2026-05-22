// Port of thurstone/global_fit.py
// GlobalAbilityCalibrator: curve-based Gauss-Newton fit that matches probabilities directly.
// Keeps the original prices in view at all times (unlike GlobalLSCalibrator, which collapses
// each race to its centred local locations first). This is the recommended global method.

import { interp } from "./inference.js";

function gradient(y, x) {
  const n = y.length;
  const out = new Float64Array(n);
  if (n === 1) { out[0] = 0; return out; }
  out[0] = (y[1] - y[0]) / (x[1] - x[0]);
  out[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
  for (let i = 1; i < n - 1; i++) out[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
  return out;
}

function clampVal(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }

function interpPriceAndSlope1d(cal, mu) {
  if (!cal.lookupCurve1dPrices) throw new Error("Calibrator has no 1D curve; call solveFromPrices first.");
  const { locs, prices } = cal.lookupCurve1dPrices;
  const dp = gradient(prices, locs);
  const muC = clampVal(mu, locs[0], locs[locs.length - 1]);
  const p = clampVal(interp(muC, locs, prices), 1e-12, 1 - 1e-12);
  return { p, dp: interp(muC, locs, dp) };
}

function interpPriceAndSlope2d(cal, mu, scale) {
  if (cal.lookupCurves2dPrices.size === 0) return interpPriceAndSlope1d(cal, mu);
  const scales = [...cal.lookupCurves2dPrices.keys()].sort((a, b) => a - b);
  if (cal.lookupCurves2dPrices.has(scale)) {
    const { locs, prices } = cal.lookupCurves2dPrices.get(scale);
    const dp = gradient(prices, locs);
    const muC = clampVal(mu, locs[0], locs[locs.length - 1]);
    const p = clampVal(interp(muC, locs, prices), 1e-12, 1 - 1e-12);
    return { p, dp: interp(muC, locs, dp) };
  }
  let idx = 0; while (idx < scales.length && scales[idx] < scale) idx++;
  let s1, s2;
  if (idx <= 0) { s1 = scales[0]; s2 = scales[Math.min(1, scales.length - 1)]; }
  else if (idx >= scales.length) { s1 = scales[scales.length - 2]; s2 = scales[scales.length - 1]; }
  else { s1 = scales[idx - 1]; s2 = scales[idx]; }
  const w = (s2 === s1) ? 0 : (scale - s1) / (s2 - s1);
  const c1 = cal.lookupCurves2dPrices.get(s1);
  const c2 = cal.lookupCurves2dPrices.get(s2);
  const dp1 = gradient(c1.prices, c1.locs), dp2 = gradient(c2.prices, c2.locs);
  const mu1 = clampVal(mu, c1.locs[0], c1.locs[c1.locs.length - 1]);
  const mu2 = clampVal(mu, c2.locs[0], c2.locs[c2.locs.length - 1]);
  const p = clampVal((1 - w) * interp(mu1, c1.locs, c1.prices) + w * interp(mu2, c2.locs, c2.prices), 1e-12, 1 - 1e-12);
  const dp = (1 - w) * interp(mu1, c1.locs, dp1) + w * interp(mu2, c2.locs, dp2);
  return { p, dp };
}

export class RaceSpec {
  constructor({ calibrator, horseIds, prices, scales = null }) {
    this.calibrator = calibrator;
    this.horseIds = [...horseIds];
    this.prices = Float64Array.from(prices);
    this.scales = scales == null ? null : Float64Array.from(scales);
  }
}

export class GlobalAbilityCalibrator {
  constructor(horseIds, { l2 = 1e-8, stepBias = 0.3, stepTheta = 0.3 } = {}) {
    this.horseIds = [...horseIds];
    this.races = [];
    this.biases = [];
    this.l2 = l2;
    this.stepBias = stepBias;
    this.stepTheta = stepTheta;
    this.theta = Object.fromEntries(this.horseIds.map(h => [h, 0.0]));
  }

  addRace(calibrator, horseIds, prices, scales = null) {
    if (calibrator.lookupCurve1dPrices == null && calibrator.lookupCurves2dPrices.size === 0) {
      calibrator.solveFromPrices(prices);
    }
    this.races.push(new RaceSpec({ calibrator, horseIds, prices, scales }));
    this.biases.push(0.0);
  }

  _predictAndSlopesForRace(rIdx) {
    const spec = this.races[rIdx];
    const cal = spec.calibrator;
    const n = spec.horseIds.length;
    const pHat = new Float64Array(n), slopes = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const mu = this.theta[spec.horseIds[i]] + this.biases[rIdx];
      const { p, dp } = (spec.scales != null && cal.lookupCurves2dPrices.size > 0)
        ? interpPriceAndSlope2d(cal, mu, spec.scales[i])
        : interpPriceAndSlope1d(cal, mu);
      pHat[i] = p; slopes[i] = dp;
    }
    return { pHat, slopes };
  }

  fit(numIters = 25) {
    for (let it = 0; it < numIters; it++) {
      // (A) per-race bias updates with theta fixed
      for (let r = 0; r < this.races.length; r++) {
        const spec = this.races[r];
        const { pHat, slopes } = this._predictAndSlopesForRace(r);
        let num = 0, denom = this.l2;
        for (let i = 0; i < pHat.length; i++) {
          const e = pHat[i] - spec.prices[i];
          num += slopes[i] * e;
          denom += slopes[i] * slopes[i];
        }
        if (denom > 0) this.biases[r] += this.stepBias * (-num / denom);
      }
      // (B) per-item theta updates with biases fixed
      for (const hid of this.horseIds) {
        let num = 0, denom = this.l2;
        for (let r = 0; r < this.races.length; r++) {
          const spec = this.races[r];
          const i = spec.horseIds.indexOf(hid);
          if (i < 0) continue;
          const cal = spec.calibrator;
          const mu = this.theta[hid] + this.biases[r];
          const { p, dp } = (spec.scales != null && cal.lookupCurves2dPrices.size > 0)
            ? interpPriceAndSlope2d(cal, mu, spec.scales[i])
            : interpPriceAndSlope1d(cal, mu);
          const e = p - spec.prices[i];
          num += dp * e;
          denom += dp * dp;
        }
        if (denom > 0) this.theta[hid] -= this.stepTheta * (num / denom);
      }
    }
  }

  rebuildAllCurves() {
    for (let r = 0; r < this.races.length; r++) {
      const spec = this.races[r];
      const muR = spec.horseIds.map(h => this.theta[h] + this.biases[r]);
      if (spec.scales != null && muR.length === spec.scales.length) {
        spec.calibrator.rebuildCurves2d(muR, Array.from(spec.scales));
      } else {
        spec.calibrator.rebuildCurves1d(muR);
      }
    }
  }

  fitWithRebuild({ outer = 3, inner = 10 } = {}) {
    for (let i = 0; i < outer; i++) {
      this.rebuildAllCurves();
      this.fit(inner);
    }
  }

  predictRace(rIdx) {
    return this._predictAndSlopesForRace(rIdx).pHat;
  }
}
