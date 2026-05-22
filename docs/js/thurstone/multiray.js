// Port of thurstone/multiray.py
// MultiRayGlobalCalibrator: items live in R^dim, each condition has (beta, V) so
// ability(item, cond) = beta + <V, Z[item]>. Fits V, beta, Z by Gauss-Newton on prices,
// reusing cached lookup curves from each AbilityCalibrator.

import { AbilityCalibrator, interp } from "./inference.js";

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

// Tiny deterministic PRNG so item/V initialisation is reproducible without external deps.
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussian(rand) {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function solveSym(A, b) {
  // tiny Gauss-Jordan for small dense systems (dim+1 <= ~4 in practice)
  const n = b.length;
  const M = A.map(row => [...row]);
  const y = [...b];
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[piv][i])) piv = r;
    if (piv !== i) { [M[i], M[piv]] = [M[piv], M[i]]; [y[i], y[piv]] = [y[piv], y[i]]; }
    const d = M[i][i];
    if (Math.abs(d) < 1e-14) continue;
    for (let j = i; j < n; j++) M[i][j] /= d;
    y[i] /= d;
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const f = M[r][i];
      if (f === 0) continue;
      for (let j = i; j < n; j++) M[r][j] -= f * M[i][j];
      y[r] -= f * y[i];
    }
  }
  return y;
}

export class ConditionSpec {
  constructor({ condId, calibrator, itemIds, prices, scales = null }) {
    this.condId = condId;
    this.calibrator = calibrator;
    this.itemIds = [...itemIds];
    this.prices = Float64Array.from(prices);
    this.scales = scales == null ? null : Float64Array.from(scales);
    this.index = Object.fromEntries(this.itemIds.map((h, i) => [h, i]));
  }
}

export class MultiRayGlobalCalibrator {
  constructor(itemIds, { dim = 2, l2Z = 1e-6, l2V = 1e-6, stepBeta = 0.3, stepV = 0.3, stepZ = 0.3, slopeFloor = 1e-10, randomState = 0 } = {}) {
    this.itemIds = [...itemIds];
    this.dim = dim;
    this.conditions = [];
    this.Z = {};
    this.V = {};
    this.beta = {};
    this.l2Z = l2Z;
    this.l2V = l2V;
    this.stepBeta = stepBeta;
    this.stepV = stepV;
    this.stepZ = stepZ;
    this.slopeFloor = slopeFloor;
    this.condIndex = {};
    const rand = mulberry32(randomState || 1);
    for (const hid of this.itemIds) {
      const v = new Float64Array(dim);
      for (let k = 0; k < dim; k++) v[k] = 0.01 * gaussian(rand);
      this.Z[hid] = v;
    }
    this._rand = rand;
  }

  _rebuildCondIndex() {
    this.condIndex = Object.fromEntries(this.conditions.map((c, i) => [c.condId, i]));
  }

  addCondition({ condId, calibrator, itemIds, prices, scales = null }) {
    if (calibrator.lookupCurve1dPrices == null && calibrator.lookupCurves2dPrices.size === 0) {
      calibrator.solveFromPrices(prices);
    }
    this.conditions.push(new ConditionSpec({ condId, calibrator, itemIds, prices, scales }));
    if (!(condId in this.V)) {
      const v = new Float64Array(this.dim);
      for (let k = 0; k < this.dim; k++) v[k] = gaussian(this._rand);
      let nrm = 0; for (let k = 0; k < this.dim; k++) nrm += v[k] * v[k]; nrm = Math.sqrt(nrm);
      if (nrm > 0) for (let k = 0; k < this.dim; k++) v[k] /= nrm;
      this.V[condId] = v;
    }
    if (!(condId in this.beta)) this.beta[condId] = 0.0;
    this._rebuildCondIndex();
  }

  ability(condId, itemId) {
    const z = this.Z[itemId], v = this.V[condId];
    let d = 0; for (let k = 0; k < this.dim; k++) d += v[k] * z[k];
    return this.beta[condId] + d;
  }

  _predictAndSlopesForCondition(cIdx) {
    const spec = this.conditions[cIdx];
    const m = spec.itemIds.length;
    const pHat = new Float64Array(m), slopes = new Float64Array(m);
    for (let k = 0; k < m; k++) {
      const mu = this.ability(spec.condId, spec.itemIds[k]);
      const { p, dp } = (spec.scales != null && spec.calibrator.lookupCurves2dPrices.size > 0)
        ? interpPriceAndSlope2d(spec.calibrator, mu, spec.scales[k])
        : interpPriceAndSlope1d(spec.calibrator, mu);
      pHat[k] = p; slopes[k] = dp;
    }
    return { pHat, slopes };
  }

  rebuildAllCurves() {
    for (const spec of this.conditions) {
      const mu = spec.itemIds.map(h => this.ability(spec.condId, h));
      if (spec.scales != null && mu.length === spec.scales.length) spec.calibrator.rebuildCurves2d(mu, Array.from(spec.scales));
      else spec.calibrator.rebuildCurves1d(mu);
    }
  }

  applyGaugeFix() {
    if (this.itemIds.length === 0) return;
    const meanZ = new Float64Array(this.dim);
    for (const h of this.itemIds) for (let k = 0; k < this.dim; k++) meanZ[k] += this.Z[h][k];
    for (let k = 0; k < this.dim; k++) meanZ[k] /= this.itemIds.length;
    for (const h of this.itemIds) for (let k = 0; k < this.dim; k++) this.Z[h][k] -= meanZ[k];
    for (const cid of Object.keys(this.V)) {
      let nrm = 0; for (let k = 0; k < this.dim; k++) nrm += this.V[cid][k] * this.V[cid][k]; nrm = Math.sqrt(nrm);
      if (nrm > 0) for (let k = 0; k < this.dim; k++) this.V[cid][k] /= nrm;
    }
  }

  fitInner(numIters) {
    for (let it = 0; it < numIters; it++) {
      const condPHat = [], condSlopes = [], condErr = [];
      for (let j = 0; j < this.conditions.length; j++) {
        const { pHat, slopes } = this._predictAndSlopesForCondition(j);
        const e = new Float64Array(pHat.length);
        for (let k = 0; k < e.length; k++) e[k] = pHat[k] - this.conditions[j].prices[k];
        condPHat.push(pHat); condSlopes.push(slopes); condErr.push(e);
      }
      // (A) update beta, V per condition with Z fixed
      for (let j = 0; j < this.conditions.length; j++) {
        const spec = this.conditions[j];
        const slopes = condSlopes[j], e = condErr[j];
        const sSafe = new Float64Array(slopes.length);
        for (let k = 0; k < slopes.length; k++) sSafe[k] = Math.abs(slopes[k]) < this.slopeFloor ? this.slopeFloor : slopes[k];
        const y = new Float64Array(e.length);
        for (let k = 0; k < e.length; k++) y[k] = -e[k] / sSafe[k];
        const m = spec.itemIds.length;
        const X = [];
        for (let k = 0; k < m; k++) {
          const row = new Array(1 + this.dim).fill(0);
          row[0] = 1.0;
          for (let q = 0; q < this.dim; q++) row[1 + q] = this.Z[spec.itemIds[k]][q];
          X.push(row);
        }
        // XtX and Xty
        const dimT = 1 + this.dim;
        const XtX = Array.from({ length: dimT }, () => new Array(dimT).fill(0));
        const Xty = new Array(dimT).fill(0);
        for (let k = 0; k < m; k++) {
          for (let r = 0; r < dimT; r++) {
            Xty[r] += X[k][r] * y[k];
            for (let c = 0; c < dimT; c++) XtX[r][c] += X[k][r] * X[k][c];
          }
        }
        for (let r = 1; r < dimT; r++) XtX[r][r] += this.l2V;
        const w = solveSym(XtX, Xty);
        this.beta[spec.condId] += this.stepBeta * w[0];
        for (let q = 0; q < this.dim; q++) this.V[spec.condId][q] += this.stepV * w[1 + q];
        // normalize V
        let nrm = 0; for (let q = 0; q < this.dim; q++) nrm += this.V[spec.condId][q] ** 2; nrm = Math.sqrt(nrm);
        if (nrm > 0) for (let q = 0; q < this.dim; q++) this.V[spec.condId][q] /= nrm;
      }
      // (B) update Z per item with (beta, V) fixed
      for (const hid of this.itemIds) {
        const rows = [], ys = [];
        for (let j = 0; j < this.conditions.length; j++) {
          const spec = this.conditions[j];
          if (!(hid in spec.index)) continue;
          const k = spec.index[hid];
          let slope = condSlopes[j][k];
          if (Math.abs(slope) < this.slopeFloor) slope = slope >= 0 ? this.slopeFloor : -this.slopeFloor;
          const eK = condPHat[j][k] - spec.prices[k];
          const yK = -eK / slope;
          rows.push([...this.V[spec.condId]]);
          ys.push(yK);
        }
        if (rows.length === 0) continue;
        const MtM = Array.from({ length: this.dim }, () => new Array(this.dim).fill(0));
        const Mty = new Array(this.dim).fill(0);
        for (let k = 0; k < rows.length; k++) {
          for (let r = 0; r < this.dim; r++) {
            Mty[r] += rows[k][r] * ys[k];
            for (let c = 0; c < this.dim; c++) MtM[r][c] += rows[k][r] * rows[k][c];
          }
        }
        for (let r = 0; r < this.dim; r++) MtM[r][r] += this.l2Z;
        const dz = solveSym(MtM, Mty);
        for (let q = 0; q < this.dim; q++) this.Z[hid][q] += this.stepZ * dz[q];
      }
    }
  }

  fitWithRebuild({ outer = 3, inner = 10 } = {}) {
    for (let i = 0; i < outer; i++) {
      this.rebuildAllCurves();
      this.fitInner(inner);
      this.applyGaugeFix();
    }
  }

  predictCondition(condId) {
    const j = this.condIndex[condId];
    return this._predictAndSlopesForCondition(j).pHat;
  }
}
