// Port of thurstone/clustering.py
// ClusterSplitter handles offsets that hang off the lattice (walkovers, equal-share).

import { Density } from "./density.js";
import { Race } from "./pricing.js";
import { winnerOfMany } from "./order_stats.js";

function intCentered(offsets) {
  const finite = offsets.filter(o => Number.isFinite(o));
  if (finite.length === 0) return [...offsets];
  const meanInt = Math.trunc(finite.reduce((a, b) => a + b, 0) / finite.length);
  return offsets.map(o => o - meanInt);
}

function divideOffsets(centered) {
  const srt = [...centered].sort((a, b) => a - b);
  if (srt.length <= 2) return srt.reduce((a, b) => a + b, 0) / srt.length;
  let bestGap = -Infinity, bestIdx = 0;
  for (let i = 0; i < srt.length - 1; i++) {
    const g = Math.abs(srt[i + 1] - srt[i]);
    if (g > bestGap) { bestGap = g; bestIdx = i; }
  }
  return 0.5 * (srt[bestIdx] + srt[bestIdx + 1]);
}

function densitiesFromOffsets(base, offsets) {
  return offsets.map(o => base.shiftFractional(o));
}

function statePricesFromDensities(densities) {
  return Array.from(new Race(densities).statePrices());
}

export class ClusterSplitter {
  constructor({ unitRatio = 3.0, maxDepth = 3 } = {}) {
    this.unitRatio = unitRatio;
    this.maxDepth = maxDepth;
  }

  extendedStatePrices(base, offsets) {
    const n = offsets.length;
    if (n === 1) return [1.0];

    // Handle +inf (cannot win)
    const posInf = [];
    for (let i = 0; i < n; i++) if (offsets[i] === Infinity) posInf.push(i);
    if (posInf.length > 0) {
      const finiteIdx = [];
      for (let i = 0; i < n; i++) if (!posInf.includes(i)) finiteIdx.push(i);
      if (finiteIdx.length === 0) return Array(n).fill(1.0 / n);
      const fp = this.extendedStatePrices(base, intCentered(finiteIdx.map(i => offsets[i])));
      const out = Array(n).fill(0.0);
      for (let j = 0; j < finiteIdx.length; j++) out[finiteIdx[j]] = fp[j];
      return out;
    }
    // Handle -inf (certain winners share equally)
    const negInf = [];
    for (let i = 0; i < n; i++) if (offsets[i] === -Infinity) negInf.push(i);
    if (negInf.length > 0) {
      const p = Array(n).fill(0.0);
      const share = 1.0 / negInf.length;
      for (const i of negInf) p[i] = share;
      return p;
    }

    const L = base.lattice.L;
    const W = base.approxSupportWidth();
    const centered = intCentered(offsets);

    const lowerBound = -L + W;
    const upperBound = L - W;
    const hangLeft = [], hangRight = [];
    for (let i = 0; i < n; i++) {
      if (centered[i] < lowerBound) hangLeft.push(i);
      if (centered[i] > upperBound) hangRight.push(i);
    }
    if (hangLeft.length === 0 && hangRight.length === 0) {
      return statePricesFromDensities(densitiesFromOffsets(base, centered));
    }

    if (this.maxDepth <= 0) {
      for (const i of hangRight) centered[i] = Infinity;
      for (const i of hangLeft) centered[i] = -Infinity;
      return this.extendedStatePrices(base, centered);
    }

    const divider = divideOffsets(centered);
    let leftIdx = [], rightIdx = [];
    for (let i = 0; i < n; i++) (centered[i] < divider ? leftIdx : rightIdx).push(i);
    if (leftIdx.length === 0 || rightIdx.length === 0) {
      if (leftIdx.length === 0) {
        let am = 0; for (let i = 1; i < centered.length; i++) if (centered[i] < centered[am]) am = i;
        leftIdx = [am]; rightIdx = []; for (let i = 0; i < n; i++) if (i !== am) rightIdx.push(i);
      } else {
        let am = 0; for (let i = 1; i < centered.length; i++) if (centered[i] > centered[am]) am = i;
        rightIdx = [am]; leftIdx = []; for (let i = 0; i < n; i++) if (i !== am) leftIdx.push(i);
      }
    }

    const densLeft = densitiesFromOffsets(base, leftIdx.map(i => centered[i]));
    const densRight = densitiesFromOffsets(base, rightIdx.map(i => centered[i]));
    if (densLeft.length === 0 || densRight.length === 0) {
      return statePricesFromDensities(densitiesFromOffsets(base, centered));
    }
    const { density: repLeft } = winnerOfMany(densLeft);
    const { density: repRight } = winnerOfMany(densRight);
    const groupPrices = new Race([repLeft, repRight]).statePrices();
    const leftShare = groupPrices[0];
    const rightShare = groupPrices[1];

    let leftRel, rightRel;
    if (leftShare <= rightShare) {
      const sub = new ClusterSplitter({ unitRatio: this.unitRatio, maxDepth: this.maxDepth - 1 });
      leftRel = sub.extendedStatePrices(base, leftIdx.map(i => centered[i]));
      rightRel = statePricesFromDensities(densRight);
      const Sr = rightRel.reduce((a, b) => a + b, 0);
      if (Sr > 0) rightRel = rightRel.map(p => p / Sr);
    } else {
      const sub = new ClusterSplitter({ unitRatio: this.unitRatio, maxDepth: this.maxDepth - 1 });
      rightRel = sub.extendedStatePrices(base, rightIdx.map(i => centered[i]));
      leftRel = statePricesFromDensities(densLeft);
      const Sl = leftRel.reduce((a, b) => a + b, 0);
      if (Sl > 0) leftRel = leftRel.map(p => p / Sl);
    }

    const out = Array(n).fill(0.0);
    for (let j = 0; j < leftIdx.length; j++) out[leftIdx[j]] = leftShare * leftRel[j];
    for (let j = 0; j < rightIdx.length; j++) out[rightIdx[j]] = rightShare * rightRel[j];

    const S = out.reduce((a, b) => a + b, 0);
    if (S <= 0) throw new Error("Extended state prices have non-positive total mass.");
    return out.map(o => o / S);
  }
}
