// Port of thurstone/pricing.py
// Race + StatePricer. The forward map: densities -> winning probabilities.

import { Density, pdfFromCdf, sumArr } from "./density.js";
import { winnerOfMany, expectedPayoffWithMultiplicity } from "./order_stats.js";

export const NAN_DIVIDEND = 2000.0;

export function cdfMin(densities) {
  const cdfs = densities.map(d => d.cdf());
  const n = cdfs[0].length;
  const prodS = new Float64Array(n); prodS.fill(1.0);
  for (const c of cdfs) for (let i = 0; i < n; i++) prodS[i] *= (1.0 - c[i]);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = 1.0 - prodS[i];
  return out;
}

export function winnerDensity(densities) {
  const lat = densities[0].lattice;
  return new Density(lat, pdfFromCdf(cdfMin(densities)));
}

export class Race {
  constructor(densities) {
    if (!densities || densities.length === 0) throw new Error("Race requires >= 1 density.");
    const L = densities[0].lattice.L, u = densities[0].lattice.unit;
    for (const d of densities) {
      if (d.lattice.L !== L || d.lattice.unit !== u) throw new Error("All densities must share the same lattice.");
    }
    this.densities = densities;
  }

  winnerDensity() { return winnerDensity(this.densities); }

  // Multiplicity-aware risk-neutral winning probabilities.
  statePrices() {
    const { density: dAll, mult: multAll } = winnerOfMany(this.densities);
    const cdfAll = dAll.cdf();
    const prices = new Float64Array(this.densities.length);
    for (let i = 0; i < this.densities.length; i++) {
      const ep = expectedPayoffWithMultiplicity(this.densities[i], dAll, multAll, null, cdfAll);
      prices[i] = sumArr(ep);
    }
    const S = sumArr(prices);
    if (S > 0) for (let i = 0; i < prices.length; i++) prices[i] /= S;
    return prices;
  }
}

export const StatePricer = {
  pricesFromDividends(dividends, nanValue = NAN_DIVIDEND) {
    const p = new Float64Array(dividends.length);
    for (let i = 0; i < dividends.length; i++) {
      let v = dividends[i];
      if (v == null || Number.isNaN(v)) v = nanValue;
      p[i] = v <= 0 ? 0.0 : 1.0 / v;
    }
    const S = sumArr(p);
    if (S > 0) for (let i = 0; i < p.length; i++) p[i] /= S;
    return p;
  },

  dividendsFromPrices(prices, multiplicity = 1.0) {
    const p = Float64Array.from(prices);
    const S = sumArr(p);
    if (S > 0) for (let i = 0; i < p.length; i++) p[i] /= S;
    const out = new Float64Array(p.length);
    for (let i = 0; i < p.length; i++) out[i] = p[i] > 0 ? 1.0 / (multiplicity * p[i]) : NaN;
    return out;
  }
};
