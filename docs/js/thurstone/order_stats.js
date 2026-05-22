// Port of thurstone/order_stats.py
// winner_of_many returns (density, multiplicity) for the field minimum.

import { Density, pdfFromCdf } from "./density.js";

export const EPS = 1e-18;
export const DEL = 1e-12;

export function conditionalWinDrawLoss(pdfA, pdfB, cdfA, cdfB) {
  const n = pdfA.length;
  const winA = new Float64Array(n);
  const draw = new Float64Array(n);
  const winB = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    winA[i] = pdfA[i] * (1.0 - cdfB[i]);
    draw[i] = pdfA[i] * pdfB[i];
    winB[i] = pdfB[i] * (1.0 - cdfA[i]);
  }
  return { winA, draw, winB };
}

function winnerOfTwo(dA, dB, multA, multB) {
  const cA = dA.cdf();
  const cB = dB.cdf();
  const n = cA.length;
  const cMin = new Float64Array(n);
  for (let i = 0; i < n; i++) cMin[i] = 1.0 - (1.0 - cA[i]) * (1.0 - cB[i]);
  const out = new Density(dA.lattice, pdfFromCdf(cMin));
  if (!multA) { multA = new Float64Array(n); multA.fill(1.0); }
  if (!multB) { multB = new Float64Array(n); multB.fill(1.0); }
  const { winA, draw, winB } = conditionalWinDrawLoss(dA.p, dB.p, cA, cB);
  const mult = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const numer = winA[i] * multA[i] + draw[i] * (multA[i] + multB[i]) + winB[i] * multB[i] + EPS;
    const denom = winA[i] + draw[i] + winB[i] + EPS;
    mult[i] = numer / denom;
  }
  return { density: out, mult };
}

export function winnerOfMany(densities) {
  if (densities.length === 0) throw new Error("winnerOfMany requires >= 1 density.");
  let d = densities[0];
  const L = d.lattice.L;
  let mult = new Float64Array(2 * L + 1);
  mult.fill(1.0);
  for (let i = 1; i < densities.length; i++) {
    const ones = new Float64Array(2 * L + 1); ones.fill(1.0);
    const r = winnerOfTwo(d, densities[i], mult, ones);
    d = r.density;
    mult = r.mult;
  }
  return { density: d, mult };
}

// "Rest of field" CDF and multiplicity given the whole field and the runner of interest.
export function getTheRest(density, densityAll, multiplicityAll, cdf = null, cdfAll = null) {
  if (cdf == null) cdf = density.cdf();
  if (cdfAll == null) {
    if (!densityAll) throw new Error("Need densityAll or cdfAll.");
    cdfAll = densityAll.cdf();
  }
  const n = cdf.length;
  const pdf = pdfFromCdf(cdf);
  const pdfAll = pdfFromCdf(cdfAll);

  const S_all = new Float64Array(n);
  const S_self = new Float64Array(n);
  const S_rest = new Float64Array(n);
  const cdfRest = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    S_all[i] = 1.0 - cdfAll[i];
    S_self[i] = 1.0 - cdf[i];
    S_rest[i] = (S_all[i] + EPS) / (S_self[i] + DEL);
    cdfRest[i] = 1.0 - S_rest[i];
  }
  // monotone non-decreasing
  let acc = cdfRest[0];
  for (let i = 1; i < n; i++) {
    if (cdfRest[i] < acc) cdfRest[i] = acc;
    else acc = cdfRest[i];
    if (cdfRest[i] > 1.0) cdfRest[i] = 1.0;
  }
  // also clamp >1 in first cell
  if (cdfRest[0] > 1.0) cdfRest[0] = 1.0;
  const pdfRest = pdfFromCdf(cdfRest);

  // multiplicity formula (mult_left / mult_right hybrid at the mode of pdf)
  const m = multiplicityAll;
  const f1 = pdf;
  const m1 = 1.0;

  const mult = new Float64Array(n);
  let kmode = 0;
  let fmax = -Infinity;
  for (let i = 0; i < n; i++) if (f1[i] > fmax) { fmax = f1[i]; kmode = i; }

  for (let i = 0; i < n; i++) {
    const numer_l = m[i] * f1[i] * S_rest[i] + m[i] * (f1[i] + S_self[i]) * pdfRest[i] - m1 * f1[i] * (S_rest[i] + pdfRest[i]);
    const denom_l = pdfRest[i] * (f1[i] + S_self[i]) + EPS;
    const mult_left = (EPS + numer_l) / denom_l;

    const T1 = (S_self[i] + EPS) / (f1[i] + DEL);
    const Trest = (S_rest[i] + EPS) / (pdfRest[i] + DEL);
    const mult_right = m[i] * Trest / (1.0 + T1) + m[i] - m1 * (1.0 + Trest) / (1.0 + T1);

    let v = i < kmode ? mult_left : mult_right;
    if (v < 0 || !Number.isFinite(v)) v = 0;
    mult[i] = v;
  }
  return { cdfRest, mult };
}

export function expectedPayoffWithMultiplicity(density, densityAll, multiplicityAll, cdf = null, cdfAll = null) {
  const { cdfRest, mult } = getTheRest(density, densityAll, multiplicityAll, cdf, cdfAll);
  const pdf = cdf == null ? density.p : pdfFromCdf(cdf);
  const pdfRest = pdfFromCdf(cdfRest);
  const cdfSelf = density.cdf();
  const { winA: win, draw } = conditionalWinDrawLoss(pdf, pdfRest, cdfSelf, cdfRest);
  const out = new Float64Array(win.length);
  for (let i = 0; i < win.length; i++) {
    let mR = mult[i];
    if (mR < 0) mR = 0;
    if (!Number.isFinite(mR)) throw new Error("Multiplicity contains non-finite values.");
    out[i] = win[i] + draw[i] / (1.0 + mR);
  }
  return out;
}
