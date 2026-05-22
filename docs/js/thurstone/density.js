// Port of thurstone/density.py
// All operations keep the lattice-aligned PDF normalized to sum 1 (or 0 for off-lattice sentinel).

import { UniformLattice } from "./lattice.js";
import { normpdf, normcdf } from "./normaldist.js";

export function cdfFromPdf(pdf) {
  const n = pdf.length;
  const out = new Float64Array(n);
  let acc = 0.0;
  let prev = 0.0;
  for (let i = 0; i < n; i++) {
    acc += pdf[i];
    let v = acc > 1.0 ? 1.0 : acc;
    if (v < prev) v = prev;       // enforce monotone non-decreasing
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    out[i] = v;
    prev = v;
  }
  return out;
}

export function pdfFromCdf(cdf) {
  const n = cdf.length;
  const out = new Float64Array(n);
  let prev = 0.0;
  for (let i = 0; i < n; i++) {
    out[i] = cdf[i] - prev;
    prev = cdf[i];
  }
  return out;
}

export function sumArr(a) {
  let s = 0.0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s;
}

export function normalizePdf(pdf) {
  const s = sumArr(pdf);
  if (s < 0) throw new Error("PDF has negative total mass.");
  if (s === 0) return pdf;        // zero-mass sentinel passes through
  const out = new Float64Array(pdf.length);
  for (let i = 0; i < pdf.length; i++) out[i] = pdf[i] / s;
  return out;
}

export class Density {
  constructor(lattice, p) {
    this.lattice = lattice;
    const arr = p instanceof Float64Array ? p : Float64Array.from(p);
    lattice.assertCompatible(arr);
    // Clip any tiny negative noise then normalize.
    for (let i = 0; i < arr.length; i++) if (arr[i] < 0) arr[i] = 0;
    this.p = normalizePdf(arr);
  }

  cdf() { return cdfFromPdf(this.p); }

  mean() {
    let m = 0.0;
    const u = this.lattice.unit;
    const L = this.lattice.L;
    for (let i = 0; i < this.p.length; i++) m += this.p[i] * u * (i - L);
    return m;
  }

  approxSupport(tol = 1e-12) {
    const idx = [];
    for (let i = 0; i < this.p.length; i++) if (this.p[i] > tol) idx.push(i);
    return idx;
  }

  approxSupportWidth(tol = 1e-12) {
    const idx = this.approxSupport(tol);
    if (idx.length === 0) return 0;
    return idx[idx.length - 1] - idx[0];
  }

  // Shift CDF right by k integer steps, then re-diff. Matches density.py exactly.
  shiftInteger(k) {
    const c = this.cdf();
    const K = c.length;
    let c2;
    if (k <= -K) {
      c2 = new Float64Array(K).fill(1.0);
    } else if (k < 0 && k > -K) {
      // np.concatenate([c[|k|:], np.full(|k|, c[-1])])
      const ak = -k;
      c2 = new Float64Array(K);
      for (let i = 0; i < K - ak; i++) c2[i] = c[i + ak];
      const fill = c[K - 1];
      for (let i = K - ak; i < K; i++) c2[i] = fill;
    } else if (k > 0 && k < K) {
      // np.concatenate([np.zeros(k), c[:-k]])
      c2 = new Float64Array(K);
      for (let i = k; i < K; i++) c2[i] = c[i - k];
    } else if (k >= K) {
      c2 = new Float64Array(K);
    } else {
      c2 = Float64Array.from(c);
    }
    return new Density(this.lattice, pdfFromCdf(c2));
  }

  // Linear blend of neighbouring integer shifts on the CDF (mass-preserving).
  shiftFractional(x) {
    const L = this.lattice.L;
    let l, u, lc, uc;
    if (x > -L + 2 && x < L - 2) {
      l = Math.floor(x);
      u = Math.ceil(x);
      const r = x - l;
      lc = 1.0 - r;
      uc = r;
    } else if (x >= L - 2) {
      l = L - 2; u = L - 1; lc = 1.0; uc = 0.0;
    } else {
      l = -L + 1; u = -L + 2; lc = 0.0; uc = 1.0;
    }
    const cL = this.shiftInteger(l).cdf();
    const cU = this.shiftInteger(u).cdf();
    const c2 = new Float64Array(cL.length);
    for (let i = 0; i < c2.length; i++) c2[i] = lc * cL[i] + uc * cU[i];
    return new Density(this.lattice, pdfFromCdf(c2));
  }

  center() {
    const m = this.mean();
    const steps = m / this.lattice.unit;
    return this.shiftFractional(-steps);
  }

  // Full discrete convolution, truncated/padded to match `keep_L`, mean drift corrected.
  convolve(other, { keepL = null, pad = false } = {}) {
    if (this.lattice.unit !== other.lattice.unit) throw new Error("Units must match.");
    const L = keepL == null ? this.lattice.L : keepL;
    if (keepL == null && this.lattice.L !== other.lattice.L) {
      throw new Error("Convolution with differing L; specify keepL.");
    }
    let p = convolve1d(this.p, other.p);
    if (p.length % 2 === 0) p = p.subarray(0, p.length - 1);
    let pMid;
    const target = 2 * L + 1;
    if (p.length > target) {
      const c = cdfFromPdf(p);
      const nExtra = Math.floor((p.length - target) / 2);
      const cTrim = c.subarray(nExtra, c.length - nExtra);
      pMid = pdfFromCdf(cTrim);
    } else if (p.length < target) {
      if (!pad) throw new Error("Resulting convolution too short; set pad=true or increase L.");
      const nExtra = target - p.length;
      const left = Math.floor(nExtra / 2);
      pMid = new Float64Array(target);
      for (let i = 0; i < p.length; i++) pMid[left + i] = p[i];
    } else {
      pMid = p;
    }
    const muSelf = this.mean();
    const muOther = other.mean();
    const newLat = new UniformLattice(L, this.lattice.unit);
    let muMid = 0.0;
    for (let i = 0; i < pMid.length; i++) muMid += pMid[i] * this.lattice.unit * (i - L);
    const muDiff = muMid - (muSelf + muOther);
    const dMid = new Density(newLat, pMid);
    return dMid.shiftFractional(-muDiff / this.lattice.unit);
  }

  // Move mass as if unit size increased by unit_ratio (coarser lattice).
  dilate(unitRatio = 2.0) {
    const L = this.lattice.L;
    const out = new Float64Array(2 * L + 1);
    for (let idx = -L; idx <= L; idx++) {
      const prob = this.p[idx + L];
      if (prob === 0) continue;
      const x = idx / unitRatio;
      let l, u, lc, uc;
      if (x > -L + 2 && x < L - 2) {
        l = Math.floor(x); u = Math.ceil(x);
        const r = x - l; lc = 1.0 - r; uc = r;
      } else if (x >= L - 2) {
        l = L - 2; u = L - 1; lc = 1.0; uc = 0.0;
      } else {
        l = -L + 1; u = -L + 2; lc = 0.0; uc = 1.0;
      }
      const li = Math.min(2 * L, Math.max(l + L, 0));
      const ui = Math.min(2 * L, Math.max(u + L, 0));
      out[li] += prob * lc;
      out[ui] += prob * uc;
    }
    return new Density(this.lattice, out);
  }

  static fromCallable(lattice, f, { center = true } = {}) {
    const x = lattice.grid();
    const p = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const v = f(x[i]);
      p[i] = v > 0 ? v : 0.0;
    }
    const d = new Density(lattice, p);
    return center ? d.center() : d;
  }

  static skewNormal(lattice, { loc = 0.0, scale = 1.0, a = 0.0 } = {}) {
    const f = (xi) => {
      const t = (xi - loc) / scale;
      return (2.0 / scale) * normpdf(t) * normcdf(a * t);
    };
    const d = Density.fromCallable(lattice, f, { center: true });
    return d.shiftFractional(loc / lattice.unit);
  }
}

// Direct linear convolution. Lengths in this package are O(1000) so this is fine.
export function convolve1d(a, b) {
  const out = new Float64Array(a.length + b.length - 1);
  for (let i = 0; i < a.length; i++) {
    const ai = a[i];
    if (ai === 0) continue;
    for (let j = 0; j < b.length; j++) out[i + j] += ai * b[j];
  }
  return out;
}
