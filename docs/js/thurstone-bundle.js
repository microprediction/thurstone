/**
 * Thurstone Library Bundle
 * All classes and functions needed by demos, without ES module imports
 * Works with file:// protocol and regular script tags
 *
 * Usage: <script src="js/thurstone-bundle.js"></script>
 *
 * Available global variables:
 * - UniformLattice, Density, Race, AbilityCalibrator, StatePricer, ClusterSplitter
 * - normpdf, normcdf, erf, erfc
 * - Helper functions and constants
 */

(function() {
  'use strict';

  // =============================================================================
  // NORMAL DISTRIBUTION (from normaldist.js)
  // =============================================================================

  const SQRT2 = Math.sqrt(2.0);
  const SQRT2PI = Math.sqrt(2.0 * Math.PI);

  // Coefficients from Cody (1969), as used by SLATEC, Cephes, Boost, etc.
  const ERF_A = [
    3.16112374387056560e+00, 1.13864154151050156e+02, 3.77485237685302021e+02,
    3.20937758913846947e+03, 1.85777706184603153e-01,
  ];
  const ERF_B = [
    2.36012909523441209e+01, 2.44024637934444173e+02, 1.28261652607737228e+03,
    2.84423683343917062e+03,
  ];
  const ERF_C = [
    5.64188496988670089e-01, 8.88314979438837594e+00, 6.61191906371416295e+01,
    2.98635138197400131e+02, 8.81952221241769090e+02, 1.71204761263407058e+03,
    2.05107837782607147e+03, 1.23033935479799725e+03, 2.15311535474403846e-08,
  ];
  const ERF_D = [
    1.57449261107098347e+01, 1.17693950891312499e+02, 5.37181101862009858e+02,
    1.62138957456669019e+03, 3.29079923573345963e+03, 4.36261909014324716e+03,
    3.43936767414372164e+03, 1.23033935480374942e+03,
  ];
  const ERF_P = [
    3.05326634961232344e-01, 3.60344899949804439e-01, 1.25781726111229246e-01,
    1.60837851487422766e-02, 6.58749161529837803e-04, 1.63153871373020978e-02,
  ];
  const ERF_Q = [
    2.56852019228982242e+00, 1.87295284992346047e+00, 5.27905102951428413e-01,
    6.05183413124413191e-02, 2.33520497626869185e-03,
  ];

  function calerf(x, jint) {
    // jint=0 -> erf, jint=1 -> erfc, jint=2 -> scaled erfcx (not needed externally).
    const y = Math.abs(x);
    let result;
    if (y <= 0.46875) {
      // erf for |x| <= 0.46875
      let ysq = 0.0;
      if (y > 1.11e-16) ysq = y * y;
      let xnum = ERF_A[4] * ysq;
      let xden = ysq;
      for (let i = 0; i < 3; i++) {
        xnum = (xnum + ERF_A[i]) * ysq;
        xden = (xden + ERF_B[i]) * ysq;
      }
      result = x * (xnum + ERF_A[3]) / (xden + ERF_B[3]);
      if (jint !== 0) result = 1.0 - result;
      if (jint === 2) result = Math.exp(ysq) * result;
      return result;
    }
    if (y <= 4.0) {
      let xnum = ERF_C[8] * y;
      let xden = y;
      for (let i = 0; i < 7; i++) {
        xnum = (xnum + ERF_C[i]) * y;
        xden = (xden + ERF_D[i]) * y;
      }
      result = (xnum + ERF_C[7]) / (xden + ERF_D[7]);
      if (jint !== 2) {
        const ysq = Math.floor(y * 16) / 16;
        const del = (y - ysq) * (y + ysq);
        result = Math.exp(-ysq * ysq) * Math.exp(-del) * result;
      }
    } else {
      // y > 4
      if (y >= 26.5430) {
        result = 0.0;
      } else {
        const ysq = 1.0 / (y * y);
        let xnum = ERF_P[5] * ysq;
        let xden = ysq;
        for (let i = 0; i < 4; i++) {
          xnum = (xnum + ERF_P[i]) * ysq;
          xden = (xden + ERF_Q[i]) * ysq;
        }
        result = ysq * (xnum + ERF_P[4]) / (xden + ERF_Q[4]);
        result = (1.0 / Math.sqrt(Math.PI) - result) / y;
        if (jint !== 2) {
          const ysqf = Math.floor(y * 16) / 16;
          const del = (y - ysqf) * (y + ysqf);
          result = Math.exp(-ysqf * ysqf) * Math.exp(-del) * result;
        }
      }
    }
    if (jint === 0) {
      result = 1.0 - result;
      if (x < 0) result = -result;
    } else if (jint === 1) {
      if (x < 0) result = 2.0 - result;
    } else {
      // erfcx
      if (x < 0) {
        const xsq = Math.floor(x * 16) / 16;
        const del = (x - xsq) * (x + xsq);
        const y2 = Math.exp(xsq * xsq) * Math.exp(del);
        result = (y2 + y2) - result;
      }
    }
    return result;
  }

  function erf(x) { return calerf(x, 0); }
  function erfc(x) { return calerf(x, 1); }

  function normpdf(x) {
    return Math.exp(-0.5 * x * x) / SQRT2PI;
  }

  function normcdf(x) {
    return 0.5 * (1.0 + erf(x / SQRT2));
  }

  // =============================================================================
  // LATTICE (from lattice.js)
  // =============================================================================

  class UniformLattice {
    constructor(L, unit) {
      this.L = L;
      this.unit = unit;
    }

    get size() { return 2 * this.L + 1; }

    grid() {
      const n = this.size;
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) out[i] = this.unit * (i - this.L);
      return out;
    }

    indexGrid() {
      const n = this.size;
      const out = new Int32Array(n);
      for (let i = 0; i < n; i++) out[i] = i - this.L;
      return out;
    }

    assertCompatible(arr) {
      if (arr.length !== this.size) {
        throw new Error(`Array length ${arr.length} incompatible with lattice size ${this.size}.`);
      }
    }
  }

  // Standard lattice defaults (mirrors thurstone/conventions.py).
  const STD_L = 500;
  const STD_UNIT = 0.1;
  const STD_SCALE = 1.0;
  const STD_A = 0.0;
  const ALT_L = 500;
  const ALT_UNIT = 0.1;
  const ALT_SCALE = 1.0;
  const ALT_A = 0.5;
  const NAN_DIVIDEND = 2000.0;

  // =============================================================================
  // DENSITY (from density.js)
  // =============================================================================

  function cdfFromPdf(pdf) {
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

  function pdfFromCdf(cdf) {
    const n = cdf.length;
    const out = new Float64Array(n);
    let prev = 0.0;
    for (let i = 0; i < n; i++) {
      out[i] = cdf[i] - prev;
      prev = cdf[i];
    }
    return out;
  }

  function sumArr(a) {
    let s = 0.0;
    for (let i = 0; i < a.length; i++) s += a[i];
    return s;
  }

  function normalizePdf(pdf) {
    const s = sumArr(pdf);
    if (s < 0) throw new Error("PDF has negative total mass.");
    if (s === 0) return pdf;        // zero-mass sentinel passes through
    const out = new Float64Array(pdf.length);
    for (let i = 0; i < pdf.length; i++) out[i] = pdf[i] / s;
    return out;
  }

  class Density {
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
  function convolve1d(a, b) {
    const out = new Float64Array(a.length + b.length - 1);
    for (let i = 0; i < a.length; i++) {
      const ai = a[i];
      if (ai === 0) continue;
      for (let j = 0; j < b.length; j++) out[i + j] += ai * b[j];
    }
    return out;
  }

  // =============================================================================
  // ORDER STATS (from order_stats.js)
  // =============================================================================

  const EPS = 1e-18;
  const DEL = 1e-12;

  function conditionalWinDrawLoss(pdfA, pdfB, cdfA, cdfB) {
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

  function winnerOfMany(densities) {
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
  function getTheRest(density, densityAll, multiplicityAll, cdf = null, cdfAll = null) {
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

  function expectedPayoffWithMultiplicity(density, densityAll, multiplicityAll, cdf = null, cdfAll = null) {
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

  // =============================================================================
  // PRICING (from pricing.js)
  // =============================================================================

  function cdfMin(densities) {
    const cdfs = densities.map(d => d.cdf());
    const n = cdfs[0].length;
    const prodS = new Float64Array(n); prodS.fill(1.0);
    for (const c of cdfs) for (let i = 0; i < n; i++) prodS[i] *= (1.0 - c[i]);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = 1.0 - prodS[i];
    return out;
  }

  function winnerDensity(densities) {
    const lat = densities[0].lattice;
    return new Density(lat, pdfFromCdf(cdfMin(densities)));
  }

  class Race {
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

  const StatePricer = {
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

  // =============================================================================
  // CLUSTERING (from clustering.js)
  // =============================================================================

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

  class ClusterSplitter {
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

  // =============================================================================
  // INFERENCE (from inference.js)
  // =============================================================================

  // numpy.interp equivalent. xp must be strictly ascending.
  function interp(x, xp, fp) {
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

  function sortAscByFirst(pairs) {
    pairs.sort((a, b) => a[0] - b[0]);
    return pairs;
  }

  function uniqueByFirst(xp, fp) {
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

  function median(arr) {
    const a = [...arr].sort((x, y) => x - y);
    const n = a.length;
    if (n === 0) return 0;
    return n % 2 ? a[(n - 1) >>> 1] : 0.5 * (a[n / 2 - 1] + a[n / 2]);
  }

  class AbilityCalibrator {
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

  // =============================================================================
  // GLOBAL EXPORTS
  // =============================================================================

  // Make classes and functions available globally
  window.UniformLattice = UniformLattice;
  window.Density = Density;
  window.Race = Race;
  window.AbilityCalibrator = AbilityCalibrator;
  window.StatePricer = StatePricer;
  window.ClusterSplitter = ClusterSplitter;

  // Export helper functions
  window.normpdf = normpdf;
  window.normcdf = normcdf;
  window.erf = erf;
  window.erfc = erfc;

  // Export constants
  window.STD_L = STD_L;
  window.STD_UNIT = STD_UNIT;
  window.STD_SCALE = STD_SCALE;
  window.STD_A = STD_A;
  window.ALT_L = ALT_L;
  window.ALT_UNIT = ALT_UNIT;
  window.ALT_SCALE = ALT_SCALE;
  window.ALT_A = ALT_A;
  window.NAN_DIVIDEND = NAN_DIVIDEND;

  // Helper functions that may be useful
  window.cdfFromPdf = cdfFromPdf;
  window.pdfFromCdf = pdfFromCdf;
  window.sumArr = sumArr;
  window.normalizePdf = normalizePdf;
  window.convolve1d = convolve1d;
  window.winnerOfMany = winnerOfMany;
  window.expectedPayoffWithMultiplicity = expectedPayoffWithMultiplicity;
  window.interp = interp;
  window.median = median;

  console.log('✅ Thurstone bundle loaded successfully!');
  console.log('Available classes: UniformLattice, Density, Race, AbilityCalibrator, StatePricer, ClusterSplitter');
  console.log('Available functions: normpdf, normcdf, erf, erfc, and many helpers');

})();