// Port of thurstone/normaldist.py
// normpdf, normcdf with the same numerical contract as Python's math.erf.
// erf uses W.J. Cody's rational approximation (1969, "Rational Chebyshev
// approximations for the error function") which delivers ~1e-15 absolute
// accuracy across the whole domain - tight enough that round-trip lattice
// computations match the Python reference to ~1e-12 relative.

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

export function erf(x) { return calerf(x, 0); }
export function erfc(x) { return calerf(x, 1); }

export function normpdf(x) {
  return Math.exp(-0.5 * x * x) / SQRT2PI;
}

export function normcdf(x) {
  return 0.5 * (1.0 + erf(x / SQRT2));
}
