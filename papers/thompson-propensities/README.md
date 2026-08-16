# All-Action Propensities for Correlated Gaussian Thompson Sampling

**Authors:** Peter Cotton
**Status:** Working manuscript — empirical results pending (August 2026)

Thompson sampling's logging propensities are posterior winner probabilities. The
familiar independent product-CDF formula is invalid under a shared posterior
parameter draw, because univariate Gaussian marginals do not determine winner
probabilities. Under a factor-plus-diagonal representation, conditioning on the
factors restores independence and a single shared product-CDF field yields all N
propensities in O(QN(L+k)) arithmetic, their posterior-mean Jacobian (a
"photo-finish" graph Laplacian), continuity and total-variation guarantees, and
exact conditional bias formulas for IPS and DR estimators fed approximate
propensities. Companion to *Scalable Probit Share Calibration*.

Build: `../build.sh thompson-propensities`
