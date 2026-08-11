"""Laplacian structure of the ability-to-probability Jacobian.

For independent lattice performances X_i ~ f(. - a_i) with the *minimum*
winning (the package convention), the outright win probability of runner i is

    p_i(a) = sum_x f_i(x) * prod_{k != i} S_k(x),

where f_i is the shifted pmf and S_k(x) = Pr(X_k > x) the survival function.
Differentiating under the sum gives, for i != j,

    d p_i / d a_j = + w_ij,
    w_ij = sum_x f_i(x) f_j(x) prod_{k != i,j} S_k(x)  > 0,

and translation invariance p(a + c*1) = p(a) forces the diagonal
d p_i / d a_i = - sum_{j != i} w_ij.  Hence the Jacobian of the forward map
is minus a weighted complete-graph Laplacian:

    Dp(a) = -L(w),   L(w) = diag(W 1) - W.

(In the max-wins convention of random-utility theory the sign flips and
Dp = +L(w); only the orientation of "better" changes.)

The Laplacian need not be formed to be applied.  With the hazard
h_i(x) = f_i(x) / S_i(x) and the shared aggregates

    q_i(x) = f_i(x) * prod_{k != i} S_k(x),
    H(x)   = sum_j h_j(x),
    G_u(x) = sum_j h_j(x) u_j,

one has, for any vector u,

    (L u)_i = sum_{j != i} w_ij (u_i - u_j)
            = sum_x q_i(x) * (u_i H(x) - G_u(x)),

because the j = i term vanishes identically.  All aggregates are shared
across i, so a Hessian-vector product costs O(n M) for n runners on an
M-point lattice, versus O(n^2 M) to form the dense weights.  This enables
Newton-CG joint calibration without ever materialising the dense Jacobian.

Numerical note: h_j is computed only where S_j >= SURVIVAL_TOL.  Near the
top of runner j's support the clamped lattice CDF can reach 1 while f_j is
still positive, making the raw ratio blow up; the true integrand there is
f_i f_j prod_{k != i,j} S_k, which is negligible whenever the masked region
carries only extreme-tail mass.  The dense routine avoids division entirely
and serves as the reference in tests.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .density import Density

SURVIVAL_TOL = 1e-12


def _pmf_and_survival(densities: Sequence[Density]) -> tuple[np.ndarray, np.ndarray]:
    """Stack pmfs and survival functions, shapes (n, M)."""
    if len(densities) < 2:
        raise ValueError("Need at least two runners.")
    F = np.stack([d.p for d in densities])
    S = np.stack([1.0 - d.cdf() for d in densities])
    return F, np.clip(S, 0.0, 1.0)


def _leave_one_out_products(S: np.ndarray) -> np.ndarray:
    """loo[i] = prod_{k != i} S[k], columnwise, via prefix/suffix products.

    Division-free, so exact zeros in S are handled correctly. O(n M).
    """
    n, M = S.shape
    prefix = np.ones((n + 1, M))
    np.cumprod(S, axis=0, out=prefix[1:])
    suffix = np.ones((n + 1, M))
    np.cumprod(S[::-1], axis=0, out=suffix[1:])
    return prefix[:n] * suffix[:n][::-1]


def outright_win_probabilities(densities: Sequence[Density]) -> np.ndarray:
    """No-tie win probabilities p_i = sum_x f_i prod_{k != i} S_k, in O(n M).

    On a lattice ties carry positive mass, so the sum over i falls short of
    one by the total tie probability; this is the smooth forward map whose
    Jacobian is -L(w), not the dead-heat-adjusted state price of Race.
    """
    F, S = _pmf_and_survival(densities)
    return np.sum(F * _leave_one_out_products(S), axis=1)


def laplacian_weights(densities: Sequence[Density]) -> np.ndarray:
    """Dense symmetric weight matrix w_ij = sum_x f_i f_j prod_{k != i,j} S_k.

    Division-free reference implementation: for each i, leave-one-out
    products are rebuilt over the remaining runners. O(n^2 M).

    Lattice pmfs are masses (density times unit), so one factor of the
    lattice unit is divided out to make w_ij a derivative with respect to
    physical ability: dp_i/da_j = +w_ij for i != j.
    """
    F, S = _pmf_and_survival(densities)
    n = F.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        rest = [k for k in range(n) if k != i]
        loo_rest = _leave_one_out_products(S[rest])  # (n-1, M): prod over rest minus one
        W[i, rest] = (F[rest] * loo_rest) @ F[i]
    W /= densities[0].lattice.unit
    return 0.5 * (W + W.T)  # symmetric up to roundoff; enforce exactly


def laplacian_dense(densities: Sequence[Density]) -> np.ndarray:
    """L(w) = diag(W 1) - W. The Jacobian of outright_win_probabilities is -L."""
    W = laplacian_weights(densities)
    return np.diag(W.sum(axis=1)) - W


def laplacian_matvec(densities: Sequence[Density], u: np.ndarray) -> np.ndarray:
    """Apply L(w) to u in O(n M) without forming the weights.

    Uses (L u)_i = sum_x q_i (u_i H - G_u); see module docstring.
    """
    F, S = _pmf_and_survival(densities)
    u = np.asarray(u, dtype=float)
    if u.shape != (F.shape[0],):
        raise ValueError("u must have one entry per runner.")
    Q = F * _leave_one_out_products(S)  # q_i(x)
    h = np.where(S >= SURVIVAL_TOL, F / np.maximum(S, SURVIVAL_TOL), 0.0)
    H = h.sum(axis=0)
    G = u @ h
    return np.sum(Q * (np.outer(u, H) - G[None, :]), axis=1) / densities[0].lattice.unit
