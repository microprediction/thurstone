import numpy as np
import pytest

from thurstone.density import Density
from thurstone.laplacian import (
    laplacian_dense,
    laplacian_matvec,
    laplacian_weights,
    outright_win_probabilities,
)
from thurstone.lattice import UniformLattice

LATTICE = UniformLattice(L=400, unit=0.05)
ABILITIES = [0.0, 0.35, -0.2, 0.6, 0.1]


def _field(abilities, lattice=LATTICE, scale=1.0, skew=0.0):
    base = Density.skew_normal(lattice, loc=0.0, scale=scale, a=skew)
    return [base.shift_fractional(a / lattice.unit) for a in abilities]


def test_weights_symmetric_positive():
    W = laplacian_weights(_field(ABILITIES))
    assert np.allclose(W, W.T)
    assert np.all(np.diag(W) == 0.0)
    off = W[~np.eye(len(ABILITIES), dtype=bool)]
    assert np.all(off > 0.0)


def test_laplacian_null_space_and_psd():
    L = laplacian_dense(_field(ABILITIES))
    n = len(ABILITIES)
    assert np.allclose(L @ np.ones(n), 0.0, atol=1e-12)
    eig = np.linalg.eigvalsh(L)
    assert eig[0] > -1e-12
    # complete graph with positive weights: connected, so lambda_2 > 0
    assert eig[1] > 0.0


def test_matvec_matches_dense():
    field = _field(ABILITIES)
    L = laplacian_dense(field)
    rng = np.random.default_rng(7)
    for _ in range(5):
        u = rng.normal(size=len(ABILITIES))
        assert np.allclose(laplacian_matvec(field, u), L @ u, rtol=1e-8, atol=1e-12)


def test_matvec_matches_dense_two_runners():
    field = _field([0.0, 0.4])
    L = laplacian_dense(field)
    u = np.array([1.0, -2.0])
    assert np.allclose(laplacian_matvec(field, u), L @ u, rtol=1e-8, atol=1e-12)


def test_matvec_matches_dense_heterogeneous():
    lattice = LATTICE
    base_narrow = Density.skew_normal(lattice, loc=0.0, scale=0.7, a=0.5)
    base_wide = Density.skew_normal(lattice, loc=0.0, scale=1.4, a=-0.3)
    field = [
        base_narrow.shift_fractional(0.3 / lattice.unit),
        base_wide.shift_fractional(-0.2 / lattice.unit),
        base_narrow,
        base_wide.shift_fractional(0.5 / lattice.unit),
    ]
    L = laplacian_dense(field)
    rng = np.random.default_rng(11)
    u = rng.normal(size=4)
    assert np.allclose(laplacian_matvec(field, u), L @ u, rtol=1e-8, atol=1e-12)


def test_jacobian_row_sums_translation_invariance():
    """p(a + c 1) = p(a) up to lattice edge effects."""
    p0 = outright_win_probabilities(_field(ABILITIES))
    p1 = outright_win_probabilities(_field([a + 0.25 for a in ABILITIES]))
    assert np.allclose(p0, p1, atol=1e-10)


def _fd_jacobian(abilities, lattice):
    n = len(abilities)
    eps = lattice.unit  # one lattice step: spans the piecewise-linear kink
    J = np.zeros((n, n))
    for j in range(n):
        up = list(abilities)
        dn = list(abilities)
        up[j] += eps
        dn[j] -= eps
        J[:, j] = (
            outright_win_probabilities(_field(up, lattice))
            - outright_win_probabilities(_field(dn, lattice))
        ) / (2 * eps)
    return J


def test_finite_difference_jacobian():
    """Central differences of the forward map recover -L(w) to O(unit)."""
    L = laplacian_dense(_field(ABILITIES))
    J = _fd_jacobian(ABILITIES, LATTICE)
    assert np.allclose(J, -L, atol=1e-2)
    mask = np.abs(L) > 1e-3
    assert np.max(np.abs((J + L)[mask] / L[mask])) < 5e-2


def test_finite_difference_jacobian_converges():
    """The FD-vs-analytic gap is discretization error: it halves with the unit."""
    err = []
    for half_width, unit in [(400, 0.05), (800, 0.025)]:
        lattice = UniformLattice(L=half_width, unit=unit)
        L = laplacian_dense(_field(ABILITIES, lattice))
        J = _fd_jacobian(ABILITIES, lattice)
        err.append(np.abs(J + L).max())
    assert err[1] < 0.6 * err[0]


def test_matvec_cost_scales_linearly():
    """The matvec touches O(n M) memory; sanity-check agreement at larger n."""
    rng = np.random.default_rng(3)
    abilities = rng.normal(scale=0.4, size=12)
    field = _field(abilities)
    u = rng.normal(size=12)
    assert np.allclose(laplacian_matvec(field, u), laplacian_dense(field) @ u, rtol=1e-8)


def test_input_validation():
    field = _field(ABILITIES)
    with pytest.raises(ValueError):
        laplacian_matvec(field, np.ones(3))
    with pytest.raises(ValueError):
        outright_win_probabilities(field[:1])
