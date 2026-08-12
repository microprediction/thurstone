"""Tests for correlated races (thurstone.correlated).

Conventions under test: min wins, lower ability = stronger, abilities in
physical units.
"""

import numpy as np
import pytest

from thurstone import (
    Density,
    FactorRace,
    Race,
    UniformLattice,
    factor_model,
    gaussian_factor_race,
    hermite_nodes,
    solve_abilities,
)
from thurstone.inference import densities_from_offsets

LAT = UniformLattice(L=400, unit=0.05)
RNG = np.random.default_rng(7)


def circle_kernel(n: int, ell: float) -> np.ndarray:
    th = 2 * np.pi * np.arange(n) / n
    d = np.abs((th[:, None] - th[None, :] + np.pi) % (2 * np.pi) - np.pi)
    return np.exp(-d / ell)


def mc_reference(mu, C, scale, n_draws, seed=9):
    """Monte Carlo win frequencies for a correlated Gaussian race (min wins)."""
    L = np.linalg.cholesky(C + 1e-9 * np.eye(len(C)))
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(mu))
    done = 0
    while done < n_draws:
        n = min(200_000, n_draws - done)
        X = np.asarray(mu)[:, None] + scale * (L @ rng.standard_normal((len(mu), n)))
        counts += np.bincount(np.argmin(X, axis=0), minlength=len(mu))
        done += n
    return counts / counts.sum()


def test_independent_limit_matches_race():
    """Zero loadings must reproduce the package's independent state prices."""
    mu = np.array([-0.6, -0.2, 0.0, 0.3, 0.7])
    base = Density.skew_normal(LAT, loc=0.0, scale=1.0, a=0.0)
    fr = FactorRace(base, mu, np.zeros((5, 1)))
    p_fr = fr.state_prices()
    dens = densities_from_offsets(base, list(mu / LAT.unit))  # offsets in steps
    p_race = Race(dens).state_prices()
    assert np.abs(p_fr - p_race).max() < 2e-3


def test_known_factor_model_matches_monte_carlo():
    """With Sigma = V V^T + diag(D) known exactly, quadrature matches MC."""
    n, k = 8, 2
    V = 0.5 * RNG.standard_normal((n, k))
    D = RNG.uniform(0.4, 0.9, n)
    C = V @ V.T + np.diag(D)
    mu = RNG.normal(0.0, 0.5, n)
    bases = [Density.skew_normal(LAT, 0.0, float(np.sqrt(d)), 0.0) for d in D]
    p = FactorRace(bases, mu, V).state_prices()
    ref = mc_reference(mu, C, 1.0, 2_000_000)
    assert np.abs(p - ref).max() < 4e-3


def test_equicorrelated_single_factor_exact():
    """Equicorrelation is exactly one factor; k=1 must already be exact."""
    n, rho = 10, 0.5
    C = rho * np.ones((n, n)) + (1 - rho) * np.eye(n)
    mu = RNG.normal(0.0, 0.5, n)
    p = gaussian_factor_race(LAT, C, 1, mu).state_prices()
    ref = mc_reference(mu, C, 1.0, 2_000_000)
    assert np.abs(p - ref).max() < 4e-3


def test_factor_model_identity_gives_no_correlation():
    """Factor analysis of C = I must not invent off-diagonal correlation."""
    V, D = factor_model(np.eye(12), 3)
    C_hat = V @ V.T + np.diag(D)
    off = C_hat - np.diag(np.diag(C_hat))
    assert np.abs(off).max() < 1e-6


def test_gumbel_min_independent_race_is_softmax():
    """Independent Gumbel-min race = Luce/softmax(-mu/scale), exactly."""
    mu = np.array([-0.8, -0.3, 0.0, 0.4, 1.0])
    scale = 0.7
    base = Density.gumbel_min(LAT, loc=0.0, scale=scale)
    p = FactorRace(base, mu, np.zeros((5, 1))).state_prices()
    z = -mu / scale
    softmax = np.exp(z - z.max())
    softmax /= softmax.sum()
    assert np.abs(p - softmax).max() < 2e-3


def test_correlated_softmax_departs_from_luce_and_sums_to_one():
    """Nonzero loadings on a Gumbel base: a non-IIA softmax generalization."""
    mu = np.array([-0.5, -0.5, 0.2, 0.2, 0.6, 0.6])
    base = Density.gumbel_min(LAT, loc=0.0, scale=0.7)
    V = np.zeros((6, 1))
    V[:2, 0] = 0.8  # the first two share an environment
    p = FactorRace(base, mu, V).state_prices()
    z = -mu / 0.7
    softmax = np.exp(z - z.max())
    softmax /= softmax.sum()
    assert abs(p.sum() - 1.0) < 1e-12
    assert np.abs(p - softmax).max() > 5e-3  # correlation must matter


def test_deletion_ensemble_matches_per_scratch_recompute():
    n = 6
    C = circle_kernel(n, 1.0)
    mu = RNG.normal(0.0, 0.5, n)
    fr = gaussian_factor_race(LAT, C, 2, mu)
    q = fr.deletion_ensemble()
    assert np.allclose(q.sum(axis=1), 1.0)
    for i in (0, 3):
        keep = np.setdiff1d(np.arange(n), [i])
        direct = fr.state_prices(keep=keep)
        assert np.abs(direct - q[i][keep]).max() < 1e-10


def test_scratch_after_calibration_favors_correlated_partner():
    """The neighbor-inheritance effect runs through the inverse map: correlated
    partners cannibalize each other, so matching equal observed win frequencies
    forces them to stronger fitted abilities; scratching one then hands its wins
    disproportionately to the partner.

    (At EQUAL abilities the effect vanishes: scratching is a marginal, and the
    survivors here are mutually independent, so redistribution is exactly
    uniform -- the deletion-semantics point.)"""
    V = np.array([[0.9], [0.9], [0.0], [0.0]])
    bases = [
        Density.skew_normal(LAT, 0.0, float(np.sqrt(max(1 - v[0] ** 2, 1e-3))), 0.0) for v in V
    ]
    # equal-ability sanity: uniform redistribution
    q0 = FactorRace(bases, np.zeros(4), V).deletion_ensemble()
    assert np.abs(q0[0] - np.array([0.0, 1 / 3, 1 / 3, 1 / 3])).max() < 1e-3

    # calibrate to equal observed frequencies, then scratch
    mu_fit = solve_abilities(bases, V, np.full(4, 0.25), n_iter=400)
    assert mu_fit[0] < mu_fit[2]  # partners fitted stronger (min wins)
    q = FactorRace(bases, mu_fit, V).deletion_ensemble()
    assert q[0][1] > q[0][2] + 5e-3
    assert q[0][1] > q[0][3] + 5e-3


def test_solve_abilities_roundtrip_under_correlation():
    n = 8
    C = circle_kernel(n, 1.2)
    mu_true = RNG.normal(0.0, 0.4, n)
    mu_true -= mu_true.mean()
    fr = gaussian_factor_race(LAT, C, 2, mu_true)
    target = fr.state_prices()
    V, D = factor_model(C, 2)
    bases = [Density.skew_normal(LAT, 0.0, float(np.sqrt(d)), 0.0) for d in D]
    mu_fit = solve_abilities(bases, V, target, n_iter=400)
    back = FactorRace(bases, mu_fit, V).state_prices()
    assert np.abs(back - target).max() < 2e-3


def test_hermite_nodes_integrate_gaussian_moments():
    # tolerances reflect node pruning (corner nodes carry x^2-weighted mass)
    F, W = hermite_nodes(2, Q=15)
    assert abs(W.sum() - 1.0) < 5e-6
    assert np.abs(W @ F).max() < 5e-6  # mean zero
    assert np.abs(W @ (F**2) - 1.0).max() < 5e-6  # unit variance


@pytest.mark.parametrize("keep", [[0, 2, 4], [1, 3]])
def test_state_prices_keep_subsets_sum_to_one(keep):
    mu = RNG.normal(0.0, 0.5, 5)
    fr = gaussian_factor_race(LAT, circle_kernel(5, 1.0), 2, mu)
    p = fr.state_prices(keep=keep)
    assert len(p) == len(keep)
    assert abs(p.sum() - 1.0) < 1e-12
