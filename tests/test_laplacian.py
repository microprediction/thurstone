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


# ---- field builders ----


def _smooth(abilities, lattice=LATTICE, scales=None, skews=None):
    n = len(abilities)
    scales = scales or [1.0] * n
    skews = skews or [0.0] * n
    out = []
    for a, s, k in zip(abilities, scales, skews):
        base = Density.skew_normal(lattice, loc=0.0, scale=s, a=k)
        out.append(base.shift_fractional(a / lattice.unit))
    return out


def _atom(x_phys, lattice=LATTICE):
    p = np.zeros(lattice.size)
    p[int(round(x_phys / lattice.unit)) + lattice.L] = 1.0
    return Density(lattice, p)


def _zero_mass(lattice=LATTICE):
    return Density(lattice, np.zeros(lattice.size))


def _two_point(x0, x1, delta, lattice=LATTICE):
    """Mass 1-delta at x0 and delta at x1: adversarial near-zero survival."""
    p = np.zeros(lattice.size)
    p[int(round(x0 / lattice.unit)) + lattice.L] = 1.0 - delta
    p[int(round(x1 / lattice.unit)) + lattice.L] = delta
    return Density(lattice, p)


# Smooth fields: differentiable forward map, so finite differences apply.
SMOOTH_CASES = {
    "baseline": dict(abilities=[0.0, 0.35, -0.2, 0.6, 0.1]),
    "pair": dict(abilities=[0.0, 0.4]),
    "identical": dict(abilities=[0.0, 0.0, 0.0, 0.0]),
    "near_equal": dict(abilities=[0.0, 1e-4, -1e-4]),
    "extreme_separation": dict(abilities=[-6.0, 0.0, 6.0]),
    "mixed_scales": dict(
        abilities=[0.0, 0.3, -0.4],
        scales=[0.5, 1.0, 2.0],
        skews=[0.0, 0.8, -0.5],
    ),
}


def _large_random(lattice=LATTICE):
    rng = np.random.default_rng(0)
    abilities = rng.normal(scale=0.6, size=25)
    scales = rng.choice([0.6, 1.0, 1.6], size=25).tolist()
    return _smooth(abilities.tolist(), lattice, scales=scales)


# Every field the machinery must handle, degenerate cases included.
FIELD_BUILDERS = {
    **{
        name: (lambda kw: lambda lat=LATTICE: _smooth(**kw, lattice=lat))(kw)
        for name, kw in SMOOTH_CASES.items()
    },
    "atom_vs_smooth": lambda lat=LATTICE: [_atom(0.1, lat)] + _smooth([0.0, 0.3], lat),
    "two_atoms_same_point": lambda lat=LATTICE: (
        [
            _atom(0.0, lat),
            _atom(0.0, lat),
        ]
        + _smooth([0.2], lat)
        + _smooth([-0.1], lat)
    ),
    "atoms_apart": lambda lat=LATTICE: [_atom(-0.5, lat), _atom(0.5, lat)] + _smooth([0.0], lat),
    "pure_atom_pair": lambda lat=LATTICE: [_atom(0.0, lat), _atom(0.3, lat)],
    "zero_mass_runner": lambda lat=LATTICE: [_zero_mass(lat)] + _smooth([0.0, 0.3], lat),
    "all_zero_mass": lambda lat=LATTICE: [_zero_mass(lat), _zero_mass(lat)],
    "off_lattice_shift": lambda lat=LATTICE: (
        [_smooth([0.0], lat)[0].shift_integer(2 * lat.L + 2)] + _smooth([0.0, 0.3], lat)
    ),
    "edge_pileup": lambda lat=LATTICE: _smooth(
        [-(lat.L - 20) * lat.unit, (lat.L - 20) * lat.unit, 0.0], lat
    ),
    "tiny_survival_masked": lambda lat=LATTICE: (
        [_two_point(0.0, 0.5, 1e-7, lat)] + _smooth([0.0, 0.2], lat)
    ),
    "tiny_survival_unmasked": lambda lat=LATTICE: (
        [_two_point(0.0, 0.5, 1e-5, lat)] + _smooth([0.0, 0.2], lat)
    ),
    "atom_left_edge": lambda lat=LATTICE: [_atom(-lat.L * lat.unit, lat)] + _smooth([0.0], lat),
    "atom_right_edge": lambda lat=LATTICE: [_atom(lat.L * lat.unit, lat)] + _smooth([0.0], lat),
    "identical_atom_pair": lambda lat=LATTICE: [_atom(0.0, lat), _atom(0.0, lat)],
    "identical_atom_trio": lambda lat=LATTICE: [_atom(0.0, lat)] * 3,
    "bimodal_gap": lambda lat=LATTICE: [
        Density(lat, _smooth([-3.0], lat)[0].p + _smooth([3.0], lat)[0].p),
        *_smooth([0.0, 0.5], lat),
    ],
    "zero_vs_smooth_pair": lambda lat=LATTICE: [_zero_mass(lat)] + _smooth([0.0], lat),
    "large_random": _large_random,
}

FIELD_NAMES = list(FIELD_BUILDERS)


# ---- matvec vs dense: the algebraic identity, on every field ----


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_matvec_matches_dense(name):
    field = FIELD_BUILDERS[name]()
    L = laplacian_dense(field)
    rng = np.random.default_rng(7)
    scale = max(np.abs(L).max(), 1.0)
    for _ in range(3):
        u = rng.normal(size=len(field))
        got = laplacian_matvec(field, u)
        want = L @ u
        assert np.allclose(got, want, rtol=1e-7, atol=1e-9 * scale), (
            f"{name}: max diff {np.abs(got - want).max():.3e}"
        )


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_matvec_linearity_and_scaling(name):
    field = FIELD_BUILDERS[name]()
    rng = np.random.default_rng(11)
    n = len(field)
    u, v = rng.normal(size=n), rng.normal(size=n)
    lhs = laplacian_matvec(field, 2.5 * u - 3.0 * v)
    rhs = 2.5 * laplacian_matvec(field, u) - 3.0 * laplacian_matvec(field, v)
    ref = np.abs(lhs).max() + np.abs(rhs).max() + 1.0
    assert np.allclose(lhs, rhs, atol=1e-10 * ref)
    # huge-magnitude u must not degrade agreement with the dense form
    big = 1e8 * u
    assert np.allclose(
        laplacian_matvec(field, big), laplacian_dense(field) @ big, rtol=1e-7, atol=1e-2
    )


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_constant_vector_annihilated_exactly(name):
    field = FIELD_BUILDERS[name]()
    out = laplacian_matvec(field, np.ones(len(field)))
    assert np.all(out == 0.0)


# ---- structural properties of the dense form, on every field ----


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_weights_symmetric_nonnegative(name):
    field = FIELD_BUILDERS[name]()
    W = laplacian_weights(field)
    assert np.all(np.isfinite(W))
    assert np.allclose(W, W.T)
    assert np.all(np.diag(W) == 0.0)
    assert np.all(W >= 0.0)


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_laplacian_psd_with_null_vector(name):
    field = FIELD_BUILDERS[name]()
    L = laplacian_dense(field)
    n = len(field)
    scale = max(np.abs(L).max(), 1.0)
    assert np.allclose(L @ np.ones(n), 0.0, atol=1e-12 * scale)
    eig = np.linalg.eigvalsh(L)
    assert eig[0] > -1e-10 * scale


def test_baseline_strictly_connected():
    """Overlapping smooth runners: all weights positive, spectral gap open."""
    field = FIELD_BUILDERS["baseline"]()
    W = laplacian_weights(field)
    off = W[~np.eye(len(field), dtype=bool)]
    assert np.all(off > 0.0)
    eig = np.linalg.eigvalsh(laplacian_dense(field))
    assert eig[1] > 0.0


# ---- degenerate-field semantics ----


def test_zero_mass_runner_disconnects():
    """Off-lattice sentinel: zero row/col, second null vector, others intact."""
    field = FIELD_BUILDERS["zero_mass_runner"]()
    W = laplacian_weights(field)
    assert np.all(W[0, :] == 0.0) and np.all(W[:, 0] == 0.0)
    L = laplacian_dense(field)
    e0 = np.zeros(len(field))
    e0[0] = 1.0
    assert np.all(L @ e0 == 0.0)
    assert np.all(laplacian_matvec(field, e0) == 0.0)
    assert outright_win_probabilities(field)[0] == 0.0
    # the smooth pair must be unaffected by the spectator
    sub = laplacian_weights(field[1:])
    assert np.allclose(W[1:, 1:], sub, rtol=1e-12)


def test_all_zero_mass():
    field = FIELD_BUILDERS["all_zero_mass"]()
    assert np.all(laplacian_weights(field) == 0.0)
    assert np.all(laplacian_matvec(field, np.array([1.0, -1.0])) == 0.0)
    assert np.all(outright_win_probabilities(field) == 0.0)


def test_two_atoms_same_point_never_win_outright():
    field = FIELD_BUILDERS["two_atoms_same_point"]()
    p = outright_win_probabilities(field)
    assert p[0] == 0.0 and p[1] == 0.0  # they always tie each other


def test_atoms_apart_dominated_runner_is_isolated():
    """An atom deterministically beaten by another has all weights zero.

    The atom at +0.5 can never finish before the atom at -0.5, so its win
    probability is frozen at 0: every derivative involving it vanishes and
    it becomes an isolated vertex of the graph (a second null direction).
    """
    field = FIELD_BUILDERS["atoms_apart"]()
    W = laplacian_weights(field)
    assert W[0, 1] == 0.0  # no overlap between the atoms
    assert W[0, 2] > 0.0  # the early atom still interacts with the smooth runner
    assert np.all(W[1, :] == 0.0)  # the dominated atom is isolated
    L = laplacian_dense(field)
    e1 = np.array([0.0, 1.0, 0.0])
    assert np.all(L @ e1 == 0.0)
    assert np.all(laplacian_matvec(field, e1) == 0.0)
    assert outright_win_probabilities(field)[1] == 0.0


def test_identical_atoms_tie_slope():
    """Two identical atoms always tie; the discrete tie-splitting slope is 1/unit.

    The continuum map is not differentiable at an atomic tie (p jumps as one
    atom moves off it); on the lattice this shows up as w = 1/unit, diverging
    as the grid refines.  A third coincident atom kills every pair product,
    so the trio's weights vanish entirely.
    """
    pair = FIELD_BUILDERS["identical_atom_pair"]()
    assert laplacian_weights(pair)[0, 1] == pytest.approx(1.0 / LATTICE.unit)
    trio = FIELD_BUILDERS["identical_atom_trio"]()
    assert np.all(laplacian_weights(trio) == 0.0)


def test_zero_mass_vs_smooth_pair():
    """A lone real runner against the off-lattice sentinel wins with certainty."""
    p = outright_win_probabilities(FIELD_BUILDERS["zero_vs_smooth_pair"]())
    assert p[0] == 0.0
    assert p[1] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_outright_probabilities_bounds(name):
    p = outright_win_probabilities(FIELD_BUILDERS[name]())
    assert np.all(p >= 0.0)
    assert p.sum() <= 1.0 + 1e-12


def test_translation_invariance():
    """p(a + c 1) = p(a) up to lattice edge effects."""
    kw = SMOOTH_CASES["baseline"]
    p0 = outright_win_probabilities(_smooth(**kw))
    p1 = outright_win_probabilities(_smooth([a + 0.25 for a in kw["abilities"]]))
    assert np.allclose(p0, p1, atol=1e-10)


# ---- finite differences: the calculus, on every smooth field ----


def _fd_jacobian(kw, lattice):
    abilities = kw["abilities"]
    n = len(abilities)
    eps = lattice.unit  # one lattice step: spans the piecewise-linear kink
    J = np.zeros((n, n))
    for j in range(n):
        up = dict(kw, abilities=[a + eps * (k == j) for k, a in enumerate(abilities)])
        dn = dict(kw, abilities=[a - eps * (k == j) for k, a in enumerate(abilities)])
        J[:, j] = (
            outright_win_probabilities(_smooth(**up, lattice=lattice))
            - outright_win_probabilities(_smooth(**dn, lattice=lattice))
        ) / (2 * eps)
    return J


@pytest.mark.parametrize("name", list(SMOOTH_CASES))
def test_finite_difference_jacobian(name):
    """Central differences of the forward map recover -L(w) to O(unit)."""
    kw = SMOOTH_CASES[name]
    L = laplacian_dense(_smooth(**kw))
    J = _fd_jacobian(kw, LATTICE)
    assert np.allclose(J, -L, atol=1e-2)
    mask = np.abs(L) > 1e-3
    if mask.any():
        assert np.max(np.abs((J + L)[mask] / L[mask])) < 5e-2


@pytest.mark.parametrize("name", list(SMOOTH_CASES))
def test_finite_difference_jacobian_converges(name):
    """The FD-vs-analytic gap is discretization error: it halves with the unit."""
    kw = SMOOTH_CASES[name]
    err = []
    for half_width, unit in [(400, 0.05), (800, 0.025)]:
        lattice = UniformLattice(L=half_width, unit=unit)
        L = laplacian_dense(_smooth(**kw, lattice=lattice))
        err.append(np.abs(_fd_jacobian(kw, lattice) + L).max())
    if err[0] < 1e-9:
        return  # already at floating-point floor (fully separated fields)
    assert err[1] < 0.6 * err[0]


def test_directional_derivative_large_field():
    """FD directional derivative matches -L u at n = 25 without forming J."""
    rng = np.random.default_rng(5)
    abilities = np.random.default_rng(0).normal(scale=0.6, size=25)
    scales = np.random.default_rng(0).choice([0.6, 1.0, 1.6], size=25).tolist()
    u = rng.normal(size=25)
    eps = LATTICE.unit
    p_up = outright_win_probabilities(_smooth((abilities + eps * u).tolist(), scales=scales))
    p_dn = outright_win_probabilities(_smooth((abilities - eps * u).tolist(), scales=scales))
    fd = (p_up - p_dn) / (2 * eps)
    lu = laplacian_matvec(_smooth(abilities.tolist(), scales=scales), u)
    assert np.allclose(fd, -lu, atol=2e-2 * max(np.abs(u).max(), 1.0))


# ---- continuum anchor: closed form for a Gaussian pair ----


def test_gaussian_pair_matches_closed_form():
    """w_12 for two unit normals is exp(-d^2/4) / (2 sqrt(pi)).

    Trapezoidal quadrature of smooth, rapidly decaying integrands is
    spectrally accurate, so the lattice weight hits the closed form at
    machine precision already on the coarsest grid.
    """
    delta = 0.3
    w_exact = np.exp(-(delta**2) / 4.0) / (2.0 * np.sqrt(np.pi))
    from math import erf

    p_exact = 0.5 * (1.0 + erf(delta / 2.0))  # P(X_1 < X_2) = Phi(delta / sqrt(2))
    for half_width, unit in [(200, 0.1), (400, 0.05), (800, 0.025)]:
        lattice = UniformLattice(L=half_width, unit=unit)
        field = _smooth([0.0, delta], lattice)
        w = laplacian_weights(field)[0, 1]
        assert abs(w - w_exact) < 1e-10 * w_exact
        # the forward map: outright win prob converges to Phi at O(unit)
        # (first order because the lattice tie mass ~ unit is excluded)
        p = outright_win_probabilities(field)
        assert abs(p[0] - p_exact) < 2.0 * unit


# ---- fuzz: random mixtures of every special ingredient ----


def _random_special_field(rng, lattice=LATTICE):
    n = int(rng.integers(2, 9))
    out = []
    for _ in range(n):
        kind = rng.choice(
            ["smooth", "narrow", "atom", "two_point", "zero", "edge"],
            p=[0.4, 0.15, 0.15, 0.15, 0.05, 0.10],
        )
        if kind == "smooth":
            d = Density.skew_normal(
                lattice, loc=0.0, scale=rng.uniform(0.3, 2.0), a=rng.uniform(-1.0, 1.0)
            ).shift_fractional(rng.normal(0.0, 2.0) / lattice.unit)
        elif kind == "narrow":
            # one-to-three lattice points wide: numerically almost an atom
            d = Density.skew_normal(
                lattice, loc=0.0, scale=rng.uniform(0.04, 0.15), a=0.0
            ).shift_fractional(rng.normal(0.0, 2.0) / lattice.unit)
        elif kind == "atom":
            d = _atom(float(rng.uniform(-15.0, 15.0)), lattice)
        elif kind == "two_point":
            x0 = float(rng.uniform(-5.0, 5.0))
            d = _two_point(x0, x0 + rng.uniform(0.1, 3.0), 10.0 ** rng.uniform(-9, -3), lattice)
        elif kind == "zero":
            d = _zero_mass(lattice)
        else:  # edge
            side = 1 if rng.random() < 0.5 else -1
            d = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0).shift_fractional(
                side * (lattice.L - int(rng.integers(0, 30)))
            )
        out.append(d)
    return out


@pytest.mark.parametrize("seed", range(25))
def test_fuzz_matvec_matches_dense(seed):
    rng = np.random.default_rng(seed)
    field = _random_special_field(rng)
    n = len(field)
    W = laplacian_weights(field)
    assert np.all(np.isfinite(W)) and np.all(W >= 0.0)
    assert np.allclose(W, W.T)
    L = np.diag(W.sum(axis=1)) - W
    scale = max(np.abs(L).max(), 1.0)
    assert np.linalg.eigvalsh(L)[0] > -1e-10 * scale
    for _ in range(2):
        u = rng.normal(size=n)
        got = laplacian_matvec(field, u)
        assert np.all(np.isfinite(got))
        assert np.allclose(got, L @ u, rtol=1e-7, atol=1e-9 * scale), (
            f"seed {seed}: max diff {np.abs(got - L @ u).max():.3e}"
        )
    assert np.all(laplacian_matvec(field, np.ones(n)) == 0.0)
    p = outright_win_probabilities(field)
    assert np.all(p >= 0.0) and p.sum() <= 1.0 + 1e-12


# ---- Newton-CG inversion: the matvec supports its intended use ----


def _cg_solve(matvec, b, iters=200, tol=1e-14):
    """Conjugate gradients for L x = b on the mean-zero subspace."""
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = float(r @ r)
    for _ in range(iters):
        Ap = matvec(p)
        denom = float(p @ Ap)
        if denom <= 0.0:
            break
        alpha = rs / denom
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(r @ r)
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def test_newton_cg_inversion_converges():
    """Joint Newton-CG inversion of prices back to abilities, matvec-only.

    Damped Newton with the O(n M) Hessian-vector product recovers the true
    ability vector (up to the translation gauge) from its own forward
    probabilities, without ever forming the dense Jacobian.
    """
    rng = np.random.default_rng(2)
    n = 8
    a_true = rng.normal(scale=0.7, size=n)
    a_true -= a_true.mean()
    scales = [0.7, 1.0, 1.3, 1.0, 0.9, 1.1, 1.0, 0.8]

    def build(a):
        return _smooth(list(a), scales=scales)

    target = outright_win_probabilities(build(a_true))
    a = np.zeros(n)
    res = np.inf
    for _ in range(30):
        field = build(a)
        r = outright_win_probabilities(field) - target
        res = np.abs(r).max()
        if res < 1e-11:
            break
        delta = _cg_solve(lambda v: laplacian_matvec(field, v), r - r.mean())
        # damped step: the analytic Laplacian is an O(unit) approximation of
        # the discrete map's derivative, so guard against overshoot
        step = 1.0
        for _ in range(20):
            trial = a + step * delta
            r_new = outright_win_probabilities(build(trial)) - target
            if np.abs(r_new).max() < res:
                a = trial - np.mean(trial)
                break
            step *= 0.5
        else:
            break
    assert res < 1e-11
    assert np.allclose(a, a_true, atol=1e-7)


# ---- validation ----


def test_input_validation():
    field = FIELD_BUILDERS["baseline"]()
    with pytest.raises(ValueError):
        laplacian_matvec(field, np.ones(3))
    with pytest.raises(ValueError):
        laplacian_matvec(field, np.array([1.0, np.nan, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        outright_win_probabilities(field[:1])
    other = UniformLattice(L=400, unit=0.1)
    mixed = [field[0], Density.skew_normal(other, loc=0.0, scale=1.0, a=0.0)]
    with pytest.raises(ValueError):
        laplacian_weights(mixed)
    corrupt = FIELD_BUILDERS["pair"]()
    corrupt[0].p = corrupt[0].p.copy()
    corrupt[0].p[0] = np.nan
    with pytest.raises(ValueError):
        laplacian_matvec(corrupt, np.zeros(2))
    negative = FIELD_BUILDERS["pair"]()
    negative[0].p = negative[0].p.copy()
    negative[0].p[0] = -0.1
    with pytest.raises(ValueError):
        laplacian_weights(negative)
