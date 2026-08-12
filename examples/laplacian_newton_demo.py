"""
Example: the Laplacian Jacobian and joint Newton-CG inversion.

The Jacobian of the map from abilities to outright win probabilities is
minus a weighted complete-graph Laplacian, and it can be applied to a
vector in O(nM) without ever being formed.  This demo shows:

1. Forward pricing of a 12-runner field with heterogeneous scales.
2. The Laplacian's structure: zero row sums (translation gauge), the
   spectral gap lambda_2, and the inverse-conditioning bound 1/lambda_2.
3. Joint Newton-CG inversion of normalized market-style prices back to
   abilities with invert_outright_probabilities, including the
   per-iteration residual history and the recovered tie-mass scale.
4. The matvec advantage at scale: O(nM) operator application versus the
   O(n^2 M) dense build at n = 100.

Run:
    python examples/laplacian_newton_demo.py
"""

import time

import numpy as np
from numpy.random import default_rng

from thurstone import (
    Density,
    LaplacianOperator,
    UniformLattice,
    invert_outright_probabilities,
    laplacian_dense,
    outright_win_probabilities,
)


def make_field(lattice, abilities, scales):
    return [
        Density.skew_normal(lattice, loc=0.0, scale=s, a=0.0).shift_fractional(a / lattice.unit)
        for a, s in zip(abilities, scales)
    ]


def main():
    rng = default_rng(7)
    lattice = UniformLattice(L=400, unit=0.05)
    n = 12
    a_true = rng.normal(scale=0.8, size=n)
    a_true -= a_true.mean()
    scales = rng.choice([0.7, 1.0, 1.4], size=n)

    # 1. Forward: abilities -> outright win probabilities
    field = make_field(lattice, a_true, scales)
    p = outright_win_probabilities(field)
    print("Forward pricing (min wins; lower ability = better):")
    print(f"  sum of outright win probabilities: {p.sum():.6f}")
    print(f"  (the deficit {1 - p.sum():.6f} is the lattice tie mass)\n")

    # 2. Structure: the Jacobian is -L(w), a weighted graph Laplacian
    L = laplacian_dense(field)
    eig = np.linalg.eigvalsh(L)
    print("Laplacian Jacobian structure:")
    print(f"  max |row sum|   : {np.abs(L @ np.ones(n)).max():.2e}  (translation gauge)")
    print(f"  lambda_2        : {eig[1]:.4f}")
    print(f"  1 / lambda_2    : {1 / eig[1]:.2f}  (inverse sensitivity bound)\n")

    # 3. Inverse: normalized market prices -> abilities, matvec only
    market = p / p.sum()  # what a market would quote: sums to one
    result = invert_outright_probabilities(
        [Density.skew_normal(lattice, loc=0.0, scale=s, a=0.0) for s in scales],
        market,
    )
    print("Joint Newton-CG inversion from normalized prices:")
    print(f"  converged  : {result.converged} ({result.message})")
    print(f"  iterations : {result.iterations}")
    print(f"  residual   : {result.residual:.2e}")
    print(f"  scale      : {result.scale:.6f}  (recovers the attainable total)")
    history = ", ".join(f"{r:.1e}" for r in result.residual_history)
    print(f"  residual history: {history}")
    err = np.abs(result.abilities - a_true).max()
    print(f"  max |recovered - true ability|: {err:.2e}\n")

    print("  runner   true a   recovered      price")
    order = np.argsort(a_true)
    for i in order:
        print(f"    {i:2d}    {a_true[i]:+.4f}   {result.abilities[i]:+.4f}   {market[i]:8.5f}")

    # 4. Scale: apply the Jacobian at n = 100 without forming it
    big_n = 100
    big = make_field(lattice, rng.normal(scale=0.5, size=big_n), np.ones(big_n))
    u = rng.normal(size=big_n)
    t0 = time.perf_counter()
    op = LaplacianOperator(big)
    for _ in range(20):
        op.matvec(u)
    t_op = (time.perf_counter() - t0) / 20
    t0 = time.perf_counter()
    laplacian_dense(big)
    t_dense = time.perf_counter() - t0
    print(
        f"\nAt n = {big_n}: operator matvec {t_op * 1e3:.2f} ms "
        f"vs dense build {t_dense * 1e3:.0f} ms per Jacobian use"
    )


if __name__ == "__main__":
    main()
