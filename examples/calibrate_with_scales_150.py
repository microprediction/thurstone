"""
Example: 2D calibration with 150 runners and per-runner scales in [15, 20].

This demonstrates AbilityCalibrator.set_scales(...) and solve_from_prices(...)
using a 2D interpolation in (loc, scale).

Optional visualization requires matplotlib:
    pip install matplotlib

Run:
    python examples/calibrate_with_scales_150.py
"""
import numpy as np
from thurstone import UniformLattice, Density
from thurstone.pricing import Race
from thurstone.inference import AbilityCalibrator


def main():
    # Lattice and symmetric base density
    lattice = UniformLattice(L=500, unit=0.1)
    base = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0)

    # Ground-truth abilities (locs, physical units) and per-runner scales
    n = 150
    true_locs = np.linspace(-6.0, 6.0, n)
    scales = np.linspace(15.0, 20.0, n)

    # Forward: build densities from (loc, scale), then compute state prices
    densities = [Density.skew_normal(lattice, loc=float(loc), scale=float(s), a=0.0)
                 for loc, s in zip(true_locs, scales)]
    prices = Race(densities).state_prices()

    # Inverse: 2D calibration in (loc, scale)
    cal = AbilityCalibrator(
        base,
        n_iter=5,
        loc_span=8.0,     # widen for large scales
        loc_step=0.2,
        scale_span=1.0,
        scale_steps=3,
        skew_a=0.0,
    )
    cal.set_scales(scales)
    est_locs = np.array(cal.solve_from_prices(prices), dtype=float)

    # Compare after median-centering (translation invariance)
    true_c = true_locs - np.median(true_locs)
    est_c = est_locs - np.median(est_locs)
    corr = float(np.corrcoef(true_c, est_c)[0, 1])
    max_abs_err = float(np.max(np.abs(true_c - est_c)))
    print(f"Correlation (median-centered): {corr:.6f}")
    print(f"Max abs error (median-centered): {max_abs_err:.3f}")

    # Optional visualization
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(true_c, est_c, s=12, alpha=0.7, label="runners")
        lim = float(max(np.max(np.abs(true_c)), np.max(np.abs(est_c))))
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x")
        ax.set_title("2D calibration: true vs estimated loc (median-centered)")
        ax.set_xlabel("true loc (centered)")
        ax.set_ylabel("estimated loc (centered)")
        ax.legend()
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        # Plotting is optional
        pass


if __name__ == "__main__":
    main()


