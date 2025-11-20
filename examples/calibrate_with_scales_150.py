"""
Example: 2D calibration with 150 runners 

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
    lattice = UniformLattice(L=250, unit=0.2)
    base = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0)

    # Ground-truth abilities (locs, physical units) and per-runner scales
    n = 150
    true_locs = np.linspace(-6.0, 6.0, n)
    scales = np.linspace(2.0, 4.0, n)

    # Forward: build densities from (loc, scale), then compute state prices
    densities = [Density.skew_normal(lattice, loc=float(loc), scale=float(s), a=0.0)
                 for loc, s in zip(true_locs, scales)]
    prices_true = Race(densities).state_prices()

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
    est_locs = np.array(cal.solve_from_prices(prices_true), dtype=float)

    # Compare after median-centering (translation invariance)
    true_c = true_locs - np.median(true_locs)
    est_c = est_locs - np.median(est_locs)
    corr_loc = float(np.corrcoef(true_c, est_c)[0, 1])
    max_abs_err_loc = float(np.max(np.abs(true_c - est_c)))
    print(f"Loc correlation (median-centered): {corr_loc:.6f}")
    print(f"Loc max abs error (median-centered): {max_abs_err_loc:.3f}")

    # Recover probabilities from estimated locs (using the same per-runner scales)
    densities_est = [Density.skew_normal(lattice, loc=float(loc), scale=float(s), a=0.0)
                     for loc, s in zip(est_locs, scales)]
    prices_est = Race(densities_est).state_prices()

    # Compare probabilities
    corr_p = float(np.corrcoef(prices_true, prices_est)[0, 1])
    max_abs_err_p = float(np.max(np.abs(prices_true - prices_est)))
    l1_err_p = float(np.sum(np.abs(prices_true - prices_est)))
    print(f"Price correlation: {corr_p:.6f}")
    print(f"Price max abs error: {max_abs_err_p:.6e}")
    print(f"Price L1 error: {l1_err_p:.6e}")

    # Optional visualization
    try:
        import matplotlib.pyplot as plt
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.scatter(true_c, est_c, s=12, alpha=0.7, label="runners")
        lim = float(max(np.max(np.abs(true_c)), np.max(np.abs(est_c))))
        ax1.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x")
        ax1.set_title("2D calibration: true vs estimated loc (median-centered)")
        ax1.set_xlabel("true loc (centered)")
        ax1.set_ylabel("estimated loc (centered)")
        ax1.legend()
        ax1.grid(alpha=0.25)
        plt.tight_layout()

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.scatter(prices_true, prices_est, s=12, alpha=0.7)
        limp = float(max(np.max(prices_true), np.max(prices_est)))
        ax2.plot([0.0, limp], [0.0, limp], "k--", lw=1)
        ax2.set_title("Probabilities: true vs recovered")
        ax2.set_xlabel("true price")
        ax2.set_ylabel("recovered price")
        ax2.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        # Plotting is optional
        pass


if __name__ == "__main__":
    main()


