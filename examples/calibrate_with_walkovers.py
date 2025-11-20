"""
Example: 1D calibration with walkovers (location-only interpolation).

This demonstrates that when abilities create clear walkovers (far-separated
clusters), the 1D inverse (location-only) can recover probabilities closely.

Run:
    python examples/calibrate_with_walkovers.py
"""
import numpy as np
from thurstone import UniformLattice, Density
from thurstone.inference import AbilityCalibrator
from thurstone.pricing import Race


def main():
    # Fine lattice and symmetric base density
    lattice = UniformLattice(L=50, unit=0.25)
    base = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0)

    # Construct abilities with pronounced separation (walkovers)
    # - Strong cluster far left
    # - Mid cluster near zero
    # - Weak cluster far right
    strong = list(np.linspace(-3.0, -1.0, 8))
    mid    = list(np.linspace(-0.5, 0.5, 14))
    weak   = list(np.linspace(1.0, 3.0, 8))
    true_ability = strong + mid + weak
    n = len(true_ability)

    # Forward: true prices from abilities (uses clustering internally)
    cal = AbilityCalibrator(base, n_iter=4)
    prices_true = np.array(cal.state_prices_from_ability(true_ability), dtype=float)

    # Also compute prices from an explicit Race built from shifted base (sanity)
    dens = [base.shift_fractional(a / lattice.unit) for a in true_ability]
    prices_race = np.array(Race(dens).state_prices(), dtype=float)

    # Inverse: 1D location-only calibration (no per-runner scales)
    est_locs = np.array(cal.solve_from_prices(prices_true), dtype=float)
    dens_est = [base.shift_fractional(a / lattice.unit) for a in est_locs]
    prices_est = np.array(Race(dens_est).state_prices(), dtype=float)

    # Metrics
    corr_p = float(np.corrcoef(prices_true, prices_est)[0, 1])
    l1_err = float(np.sum(np.abs(prices_true - prices_est)))
    max_err = float(np.max(np.abs(prices_true - prices_est)))
    print(f"n = {n}")
    print(f"Price correlation (true vs recovered): {corr_p:.6f}")
    print(f"Price L1 error: {l1_err:.6e}")
    print(f"Price max abs error: {max_err:.6e}")

    # Show walkover nature (min/max)
    print(f"True price min/max:  {prices_true.min():.3e} / {prices_true.max():.3e}")
    print(f"Recov price min/max: {prices_est.min():.3e} / {prices_est.max():.3e}")

    # Optional visualization
    try:
        import matplotlib.pyplot as plt

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.scatter(prices_true, prices_est, s=14, alpha=0.7)
        lim = float(max(np.max(prices_true), np.max(prices_est)))
        ax1.plot([0.0, lim], [0.0, lim], "k--", lw=1)
        ax1.set_title("Walkovers: true vs recovered probabilities")
        ax1.set_xlabel("true price")
        ax1.set_ylabel("recovered price")
        ax1.grid(alpha=0.25)
        plt.tight_layout()

        # Plot estimated vs true ability (centered for comparability)
        t_c = np.array(true_ability) - np.median(true_ability)
        e_c = est_locs - np.median(est_locs)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.scatter(t_c, e_c, s=14, alpha=0.7)
        lim2 = float(max(np.max(np.abs(t_c)), np.max(np.abs(e_c))))
        ax2.plot([-lim2, lim2], [-lim2, lim2], "k--", lw=1)
        ax2.set_title("Walkovers: true vs recovered loc (median-centered)")
        ax2.set_xlabel("true loc (centered)")
        ax2.set_ylabel("recovered loc (centered)")
        ax2.grid(alpha=0.25)
        plt.tight_layout()

        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()


