"""
Example: Global LS calibration via per-race relative abilities.

- 20 horses with true global abilities (locs) in [-3, 3]
- 5 races, each with 8 randomly chosen horses
- For each race: invert prices -> centered per-race abilities
- Stitch globally with LS across overlapping horses (slope-weighted)

Run:
    python examples/global_calibration_ls_demo.py
"""
import numpy as np
from numpy.random import default_rng

from thurstone import UniformLattice, Density, AbilityCalibrator, GlobalLSCalibrator

NUM_HORSES = 500
NUM_RACES = 100
RACE_SIZE = 20

def main():
    rng = default_rng(123)
    # Base lattice and density
    lattice = UniformLattice(L=400, unit=0.1)
    base = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0)

    # Global ground truth
    num_horses = NUM_HORSES
    horse_ids = [f"H{i}" for i in range(num_horses)]
    true_theta = np.linspace(-3.0, 3.0, num_horses)
    rng.shuffle(true_theta)

    # Build races
    num_races = NUM_RACES
    race_size = RACE_SIZE
    race_bias = np.zeros(num_races)  # no bias needed for this LS demo

    calibrators = []
    race_horse_ids = []
    race_prices = []

    for r in range(num_races):
        idx = rng.choice(num_horses, size=race_size, replace=False)
        ids_r = [horse_ids[i] for i in idx]
        mu_r = [float(true_theta[i] + race_bias[r]) for i in idx]
        cal_r = AbilityCalibrator(base, n_iter=3)
        # Enhancing interpolation stability
        cal_r.offset_grid = list(range(-80, 81))
        cal_r.loc_span = 6.0
        cal_r.loc_step = 0.1
        p_r = np.array(cal_r.state_prices_from_ability(mu_r), dtype=float)
        # Build curves once (cheap)
        cal_r.solve_from_prices(p_r)
        calibrators.append(cal_r)
        race_horse_ids.append(ids_r)
        race_prices.append(p_r)

    # Global LS stitching
    gls = GlobalLSCalibrator(horse_ids=horse_ids)
    for r in range(num_races):
        gls.add_race(calibrators[r], race_horse_ids[r], race_prices[r])
    # One or two relinearization passes are usually enough here
    gls.fit(use_slope_weights=True, ridge=1e-8, weight_cap=1.0)

    # Evaluate recovery up to translation
    est_theta = np.array([gls.theta[hid] for hid in horse_ids], dtype=float)
    est_c = est_theta - np.median(est_theta)
    true_c = true_theta - np.median(true_theta)
    corr = float(np.corrcoef(true_c, est_c)[0, 1])
    mae = float(np.mean(np.abs(true_c - est_c)))
    print(f"Global LS ability correlation (centered): {corr:.6f}")
    print(f"Global LS ability MAE (centered):        {mae:.4f}")

    # Per-race reconstruction from global theta (using per-race median alignment)
    total_l2 = 0.0
    print("\nPer-race probabilities (true vs fitted via LS):")
    for r in range(num_races):
        ids = race_horse_ids[r]
        p_true = race_prices[r]
        # Align race by per-race median of (local inversion - global theta) as a bias estimate
        local_inv = np.array(calibrators[r].solve_from_prices(p_true), dtype=float)
        br = float(np.median(local_inv - np.array([gls.theta[h] for h in ids], dtype=float)))
        mu_fit = [float(gls.theta[h] + br) for h in ids]
        p_fit = np.array(calibrators[r].state_prices_from_ability(mu_fit), dtype=float)
        l2 = float(np.linalg.norm(p_true - p_fit))
        total_l2 += l2
        print(f"\nRace {r}:")
        print(f"  horses: {ids}")
        print(f"  true:   {np.array2string(p_true, precision=4, floatmode='fixed')}")
        print(f"  fitted: {np.array2string(p_fit, precision=4, floatmode='fixed')}")
        print(f"  L2(true,fitted): {l2:.6e}")
    print(f"\nAggregate L2(true,fitted) across races: {total_l2:.6e}")

    # Plot true vs inferred (centered)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(true_c, est_c, s=24, alpha=0.8)
        lim = float(max(np.max(np.abs(true_c)), np.max(np.abs(est_c))))
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x")
        ax.set_title("Global LS: true vs inferred loc (centered)")
        ax.set_xlabel("true loc (centered)")
        ax.set_ylabel("inferred loc (centered)")
        ax.grid(alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()


