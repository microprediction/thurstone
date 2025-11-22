"""
Example: Global ability calibration with noisy, inconsistent probabilities.

- 20 horses with true global abilities (locs) in [-3, 3]
- 5 races, 8 randomly chosen horses each
- True prices are generated from the forward model, then perturbed with noise
- GlobalAbilityCalibrator is used to recover abilities from noisy prices

Run:
    python examples/global_calibration_noisy.py
"""
import numpy as np
from numpy.random import default_rng

from thurstone import UniformLattice, Density
from thurstone.inference import AbilityCalibrator
from thurstone.global_fit import GlobalAbilityCalibrator


NUM_HORSES = 1000
NUM_RACES = 500
RACE_SIZE = 20


def softmax_noise(p: np.ndarray, rng, sigma: float = 0.25, eps: float = 1e-12) -> np.ndarray:
    """Add zero-mean Gaussian noise in log space and re-normalize by softmax."""
    logits = np.log(np.clip(p, eps, 1.0))
    noisy = logits + rng.normal(0.0, sigma, size=p.shape)
    ex = np.exp(noisy - np.max(noisy))
    return ex / np.sum(ex)


def main():
    rng = default_rng(7)
    # Base lattice and density
    lattice = UniformLattice(L=400, unit=0.1)
    base = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0)

    # Global ground truth
    num_horses = NUM_HORSES
    horse_ids = [f"H{i}" for i in range(num_horses)]
    true_theta = np.linspace(-3.0, 3.0, num_horses)
    rng.shuffle(true_theta)

    # Races
    num_races = NUM_RACES
    race_size = RACE_SIZE
    race_bias = rng.uniform(-0.2, 0.2, size=num_races)

    # Create per-race calibrators and noisy prices
    cal_per_race = []
    race_horse_ids = []
    race_prices_true = []
    race_prices_noisy = []
    for r in range(num_races):
        idx = rng.choice(num_horses, size=race_size, replace=False)
        ids_r = [horse_ids[i] for i in idx]
        mu_r = [float(true_theta[i] + race_bias[r]) for i in idx]
        cal_r = AbilityCalibrator(base, n_iter=3)
        # Steeper, well-bracketed curve for robust interpolation
        cal_r.offset_grid = list(range(-120, 121))
        cal_r.loc_span = 8.0
        cal_r.loc_step = 0.05
        p_true = np.array(cal_r.state_prices_from_ability(mu_r), dtype=float)
        # Populate lookup curves for this race (cheap)
        cal_r.solve_from_prices(p_true)
        # Add noise
        p_noisy = softmax_noise(p_true, rng, sigma=0.05)

        cal_per_race.append(cal_r)
        race_horse_ids.append(ids_r)
        race_prices_true.append(p_true)
        race_prices_noisy.append(p_noisy)

    # Global fit (with biases enabled for robustness)
    gcal = GlobalAbilityCalibrator(horse_ids=horse_ids)
    # Theta-first warm start from per-race inversions of noisy prices (median-centered)
    theta_accum = {hid: [] for hid in horse_ids}
    for r in range(num_races):
        gcal.add_race(cal_per_race[r], race_horse_ids[r], race_prices_noisy[r])
        est_r = np.array(cal_per_race[r].solve_from_prices(race_prices_noisy[r]), dtype=float)
        est_r_c = est_r - np.median(est_r)
        for hid, loc in zip(race_horse_ids[r], est_r_c):
            theta_accum[hid].append(float(loc))
    init_theta = {}
    for hid in horse_ids:
        vals = theta_accum[hid]
        init_theta[hid] = float(np.mean(vals)) if len(vals) else 0.0
    gcal.theta.update(init_theta)

    # Re-linearize between LS sweeps (noisy → more iterations)
    gcal.step_theta = 0.3
    gcal.step_bias = 0.3
    # A few theta-only outer passes to stabilize, then full passes
    gcal.fit_with_rebuild_theta_only(num_outer_iters=3, num_inner_iters=20)
    gcal.fit_with_rebuild(num_outer_iters=7, num_inner_iters=40)

    # Evaluate recovery up to translation
    est_theta = np.array([gcal.theta[hid] for hid in horse_ids], dtype=float)
    est_c = est_theta - np.median(est_theta)
    true_c = true_theta - np.median(true_theta)
    corr = float(np.corrcoef(true_c, est_c)[0, 1])
    mae = float(np.mean(np.abs(true_c - est_c)))
    print(f"Global ability correlation (centered): {corr:.6f}")
    print(f"Global ability MAE (centered):        {mae:.4f}")

    # Show per-race true vs noisy vs fitted
    total_l2 = 0.0
    print("\nPer-race probabilities (true vs noisy vs fitted):")
    for r in range(num_races):
        ids = race_horse_ids[r]
        p_true = race_prices_true[r]
        p_noisy = race_prices_noisy[r]
        p_fit = gcal.predict_race(r)
        l2 = float(np.linalg.norm(p_true - p_fit))
        total_l2 += l2
        print(f"\nRace {r}:")
        print(f"  horses: {ids}")
        print(f"  true:   {np.array2string(p_true,  precision=4, floatmode='fixed')}")
        print(f"  noisy:  {np.array2string(p_noisy, precision=4, floatmode='fixed')}")
        print(f"  fitted: {np.array2string(p_fit,   precision=4, floatmode='fixed')}")
        print(f"  L2(true,fitted): {l2:.6e}")
    print(f"\nAggregate L2(true,fitted) across races: {total_l2:.6e}")

    # Plot true vs inferred global locations (median-centered)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(true_c, est_c, s=22, alpha=0.8)
        lim = float(max(np.max(np.abs(true_c)), np.max(np.abs(est_c))))
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x")
        ax.set_title("Global calibration (noisy): true vs inferred loc (centered)")
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


