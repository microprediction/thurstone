"""
Example: Compare two global ability calibration methods on the same synthetic data:
  - GlobalAbilityCalibrator (curve-based Gauss–Newton with relinearization)
  - GlobalLSCalibrator (relative-then-weighted-average LS)

Run:
    python examples/global_calibration_compare.py
"""
import numpy as np
from numpy.random import default_rng

from thurstone import (
    UniformLattice,
    Density,
    AbilityCalibrator,
    GlobalAbilityCalibrator,
    GlobalLSCalibrator,
)


def softmax_noise(p: np.ndarray, rng, sigma: float = 0.15, eps: float = 1e-12) -> np.ndarray:
    """Add zero-mean Gaussian noise in log space and re-normalize by softmax."""
    logits = np.log(np.clip(p, eps, 1.0))
    noisy = logits + rng.normal(0.0, sigma, size=p.shape)
    ex = np.exp(noisy - np.max(noisy))
    return ex / np.sum(ex)


def build_synthetic(rng, num_horses=80, num_races=20, race_size=12, bias_range=0.2):
    # Base lattice and density
    lattice = UniformLattice(L=400, unit=0.1)
    base = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0)
    # Ground-truth global abilities
    horse_ids = [f"H{i}" for i in range(num_horses)]
    true_theta = np.linspace(-3.0, 3.0, num_horses)
    rng.shuffle(true_theta)
    # Race specs
    race_bias = rng.uniform(-bias_range, bias_range, size=num_races)
    calibrators = []
    race_horse_ids = []
    race_prices_true = []
    race_prices_noisy = []
    for r in range(num_races):
        idx = rng.choice(num_horses, size=race_size, replace=False)
        ids_r = [horse_ids[i] for i in idx]
        mu_r = [float(true_theta[i] + race_bias[r]) for i in idx]
        cal_r = AbilityCalibrator(base, n_iter=3)
        # Robust interpolation settings
        cal_r.offset_grid = list(range(-100, 101))
        cal_r.loc_span = 7.0
        cal_r.loc_step = 0.1
        p_true = np.array(cal_r.state_prices_from_ability(mu_r), dtype=float)
        # Ensure curves exist (for both methods)
        cal_r.solve_from_prices(p_true)
        p_noisy = softmax_noise(p_true, rng, sigma=0.15)
        calibrators.append(cal_r)
        race_horse_ids.append(ids_r)
        race_prices_true.append(p_true)
        race_prices_noisy.append(p_noisy)
    return (lattice, base, horse_ids, true_theta, calibrators, race_horse_ids, race_prices_true, race_prices_noisy)


def eval_centered(true_theta: np.ndarray, est_theta: np.ndarray):
    est_c = est_theta - np.median(est_theta)
    true_c = true_theta - np.median(true_theta)
    corr = float(np.corrcoef(true_c, est_c)[0, 1])
    mae = float(np.mean(np.abs(true_c - est_c)))
    return corr, mae, true_c, est_c


def main():
    rng = default_rng(2025)
    (
        lattice,
        base,
        horse_ids,
        true_theta,
        calibrators,
        race_horse_ids,
        race_prices_true,
        race_prices_noisy,
    ) = build_synthetic(rng)

    # Method A: Curve-based global fit with relinearization
    gfit = GlobalAbilityCalibrator(horse_ids=horse_ids)
    for r in range(len(calibrators)):
        gfit.add_race(calibrators[r], race_horse_ids[r], race_prices_noisy[r])
    gfit.step_theta = 0.25
    gfit.step_bias = 0.25
    gfit.fit_with_rebuild_theta_only(num_outer_iters=2, num_inner_iters=15)
    gfit.fit_with_rebuild(num_outer_iters=5, num_inner_iters=30)
    est_fit = np.array([gfit.theta[h] for h in horse_ids], dtype=float)
    corr_fit, mae_fit, true_c, est_fit_c = eval_centered(true_theta, est_fit)

    # Method B: Relative-then-LS (slope-weighted)
    gls = GlobalLSCalibrator(horse_ids=horse_ids)
    for r in range(len(calibrators)):
        gls.add_race(calibrators[r], race_horse_ids[r], race_prices_noisy[r])
    gls.fit(use_slope_weights=True, ridge=1e-8, weight_cap=1.0)
    est_ls = np.array([gls.theta[h] for h in horse_ids], dtype=float)
    corr_ls, mae_ls, _, est_ls_c = eval_centered(true_theta, est_ls)

    print("Global ability recovery (centered):")
    print(f"  Curve-based (GN):  corr={corr_fit:.4f}, MAE={mae_fit:.4f}")
    print(f"  Relative LS:       corr={corr_ls:.4f}, MAE={mae_ls:.4f}")

    # Optional plots
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        for ax, est_c, title in [
            (axes[0], est_fit_c, "Curve-based GN"),
            (axes[1], est_ls_c, "Relative LS"),
        ]:
            ax.scatter(true_c, est_c, s=18, alpha=0.8)
            lim = float(max(np.max(np.abs(true_c)), np.max(np.abs(est_c))))
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x")
            ax.set_title(title)
            ax.set_xlabel("true loc (centered)")
            ax.grid(alpha=0.25)
            ax.legend()
        axes[0].set_ylabel("inferred loc (centered)")
        fig.suptitle("Global abilities: true vs inferred (centered)")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()


