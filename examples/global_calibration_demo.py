"""
Example: Global ability calibration across multiple races using cached lookup curves.

- 20 horses with true global abilities (locs) in [-3, 3]
- 5 races, each with 8 randomly chosen horses
- Each race has a small bias added to all locs
- We generate prices from the forward model, then recover global abilities
  using GlobalAbilityCalibrator over all races simultaneously.

Run:
    python examples/global_calibration_demo.py
"""
import numpy as np
from numpy.random import default_rng

from thurstone import UniformLattice, Density
from thurstone.inference import AbilityCalibrator
from thurstone.global_fit import GlobalAbilityCalibrator


def main():
    rng = default_rng(42)
    # Base lattice and density
    lattice = UniformLattice(L=400, unit=0.1)
    base = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0)

    # Global ground truth
    num_horses = 20
    horse_ids = [f"H{i}" for i in range(num_horses)]
    true_theta = np.linspace(-3.0, 3.0, num_horses)
    rng.shuffle(true_theta)

    # Build races
    num_races = 5
    race_size = 8
    race_bias = rng.uniform(-0.2, 0.2, size=num_races)

    # Create per-race calibrators and generate prices
    cal_per_race = []
    race_horse_ids = []
    race_prices = []
    for r in range(num_races):
        # Pick subset
        idx = rng.choice(num_horses, size=race_size, replace=False)
        ids_r = [horse_ids[i] for i in idx]
        mu_r = [float(true_theta[i] + race_bias[r]) for i in idx]
        cal_r = AbilityCalibrator(base, n_iter=3)
        # Make the implicit curve steeper and better bracketed
        cal_r.offset_grid = list(range(-80, 81))
        cal_r.loc_span = 6.0
        cal_r.loc_step = 0.1
        prices_r = np.array(cal_r.state_prices_from_ability(mu_r), dtype=float)
        # Populate lookup curves for this race (cheap)
        cal_r.solve_from_prices(prices_r)
        cal_per_race.append(cal_r)
        race_horse_ids.append(ids_r)
        race_prices.append(prices_r)

    # Global fit
    gcal = GlobalAbilityCalibrator(horse_ids=horse_ids)
    for r in range(num_races):
        gcal.add_race(cal_per_race[r], race_horse_ids[r], race_prices[r])

    # Gauss–Newton-style: rebuild curves around current global params between LS sweeps
    gcal.fit_with_rebuild(num_outer_iters=5, num_inner_iters=20)

    # Evaluate recovery up to translation
    est_theta = np.array([gcal.theta[hid] for hid in horse_ids], dtype=float)
    est_c = est_theta - np.median(est_theta)
    true_c = true_theta - np.median(true_theta)
    corr = float(np.corrcoef(true_c, est_c)[0, 1])
    mae = float(np.mean(np.abs(true_c - est_c)))
    print(f"Global ability correlation (centered): {corr:.6f}")
    print(f"Global ability MAE (centered):        {mae:.4f}")

    # Verify race reconstructions and display true vs fitted per race
    total_l2 = 0.0
    print("\nPer-race probabilities (true vs fitted):")
    for r in range(num_races):
        ids = race_horse_ids[r]
        p_true = race_prices[r]
        p_fit = gcal.predict_race(r)
        l1 = float(np.sum(np.abs(p_true - p_fit)))
        l2 = float(np.linalg.norm(p_true - p_fit))
        total_l2 += l2
        print(f"\nRace {r}:")
        print(f"  horses: {ids}")
        print(f"  true:   {np.array2string(p_true, precision=4, floatmode='fixed')}")
        print(f"  fitted: {np.array2string(p_fit, precision=4, floatmode='fixed')}")
        print(f"  L1: {l1:.6e}  L2: {l2:.6e}")
    print(f"\nAggregate L2 error across races: {total_l2:.6e}")

    # Simple success heuristic for this demo
    if corr > 0.99 and mae < 0.15:
        print("Success: recovered global abilities match the truth (up to translation).")
    else:
        print("Note: correlation below target; increase num_iters or adjust lattice/grid if needed.")


if __name__ == "__main__":
    main()


