r"""
Kalman tracker demo (synthetic):

Pipeline
--------
1) Simulate a dynamic world of latent abilities and bookmaker pricing.
2) Invert prices per race to get centered ability observations.
3) Feed observations into a per-horse 1-D Kalman filter.
4) Run EM to estimate process and observation variances.
5) Smooth and plot trajectories for a few horses vs truth and bookmaker μ̂.
"""
from __future__ import annotations

import numpy as np
import math
from typing import List

from thurstone import UniformLattice, Density, KalmanAbilityTracker
from thurstone.sim_world import simulate_world, sigma_true


# Configuration
RNG_SEED: int = 7
NUM_HORSES: int = 20
NUM_RACES: int = 50
RACE_SIZE_RANGE: tuple[int, int] = (8, 14)
HORIZON_DAYS: float = 2000
ALPHA_RW: float = 0.03          # σ_true(dt) = sqrt(ALPHA_RW * dt) ⇒ q_true = ALPHA_RW
SIGMA0_INIT: float = 1.0
BOOKMAKER_TAU_REL: float = 0.5      # per-horse noise std
BOOKMAKER_TAU_BIAS: float = 1.0     # race-level bias std

LATTICE_L: int = 400
LATTICE_UNIT: float = 0.1
CAL_N_ITER: int = 3


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    # Simulate world
    races, true_theta, true_times, book_theta, book_times, sigma_gt = simulate_world(
        rng,
        n_horses=NUM_HORSES,
        n_races=NUM_RACES,
        race_size_range=RACE_SIZE_RANGE,
        horizon_days=HORIZON_DAYS,
        alpha=ALPHA_RW,
        sigma0=SIGMA0_INIT,
        bookmaker_rel_tau=BOOKMAKER_TAU_REL,
        bookmaker_bias_tau=BOOKMAKER_TAU_BIAS,
    )

    # Base density for inversion
    lattice = UniformLattice(L=LATTICE_L, unit=LATTICE_UNIT)
    base = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0)

    # Initialize Kalman tracker with reasonable priors
    tracker = KalmanAbilityTracker(
        base_density=base,
        ability_calibrator_kwargs=dict(n_iter=CAL_N_ITER),
        process_var_per_time=ALPHA_RW,      # q prior
        obs_var=BOOKMAKER_TAU_REL ** 2,         # r prior (relative noise)
        init_var=10.0,
    )

    # Feed races chronologically
    for race in races:
        tracker.update_race(race)

    # Learn q and r via EM, then (optionally) refit if desired
    # Re-estimate only the process variance q, holding the observation variance r fixed to the true BOOKMAKER_TAU^2
    tracker.fit_em(num_iters=10, fix_obs_var=BOOKMAKER_TAU_REL ** 2)

    # Visualization
    try:
        import matplotlib.pyplot as plt
        xs = np.linspace(0.0, 40.0, 200)
        plt.figure(figsize=(6, 4))
        plt.plot(xs, [sigma_gt(float(x)) for x in xs], label="σ_true(dt)")
        # For Kalman RW: σ(dt) = sqrt(q * dt)
        q_hat = tracker.process_var_per_time
        plt.plot(xs, [math.sqrt(max(q_hat * float(x), 1e-12)) for x in xs], label="σ_est(dt) from q̂")
        plt.xlabel("Δt")
        plt.ylabel("σ(Δt)")
        plt.title("Stickiness function: true vs estimated (Kalman)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot a few horses
        candidates = [h for h, v in true_times.items() if len(v) >= 2]
        rng_local = np.random.default_rng(123)
        sample: List[str] = candidates[:6] if len(candidates) >= 6 else candidates
        if len(sample) < 6 and len(candidates) > 6:
            sample = rng_local.choice(candidates, size=6, replace=False).tolist()
        if sample:
            nrows = int(np.ceil(len(sample) / 3))
            ncols = min(3, len(sample))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 2.8*nrows), squeeze=False, sharex=False)
            for k, hid in enumerate(sample):
                r = k // ncols
                c = k % ncols
                ax = axes[r][c]
                # true
                tt = true_times.get(hid, np.array([]))
                th = true_theta.get(hid, np.array([]))
                if tt.size > 0:
                    ax.plot(tt, th, '-o', label='true θ', alpha=0.9)
                # bookmaker μ̂
                bt = book_times.get(hid, np.array([]))
                bh = book_theta.get(hid, np.array([]))
                if bt.size > 0:
                    ax.plot(bt, bh, 'x-', label='book μ̂', alpha=0.7)
                # Kalman smoothed, zero-mean anchored across horses per time
                anchored = tracker.smooth_horses_zero_mean()
                if hid in anchored:
                    st, smu = anchored[hid]
                    ax.plot(st, smu, '-.', label='Kalman θ̂ (smoothed, zero-mean)', alpha=0.9)
                ax.set_title(hid)
                ax.set_xlabel('time')
                ax.set_ylabel('ability')
                ax.grid(True, alpha=0.2)
                ax.legend(fontsize=8)
            plt.tight_layout()
            plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()


