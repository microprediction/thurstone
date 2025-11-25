r"""
Dynamic calibration demo (synthetic):

Generative story
----------------
- There are H horses with latent abilities μ_h(t).
- Each horse follows a random‑walk in continuous time:
      μ_h(t2) = μ_h(t1) + ε,   ε ~ N(0, σ_true(Δt)^2),  Δt = t2 - t1
- Races occur at times t_r with random subsets of horses.
- Bookmakers form noisy ability estimates:  mu_hat_h = μ_h + η_h,  η_h ~ N(0, τ^2)
- Posted (observed) prices are state prices from mu_hat:  p_obs = state_prices_from_ability(mu_hat_r)

Estimation
----------
1) Use DynamicThurstoneCalibrator.fit_abilities(sigma_function=None) to obtain per‑race static
   abilities from observed prices (via inverse).
2) Learn a piecewise‑constant σ(Δt) from increments of these static abilities (fit_sigma).
3) Smooth each horse’s trajectory with a random‑walk prior using the learned σ(Δt).
4) Evaluate recovered dynamic abilities against ground truth; compare learned σ(Δt) with σ_true(Δt).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional

import numpy as np

from thurstone import UniformLattice, Density
from thurstone.inference import AbilityCalibrator
from thurstone.dynamic import RaceObservation, DynamicThurstoneCalibrator

# ------------------------------------------------------------------
# Configuration – central place for key parameters
# ------------------------------------------------------------------
RNG_SEED: int = 7
NUM_HORSES: int = 20
NUM_RACES: int = 50
RACE_SIZE_RANGE: tuple[int, int] = (8, 14)
HORIZON_DAYS: float = 2000
ALPHA_RW: float = 0.03            # σ_true(dt) = sqrt(ALPHA_RW * dt)
SIGMA0_INIT: float = 1.0          # initial ability std
BOOKMAKER_TAU: float = 0.1       # bookmaker ability noise (in ability space)

# Forward model (fixed emission)
LATTICE_L: int = 400
LATTICE_UNIT: float = 0.1
CAL_N_ITER: int = 3

# Sigma learning and smoothing
SIGMA_N_BINS: int = 10
SIGMA_MIN_POINTS: int = 50
SIGMA_MONOTONE: bool = True
OBS_VAR: float = BOOKMAKER_TAU ** 2
SMOOTH_SIGMA_SCALE: float = 1.4   # >1 to reduce over-smoothing by increasing σ(Δt)


def sigma_true(dt: float, alpha: float = 0.04) -> float:
    """Ground‑truth stickiness: random‑walk std scales like sqrt(alpha * dt)."""
    return math.sqrt(max(alpha * max(dt, 0.0), 1e-12))


def simulate_schedule(
    rng: np.random.Generator,
    n_horses: int,
    n_races: int,
    race_size_range: Tuple[int, int],
    horizon_days: float,
) -> Tuple[List[float], List[List[int]]]:
    """Return sorted race times and a list of participant index sets per race."""
    times = np.sort(rng.uniform(0.0, horizon_days, size=n_races)).tolist()
    fields: List[List[int]] = []
    lo, hi = race_size_range
    for _ in range(n_races):
        m = int(rng.integers(low=lo, high=hi + 1))
        m = max(2, min(m, n_horses))
        field = rng.choice(n_horses, size=m, replace=False).tolist()
        fields.append(field)
    return times, fields


def simulate_world(
    rng: np.random.Generator,
    n_horses: int = 60,
    n_races: int = 90,
    race_size_range: Tuple[int, int] = (8, 14),
    horizon_days: float = 240.0,
    alpha: float = 0.04,          # RW diffusion scale in σ_true
    sigma0: float = 0.8,          # initial ability std
    bookmaker_tau: float = 0.5,   # bookmaker ability noise std (per race)
) -> Tuple[
    List[RaceObservation],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    callable,
]:
    """
    Simulate a dataset:
      - returns races with observed (bookmaker) prices,
      - true per‑horse abilities and times at their races,
      - reference to sigma_true(dt).
    """
    times, fields = simulate_schedule(
        rng, n_horses=n_horses, n_races=n_races,
        race_size_range=race_size_range, horizon_days=horizon_days
    )
    horse_ids = [f"H{i}" for i in range(n_horses)]

    # Build per‑horse race index lists
    idx_per_horse: Dict[int, List[int]] = {h: [] for h in range(n_horses)}
    for r_i, fld in enumerate(fields):
        for h in fld:
            idx_per_horse[h].append(r_i)
    for h in range(n_horses):
        idx_per_horse[h].sort(key=lambda r_i: times[r_i])

    # Simulate true abilities μ at race times
    mu_at_race: Dict[Tuple[int, int], float] = {}
    true_theta: Dict[str, List[float]] = {hid: [] for hid in horse_ids}
    true_times: Dict[str, List[float]] = {hid: [] for hid in horse_ids}
    for h in range(n_horses):
        mu = float(rng.normal(0.0, sigma0))
        prev_t: Optional[float] = None
        for r_i in idx_per_horse[h]:
            t = times[r_i]
            if prev_t is not None:
                dt = t - prev_t
                mu += float(rng.normal(0.0, sigma_true(dt, alpha=alpha)))
            prev_t = t
            mu_at_race[(h, r_i)] = mu
            hid = horse_ids[h]
            true_theta[hid].append(mu)
            true_times[hid].append(t)

    # Forward model (also used by bookmaker)
    lattice = UniformLattice(L=400, unit=0.1)
    base = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0)
    forward = AbilityCalibrator(base, n_iter=3)

    # Build observed bookmaker prices from noisy abilities:
    races: List[RaceObservation] = []
    for r_i, fld in enumerate(fields):
        ids = [horse_ids[h] for h in fld]
        mu_true = np.array([mu_at_race[(h, r_i)] for h in fld], dtype=float)
        # true probabilities (fair)
        p_true = np.asarray(forward.state_prices_from_ability(mu_true.tolist()), dtype=float)
        # bookmaker noisy ability view (Gaussian in ability space), then price
        mu_hat = mu_true + rng.normal(0.0, bookmaker_tau, size=len(fld))
        p_obs = np.asarray(forward.state_prices_from_ability(mu_hat.tolist()), dtype=float)
        winner_idx = int(rng.choice(len(fld), p=p_true / max(np.sum(p_true), 1e-12)))
        winner_id = ids[winner_idx]
        races.append(
            RaceObservation(
                race_id=f"R{r_i}",
                time=times[r_i],
                horse_ids=ids,
                prices=p_obs,
                winner=winner_id,
            )
        )

    true_theta_np = {h: np.asarray(v, dtype=float) for h, v in true_theta.items()}
    true_times_np = {h: np.asarray(v, dtype=float) for h, v in true_times.items()}
    return races, true_theta_np, true_times_np, (lambda dt: sigma_true(dt, alpha=alpha))


def evaluate_correlation_centered(
    est: Dict[str, np.ndarray],
    est_times: Dict[str, np.ndarray],
    tru: Dict[str, np.ndarray],
    tru_times: Dict[str, np.ndarray],
) -> float:
    """Compute global correlation on concatenated per‑horse median‑centered series."""
    est_all: List[float] = []
    tru_all: List[float] = []
    for h, e in est.items():
        if h not in tru:
            continue
        te = est_times.get(h, np.array([]))
        tt = tru_times.get(h, np.array([]))
        if len(te) == 0 or len(tt) == 0:
            continue
        # align by index (both sequences are time‑ordered)
        L = min(len(e), len(tru[h]))
        if L < 2:
            continue
        ee = e[:L] - np.median(e[:L])
        ttv = tru[h][:L] - np.median(tru[h][:L])
        est_all.append(ee)
        tru_all.append(ttv)
    if not est_all:
        return float("nan")
    E = np.concatenate(est_all)
    T = np.concatenate(tru_all)
    if E.size < 2:
        return float("nan")
    return float(np.corrcoef(T, E)[0, 1])


def evaluate_sigma_rmse(sigma_est, sigma_gt, dt_min: float = 0.5, dt_max: float = 40.0, n: int = 80) -> float:
    xs = np.linspace(dt_min, dt_max, n)
    se = np.array([sigma_est(float(x)) for x in xs], dtype=float)
    sg = np.array([sigma_gt(float(x)) for x in xs], dtype=float)
    return float(np.sqrt(np.mean((se - sg) ** 2)))


def estimate_book_factor_band(
    dyn: DynamicThurstoneCalibrator,
    races: List[RaceObservation],
    base: Density,
    use_smoothed: bool = True,
    pmin: float = 0.02,
    pmax: float = 0.98,
) -> Tuple[float, float, float]:
    """
    Infer bookmaker multiplicative error band from market vs model fair probs.
    Uses dyn.theta_/times_ to build per-race abilities, then compares:
        factor = m_i / q_i
    Returns (q10, median, q90) of factor over all horses/races.
    """
    cal = AbilityCalibrator(base, n_iter=3)
    factors: List[float] = []
    # Build lookup from horse -> {time -> ability}
    for race in races:
        # assemble ability vector for this race from dyn.theta_ at this race time
        a_vec: List[float] = []
        for h in race.horse_ids:
            t_arr = dyn.times_.get(h)
            th_arr = dyn.theta_.get(h)
            if t_arr is None or th_arr is None:
                a_vec.append(0.0)
                continue
            # find exact time match (times are from schedule)
            idx = int(np.where(t_arr == race.time)[0][0])
            a_vec.append(float(th_arr[idx]))
        q_model = np.asarray(cal.state_prices_from_ability(a_vec), dtype=float)
        m_obs = np.asarray(race.prices, dtype=float)
        # use odds space for stability: u = odds(m)/odds(q) = exp(logit(m)-logit(q))
        # also restrict to mid-probability range to avoid tail blow-ups
        valid = (
            (q_model > pmin) & (q_model < pmax) &
            (m_obs > pmin) & (m_obs < pmax) &
            np.isfinite(q_model) & np.isfinite(m_obs)
        )
        if np.any(valid):
            logit_m = np.log(m_obs[valid]) - np.log1p(-m_obs[valid])
            logit_q = np.log(q_model[valid]) - np.log1p(-q_model[valid])
            u = np.exp(logit_m - logit_q)  # multiplicative factor on odds
            # Keep only finite, positive
            u = u[np.isfinite(u) & (u > 0)]
            factors.extend(u.tolist())
    if not factors:
        return 1.0, 1.0, 1.0
    f_arr = np.asarray(factors, dtype=float)
    q10 = float(np.quantile(f_arr, 0.10))
    med = float(np.median(f_arr))
    q90 = float(np.quantile(f_arr, 0.90))
    return q10, med, q90


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    # --- simulate world
    races, true_theta, true_times, sigma_gt = simulate_world(
        rng,
        n_horses=NUM_HORSES,
        n_races=NUM_RACES,
        race_size_range=RACE_SIZE_RANGE,
        horizon_days=HORIZON_DAYS,
        alpha=ALPHA_RW,
        sigma0=SIGMA0_INIT,
        bookmaker_tau=BOOKMAKER_TAU,
    )

    # base model for inference
    lattice = UniformLattice(L=LATTICE_L, unit=LATTICE_UNIT)
    base = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0)

    dyn = DynamicThurstoneCalibrator(
        base_density=base,
        races=races,
        ability_calibrator_kwargs=dict(n_iter=CAL_N_ITER),
        bookmaker_sigma=BOOKMAKER_TAU,
    )

    # 0) Bookmaker-only baseline: invert prices without any result information
    dyn_book = DynamicThurstoneCalibrator(
        base_density=base,
        races=races,
        ability_calibrator_kwargs=dict(n_iter=CAL_N_ITER),
        bookmaker_sigma=BOOKMAKER_TAU,
    )
    dyn_book.fit_abilities(sigma_function=None)
    r_book = evaluate_correlation_centered(dyn_book.theta_, dyn_book.times_, true_theta, true_times)
    print(f"[Book]    correlation (centered): {r_book:.4f}")

    # 1) Static per‑race inverse from (noisy) bookmaker prices, then refine with results
    dyn.fit_abilities_with_results(refine_steps=1, refine_step_size=0.6, refine_eps=0.05)
    r_raw = evaluate_correlation_centered(dyn.theta_, dyn.times_, true_theta, true_times)
    print(f"[Raw+Res] correlation (centered): {r_raw:.4f}")

    # 2) Learn σ(Δt) from current trajectories using a parametric model:
    #    Var(Δθ) ≈ 2*τ^2 + α*Δt  ⇒  τ̂^2 = intercept/2,  α̂ = slope
    sigma_est, alpha_hat, meas_var_hat = dyn.fit_sigma_parametric(
        meas_var=None,
        dt_min=0.0,
        dt_max=40.0,
    )
    print(f"[Sigma]   tau_hat (parametric): {math.sqrt(max(meas_var_hat,0.0)):.4f}")
    # Optional: compare to a binned estimator using τ̂^2 from the parametric fit
    # sigma_est, _ = dyn.fit_sigma(n_bins=SIGMA_N_BINS, min_points=SIGMA_MIN_POINTS,
    #                              monotone=SIGMA_MONOTONE, meas_var=meas_var_hat), meas_var_hat
    rmse_sigma = evaluate_sigma_rmse(sigma_est, sigma_gt, dt_min=0.5, dt_max=40.0, n=80)
    print(f"[Sigma]   RMSE vs true σ(dt):    {rmse_sigma:.4f}")

    # 3) Smooth abilities using learned σ(Δt)
    # Use inferred measurement variance for observation noise and scale σ(Δt) to avoid over-smoothing.
    def sigma_scaled(dt: float) -> float:
        return float(SMOOTH_SIGMA_SCALE) * float(sigma_est(float(dt)))
    dyn.fit_abilities(sigma_function=sigma_scaled, obs_var=float(meas_var_hat))
    r_smooth = evaluate_correlation_centered(dyn.theta_, dyn.times_, true_theta, true_times)
    print(f"[Smooth]  correlation (centered): {r_smooth:.4f}")
    # 4) Infer bookmaker error factor band:
    #    - from book-only abilities (pure market comparison)
    #    - from smoothed abilities (after using σ̂ and results)
    q10b, medb, q90b = estimate_book_factor_band(dyn_book, races, base)
    print(f"[BookErr/book]   factor band (10/50/90%): [{q10b:.2f}, {medb:.2f}, {q90b:.2f}]  (target ≈ [0.8, 1.3])")
    q10s, meds, q90s = estimate_book_factor_band(dyn, races, base)
    print(f"[BookErr/smooth] factor band (10/50/90%): [{q10s:.2f}, {meds:.2f}, {q90s:.2f}]")

    # Optional visualization
    try:
        import matplotlib.pyplot as plt

        xs = np.linspace(0.0, 40.0, 200)
        plt.figure(figsize=(6, 4))
        plt.plot(xs, [sigma_gt(float(x)) for x in xs], label="σ_true(dt)")
        plt.plot(xs, [sigma_est(float(x)) for x in xs], label="σ_est(dt)")
        plt.xlabel("Δt")
        plt.ylabel("σ(Δt)")
        plt.title("Stickiness function: true vs estimated")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()


