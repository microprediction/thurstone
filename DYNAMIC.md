## Dynamic calibration in Thurstone

This note explains the dynamic ability model, how bookmaker uncertainty is modeled, how we invert prices to abilities race‑by‑race, how we learn the stickiness function σ(Δt), and how we smooth trajectories.

### Overview
- We observe races over time. Each race provides risk‑neutral winning probabilities (state prices) for the entered horses.
- Abilities are latent and evolve slowly. Prices are produced by a forward Thurstone model given abilities.
- We invert prices to obtain per‑race ability estimates, then stitch them into per‑horse time series and learn the time‑evolution noise scale σ(Δt). Finally, we smooth each trajectory with a random‑walk prior.

Relevant code:
- `thurstone/dynamic.py`: `RaceObservation`, `DynamicThurstoneCalibrator` and helpers.
- `examples/dynamic_calibration_demo.py`: synthetic demo using the full pipeline.

## Generative story (conceptual)

Let θ_h(t) denote the latent ability of horse h at time t.

- Random‑walk prior:
  - For two race times t1 < t2 on the same horse, define Δt = t2 − t1 and
    - θ_h(t2) = θ_h(t1) + ε, with ε ∼ N(0, σ_true(Δt)^2).
- Race pricing:
  - Given a vector of abilities θ for the entrants, a fixed base lattice density generates performances; the winner density and multiplicity yield risk‑neutral winning probabilities.
- Bookmaker uncertainty (predictive pricing):
  - The bookmaker does not know θ exactly. Instead, they have uncertainty on θ with std τ (in ability units).
  - They price using a predictive density obtained by convolving the base performance density with a zero‑mean Gaussian over ability offsets of std `bookmaker_sigma = τ`.

In the demo we simulate this world, producing “observed” prices that are consistent with predictive pricing.

## Inversion: prices → per‑race abilities

`DynamicThurstoneCalibrator` inverts prices for each race using `AbilityCalibrator`. Two options:
- If `bookmaker_sigma == 0`, inversion uses the original base density.
- If `bookmaker_sigma > 0`, inversion uses a predictive base density constructed as a mixture of shifted base densities with discrete Gaussian weights in ability space (±4σ window). This matches the bookmaker’s predictive pricing story and prevents systematic bias in inversion.

API:
- Set `bookmaker_sigma` on `DynamicThurstoneCalibrator` to the assumed bookmaker ability noise.
- `_predictive_base_density()` builds the predictive density.
- `_new_calibrator()` constructs an `AbilityCalibrator` using the predictive density.
- `_initial_abilities_from_prices()` runs the inversion race‑by‑race and aggregates time‑aligned per‑horse series.

## Learning σ(Δt)

We use the differences between successive ability estimates for each horse to learn how ability variance accumulates with elapsed time.

Let Δθ = θ_j − θ_{j−1} across consecutive observations for the same horse at gaps Δt_j.

Two estimators are supported:

### 1) Parametric estimator (recommended)
We fit

\[ \mathrm{Var}(Δθ) \approx 2\,τ^2 + α\,Δt. \]

- Here τ^2 is the measurement variance in ability space (per observation), and α controls the diffusion (random‑walk) scale.
- Fit by least‑squares on (Δt_j, (Δθ_j)^2), recovering:
  - τ̂^2 = max(intercept/2, 0)
  - α̂ = max(slope, 0)
- The stickiness function is then

\[ \hat{σ}(Δt) = \sqrt{\max(α̂\,Δt, \varepsilon)}. \]

Advantages:
- Removes bias when no truly small Δt samples exist.
- Smooth, monotone σ̂(Δt) by construction.

Code: `fit_sigma_parametric()` returns `(sigma_fn, alpha_hat, meas_var_hat)`.

### 2) Binned (piecewise constant) estimator
- Bin Δt by quantiles, compute mean of (Δθ)^2 per bin, subtract `2*meas_var` (known or τ̂^2 from the parametric fit), and take square roots.
- Optionally enforce monotonicity across bins.

Code: `fit_sigma(n_bins, min_points, monotone, meas_var)` returns a piecewise constant σ̂.

Tip: A good workflow is to estimate τ̂^2 with the parametric fit, then optionally refine a binned σ̂ using that τ̂^2.

## Smoothing trajectories

Given raw abilities m_j at times t_j (per horse) and a stickiness function σ(Δt), we solve the MAP estimate of a random‑walk model:

minimize over θ_j:

\[ \sum_j \frac{(θ_j - m_j)^2}{\mathrm{obs\_var}} \;+\; \sum_{j>1} \frac{(θ_j - θ_{j-1})^2}{σ(Δt_j)^2}. \]

This yields a symmetric tridiagonal system A θ = b solved in closed form. The `obs_var` should reflect per‑race measurement noise; in practice use the τ̂^2 estimated above.

Code: `_smooth_trajectory()` implements the solver; `fit_abilities(sigma_function, obs_var)` applies it per horse.

## Winner‑informed refinement (optional)

`fit_abilities_with_results(refine_steps, step, eps)` optionally nudges per‑race ability vectors in the direction that increases the log probability of the observed winner (finite‑difference gradient, re‑centered after each step). This can help disambiguate tight races but is not required for σ(Δt) learning.

## Practical guidance

- Predictive inversion: If you know or assume the bookmaker “adds variance,” set `bookmaker_sigma > 0`. This aligns inversion with pricing and reduces bias.
- Parametric σ̂(Δt): Prefer `fit_sigma_parametric()`. It is robust to a lack of tiny Δt samples and produces smooth σ̂.
- Observation variance: Use `obs_var = τ̂^2` (from parametric fit) when smoothing.
- Small‑Δt behavior: σ̂(0) = 0 under the parametric model; avoid evaluating exactly at 0 in plots or RMSE. Consider starting from a small `dt_min`.
- Data coverage: Increase the number of races or horses, or ensure varied intervals, to stabilize both τ̂ and α̂.

## Example workflow (see `examples/dynamic_calibration_demo.py`)

1. Simulate a schedule and dynamic abilities; generate observed prices via predictive pricing with known τ.
2. Construct `DynamicThurstoneCalibrator` with `bookmaker_sigma=τ`.
3. Invert prices to get per‑race abilities; optionally refine with results.
4. Fit σ̂(Δt) using `fit_sigma_parametric()`; record τ̂ and α̂.
5. Smooth per‑horse trajectories using `fit_abilities(sigma_function=σ̂, obs_var=τ̂^2)`.
6. Evaluate: correlation between centered smoothed abilities and ground truth, and RMSE between σ̂ and σ_true over a reasonable Δt range.

## Why this works

- The inversion step provides noisy, per‑race estimates of abilities that reflect both process noise (true evolution) and measurement noise (bookmaker/pricing uncertainty).
- The variance decomposition Var(Δθ) = 2τ^2 + αΔt separates these effects by exploiting the linear growth of random‑walk variance with elapsed time.
- Smoothing then fuses per‑race estimates into coherent trajectories by penalizing large changes unless supported by σ̂(Δt).

## References and related components

- Base lattice and pricing rely on the same machinery as single‑race calibration (winner density, multiplicity, etc.).
- `AbilityCalibrator` provides forward pricing and interpolation‑based inverse calibration consistent with the forward model.


