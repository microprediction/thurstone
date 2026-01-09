use crate::density::Density;
use ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

// --- Public containers --------------------------------------------------------

/// Minimal container for one race observation.
/// - race_id: opaque identifier for the race
/// - time: numeric time (e.g., days since epoch). Must be comparable across races
/// - horse_ids: list of horse identifiers in post order (only used for alignment)
/// - prices: risk-neutral winning probabilities for the same order of horse_ids
/// - winner: optional winner id (future extensions may use ranks/margins)
#[pyclass]
#[derive(Clone)]
pub struct RaceObservation {
    #[pyo3(get, set)]
    pub race_id: String,

    #[pyo3(get, set)]
    pub time: f64,

    #[pyo3(get, set)]
    pub horse_ids: Vec<String>,

    #[pyo3(get, set)]
    pub prices: Vec<f64>,

    #[pyo3(get, set)]
    pub winner: Option<String>,
}

#[pymethods]
impl RaceObservation {
    #[new]
    #[pyo3(signature = (race_id, time, horse_ids, prices, winner=None))]
    fn new(
        race_id: String,
        time: f64,
        horse_ids: Vec<String>,
        prices: Vec<f64>,
        winner: Option<String>,
    ) -> Self {
        RaceObservation {
            race_id,
            time,
            horse_ids,
            prices,
            winner,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RaceObservation(race_id='{}', time={}, n_horses={})",
            self.race_id,
            self.time,
            self.horse_ids.len()
        )
    }
}

// --- Sigma Function Classes ---------------------------------------------------

/// Callable object that computes sigma(dt) = sqrt(max(alpha * dt, 1e-12))
#[pyclass]
#[derive(Clone)]
pub struct ParametricSigmaFn {
    alpha: f64,
}

#[pymethods]
impl ParametricSigmaFn {
    fn __call__(&self, dt: f64) -> f64 {
        (self.alpha * dt).max(1e-12).sqrt()
    }

    fn __repr__(&self) -> String {
        format!("ParametricSigmaFn(alpha={})", self.alpha)
    }
}

/// Callable object that computes sigma(dt) using piecewise-constant bins
#[pyclass]
#[derive(Clone)]
pub struct PiecewiseSigmaFn {
    edges: Vec<f64>,
    vals: Vec<f64>,
}

#[pymethods]
impl PiecewiseSigmaFn {
    fn __call__(&self, dt: f64) -> f64 {
        // Binary search for the correct bin
        let mut k = 0;
        for (i, &edge) in self.edges.iter().enumerate().skip(1) {
            if dt < edge {
                k = i - 1;
                break;
            }
        }
        if dt >= *self.edges.last().unwrap() {
            k = self.vals.len() - 1;
        }
        k = k.min(self.vals.len() - 1);
        self.vals[k]
    }

    fn __repr__(&self) -> String {
        format!("PiecewiseSigmaFn(n_bins={})", self.vals.len())
    }
}

// --- Dynamic Thurstone Calibrator ---------------------------------------------

/// Dynamic Thurstone-style calibrator on top of the static AbilityCalibrator.
///
/// Pipeline:
///   1) For each race, run AbilityCalibrator.solve_from_prices(prices) to get
///      "raw" per-race abilities for the entrants.
///   2) Assemble per-horse trajectories (times and abilities) across races.
///   3) Optionally smooth each trajectory with a random-walk prior whose
///      increment std is σ(Δt).
///   4) Optionally estimate σ(Δt) from observed ability increments.
///
/// If bookmaker_sigma > 0, all price→ability inversions are performed using a
/// predictive base density obtained by convolving the base density with a
/// zero-mean Gaussian over ability offsets (in ability units). This models a
/// bookmaker adding their own uncertainty before pricing.
#[pyclass]
pub struct DynamicThurstoneCalibrator {
    #[pyo3(get)]
    pub base_density: Py<Density>,

    #[pyo3(get)]
    pub races: Vec<Py<RaceObservation>>,

    #[pyo3(get)]
    pub ability_calibrator_kwargs: Py<PyDict>,

    #[pyo3(get, set)]
    pub bookmaker_sigma: f64,

    // learned / produced attributes
    theta_: HashMap<String, Array1<f64>>,
    times_: HashMap<String, Array1<f64>>,

    // piecewise sigma(Δt) if learned via fit_sigma
    sigma_edges_: Option<Array1<f64>>,
    sigma_vals_: Option<Array1<f64>>,
}

#[pymethods]
impl DynamicThurstoneCalibrator {
    #[new]
    #[pyo3(signature = (base_density, races, ability_calibrator_kwargs=None, bookmaker_sigma=0.0))]
    fn new(
        py: Python<'_>,
        base_density: Py<Density>,
        races: Vec<Py<RaceObservation>>,
        ability_calibrator_kwargs: Option<Py<PyDict>>,
        bookmaker_sigma: f64,
    ) -> PyResult<Self> {
        let kwargs = if let Some(k) = ability_calibrator_kwargs {
            k
        } else {
            PyDict::new(py).into()
        };

        // Sort races by time
        let mut sorted_races = races;
        sorted_races.sort_by(|a, b| {
            let a_time = a.borrow(py).time;
            let b_time = b.borrow(py).time;
            a_time.partial_cmp(&b_time).unwrap()
        });

        Ok(DynamicThurstoneCalibrator {
            base_density,
            races: sorted_races,
            ability_calibrator_kwargs: kwargs,
            bookmaker_sigma,
            theta_: HashMap::new(),
            times_: HashMap::new(),
            sigma_edges_: None,
            sigma_vals_: None,
        })
    }

    // --- bookmaker predictive density -----------------------------------------

    /// Density used when inverting prices. If bookmaker_sigma == 0, returns the
    /// original base density. Otherwise returns a mixture of shifted base
    /// densities with Gaussian weights over ability offsets.
    fn _predictive_base_density(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.bookmaker_sigma <= 0.0 {
            return Ok(self.base_density.clone_ref(py).into());
        }

        let base_ref = self.base_density.borrow(py);
        let unit = base_ref.lattice.borrow(py).unit_value();

        // express bookmaker std in lattice steps
        let sigma_steps = self.bookmaker_sigma / unit.max(1e-12);
        let max_steps = (4.0 * sigma_steps).ceil() as i32;

        if max_steps <= 0 {
            return Ok(self.base_density.clone_ref(py).into());
        }

        let offsets_steps: Vec<f64> = (-max_steps..=max_steps).map(|i| i as f64).collect();

        // Gaussian over ability offsets (ability units)
        let offsets_ability: Vec<f64> = offsets_steps.iter().map(|&o| o * unit).collect();
        let mut w: Vec<f64> = offsets_ability
            .iter()
            .map(|&oa| (-0.5 * (oa / self.bookmaker_sigma).powi(2)).exp())
            .collect();

        let w_sum: f64 = w.iter().sum();
        if w_sum <= 0.0 {
            return Ok(self.base_density.clone_ref(py).into());
        }

        w.iter_mut().for_each(|wi| *wi /= w_sum);

        let base_p_len = base_ref.p.len();
        drop(base_ref);

        let mut pdf_pred = Array1::<f64>::zeros(base_p_len);

        for (o_steps, weight) in offsets_steps.iter().zip(w.iter()) {
            if *weight == 0.0 {
                continue;
            }
            let shifted = self
                .base_density
                .borrow(py)
                .shift_fractional(py, *o_steps)?;
            let shifted_p = &shifted.p;
            pdf_pred = pdf_pred + *weight * shifted_p;
        }

        let pdf_pred_py = pdf_pred.to_pyarray(py);
        let lattice = self.base_density.borrow(py).lattice.clone_ref(py);

        // Create Density by calling the Python constructor
        let density_class = py.import("thurstone.density")?.getattr("Density")?;
        let pred_density = density_class.call1((lattice, pdf_pred_py))?;
        Ok(pred_density.unbind())
    }

    /// Factory for AbilityCalibrator using the predictive base density.
    fn _new_calibrator(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let pred_base = self._predictive_base_density(py)?;

        // Create AbilityCalibrator using Python constructor
        let calibrator_class = py
            .import("thurstone.inference")?
            .getattr("AbilityCalibrator")?;

        // Build kwargs
        let kwargs = self.ability_calibrator_kwargs.bind(py);

        // Call constructor with base density and kwargs
        let calibrator = calibrator_class.call((pred_base,), Some(&kwargs))?;

        Ok(calibrator.unbind())
    }

    // --- indexing -------------------------------------------------------------

    /// Map horse_id -> sorted list of race indices it appears in.
    fn _build_horse_index(&self, py: Python<'_>) -> HashMap<String, Vec<usize>> {
        let mut idx: HashMap<String, Vec<usize>> = HashMap::new();

        for (r_i, race_py) in self.races.iter().enumerate() {
            let race = race_py.borrow(py);
            for h in &race.horse_ids {
                idx.entry(h.clone()).or_insert_with(Vec::new).push(r_i);
            }
        }

        // Sort each horse's race indices by time
        for (_h, ndxs) in idx.iter_mut() {
            ndxs.sort_by(|&i, &j| {
                let ti = self.races[i].borrow(py).time;
                let tj = self.races[j].borrow(py).time;
                ti.partial_cmp(&tj).unwrap()
            });
        }

        idx
    }

    // --- initial per-race abilities ------------------------------------------

    /// Use AbilityCalibrator race-by-race to get raw abilities (no smoothing).
    ///
    /// Returns
    /// -------
    /// theta_raw : dict[horse_id] -> np.ndarray (time-ordered abilities)
    /// times     : dict[horse_id] -> np.ndarray (aligned times)
    fn _initial_abilities_from_prices<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        HashMap<String, Bound<'py, PyArray1<f64>>>,
        HashMap<String, Bound<'py, PyArray1<f64>>>,
    )> {
        // return todo!();
        let horse_index = self._build_horse_index(py);

        // per-race ability vectors (aligned to race.horse_ids)
        let mut race_abilities: Vec<HashMap<String, f64>> = Vec::new();

        for race_py in &self.races {
            let race = race_py.borrow(py);
            let cal = self._new_calibrator(py)?;

            // race.prices should be risk-neutral winning probabilities
            let prices_py = race.prices.clone();
            let ability_arr = cal
                .bind(py)
                .call_method1("solve_from_prices", (prices_py,))?;
            let ability_np: &Bound<'_, PyArray1<f64>> = ability_arr.downcast()?;
            let mut ability_vec = ability_np.readonly().as_array().to_owned();

            // Center per-race to remove translation ambiguity
            let median = median_f64(&ability_vec);
            ability_vec.mapv_inplace(|a| a - median);

            let per_horse: HashMap<String, f64> = race
                .horse_ids
                .iter()
                .zip(ability_vec.iter())
                .map(|(h, &a)| (h.clone(), a))
                .collect();

            race_abilities.push(per_horse);
        }

        let mut theta_raw: HashMap<String, Vec<f64>> = HashMap::new();
        let mut times: HashMap<String, Vec<f64>> = HashMap::new();

        for h in horse_index.keys() {
            theta_raw.insert(h.clone(), Vec::new());
            times.insert(h.clone(), Vec::new());
        }

        for (h, ndxs) in &horse_index {
            for &i in ndxs {
                theta_raw.get_mut(h).unwrap().push(race_abilities[i][h]);
                times
                    .get_mut(h)
                    .unwrap()
                    .push(self.races[i].borrow(py).time);
            }
        }

        let theta_np: HashMap<String, Bound<'_, PyArray1<f64>>> = theta_raw
            .into_iter()
            .map(|(h, v)| (h, PyArray1::from_vec(py, v)))
            .collect();

        let times_np: HashMap<String, Bound<'_, PyArray1<f64>>> = times
            .into_iter()
            .map(|(h, v)| (h, PyArray1::from_vec(py, v)))
            .collect();

        Ok((theta_np, times_np))
    }

    // --- trajectory smoother --------------------------------------------------

    /// Smooth raw abilities m_j at times t_j with a random-walk prior:
    ///
    ///     sum_j (θ_j - m_j)^2 / obs_var
    ///   + sum_{j>1} (θ_j - θ_{j-1})^2 / σ(Δt_j)^2
    ///
    /// This yields a symmetric tridiagonal linear system A θ = b.
    #[staticmethod]
    fn _smooth_trajectory<'py>(
        py: Python<'py>,
        times: PyReadonlyArray1<f64>,
        m: PyReadonlyArray1<f64>,
        sigma_fn: &Bound<'_, PyAny>,
        obs_var: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let times_arr = times.as_array();
        let m_arr = m.as_array();
        let j = m_arr.len();

        if j <= 1 {
            return Ok(m_arr.to_owned().to_pyarray(py));
        }

        let mut dt = Array1::zeros(j - 1);
        for i in 0..j - 1 {
            dt[i] = times_arr[i + 1] - times_arr[i];
        }

        let lam_obs = 1.0 / obs_var;

        // process precision per gap
        let mut lam_proc = Array1::zeros(j - 1);
        for (i, &d) in dt.iter().enumerate() {
            let sigma_d: f64 = sigma_fn.call1((d,))?.extract()?;
            lam_proc[i] = 1.0 / sigma_d.max(1e-6).powi(2);
        }

        let mut a = ndarray::Array2::zeros((j, j));
        let b = lam_obs * m_arr.to_owned();

        // first row
        a[[0, 0]] = lam_obs + lam_proc[0];
        a[[0, 1]] = -lam_proc[0];

        // interior rows
        for ji in 1..j - 1 {
            let lp = lam_proc[ji - 1];
            let ln = lam_proc[ji];
            a[[ji, ji - 1]] = -lp;
            a[[ji, ji]] = lam_obs + lp + ln;
            a[[ji, ji + 1]] = -ln;
        }

        // last row
        a[[j - 1, j - 2]] = -lam_proc[j - 2];
        a[[j - 1, j - 1]] = lam_obs + lam_proc[j - 2];

        // Solve tridiagonal system using Thomas algorithm
        let result = solve_tridiagonal_system(&a, &b)?;
        Ok(result.to_pyarray(py))
    }

    // --- public API -----------------------------------------------------------

    /// Fit (or just stage) dynamic abilities θ_{h,j}.
    /// If sigma_function is None, store per-race static abilities.
    /// Else smooth each horse trajectory with random-walk prior using σ(Δt).
    #[pyo3(signature = (sigma_function=None, obs_var=1.0))]
    fn fit_abilities(
        &mut self,
        py: Python<'_>,
        sigma_function: Option<&Bound<'_, PyAny>>,
        obs_var: f64,
    ) -> PyResult<()> {
        let (theta_raw, times) = self._initial_abilities_from_prices(py)?;

        if sigma_function.is_none() {
            self.theta_ = theta_raw
                .into_iter()
                .map(|(k, x)| (k, x.to_owned_array()))
                .collect();

            self.times_ = times
                .into_iter()
                .map(|(k, x)| (k, x.to_owned_array()))
                .collect();

            return Ok(());
        }

        let sigma_fn = sigma_function.unwrap();
        let mut theta_smooth: HashMap<String, Array1<f64>> = HashMap::new();

        for (h, m) in &theta_raw {
            let t = &times[h];
            if m.len()? <= 1 {
                theta_smooth.insert(h.clone(), m.to_owned_array());
            } else {
                let t_py = t;
                let m_py = m;
                let smoothed_py = Self::_smooth_trajectory(
                    py,
                    t_py.readonly(),
                    m_py.readonly(),
                    sigma_fn,
                    obs_var,
                )?;
                let smoothed = smoothed_py.readonly().as_array().to_owned();
                theta_smooth.insert(h.clone(), smoothed);
            }
        }

        self.theta_ = theta_smooth;
        self.times_ = times
            .into_iter()
            .map(|(k, x)| (k, x.to_owned_array()))
            .collect();

        Ok(())
    }

    /// Learn a piecewise-constant σ(Δt) from current θ trajectories.
    /// Requires self.theta_ and self.times_ (e.g., after fit_abilities(None)).
    #[pyo3(signature = (n_bins=5, min_points=20, monotone=true, meas_var=0.0))]
    fn fit_sigma(
        &mut self,
        py: Python<'_>,
        n_bins: usize,
        min_points: usize,
        monotone: bool,
        meas_var: f64,
    ) -> PyResult<Py<PyAny>> {
        let mut dts: Vec<f64> = Vec::new();
        let mut dtheta2: Vec<f64> = Vec::new();

        for (h, theta) in &self.theta_ {
            let t = &self.times_[h];
            if theta.len() <= 1 {
                continue;
            }

            for i in 0..theta.len() - 1 {
                let dt = t[i + 1] - t[i];
                let dth = theta[i + 1] - theta[i];
                dts.push(dt);
                dtheta2.push(dth * dth);
            }
        }

        if dts.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No ability increments to fit sigma(Δt).",
            ));
        }

        let dts_arr = Array1::from_vec(dts);
        let dtheta2_arr = Array1::from_vec(dtheta2);

        // bin edges from quantiles
        let mut quantiles = Vec::new();
        for i in 0..=n_bins {
            quantiles.push(i as f64 / n_bins as f64);
        }

        let edges = quantile(&dts_arr, &quantiles);
        let mut edges_arr = Array1::from_vec(edges);
        edges_arr[0] -= 1e-9;
        edges_arr[n_bins] += 1e-9;

        let mut sigma_vals: Vec<f64> = Vec::new();

        for k in 0..n_bins {
            let lo = edges_arr[k];
            let hi = edges_arr[k + 1];

            let mut mask_sum = 0;
            let mut var_sum = 0.0;

            for (i, &dt) in dts_arr.iter().enumerate() {
                if dt >= lo && dt < hi {
                    mask_sum += 1;
                    var_sum += dtheta2_arr[i];
                }
            }

            let var_k = if mask_sum < (min_points / n_bins.max(1)).max(1) {
                dtheta2_arr.mean().unwrap()
            } else {
                var_sum / mask_sum as f64
            };

            let var_k = (var_k - 2.0 * meas_var).max(1e-12);
            sigma_vals.push(var_k.sqrt());
        }

        let mut sigma_vals_arr = Array1::from_vec(sigma_vals);

        if monotone && sigma_vals_arr.len() > 1 {
            // enforce non-decreasing σ with Δt (simple pooled adjacent violators)
            for k in 1..sigma_vals_arr.len() {
                if sigma_vals_arr[k] < sigma_vals_arr[k - 1] {
                    sigma_vals_arr[k] = sigma_vals_arr[k - 1];
                }
            }
        }

        self.sigma_edges_ = Some(edges_arr.clone());
        self.sigma_vals_ = Some(sigma_vals_arr.clone());

        // Create PiecewiseSigmaFn callable object
        let sigma_fn = Py::new(
            py,
            PiecewiseSigmaFn {
                edges: edges_arr.to_vec(),
                vals: sigma_vals_arr.to_vec(),
            },
        )?;

        Ok(sigma_fn.into())
    }

    // --- helpers for measurement-noise calibration ----------------------------

    /// Collect (Δt, (Δθ)^2) from current stored per-horse trajectories.
    /// If dt_min/dt_max are provided, restrict to that window.
    #[pyo3(signature = (dt_min=None, dt_max=None))]
    fn _collect_increments<'py>(
        &self,
        py: Python<'py>,
        dt_min: Option<f64>,
        dt_max: Option<f64>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let mut dts: Vec<f64> = Vec::new();
        let mut dtheta2: Vec<f64> = Vec::new();

        for (h, theta) in &self.theta_ {
            let t = &self.times_[h];
            if theta.len() <= 1 {
                continue;
            }

            for i in 0..theta.len() - 1 {
                let dt = t[i + 1] - t[i];
                let dth = theta[i + 1] - theta[i];

                let mut include = true;
                if let Some(min) = dt_min {
                    if dt < min {
                        include = false;
                    }
                }
                if let Some(max) = dt_max {
                    if dt > max {
                        include = false;
                    }
                }

                if include {
                    dts.push(dt);
                    dtheta2.push(dth * dth);
                }
            }
        }

        (PyArray1::from_vec(py, dts), PyArray1::from_vec(py, dtheta2))
    }

    /// Estimate measurement variance τ^2 from smallest-Δt increments, then fit
    /// σ(Δt) via the binned estimator with meas_var=τ̂^2.
    ///
    /// Returns (sigma_fn, meas_var_hat).
    #[pyo3(signature = (n_bins=5, min_points=20, monotone=true, small_quantile=0.2, dt_min=None, dt_max=None))]
    fn fit_sigma_autocalibrate(
        &mut self,
        py: Python<'_>,
        n_bins: usize,
        min_points: usize,
        monotone: bool,
        small_quantile: f64,
        dt_min: Option<f64>,
        dt_max: Option<f64>,
    ) -> PyResult<(Py<PyAny>, f64)> {
        let (dts_py, dtheta2_py) = self._collect_increments(py, dt_min, dt_max);
        let dts = dts_py.readonly().as_array().to_owned();
        let dtheta2 = dtheta2_py.readonly().as_array().to_owned();

        if dts.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No ability increments to autocalibrate meas_var.",
            ));
        }

        let q = small_quantile.clamp(0.01, 0.9);
        let cutoff = quantile(&dts, &[q])[0];

        let mut meas_var_hat = 0.0;
        let mut small_count = 0;
        let mut small_sum = 0.0;

        for (i, &dt) in dts.iter().enumerate() {
            if dt <= cutoff {
                small_count += 1;
                small_sum += dtheta2[i];
            }
        }

        if small_count > 0 {
            meas_var_hat = 0.5 * (small_sum / small_count as f64);
            meas_var_hat = meas_var_hat.max(0.0);
        }

        let sigma_fn = self.fit_sigma(py, n_bins, min_points, monotone, meas_var_hat)?;

        Ok((sigma_fn, meas_var_hat))
    }

    // --- simple per-race refinement using observed winner ---------------------

    /// One finite-difference gradient step to increase log probability of winner.
    /// If winner_id is None or not in horse_ids, return ability unchanged.
    #[pyo3(signature = (ability, horse_ids, winner_id, step=0.5, eps=0.05))]
    fn _refine_with_result_once<'py>(
        &self,
        py: Python<'py>,
        ability: PyReadonlyArray1<f64>,
        horse_ids: Vec<String>,
        winner_id: Option<String>,
        step: f64,
        eps: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut ability_arr = ability.as_array().to_owned();

        if winner_id.is_none() {
            return Ok(ability_arr.to_pyarray(py));
        }

        let winner = winner_id.unwrap();
        let winner_idx = match horse_ids.iter().position(|h| h == &winner) {
            Some(idx) => idx,
            None => return Ok(ability_arr.to_pyarray(py)),
        };

        let cal = self._new_calibrator(py)?;
        let ability_py = ability_arr.to_vec();
        let base_probs_arr = cal
            .bind(py)
            .call_method1("state_prices_from_ability", (ability_py,))?;
        let base_probs_np: &Bound<'_, PyArray1<f64>> = base_probs_arr.downcast()?;
        let base_probs_ro = base_probs_np.readonly();
        let base_probs = base_probs_ro.as_array();

        let loss0 = -(base_probs[winner_idx].max(1e-15).ln());

        let mut grad = Array1::zeros(ability_arr.len());

        for i in 0..ability_arr.len() {
            let mut a_pert = ability_arr.clone();
            a_pert[i] -= eps; // negative shift = better ability
            let a_pert_py = a_pert.to_vec();
            let p_arr = cal
                .bind(py)
                .call_method1("state_prices_from_ability", (a_pert_py,))?;
            let p_np: &Bound<'_, PyArray1<f64>> = p_arr.downcast()?;
            let p_ro = p_np.readonly();
            let p = p_ro.as_array();
            let li = -(p[winner_idx].max(1e-15).ln());
            grad[i] = (li - loss0) / (-eps);
        }

        // update and re-center to preserve translation invariance
        ability_arr = ability_arr - step * &grad;
        let median = median_f64(&ability_arr);
        ability_arr.mapv_inplace(|a| a - median);

        Ok(ability_arr.to_pyarray(py))
    }

    /// Build per-race raw abilities from prices, then nudge each race's vector
    /// to better explain the observed winner (if provided).
    /// Stores the refined, time-aligned per-horse trajectories.
    #[pyo3(signature = (refine_steps=1, refine_step_size=0.5, refine_eps=0.05))]
    fn fit_abilities_with_results(
        &mut self,
        py: Python<'_>,
        refine_steps: usize,
        refine_step_size: f64,
        refine_eps: f64,
    ) -> PyResult<()> {
        let horse_index = self._build_horse_index(py);
        let mut race_abilities: Vec<HashMap<String, f64>> = Vec::new();

        for race_py in &self.races {
            let race = race_py.borrow(py);
            let cal = self._new_calibrator(py)?;
            let prices_py = race.prices.clone();
            let ability_arr = cal
                .bind(py)
                .call_method1("solve_from_prices", (prices_py,))?;
            let ability_np: &Bound<'_, PyArray1<f64>> = ability_arr.downcast()?;
            let mut ability_vec = ability_np.readonly().as_array().to_owned();

            // Center per-race to remove translation ambiguity before refinement
            let median = median_f64(&ability_vec);
            ability_vec.mapv_inplace(|a| a - median);

            let mut a = ability_vec.clone();

            for _ in 0..refine_steps {
                let a_py = a.to_pyarray(py);
                a = self
                    ._refine_with_result_once(
                        py,
                        a_py.readonly(),
                        race.horse_ids.clone(),
                        race.winner.clone(),
                        refine_step_size,
                        refine_eps,
                    )?
                    .to_owned_array();
            }

            let per_horse: HashMap<String, f64> = race
                .horse_ids
                .iter()
                .zip(a.iter())
                .map(|(h, &x)| (h.clone(), x))
                .collect();

            race_abilities.push(per_horse);
        }

        let mut theta_raw: HashMap<String, Vec<f64>> = HashMap::new();
        let mut times: HashMap<String, Vec<f64>> = HashMap::new();

        for h in horse_index.keys() {
            theta_raw.insert(h.clone(), Vec::new());
            times.insert(h.clone(), Vec::new());
        }

        for (h, ndxs) in &horse_index {
            for &i in ndxs {
                theta_raw.get_mut(h).unwrap().push(race_abilities[i][h]);
                times
                    .get_mut(h)
                    .unwrap()
                    .push(self.races[i].borrow(py).time);
            }
        }

        self.theta_ = theta_raw
            .into_iter()
            .map(|(h, v)| (h, Array1::from_vec(v)))
            .collect();
        self.times_ = times
            .into_iter()
            .map(|(h, v)| (h, Array1::from_vec(v)))
            .collect();

        Ok(())
    }

    /// Fit α in Var(Δθ) ≈ 2*meas_var + α*Δt by least squares.
    /// If meas_var is None, also fit intercept and set meas_var_hat = max(intercept/2, 0).
    /// Returns (sigma_fn, alpha_hat, meas_var_hat) where sigma_fn(dt)=sqrt(max(α dt, ε)).
    #[pyo3(signature = (meas_var=None, dt_min=None, dt_max=None))]
    fn fit_sigma_parametric(
        &self,
        py: Python<'_>,
        meas_var: Option<f64>,
        dt_min: Option<f64>,
        dt_max: Option<f64>,
    ) -> PyResult<(Py<PyAny>, f64, f64)> {
        let mut dts: Vec<f64> = Vec::new();
        let mut dtheta2: Vec<f64> = Vec::new();

        for (h, theta) in &self.theta_ {
            let t = &self.times_[h];
            if theta.len() <= 1 {
                continue;
            }

            for i in 0..theta.len() - 1 {
                let dt_h = t[i + 1] - t[i];
                let dth = theta[i + 1] - theta[i];

                let mut include = true;
                if let Some(min) = dt_min {
                    if dt_h < min {
                        include = false;
                    }
                }
                if let Some(max) = dt_max {
                    if dt_h > max {
                        include = false;
                    }
                }

                if include {
                    dts.push(dt_h);
                    dtheta2.push(dth * dth);
                }
            }
        }

        if dts.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No ability increments to fit parametric σ(Δt).",
            ));
        }

        let x = Array1::from_vec(dts);
        let y = Array1::from_vec(dtheta2);

        let (alpha_hat, meas_var_hat) = if let Some(mv) = meas_var {
            // meas_var is provided: fit slope only
            // y_adj = max(y - 2*meas_var, 0)
            // alpha = dot(x, y_adj) / dot(x, x)
            let y_adj: Array1<f64> = y.mapv(|yi| (yi - 2.0 * mv).max(0.0));
            let denom = x.iter().map(|&xi| xi * xi).sum::<f64>();
            let alpha = if denom <= 0.0 {
                0.0
            } else {
                let numer = x
                    .iter()
                    .zip(y_adj.iter())
                    .map(|(&xi, &yi)| xi * yi)
                    .sum::<f64>();
                (numer / denom).max(0.0)
            };
            (alpha, mv)
        } else {
            // meas_var is None: fit both intercept and slope
            // X = [ones(n), x]
            // beta = lstsq(X, y)
            // a = max(beta[0], 0)
            // b = max(beta[1], 0)
            // meas_var_hat = max(a * 0.5, 0)
            // alpha_hat = b
            let n = x.len();

            // Build design matrix X = [ones, x]
            let mut x_mat = ndarray::Array2::zeros((n, 2));
            for i in 0..n {
                x_mat[[i, 0]] = 1.0;
                x_mat[[i, 1]] = x[i];
            }

            // Solve least squares: X^T X beta = X^T y
            let xt = x_mat.t();
            let xtx = xt.dot(&x_mat);
            let xty = xt.dot(&y);

            // Solve 2x2 system
            let det = xtx[[0, 0]] * xtx[[1, 1]] - xtx[[0, 1]] * xtx[[1, 0]];
            let (beta0, beta1) = if det.abs() < 1e-15 {
                // Singular matrix, fallback to simple mean
                (y.mean().unwrap_or(0.0), 0.0)
            } else {
                let inv_det = 1.0 / det;
                let b0 = inv_det * (xtx[[1, 1]] * xty[0] - xtx[[0, 1]] * xty[1]);
                let b1 = inv_det * (-xtx[[1, 0]] * xty[0] + xtx[[0, 0]] * xty[1]);
                (b0, b1)
            };

            let a = beta0.max(0.0);
            let b = beta1.max(0.0);
            let mv_hat = (a * 0.5).max(0.0);
            (b, mv_hat)
        };

        // Create a ParametricSigmaFn callable object
        let sigma_fn = Py::new(py, ParametricSigmaFn { alpha: alpha_hat })?;
        Ok((sigma_fn.into(), alpha_hat, meas_var_hat))
    }

    // --- getters for learned attributes ---------------------------------------

    #[getter]
    fn theta_<'py>(&self, py: Python<'py>) -> HashMap<String, Bound<'py, PyArray1<f64>>> {
        self.theta_
            .iter()
            .map(|(k, v)| (k.clone(), v.to_pyarray(py)))
            .collect()
    }

    #[getter]
    fn times_<'py>(&self, py: Python<'py>) -> HashMap<String, Bound<'py, PyArray1<f64>>> {
        self.times_
            .iter()
            .map(|(k, v)| (k.clone(), v.to_pyarray(py)))
            .collect()
    }

    #[getter]
    fn sigma_edges_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.sigma_edges_.as_ref().map(|v| v.to_pyarray(py))
    }

    #[getter]
    fn sigma_vals_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.sigma_vals_.as_ref().map(|v| v.to_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "DynamicThurstoneCalibrator(n_races={}, bookmaker_sigma={})",
            self.races.len(),
            self.bookmaker_sigma
        )
    }
}

// --- Helper functions ---------------------------------------------------------

/// Compute median of an array
fn median_f64(arr: &Array1<f64>) -> f64 {
    let mut sorted = arr.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Compute quantiles of an array
fn quantile(arr: &Array1<f64>, quantiles: &[f64]) -> Vec<f64> {
    let mut sorted = arr.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();

    quantiles
        .iter()
        .map(|&q| {
            let idx = (q * (n - 1) as f64).round() as usize;
            sorted[idx.min(n - 1)]
        })
        .collect()
}

/// Solve tridiagonal system Ax = b using Thomas algorithm
fn solve_tridiagonal_system(a: &ndarray::Array2<f64>, b: &Array1<f64>) -> PyResult<Array1<f64>> {
    let n = b.len();

    // Extract diagonal and off-diagonal elements
    let mut diag = Array1::zeros(n);
    let mut lower = Array1::zeros(n - 1);
    let mut upper = Array1::zeros(n - 1);

    for i in 0..n {
        diag[i] = a[[i, i]];
        if i < n - 1 {
            lower[i] = a[[i + 1, i]];
            upper[i] = a[[i, i + 1]];
        }
    }

    // Thomas algorithm (tridiagonal matrix algorithm)
    let mut c_prime = Array1::zeros(n - 1);
    let mut d_prime = Array1::zeros(n);

    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = b[0] / diag[0];

    for i in 1..n - 1 {
        let m = 1.0 / (diag[i] - lower[i - 1] * c_prime[i - 1]);
        c_prime[i] = upper[i] * m;
        d_prime[i] = (b[i] - lower[i - 1] * d_prime[i - 1]) * m;
    }

    d_prime[n - 1] =
        (b[n - 1] - lower[n - 2] * d_prime[n - 2]) / (diag[n - 1] - lower[n - 2] * c_prime[n - 2]);

    // Back substitution
    let mut x = Array1::zeros(n);
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}
