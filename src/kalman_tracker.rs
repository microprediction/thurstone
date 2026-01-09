use crate::density::Density;
use crate::dynamic::RaceObservation;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Filter state for a single horse
#[pyclass]
#[derive(Clone)]
pub struct HorseFilterState {
    #[pyo3(get, set)]
    pub mean: f64,

    #[pyo3(get, set)]
    pub var: f64,

    #[pyo3(get, set)]
    pub last_time: f64,
}

#[pymethods]
impl HorseFilterState {
    #[new]
    fn new(mean: f64, var: f64, last_time: f64) -> Self {
        HorseFilterState {
            mean,
            var,
            last_time,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HorseFilterState(mean={:.3}, var={:.3}, last_time={:.1})",
            self.mean, self.var, self.last_time
        )
    }
}

/// Per-horse 1-D Kalman tracker over latent abilities using price-implied abilities as observations.
///
/// Model (for each horse h):
///     x_{j} = x_{j-1} + w_j,     w_j ~ N(0, q * Δt_j)
///     y_{j} = x_{j} + v_j,       v_j ~ N(0, r)
///
/// where:
///   - x_j is the latent ability at the j-th race for this horse,
///   - y_j is the centered, price-implied ability from the race inversion,
///   - Δt_j is the time between successive races for this horse,
///   - q is process variance per unit time,
///   - r is observation variance.
#[pyclass]
pub struct KalmanAbilityTracker {
    #[pyo3(get)]
    pub base_density: Py<Density>,

    #[pyo3(get)]
    pub ability_calibrator_kwargs: Py<PyDict>,

    #[pyo3(get, set)]
    pub process_var_per_time: f64,

    #[pyo3(get, set)]
    pub obs_var: f64,

    #[pyo3(get, set)]
    pub init_var: f64,

    #[pyo3(get, set)]
    pub min_dt: f64,

    #[pyo3(get, set)]
    pub min_var: f64,

    // Internal state
    state_: HashMap<String, HorseFilterState>,
    history_: HashMap<String, Vec<(f64, f64)>>, // (time, observation)
    centers_: HashMap<String, Vec<(f64, f64)>>, // (time, center)
}

#[pymethods]
impl KalmanAbilityTracker {
    #[new]
    #[pyo3(signature = (base_density, ability_calibrator_kwargs=None, process_var_per_time=0.1, obs_var=0.5, init_var=10.0, min_dt=1e-3, min_var=1e-9))]
    fn new(
        py: Python<'_>,
        base_density: Py<Density>,
        ability_calibrator_kwargs: Option<Py<PyDict>>,
        process_var_per_time: f64,
        obs_var: f64,
        init_var: f64,
        min_dt: f64,
        min_var: f64,
    ) -> Self {
        let kwargs = if let Some(k) = ability_calibrator_kwargs {
            k
        } else {
            PyDict::new(py).into()
        };

        KalmanAbilityTracker {
            base_density,
            ability_calibrator_kwargs: kwargs,
            process_var_per_time,
            obs_var,
            init_var,
            min_dt,
            min_var,
            state_: HashMap::new(),
            history_: HashMap::new(),
            centers_: HashMap::new(),
        }
    }

    /// Invert prices for a race, center the ability vector to remove translation gauge,
    /// and feed observations into per-horse Kalman updates. Races should be processed
    /// in chronological order for best results.
    fn update_race(&mut self, py: Python<'_>, race: &RaceObservation) -> PyResult<()> {
        // Create AbilityCalibrator
        let calibrator_class = py
            .import("thurstone.inference")?
            .getattr("AbilityCalibrator")?;

        let kwargs = self.ability_calibrator_kwargs.bind(py);
        let cal = calibrator_class.call((self.base_density.clone_ref(py),), Some(&kwargs))?;

        // Solve for abilities
        let ability_arr = cal.call_method1("solve_from_prices", (race.prices.clone(),))?;
        let ability_np = ability_arr.downcast::<PyArray1<f64>>()?;
        let ability_vec = ability_np.to_owned_array();

        // Center abilities
        let center = median_f64(&ability_vec);
        let ability_centered: Array1<f64> = ability_vec.mapv(|x| x - center);

        // Update each horse
        for (horse_id, &ability_obs) in race.horse_ids.iter().zip(ability_centered.iter()) {
            let t = race.time;
            self._kf_update(horse_id.clone(), t, ability_obs)?;
            // Record the removed race center for reconstruction
            self.centers_
                .entry(horse_id.clone())
                .or_insert_with(Vec::new)
                .push((t, center));
        }

        Ok(())
    }

    /// Return current filter state for a horse, or None if unseen
    fn get_horse_state(&self, py: Python<'_>, horse_id: String) -> Option<Py<HorseFilterState>> {
        self.state_
            .get(&horse_id)
            .map(|s| Py::new(py, s.clone()).unwrap())
    }

    /// One 1-D Kalman predict+update step for a single horse
    fn _kf_update(&mut self, horse_id: String, time: f64, y: f64) -> PyResult<()> {
        let (mean_prev, var_prev, dt) = if let Some(state) = self.state_.get(&horse_id) {
            let dt = time - state.last_time;
            let dt = if dt < 0.0 { 0.0 } else { dt };
            (state.mean, state.var, dt)
        } else {
            (0.0, self.init_var, 0.0)
        };

        let dt_eff = dt.max(self.min_dt);
        let q = self.process_var_per_time * dt_eff;

        // Predict
        let mean_pred = mean_prev;
        let var_pred = var_prev + q;

        // Update
        let r = self.obs_var;
        let s = var_pred + r;
        let k = if s <= 0.0 { 0.0 } else { var_pred / s };

        let mean_new = mean_pred + k * (y - mean_pred);
        let var_new = ((1.0 - k) * var_pred).max(self.min_var);

        self.state_.insert(
            horse_id.clone(),
            HorseFilterState {
                mean: mean_new,
                var: var_new,
                last_time: time,
            },
        );

        self.history_
            .entry(horse_id.clone())
            .or_insert_with(Vec::new)
            .push((time, y));

        Ok(())
    }

    /// Batch EM across horses to estimate process_var_per_time (q) and obs_var (r).
    /// Uses filter + RTS smoother per horse to accumulate sufficient statistics.
    #[pyo3(signature = (num_iters=10, fix_obs_var=None))]
    fn fit_em(
        &mut self,
        py: Python<'_>,
        num_iters: usize,
        fix_obs_var: Option<f64>,
    ) -> PyResult<()> {
        if self.history_.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No observations stored; call update_race first.",
            ));
        }

        let mut q = self.process_var_per_time;
        let mut r = fix_obs_var.unwrap_or(self.obs_var);

        for _ in 0..num_iters.max(1) {
            let mut r_num = 0.0;
            let mut r_den = 0.0;
            let mut q_num = 0.0;
            let mut q_den = 0.0;

            for obs_list in self.history_.values() {
                if obs_list.is_empty() {
                    continue;
                }

                let times: Array1<f64> =
                    Array1::from_vec(obs_list.iter().map(|(t, _)| *t).collect());
                let ys: Array1<f64> = Array1::from_vec(obs_list.iter().map(|(_, y)| *y).collect());

                let (rn, rd, qn, qd) =
                    self._em_stats_single_horse(times.to_pyarray(py), ys.to_pyarray(py), q, r)?;

                if fix_obs_var.is_none() {
                    r_num += rn;
                    r_den += rd;
                }
                q_num += qn;
                q_den += qd;
            }

            if fix_obs_var.is_none() && r_den > 0.0 {
                r = r_num / r_den;
            }
            if q_den > 0.0 {
                q = q_num / q_den;
            }
        }

        self.process_var_per_time = q;
        self.obs_var = r;

        Ok(())
    }

    /// Return (times, smoothed_means, smoothed_vars) for a horse using the current
    /// process_var_per_time and obs_var. None if horse has no observations.
    fn smooth_horse<'py>(
        &self,
        py: Python<'py>,
        horse_id: String,
    ) -> Option<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let obs_list = self.history_.get(&horse_id)?;
        if obs_list.is_empty() {
            return None;
        }

        let times: Array1<f64> = Array1::from_vec(obs_list.iter().map(|(t, _)| *t).collect());
        let ys: Array1<f64> = Array1::from_vec(obs_list.iter().map(|(_, y)| *y).collect());

        let (m_s, p_s) = self
            ._smooth_trajectory(py, times.to_pyarray(py), ys.to_pyarray(py))
            .ok()?;

        Some((times.to_pyarray(py), m_s, p_s))
    }

    /// Return absolute (times, smoothed_means_plus_center, smoothed_vars), by adding
    /// back the per-race centers that were removed during observation construction.
    fn smooth_horse_abs<'py>(
        &self,
        py: Python<'py>,
        horse_id: String,
    ) -> Option<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let (times_py, m_s_py, p_s_py) = self.smooth_horse(py, horse_id.clone())?;
        let times = times_py.to_owned_array();
        let m_s = m_s_py.to_owned_array();
        let p_s = p_s_py.to_owned_array();

        let centers = self.centers_.get(&horse_id)?;
        if centers.len() != times.len() {
            // Cannot reconstruct; return relative
            return Some((times.to_pyarray(py), m_s.to_pyarray(py), p_s.to_pyarray(py)));
        }

        let c_arr: Array1<f64> = Array1::from_vec(centers.iter().map(|(_, c)| *c).collect());
        let m_s_abs = m_s + &c_arr;

        Some((
            times.to_pyarray(py),
            m_s_abs.to_pyarray(py),
            p_s.to_pyarray(py),
        ))
    }

    /// Produce zero-mean anchored trajectories on race times
    fn smooth_horses_zero_mean(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        // First pass: collect relative smooths
        let mut rel: HashMap<String, (Array1<f64>, Array1<f64>)> = HashMap::new();

        for (h, obs) in &self.history_ {
            if obs.is_empty() {
                continue;
            }
            if let Some((times_py, means_py, _)) = self.smooth_horse(py, h.clone()) {
                let times = times_py.to_owned_array();
                let means = means_py.to_owned_array();
                rel.insert(h.clone(), (times, means));
            }
        }

        // Aggregate per-time means
        let mut time_accum_sum: HashMap<u64, f64> = HashMap::new();
        let mut time_accum_cnt: HashMap<u64, usize> = HashMap::new();

        for (times, means) in rel.values() {
            for (&t, &m) in times.iter().zip(means.iter()) {
                let t_key = (t * 1e9) as u64; // Use integer key for HashMap
                *time_accum_sum.entry(t_key).or_insert(0.0) += m;
                *time_accum_cnt.entry(t_key).or_insert(0) += 1;
            }
        }

        let mut time_mean: HashMap<u64, f64> = HashMap::new();
        for (t_key, sum) in &time_accum_sum {
            let cnt = time_accum_cnt[t_key];
            time_mean.insert(*t_key, sum / cnt.max(1) as f64);
        }

        // Subtract the per-time mean
        let result_dict = PyDict::new(py);
        for (h, (times, means)) in rel {
            let adj: Array1<f64> = times
                .iter()
                .zip(means.iter())
                .map(|(&t, &m)| {
                    let t_key = (t * 1e9) as u64;
                    m - time_mean.get(&t_key).unwrap_or(&0.0)
                })
                .collect();

            let tuple = (times.to_pyarray(py), adj.to_pyarray(py));
            result_dict.set_item(h, tuple)?;
        }

        Ok(result_dict.into())
    }

    fn _em_stats_single_horse(
        &self,
        times: Bound<'_, PyArray1<f64>>,
        ys: Bound<'_, PyArray1<f64>>,
        q: f64,
        r: f64,
    ) -> PyResult<(f64, f64, f64, f64)> {
        let j = ys.len()?;
        if j == 0 {
            return Ok((0.0, 0.0, 0.0, 0.0));
        }

        // Forward filter
        let mut m_f = Array1::zeros(j);
        let mut p_f = Array1::zeros(j);
        let mut m_pred = Array1::zeros(j);
        let mut p_pred = Array1::zeros(j);

        let ys = ys.to_owned_array();
        let times = times.to_owned_array();

        // Prior
        let m_prev = 0.0;
        let p_prev = self.init_var;

        // First observation j=0
        m_pred[0] = m_prev;
        p_pred[0] = p_prev;
        let s0 = p_pred[0] + r;
        let k0 = if s0 <= 0.0 { 0.0 } else { p_pred[0] / s0 };
        m_f[0] = m_pred[0] + k0 * (ys[0] - m_pred[0]);
        p_f[0] = (1.0 - k0) * p_pred[0];

        // Forward pass
        for j_idx in 1..j {
            let dt = times[j_idx] - times[j_idx - 1];
            let dt = if dt < 0.0 { 0.0 } else { dt };
            let dt_eff = dt.max(self.min_dt);
            let qj = q * dt_eff;

            m_pred[j_idx] = m_f[j_idx - 1];
            p_pred[j_idx] = p_f[j_idx - 1] + qj;
            let sj = p_pred[j_idx] + r;
            let kj = if sj <= 0.0 { 0.0 } else { p_pred[j_idx] / sj };
            m_f[j_idx] = m_pred[j_idx] + kj * (ys[j_idx] - m_pred[j_idx]);
            p_f[j_idx] = (1.0 - kj) * p_pred[j_idx];
        }

        // RTS smoother
        let mut m_s = m_f.clone();
        let mut p_s = p_f.clone();
        let mut c = Array1::zeros(j);

        for j_idx in (1..j).rev() {
            let dt = times[j_idx] - times[j_idx - 1];
            let dt = if dt < 0.0 { 0.0 } else { dt };
            let dt_eff = dt.max(self.min_dt);
            let qj = q * dt_eff;
            let p_pred_j = p_f[j_idx - 1] + qj;
            let a = if p_pred_j <= 0.0 {
                0.0
            } else {
                p_f[j_idx - 1] / p_pred_j
            };
            m_s[j_idx - 1] = m_f[j_idx - 1] + a * (m_s[j_idx] - m_f[j_idx - 1]);
            p_s[j_idx - 1] = p_f[j_idx - 1] + a * a * (p_s[j_idx] - p_pred_j);
            c[j_idx] = a * p_s[j_idx];
        }

        // Stats for r
        let mut r_num = 0.0;
        let mut r_den = 0.0;
        for j_idx in 0..j {
            let err = ys[j_idx] - m_s[j_idx];
            r_num += err * err + p_s[j_idx];
            r_den += 1.0;
        }

        // Stats for q
        let mut q_num = 0.0;
        let mut q_den = 0.0;
        for j_idx in 1..j {
            let dt = times[j_idx] - times[j_idx - 1];
            let dt = if dt < 0.0 { 0.0 } else { dt };
            let dt_eff = dt.max(self.min_dt);
            let ex2 = m_s[j_idx] * m_s[j_idx] + p_s[j_idx];
            let ex_prev2 = m_s[j_idx - 1] * m_s[j_idx - 1] + p_s[j_idx - 1];
            let exx_prev = m_s[j_idx] * m_s[j_idx - 1] + c[j_idx];
            let e_delta2 = ex2 + ex_prev2 - 2.0 * exx_prev;
            q_num += e_delta2 / dt_eff;
            q_den += 1.0;
        }

        Ok((r_num, r_den, q_num, q_den))
    }

    fn _smooth_trajectory<'py>(
        &self,
        py: Python<'py>,
        times: Bound<'py, PyArray1<f64>>,
        ys: Bound<'py, PyArray1<f64>>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let j = ys.len()?;
        let q = self.process_var_per_time;
        let r = self.obs_var;

        let mut m_f = Array1::zeros(j);
        let mut p_f = Array1::zeros(j);
        let mut m_pred = Array1::zeros(j);
        let mut p_pred = Array1::zeros(j);

        let ys = ys.to_owned_array();
        let times = times.to_owned_array();

        // Prior
        let m_prev = 0.0;
        let p_prev = self.init_var;
        m_pred[0] = m_prev;
        p_pred[0] = p_prev;
        let s0 = p_pred[0] + r;
        let k0 = if s0 <= 0.0 { 0.0 } else { p_pred[0] / s0 };
        m_f[0] = m_pred[0] + k0 * (ys[0] - m_pred[0]);
        p_f[0] = (1.0 - k0) * p_pred[0];

        for j_idx in 1..j {
            let dt = times[j_idx] - times[j_idx - 1];
            let dt = if dt < 0.0 { 0.0 } else { dt };
            let dt_eff = dt.max(self.min_dt);
            let qj = q * dt_eff;
            m_pred[j_idx] = m_f[j_idx - 1];
            p_pred[j_idx] = p_f[j_idx - 1] + qj;
            let sj = p_pred[j_idx] + r;
            let kj = if sj <= 0.0 { 0.0 } else { p_pred[j_idx] / sj };
            m_f[j_idx] = m_pred[j_idx] + kj * (ys[j_idx] - m_pred[j_idx]);
            p_f[j_idx] = (1.0 - kj) * p_pred[j_idx];
        }

        // RTS
        let mut m_s = m_f.clone();
        let mut p_s = p_f.clone();
        for j_idx in (1..j).rev() {
            let dt = times[j_idx] - times[j_idx - 1];
            let dt = if dt < 0.0 { 0.0 } else { dt };
            let dt_eff = dt.max(self.min_dt);
            let qj = q * dt_eff;
            let p_pred_j = p_f[j_idx - 1] + qj;
            let a = if p_pred_j <= 0.0 {
                0.0
            } else {
                p_f[j_idx - 1] / p_pred_j
            };
            m_s[j_idx - 1] = m_f[j_idx - 1] + a * (m_s[j_idx] - m_f[j_idx - 1]);
            p_s[j_idx - 1] = p_f[j_idx - 1] + a * a * (p_s[j_idx] - p_pred_j);
        }

        Ok((m_s.into_pyarray(py), p_s.into_pyarray(py)))
    }

    fn __repr__(&self) -> String {
        format!(
            "KalmanAbilityTracker(process_var_per_time={:.3}, obs_var={:.3}, n_horses={})",
            self.process_var_per_time,
            self.obs_var,
            self.state_.len()
        )
    }
}

// Helper functions

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
