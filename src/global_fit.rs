use crate::inference::AbilityCalibrator;
use ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---- Helper functions ----

/// Interpolate price and d price / d mu from cached 1D curve.
fn interp_price_and_slope_1d(
    py: Python<'_>,
    cal: &AbilityCalibrator,
    mu: f64,
) -> PyResult<(f64, f64)> {
    let lookup_curve = cal
        .lookup_curve_1d_prices(py)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "AbilityCalibrator has no 1D lookup curve. Run solve_from_prices first.",
            )
        })?;

    let lookup_curve_ref = lookup_curve.borrow(py);
    let locs_py = lookup_curve_ref.locs(py);
    let prices_py = lookup_curve_ref.prices(py);

    let locs = locs_py.bind(py).readonly();
    let prices = prices_py.bind(py).readonly();

    let locs_arr = locs.as_array();
    let prices_arr = prices.as_array();

    // Compute gradient (np.gradient equivalent)
    let dprices = gradient(prices_arr, locs_arr);

    let loc_min = locs_arr
        .iter()
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let loc_max = locs_arr
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mu_c = mu.max(loc_min).min(loc_max);

    let p = interp_1d(mu_c, locs_arr, prices_arr);
    let p = p.max(1e-12).min(1.0 - 1e-12);

    let dprices_arr = Array1::from_vec(dprices);
    let dp = interp_1d(mu_c, locs_arr, dprices_arr.view());

    Ok((p, dp))
}

/// Interpolate price and slope across location and scale using cached 2D curves.
fn interp_price_and_slope_2d(
    py: Python<'_>,
    cal: &AbilityCalibrator,
    mu: f64,
    scale: f64,
) -> PyResult<(f64, f64)> {
    let lookup_curves = cal.lookup_curves_2d_prices(py);

    if lookup_curves.is_empty() {
        return interp_price_and_slope_1d(py, cal, mu);
    }

    let mut scales: Vec<f64> = lookup_curves
        .keys()
        .map(|s| s.parse::<f64>().unwrap())
        .collect();
    scales.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let s_key = format!("{}", scale);

    // Check if we have exact match
    if let Some((locs_py, prices_py)) = lookup_curves.get(&s_key) {
        let locs = locs_py.bind(py).readonly();
        let prices = prices_py.bind(py).readonly();

        let locs_arr = locs.as_array();
        let prices_arr = prices.as_array();

        let dprices = gradient(prices_arr, locs_arr);

        let loc_min = locs_arr
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let loc_max = locs_arr
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let mu_c = mu.max(loc_min).min(loc_max);

        let p = interp_1d(mu_c, locs_arr, prices_arr);
        let p = p.max(1e-12).min(1.0 - 1e-12);

        let dprices_arr = Array1::from_vec(dprices);
        let dp = interp_1d(mu_c, locs_arr, dprices_arr.view());

        return Ok((p, dp));
    }

    // Find bounding scales for interpolation
    let s_arr = Array1::from_vec(scales.clone());
    let idx = s_arr
        .iter()
        .position(|&s| s >= scale)
        .unwrap_or(s_arr.len());

    let (s1, s2) = if idx == 0 {
        (s_arr[0], s_arr[1.min(s_arr.len() - 1)])
    } else if idx >= s_arr.len() {
        (s_arr[s_arr.len() - 2], s_arr[s_arr.len() - 1])
    } else {
        (s_arr[idx - 1], s_arr[idx])
    };

    let w = if (s2 - s1).abs() < 1e-12 {
        0.0
    } else {
        (scale - s1) / (s2 - s1)
    };

    // Interpolate at s1
    let s1_key = format!("{}", s1);
    let (locs1_py, prices1_py) = lookup_curves.get(&s1_key).unwrap();
    let locs1 = locs1_py.bind(py).readonly();
    let prices1 = prices1_py.bind(py).readonly();

    let locs1_arr = locs1.as_array();
    let prices1_arr = prices1.as_array();

    let dprices1 = gradient(prices1_arr, locs1_arr);

    let loc1_min = locs1_arr
        .iter()
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let loc1_max = locs1_arr
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mu1 = mu.max(loc1_min).min(loc1_max);

    let p1 = interp_1d(mu1, locs1_arr, prices1_arr);

    let dprices1_arr = Array1::from_vec(dprices1);
    let dp1 = interp_1d(mu1, locs1_arr, dprices1_arr.view());

    // Interpolate at s2
    let s2_key = format!("{}", s2);
    let (locs2_py, prices2_py) = lookup_curves.get(&s2_key).unwrap();
    let locs2 = locs2_py.bind(py).readonly();
    let prices2 = prices2_py.bind(py).readonly();

    let locs2_arr = locs2.as_array();
    let prices2_arr = prices2.as_array();

    let dprices2 = gradient(prices2_arr, locs2_arr);

    let loc2_min = locs2_arr
        .iter()
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let loc2_max = locs2_arr
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mu2 = mu.max(loc2_min).min(loc2_max);

    let p2 = interp_1d(mu2, locs2_arr, prices2_arr);

    let dprices2_arr = Array1::from_vec(dprices2);
    let dp2 = interp_1d(mu2, locs2_arr, dprices2_arr.view());

    // Blend between s1 and s2
    let p = (1.0 - w) * p1 + w * p2;
    let p = p.max(1e-12).min(1.0 - 1e-12);
    let dp = (1.0 - w) * dp1 + w * dp2;

    Ok((p, dp))
}

// ---- Data structures ----

/// Specification for a race
#[pyclass]
pub struct RaceSpec {
    #[pyo3(get)]
    calibrator: Py<AbilityCalibrator>,
    #[pyo3(get)]
    horse_ids: Vec<String>,
    #[pyo3(get)]
    prices: Py<PyArray1<f64>>,
    #[pyo3(get)]
    scales: Option<Py<PyArray1<f64>>>,
}

/// Global ability calibrator for fitting abilities across multiple races
#[pyclass]
pub struct GlobalAbilityCalibrator {
    #[pyo3(get)]
    horse_ids: Vec<String>,
    #[pyo3(get)]
    races: Vec<Py<RaceSpec>>,
    #[pyo3(get)]
    theta: HashMap<String, f64>,
    #[pyo3(get)]
    biases: Vec<f64>,
    #[pyo3(get, set)]
    l2: f64,
    #[pyo3(get, set)]
    step_bias: f64,
    #[pyo3(get, set)]
    step_theta: f64,
}

#[pymethods]
impl GlobalAbilityCalibrator {
    #[new]
    #[pyo3(signature = (horse_ids, l2=1e-8, step_bias=0.3, step_theta=0.3))]
    fn new(horse_ids: Vec<String>, l2: f64, step_bias: f64, step_theta: f64) -> Self {
        let theta: HashMap<String, f64> = horse_ids
            .iter()
            .map(|hid| (hid.clone(), 0.0))
            .collect();

        GlobalAbilityCalibrator {
            horse_ids,
            races: Vec::new(),
            theta,
            biases: Vec::new(),
            l2,
            step_bias,
            step_theta,
        }
    }

    /// Add a race to the calibrator
    #[pyo3(signature = (calibrator, horse_ids, prices, scales=None))]
    fn add_race(
        &mut self,
        py: Python<'_>,
        calibrator: Py<AbilityCalibrator>,
        horse_ids: Vec<String>,
        prices: PyReadonlyArray1<f64>,
        scales: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        let prices_arr = prices.as_array().to_owned().to_pyarray(py).unbind();
        let scales_arr = scales.map(|s| s.as_array().to_owned().to_pyarray(py).unbind());

        // Check if calibrator needs initialization
        {
            let cal_ref = calibrator.borrow(py);
            let has_1d = cal_ref.lookup_curve_1d_prices(py).is_some();
            let has_2d = !cal_ref.lookup_curves_2d_prices(py).is_empty();

            if !has_1d && !has_2d {
                drop(cal_ref);
                let mut cal_ref = calibrator.borrow_mut(py);
                let prices_vec: Vec<f64> = prices.as_array().to_vec();
                cal_ref.solve_from_prices(py, prices_vec, None)?;
            }
        }

        let race_spec = RaceSpec {
            calibrator: calibrator.clone_ref(py),
            horse_ids,
            prices: prices_arr,
            scales: scales_arr,
        };

        self.races.push(Py::new(py, race_spec)?);
        self.biases.push(0.0);

        Ok(())
    }

    /// Predict prices and slopes for a race
    fn _predict_and_slopes_for_race(
        &self,
        py: Python<'_>,
        r_idx: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let spec_py = &self.races[r_idx];
        let spec = spec_py.borrow(py);
        let cal = &spec.calibrator;

        let n = spec.horse_ids.len();
        let mut p_hat = vec![0.0; n];
        let mut slopes = vec![0.0; n];

        let cal_ref = cal.borrow(py);
        let has_2d = !cal_ref.lookup_curves_2d_prices(py).is_empty();
        drop(cal_ref);

        for (i, hid) in spec.horse_ids.iter().enumerate() {
            let mu = self.theta[hid] + self.biases[r_idx];

            let (p, dp) = if let Some(ref scales_py) = spec.scales {
                let scales = scales_py.bind(py).readonly();
                let scales_arr = scales.as_array();
                if has_2d && i < scales_arr.len() {
                    interp_price_and_slope_2d(py, &cal.borrow(py), mu, scales_arr[i])?
                } else {
                    interp_price_and_slope_1d(py, &cal.borrow(py), mu)?
                }
            } else {
                interp_price_and_slope_1d(py, &cal.borrow(py), mu)?
            };

            p_hat[i] = p;
            slopes[i] = dp;
        }

        Ok((p_hat, slopes))
    }

    /// Fit the model for a given number of iterations
    #[pyo3(signature = (num_iters=25))]
    fn fit(&mut self, py: Python<'_>, num_iters: usize) -> PyResult<()> {
        for _ in 0..num_iters {
            // Update biases
            for r in 0..self.races.len() {
                let spec_py = &self.races[r];
                let spec = spec_py.borrow(py);

                let (p_hat, slopes) = self._predict_and_slopes_for_race(py, r)?;

                let prices = spec.prices.bind(py).readonly();
                let prices_arr = prices.as_array();

                // Compute error
                let e: Vec<f64> = p_hat
                    .iter()
                    .zip(prices_arr.iter())
                    .map(|(&ph, &p)| ph - p)
                    .collect();

                // Compute denominator
                let denom = slopes.iter().map(|&s| s * s).sum::<f64>() + self.l2;

                if denom > 0.0 {
                    let delta = -slopes
                        .iter()
                        .zip(e.iter())
                        .map(|(&s, &err)| s * err)
                        .sum::<f64>()
                        / denom;
                    self.biases[r] += self.step_bias * delta;
                }
            }

            // Update theta
            for hid in &self.horse_ids.clone() {
                let mut num = 0.0;
                let mut den = self.l2;

                for r in 0..self.races.len() {
                    let spec_py = &self.races[r];
                    let spec = spec_py.borrow(py);

                    if !spec.horse_ids.contains(hid) {
                        continue;
                    }

                    let i = spec.horse_ids.iter().position(|id| id == hid).unwrap();
                    let cal = &spec.calibrator;
                    let cal_ref = cal.borrow(py);

                    let mu = self.theta[hid] + self.biases[r];

                    let has_2d = !cal_ref.lookup_curves_2d_prices(py).is_empty();

                    let (p, dp) = if let Some(ref scales_py) = spec.scales {
                        let scales = scales_py.bind(py).readonly();
                        let scales_arr = scales.as_array();
                        if has_2d && i < scales_arr.len() {
                            drop(cal_ref);
                            interp_price_and_slope_2d(py, &cal.borrow(py), mu, scales_arr[i])?
                        } else {
                            drop(cal_ref);
                            interp_price_and_slope_1d(py, &cal.borrow(py), mu)?
                        }
                    } else {
                        drop(cal_ref);
                        interp_price_and_slope_1d(py, &cal.borrow(py), mu)?
                    };

                    let prices = spec.prices.bind(py).readonly();
                    let e = p - prices.as_array()[i];

                    num += dp * e;
                    den += dp * dp;
                }

                if den > 0.0 {
                    *self.theta.get_mut(hid).unwrap() -= self.step_theta * (num / den);
                }
            }
        }

        Ok(())
    }

    /// Predict prices for a race
    fn predict_race<'py>(&self, py: Python<'py>, r_idx: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (p_hat, _) = self._predict_and_slopes_for_race(py, r_idx)?;
        Ok(Array1::from_vec(p_hat).to_pyarray(py))
    }

    /// Rebuild all curves
    fn rebuild_all_curves(&mut self, py: Python<'_>) -> PyResult<()> {
        for r in 0..self.races.len() {
            let spec_py = &self.races[r];
            let spec = spec_py.borrow(py);

            let mu_r: Vec<f64> = spec
                .horse_ids
                .iter()
                .map(|hid| self.theta[hid] + self.biases[r])
                .collect();

            let cal = &spec.calibrator;
            let mut cal_ref = cal.borrow_mut(py);

            if let Some(ref scales_py) = spec.scales {
                let scales = scales_py.bind(py).readonly();
                let scales_vec = scales.as_array().to_vec();
                if mu_r.len() == scales_vec.len() {
                    cal_ref.rebuild_curves_from_field_2d(py, mu_r, scales_vec)?;
                } else {
                    cal_ref.rebuild_curves_from_field_1d(py, mu_r)?;
                }
            } else {
                cal_ref.rebuild_curves_from_field_1d(py, mu_r)?;
            }
        }

        Ok(())
    }

    /// Fit with curve rebuilding
    #[pyo3(signature = (num_outer_iters=3, num_inner_iters=10))]
    fn fit_with_rebuild(
        &mut self,
        py: Python<'_>,
        num_outer_iters: usize,
        num_inner_iters: usize,
    ) -> PyResult<()> {
        for _ in 0..num_outer_iters {
            self.rebuild_all_curves(py)?;
            self.fit(py, num_inner_iters)?;
        }
        Ok(())
    }

    /// Fit theta only (hold race biases fixed)
    #[pyo3(signature = (num_iters=25))]
    fn fit_theta_only(&mut self, py: Python<'_>, num_iters: usize) -> PyResult<()> {
        for _ in 0..num_iters {
            for hid in &self.horse_ids.clone() {
                let mut num = 0.0;
                let mut den = self.l2;

                for r in 0..self.races.len() {
                    let spec_py = &self.races[r];
                    let spec = spec_py.borrow(py);

                    if !spec.horse_ids.contains(hid) {
                        continue;
                    }

                    let i = spec.horse_ids.iter().position(|id| id == hid).unwrap();
                    let cal = &spec.calibrator;
                    let cal_ref = cal.borrow(py);

                    let mu = self.theta[hid] + self.biases[r];

                    let has_2d = !cal_ref.lookup_curves_2d_prices(py).is_empty();

                    let (p, dp) = if let Some(ref scales_py) = spec.scales {
                        let scales = scales_py.bind(py).readonly();
                        let scales_arr = scales.as_array();
                        if has_2d && i < scales_arr.len() {
                            drop(cal_ref);
                            interp_price_and_slope_2d(py, &cal.borrow(py), mu, scales_arr[i])?
                        } else {
                            drop(cal_ref);
                            interp_price_and_slope_1d(py, &cal.borrow(py), mu)?
                        }
                    } else {
                        drop(cal_ref);
                        interp_price_and_slope_1d(py, &cal.borrow(py), mu)?
                    };

                    let prices = spec.prices.bind(py).readonly();
                    let e = p - prices.as_array()[i];

                    num += dp * e;
                    den += dp * dp;
                }

                if den > 0.0 {
                    *self.theta.get_mut(hid).unwrap() -= self.step_theta * (num / den);
                }
            }
        }

        Ok(())
    }

    /// Fit with rebuild (theta only)
    #[pyo3(signature = (num_outer_iters=3, num_inner_iters=10))]
    fn fit_with_rebuild_theta_only(
        &mut self,
        py: Python<'_>,
        num_outer_iters: usize,
        num_inner_iters: usize,
    ) -> PyResult<()> {
        for _ in 0..num_outer_iters {
            self.rebuild_all_curves(py)?;
            self.fit_theta_only(py, num_inner_iters)?;
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "GlobalAbilityCalibrator(horse_ids=[...], races={}, l2={}, step_bias={}, step_theta={})",
            self.races.len(),
            self.l2,
            self.step_bias,
            self.step_theta
        )
    }
}

// ---- Helper functions for numerical operations ----

/// Compute gradient similar to numpy.gradient
fn gradient(y: ArrayView1<f64>, x: ArrayView1<f64>) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0.0];
    }

    let mut grad = vec![0.0; n];

    // First point: forward difference
    grad[0] = (y[1] - y[0]) / (x[1] - x[0]);

    // Interior points: centered difference
    for i in 1..n - 1 {
        grad[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
    }

    // Last point: backward difference
    grad[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);

    grad
}

/// Linear interpolation
fn interp_1d(x: f64, xp: ArrayView1<f64>, fp: ArrayView1<f64>) -> f64 {
    let n = xp.len();

    if n == 0 {
        return 0.0;
    }

    if x <= xp[0] {
        return fp[0];
    }

    if x >= xp[n - 1] {
        return fp[n - 1];
    }

    // Binary search for the right interval
    let mut left = 0;
    let mut right = n - 1;

    while right - left > 1 {
        let mid = (left + right) / 2;
        if x < xp[mid] {
            right = mid;
        } else {
            left = mid;
        }
    }

    let x0 = xp[left];
    let x1 = xp[right];
    let y0 = fp[left];
    let y1 = fp[right];

    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}
