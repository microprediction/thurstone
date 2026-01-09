use ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Per-race data for global LS calibration
#[pyclass]
pub struct RaceLS {
    #[pyo3(get)]
    pub calibrator: Py<PyAny>,

    #[pyo3(get)]
    pub horse_ids: Vec<String>,

    #[pyo3(get)]
    pub prices: Vec<f64>,

    #[pyo3(get)]
    pub scales: Option<Vec<f64>>,

    #[pyo3(get)]
    pub local_locs: Option<Vec<f64>>,
}

#[pymethods]
impl RaceLS {
    #[new]
    #[pyo3(signature = (calibrator, horse_ids, prices, scales=None, local_locs=None))]
    fn new(
        calibrator: Py<PyAny>,
        horse_ids: Vec<String>,
        prices: Vec<f64>,
        scales: Option<Vec<f64>>,
        local_locs: Option<Vec<f64>>,
    ) -> Self {
        RaceLS {
            calibrator,
            horse_ids,
            prices,
            scales,
            local_locs,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RaceLS(n_horses={}, has_scales={}, has_local_locs={})",
            self.horse_ids.len(),
            self.scales.is_some(),
            self.local_locs.is_some()
        )
    }
}

/// Global least squares calibrator
///
/// Relative-then-global least squares:
///   1) Per race: invert prices -> local locs (once), center per race to remove translation
///   2) Stitch globally via (slope-weighted) LS across overlapping runners
#[pyclass]
pub struct GlobalLSCalibrator {
    #[pyo3(get)]
    pub horse_ids: Vec<String>,

    #[pyo3(get)]
    pub races: Vec<Py<RaceLS>>,

    theta_: HashMap<String, f64>,
}

#[pymethods]
impl GlobalLSCalibrator {
    #[new]
    fn new(horse_ids: Vec<String>) -> Self {
        let theta_: HashMap<String, f64> = horse_ids.iter().map(|h| (h.clone(), 0.0)).collect();

        GlobalLSCalibrator {
            horse_ids,
            races: Vec::new(),
            theta_,
        }
    }

    /// Add a race to the calibrator
    #[pyo3(signature = (calibrator, horse_ids, prices, scales=None))]
    fn add_race(
        &mut self,
        py: Python<'_>,
        calibrator: Py<PyAny>,
        horse_ids: Vec<String>,
        prices: Vec<f64>,
        scales: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let cal = calibrator.bind(py);

        // Ensure lookup curves exist at least once
        let lookup_curve_1d_prices = cal.getattr("lookup_curve_1d_prices")?;
        let lookup_curves_2d_prices = cal.getattr("lookup_curves_2d_prices")?;

        let is_none = lookup_curve_1d_prices.is_none();
        let is_empty = if let Ok(dict) = lookup_curves_2d_prices.downcast::<PyDict>() {
            dict.is_empty()
        } else {
            true
        };

        if is_none && is_empty {
            cal.call_method1("solve_from_prices", (prices.clone(),))?;
        }

        // Precompute local raw locs once for this race
        let local_locs_arr = cal.call_method1("solve_from_prices", (prices.clone(),))?;
        let local_locs_np = local_locs_arr.downcast::<PyArray1<f64>>()?;
        let local_locs = local_locs_np.readonly().as_array().to_vec();

        let race = RaceLS {
            calibrator: calibrator.clone_ref(py),
            horse_ids,
            prices,
            scales,
            local_locs: Some(local_locs),
        };

        self.races.push(Py::new(py, race)?);
        Ok(())
    }

    /// Return median-centered per-race locs (translation-free)
    fn _invert_and_center<'py>(&self, py: Python<'py>, race: &RaceLS) -> Bound<'py, PyArray1<f64>> {
        let local_locs = if let Some(ref locs) = race.local_locs {
            Array1::from_vec(locs.clone())
        } else {
            // This should not happen if add_race precomputed local_locs
            Array1::zeros(race.prices.len())
        };

        let median = median_f64(&local_locs);
        local_locs.mapv(|x| x - median).to_pyarray(py)
    }

    /// Approximate |dp/dmu| using cached curves; fall back to 1.0
    fn _slope_weight(
        &self,
        py: Python<'_>,
        cal: &Bound<'_, PyAny>,
        loc: f64,
        scale: Option<f64>,
    ) -> PyResult<f64> {
        let result: PyResult<f64> = (|| {
            // Try 2D curves if scale is provided
            if let Some(scale_val) = scale {
                let lookup_curves_2d_prices = cal.getattr("lookup_curves_2d_prices")?;
                if let Ok(dict) = lookup_curves_2d_prices.downcast::<PyDict>() {
                    if !dict.is_empty() {
                        // Get scales and find nearest
                        let scales: Vec<f64> = dict
                            .keys()
                            .iter()
                            .filter_map(|k| k.extract::<f64>().ok())
                            .collect();
                        if !scales.is_empty() {
                            let mut sorted_scales = scales.clone();
                            sorted_scales.sort_by(|a, b| a.partial_cmp(b).unwrap());

                            // Binary search for nearest scale
                            let idx = sorted_scales
                                .binary_search_by(|s| {
                                    s.partial_cmp(&scale_val)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .unwrap_or_else(|i| i);

                            let s_sel = if idx == 0 {
                                sorted_scales[0]
                            } else if idx >= sorted_scales.len() {
                                *sorted_scales.last().unwrap()
                            } else {
                                let diff_before = (scale_val - sorted_scales[idx - 1]).abs();
                                let diff_after = (sorted_scales[idx] - scale_val).abs();
                                if diff_before <= diff_after {
                                    sorted_scales[idx - 1]
                                } else {
                                    sorted_scales[idx]
                                }
                            };

                            // Get locs and prices for this scale
                            let curve = dict.get_item(s_sel)?;
                            if let Some(curve_val) = curve {
                                let locs_item = curve_val.get_item(0)?;
                                let prices_item = curve_val.get_item(1)?;

                                let locs_arr = locs_item.downcast::<PyArray1<f64>>()?;
                                let prices_arr = prices_item.downcast::<PyArray1<f64>>()?;

                                let locs = locs_arr.to_owned_array();
                                let prices = prices_arr.to_owned_array();

                                // Compute gradient
                                let dprices = gradient(&prices, &locs);
                                let mu_c = loc.clamp(
                                    *locs
                                        .iter()
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .unwrap(),
                                    *locs
                                        .iter()
                                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                                        .unwrap(),
                                );
                                let dp = interp(&locs, &dprices, mu_c);
                                return Ok(dp.abs() + 1e-12);
                            }
                        }
                    }
                }
            }

            // Try 1D curve
            let lookup_curve_1d_prices = cal.getattr("lookup_curve_1d_prices")?;
            if !lookup_curve_1d_prices.is_none() {
                let curve = lookup_curve_1d_prices.downcast::<PyDict>()?;
                let locs_item = curve.get_item("locs")?.unwrap();
                let prices_item = curve.get_item("prices")?.unwrap();

                let locs_arr = locs_item.downcast::<PyArray1<f64>>()?;
                let prices_arr = prices_item.downcast::<PyArray1<f64>>()?;

                let locs = locs_arr.to_owned_array();
                let prices = prices_arr.to_owned_array();

                let dprices = gradient(&prices, &locs);
                let mu_c = loc.clamp(
                    *locs
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap(),
                    *locs
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap(),
                );
                let dp = interp(&locs, &dprices, mu_c);
                return Ok(dp.abs() + 1e-12);
            }

            Err(pyo3::exceptions::PyValueError::new_err("No curves found"))
        })();

        Ok(result.unwrap_or(1.0))
    }

    /// Solve for global theta via one-pass (slope-weighted) LS on centered per-race inversions
    #[pyo3(signature = (use_slope_weights=true, ridge=0.0, weight_cap=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        use_slope_weights: bool,
        ridge: f64,
        weight_cap: Option<f64>,
    ) -> PyResult<()> {
        let mut sum_w_y: HashMap<String, f64> =
            self.horse_ids.iter().map(|h| (h.clone(), 0.0)).collect();
        let mut sum_w: HashMap<String, f64> =
            self.horse_ids.iter().map(|h| (h.clone(), ridge)).collect();

        for race_py in &self.races {
            let race = race_py.borrow(py);
            let centered = self._invert_and_center(py, &race).to_owned_array();

            // Accumulate per horse
            for (j, hid) in race.horse_ids.iter().enumerate() {
                let w = if use_slope_weights {
                    let sc = race.scales.as_ref().map(|s| s[j]);
                    // Use raw local loc for slope weight (not centered)
                    let loc_for_slope = if let Some(ref locs) = race.local_locs {
                        locs[j]
                    } else {
                        centered[j]
                    };
                    let mut weight =
                        self._slope_weight(py, &race.calibrator.bind(py), loc_for_slope, sc)?;
                    if let Some(cap) = weight_cap {
                        weight = weight.min(cap);
                    }
                    weight
                } else {
                    1.0
                };

                *sum_w_y.get_mut(hid).unwrap() += w * centered[j];
                *sum_w.get_mut(hid).unwrap() += w;
            }
        }

        // Closed-form per-horse since design is diagonal by construction
        for hid in &self.horse_ids {
            let denom = sum_w[hid];
            self.theta_.insert(
                hid.clone(),
                if denom > 0.0 {
                    sum_w_y[hid] / denom
                } else {
                    0.0
                },
            );
        }

        // Fix gauge: center global theta to zero median
        let theta_values: Vec<f64> = self.horse_ids.iter().map(|h| self.theta_[h]).collect();
        let med = median_f64(&Array1::from_vec(theta_values));

        for hid in &self.horse_ids {
            *self.theta_.get_mut(hid).unwrap() -= med;
        }

        Ok(())
    }

    #[getter]
    fn theta<'py>(&self, py: Python<'py>) -> Py<PyDict> {
        let dict = PyDict::new(py);
        for (k, v) in &self.theta_ {
            dict.set_item(k, v).unwrap();
        }
        dict.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "GlobalLSCalibrator(n_horses={}, n_races={})",
            self.horse_ids.len(),
            self.races.len()
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

fn gradient(y: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    let n = y.len();
    let mut grad = Array1::zeros(n);

    if n == 0 {
        return grad;
    }

    if n == 1 {
        grad[0] = 0.0;
        return grad;
    }

    // Forward difference at start
    grad[0] = (y[1] - y[0]) / (x[1] - x[0]);

    // Central difference in middle
    for i in 1..n - 1 {
        grad[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
    }

    // Backward difference at end
    grad[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);

    grad
}

fn interp(x: &Array1<f64>, y: &Array1<f64>, x_new: f64) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return y[0];
    }

    // Find the bracketing interval
    if x_new <= x[0] {
        return y[0];
    }
    if x_new >= x[n - 1] {
        return y[n - 1];
    }

    // Binary search for the interval
    let mut left = 0;
    let mut right = n - 1;
    while right - left > 1 {
        let mid = (left + right) / 2;
        if x[mid] <= x_new {
            left = mid;
        } else {
            right = mid;
        }
    }

    // Linear interpolation
    let t = (x_new - x[left]) / (x[right] - x[left]);
    y[left] + t * (y[right] - y[left])
}
