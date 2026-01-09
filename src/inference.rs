use crate::clustering::ClusterSplitter;
use crate::density::Density;
use crate::order_stats::{expected_payoff_with_multiplicity, winner_of_many};
use crate::pricing::{Race, StatePricer};
use ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---- Helper functions ----

/// Create densities by shifting base density by offsets
fn densities_from_offsets(
    py: Python<'_>,
    base: &Density,
    offsets: &[f64],
) -> PyResult<Vec<Py<Density>>> {
    let mut out = Vec::new();
    for &o in offsets {
        let shifted = base.shift_fractional(py, o)?;
        out.push(Py::new(py, shifted)?);
    }
    Ok(out)
}

/// Convert densities to state prices
fn state_prices_from_densities(
    py: Python<'_>,
    densities: Vec<Py<Density>>,
) -> PyResult<Vec<f64>> {
    let race = Race::new(py, densities, None)?;
    let prices_arr = race.state_prices(py)?;
    let prices_ro = prices_arr.readonly();
    Ok(prices_ro.as_array().to_vec())
}

/// Compute state prices from base density and offsets
#[pyfunction]
pub fn implicit_state_prices<'py>(
    py: Python<'py>,
    base: &Density,
    offsets: Vec<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let dens = densities_from_offsets(py, base, &offsets)?;
    let prices = state_prices_from_densities(py, dens)?;
    Ok(Array1::from_vec(prices).to_pyarray(py))
}

// ---- Calibration structures ----

/// Cached lookup curves for 1D calibration
#[pyclass]
pub struct LookupCurve1D {
    #[pyo3(get)]
    locs: Py<PyArray1<f64>>,
    #[pyo3(get)]
    prices: Py<PyArray1<f64>>,
}

impl LookupCurve1D {
    /// Accessor for locs (for internal Rust use)
    pub fn locs(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.locs.clone_ref(py)
    }

    /// Accessor for prices (for internal Rust use)
    pub fn prices(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.prices.clone_ref(py)
    }
}

/// Cached lookup curves for 1D inverse
#[pyclass]
pub struct LookupCurve1DInverse {
    #[pyo3(get)]
    prices: Py<PyArray1<f64>>,
    #[pyo3(get)]
    offsets: Py<PyArray1<f64>>,
}

/// Ability calibrator for converting between prices and abilities
#[pyclass]
pub struct AbilityCalibrator {
    #[pyo3(get)]
    base: Py<Density>,

    #[pyo3(get)]
    offset_grid: Vec<f64>,

    #[pyo3(get, set)]
    n_iter: usize,

    scales: Option<Py<PyArray1<f64>>>,

    #[pyo3(get, set)]
    scale_span: f64,

    #[pyo3(get, set)]
    scale_steps: usize,

    #[pyo3(get, set)]
    loc_span: f64,

    #[pyo3(get, set)]
    loc_step: f64,

    #[pyo3(get, set)]
    skew_a: f64,

    // Cached curves
    lookup_curve_1d_prices: Option<Py<LookupCurve1D>>,
    lookup_curve_1d_inverse: Option<Py<LookupCurve1DInverse>>,
    lookup_curves_2d_prices: HashMap<String, (Py<PyArray1<f64>>, Py<PyArray1<f64>>)>,
    lookup_curves_2d_inverse: HashMap<String, (Py<PyArray1<f64>>, Py<PyArray1<f64>>)>,
}

#[pymethods]
impl AbilityCalibrator {
    #[new]
    #[pyo3(signature = (
        base,
        offset_grid=None,
        n_iter=3,
        scales=None,
        scale_span=0.5,
        scale_steps=3,
        loc_span=5.0,
        loc_step=0.25,
        skew_a=0.0
    ))]
    fn new(
        py: Python<'_>,
        base: Py<Density>,
        offset_grid: Option<Vec<f64>>,
        n_iter: usize,
        scales: Option<Py<PyArray1<f64>>>,
        scale_span: f64,
        scale_steps: usize,
        loc_span: f64,
        loc_step: f64,
        skew_a: f64,
    ) -> PyResult<Self> {
        let offset_grid = if let Some(grid) = offset_grid {
            grid
        } else {
            // Create default grid from base lattice L
            let base_ref = base.borrow(py);
            let l = base_ref.lattice_l(py);
            let l_val = l as i32;
            let mut grid: Vec<f64> = ((-l_val / 2)..(l_val / 2))
                .map(|i| i as f64)
                .collect();
            grid.reverse();
            grid
        };

        Ok(AbilityCalibrator {
            base,
            offset_grid,
            n_iter,
            scales,
            scale_span,
            scale_steps,
            loc_span,
            loc_step,
            skew_a,
            lookup_curve_1d_prices: None,
            lookup_curve_1d_inverse: None,
            lookup_curves_2d_prices: HashMap::new(),
            lookup_curves_2d_inverse: HashMap::new(),
        })
    }

    /// Set scales array for 2D calibration
    fn set_scales(&mut self, py: Python<'_>, scales: PyReadonlyArray1<f64>) {
        self.scales = Some(scales.as_array().to_owned().to_pyarray(py).unbind());
    }

    /// Create density for given location and scale
    fn density_for(&self, py: Python<'_>, loc: f64, scale: f64) -> PyResult<Py<Density>> {
        let base_ref = self.base.borrow(py);
        let lattice = self.base.borrow(py).lattice.clone_ref(py);
        drop(base_ref);

        let density = Density::skew_normal(py, lattice, loc, scale, self.skew_a)?;
        Py::new(py, density)
    }

    /// Rebuild 1D lookup curves from field locations
    pub fn rebuild_curves_from_field_1d(&mut self, py: Python<'_>, locs: Vec<f64>) -> PyResult<()> {
        let base_ref = self.base.borrow(py);
        let unit = base_ref.lattice.borrow(py).unit_value();
        drop(base_ref);

        // Convert physical locs to lattice steps
        let offsets: Vec<f64> = locs.iter().map(|&loc| loc / unit).collect();
        let grid = &self.offset_grid;

        // Build current field densities with multiplicity
        let current_densities = densities_from_offsets(py, &self.base.borrow(py), &offsets)?;
        let (density_all, mult_all) = winner_of_many(py, current_densities)?;
        let cdf_all = density_all.borrow(py).cdf(py);

        // Build global implicit curve p(g)
        let mut implied_prices = Vec::new();
        for &g in grid {
            let d_g = self.base.borrow(py).shift_fractional(py, g)?;
            let payoff_vec = expected_payoff_with_multiplicity(
                py,
                &d_g,
                &density_all.borrow(py),
                mult_all.readonly(),
                None,
                Some(cdf_all.readonly()),
            )?;
            let payoff_sum: f64 = payoff_vec.readonly().as_array().sum();
            implied_prices.push(payoff_sum);
        }

        let implied_prices_arr = Array1::from_vec(implied_prices);

        // Cache price curve in physical units (ascending by loc)
        let locs_phys: Vec<f64> = grid.iter().map(|&g| unit * g).collect();
        let locs_phys_arr = Array1::from_vec(locs_phys);

        let mut indices: Vec<usize> = (0..locs_phys_arr.len()).collect();
        indices.sort_by(|&i, &j| locs_phys_arr[i].partial_cmp(&locs_phys_arr[j]).unwrap());

        let locs_sorted: Vec<f64> = indices.iter().map(|&i| locs_phys_arr[i]).collect();
        let prices_sorted: Vec<f64> = indices.iter().map(|&i| implied_prices_arr[i]).collect();

        let curve_1d = LookupCurve1D {
            locs: Array1::from_vec(locs_sorted).to_pyarray(py).unbind(),
            prices: Array1::from_vec(prices_sorted).to_pyarray(py).unbind(),
        };
        self.lookup_curve_1d_prices = Some(Py::new(py, curve_1d)?);

        // Cache inverse curve (price -> offset steps), monotone in price
        let mut pairs: Vec<(f64, f64)> = implied_prices_arr
            .iter()
            .zip(grid.iter())
            .map(|(&p, &g)| (p, g))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let xp: Vec<f64> = pairs.iter().map(|(p, _)| *p).collect();
        let fp: Vec<f64> = pairs.iter().map(|(_, g)| *g).collect();

        // Get unique prices
        let (xp_unique, fp_unique) = unique_pairs(&xp, &fp);

        let curve_1d_inv = LookupCurve1DInverse {
            prices: Array1::from_vec(xp_unique).to_pyarray(py).unbind(),
            offsets: Array1::from_vec(fp_unique).to_pyarray(py).unbind(),
        };
        self.lookup_curve_1d_inverse = Some(Py::new(py, curve_1d_inv)?);

        Ok(())
    }

    /// Rebuild 2D lookup curves from field locations and scales
    pub fn rebuild_curves_from_field_2d(
        &mut self,
        py: Python<'_>,
        locs: Vec<f64>,
        scales: Vec<f64>,
    ) -> PyResult<()> {
        let n = locs.len();

        // Build current field densities
        let mut current_densities = Vec::new();
        for j in 0..n {
            let d = self.density_for(py, locs[j], scales[j])?;
            current_densities.push(d);
        }

        let (density_all, mult_all) = winner_of_many(py, current_densities)?;
        let cdf_all = density_all.borrow(py).cdf(py);

        // Location grid (physical units)
        let loc_grid_vec: Vec<f64> = arange(-self.loc_span, self.loc_span + self.loc_step, self.loc_step);
        let loc_grid = Array1::from_vec(loc_grid_vec.clone());

        // Use unique scales present
        let mut unique_s_values: Vec<f64> = scales.iter().copied().collect();
        unique_s_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_s_values.dedup();

        self.lookup_curves_2d_prices.clear();
        self.lookup_curves_2d_inverse.clear();

        for s in unique_s_values {
            let mut p_curve = Vec::new();
            for &loc_candidate in &loc_grid_vec {
                let d_i_candidate = self.density_for(py, loc_candidate, s)?;
                let payoff_vec = expected_payoff_with_multiplicity(
                    py,
                    &d_i_candidate.borrow(py),
                    &density_all.borrow(py),
                    mult_all.readonly(),
                    None,
                    Some(cdf_all.readonly()),
                )?;
                let payoff_sum: f64 = payoff_vec.readonly().as_array().sum();
                p_curve.push(payoff_sum);
            }

            let p_curve_arr = Array1::from_vec(p_curve.clone());

            // Cache price curve per scale (loc -> price), sorted by loc
            let mut indices: Vec<usize> = (0..loc_grid.len()).collect();
            indices.sort_by(|&i, &j| loc_grid[i].partial_cmp(&loc_grid[j]).unwrap());

            let locs_sorted: Vec<f64> = indices.iter().map(|&i| loc_grid[i]).collect();
            let prices_sorted: Vec<f64> = indices.iter().map(|&i| p_curve_arr[i]).collect();

            let s_key = format!("{}", s);
            self.lookup_curves_2d_prices.insert(
                s_key.clone(),
                (
                    Array1::from_vec(locs_sorted).to_pyarray(py).unbind(),
                    Array1::from_vec(prices_sorted).to_pyarray(py).unbind(),
                ),
            );

            // Cache inverse (price -> loc)
            let mut pairs: Vec<(f64, f64)> = p_curve
                .iter()
                .zip(loc_grid_vec.iter())
                .map(|(&p, &l)| (p, l))
                .collect();
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let xp: Vec<f64> = pairs.iter().map(|(p, _)| *p).collect();
            let fp: Vec<f64> = pairs.iter().map(|(_, l)| *l).collect();

            let (xp_unique, fp_unique) = unique_pairs(&xp, &fp);

            self.lookup_curves_2d_inverse.insert(
                s_key,
                (
                    Array1::from_vec(xp_unique).to_pyarray(py).unbind(),
                    Array1::from_vec(fp_unique).to_pyarray(py).unbind(),
                ),
            );
        }

        Ok(())
    }

    /// Solve for abilities from state prices
    #[pyo3(signature = (prices, initial_offsets=None))]
    pub fn solve_from_prices<'py>(
        &mut self,
        py: Python<'py>,
        prices: Vec<f64>,
        initial_offsets: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let prices_arr = Array1::from_vec(prices.clone());
        let n = prices_arr.len();

        // Check if we should use 2D path
        let use_2d = if let Some(ref scales_py) = self.scales {
            let scales_arr = scales_py.bind(py).readonly();
            scales_arr.as_array().len() == n
        } else {
            false
        };

        if !use_2d {
            // ---- 1D global-curve inversion ----
            let base_ref = self.base.borrow(py);
            let unit = base_ref.lattice.borrow(py).unit_value();
            drop(base_ref);

            let initial_offsets = initial_offsets.unwrap_or_else(|| vec![0.0; n]);
            let mut offsets: Vec<f64> = initial_offsets.iter().map(|&o| o / unit).collect();
            let grid = &self.offset_grid;

            for _ in 0..self.n_iter {
                // Build densities from current offsets
                let current_densities = densities_from_offsets(py, &self.base.borrow(py), &offsets)?;
                let (density_all, mult_all) = winner_of_many(py, current_densities)?;
                let cdf_all = density_all.borrow(py).cdf(py);

                // Build implicit curve p(g)
                let mut implied_prices = Vec::new();
                for &g in grid.iter() {
                    let d_g = self.base.borrow(py).shift_fractional(py, g)?;
                    let payoff_vec = expected_payoff_with_multiplicity(
                        py,
                        &d_g,
                        &density_all.borrow(py),
                        mult_all.readonly(),
                        None,
                        Some(cdf_all.readonly()),
                    )?;
                    let payoff_sum: f64 = payoff_vec.readonly().as_array().sum();
                    implied_prices.push(payoff_sum);
                }

                let implied_prices_arr = Array1::from_vec(implied_prices.clone());

                // Cache price curve
                let locs_phys: Vec<f64> = grid.iter().map(|&g| unit * g).collect();
                let locs_phys_arr = Array1::from_vec(locs_phys.clone());

                let mut indices: Vec<usize> = (0..locs_phys_arr.len()).collect();
                indices.sort_by(|&i, &j| locs_phys_arr[i].partial_cmp(&locs_phys_arr[j]).unwrap());

                let locs_sorted: Vec<f64> = indices.iter().map(|&i| locs_phys_arr[i]).collect();
                let prices_sorted: Vec<f64> = indices.iter().map(|&i| implied_prices_arr[i]).collect();

                let curve_1d = LookupCurve1D {
                    locs: Array1::from_vec(locs_sorted).to_pyarray(py).unbind(),
                    prices: Array1::from_vec(prices_sorted).to_pyarray(py).unbind(),
                };
                self.lookup_curve_1d_prices = Some(Py::new(py, curve_1d)?);

                // Sort by price for monotonic inversion
                let mut pairs: Vec<(f64, f64)> = implied_prices
                    .iter()
                    .zip(grid.iter())
                    .map(|(&p, &g)| (p, g))
                    .collect();
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                let xp: Vec<f64> = pairs.iter().map(|(p, _)| *p).collect();
                let fp: Vec<f64> = pairs.iter().map(|(_, g)| *g).collect();

                let (xp_unique, fp_unique) = unique_pairs(&xp, &fp);

                let curve_1d_inv = LookupCurve1DInverse {
                    prices: Array1::from_vec(xp_unique.clone()).to_pyarray(py).unbind(),
                    offsets: Array1::from_vec(fp_unique.clone()).to_pyarray(py).unbind(),
                };
                self.lookup_curve_1d_inverse = Some(Py::new(py, curve_1d_inv)?);

                // Invert all prices
                let xp_arr = Array1::from_vec(xp_unique);
                let fp_arr = Array1::from_vec(fp_unique);

                let p_min = xp_arr.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let p_max = xp_arr.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                offsets = prices_arr
                    .iter()
                    .map(|&p| {
                        let p_clamped = p.max(p_min).min(p_max);
                        interp(p_clamped, &xp_arr, &fp_arr)
                    })
                    .collect();
            }

            // Return abilities in physical units
            let abilities: Vec<f64> = offsets.iter().map(|&o| o * unit).collect();
            Ok(Array1::from_vec(abilities).to_pyarray(py))
        } else {
            // ---- 2D calibration in (loc, scale) per runner ----
            let scales_py = self.scales.as_ref().unwrap();
            let scales_arr = scales_py.bind(py).readonly();
            let scales: Vec<f64> = scales_arr.as_array().to_vec();

            let base_ref = self.base.borrow(py);
            let unit = base_ref.lattice.borrow(py).unit_value();
            drop(base_ref);

            // Initialize locs (abilities) in physical units
            let mut locs: Vec<f64> = if let Some(init_offsets) = initial_offsets {
                init_offsets.iter().map(|&o| o * unit).collect()
            } else {
                vec![0.0; n]
            };

            // Location grid (physical units)
            let loc_grid_vec: Vec<f64> = arange(-self.loc_span, self.loc_span + self.loc_step, self.loc_step);

            for _ in 0..self.n_iter {
                // Precompute field distribution once per iteration
                let mut current_densities = Vec::new();
                for j in 0..n {
                    let d = self.density_for(py, locs[j], scales[j])?;
                    current_densities.push(d);
                }

                let (density_all, mult_all) = winner_of_many(py, current_densities)?;
                let cdf_all = density_all.borrow(py).cdf(py);

                // Build scale grids for each runner
                let mut per_runner_sg = Vec::new();
                let mut all_s_values = Vec::new();

                for &si in &scales {
                    let sg = self.scale_grid_for(si);
                    all_s_values.extend(sg.iter());
                    per_runner_sg.push(sg);
                }

                let mut unique_s_values: Vec<f64> = all_s_values;
                unique_s_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                unique_s_values.dedup();

                // Build curve for each unique scale
                let mut scale_cache: HashMap<String, (Array1<f64>, Array1<f64>)> = HashMap::new();

                for s in unique_s_values {
                    let mut p_curve = Vec::new();
                    for &loc_candidate in &loc_grid_vec {
                        let d_i_candidate = self.density_for(py, loc_candidate, s)?;
                        let payoff_vec = expected_payoff_with_multiplicity(
                            py,
                            &d_i_candidate.borrow(py),
                            &density_all.borrow(py),
                            mult_all.readonly(),
                            None,
                            Some(cdf_all.readonly()),
                        )?;
                        let payoff_sum: f64 = payoff_vec.readonly().as_array().sum();
                        p_curve.push(payoff_sum);
                    }

                    let p_curve_arr = Array1::from_vec(p_curve.clone());
                    let loc_grid_arr = Array1::from_vec(loc_grid_vec.clone());

                    // Cache price curve (loc -> price), sorted by loc
                    let mut indices: Vec<usize> = (0..loc_grid_arr.len()).collect();
                    indices.sort_by(|&i, &j| loc_grid_arr[i].partial_cmp(&loc_grid_arr[j]).unwrap());

                    let locs_sorted: Vec<f64> = indices.iter().map(|&i| loc_grid_arr[i]).collect();
                    let prices_sorted: Vec<f64> = indices.iter().map(|&i| p_curve_arr[i]).collect();

                    let s_key = format!("{}", s);
                    self.lookup_curves_2d_prices.insert(
                        s_key.clone(),
                        (
                            Array1::from_vec(locs_sorted).to_pyarray(py).unbind(),
                            Array1::from_vec(prices_sorted).to_pyarray(py).unbind(),
                        ),
                    );

                    // Sort by price for inversion
                    let mut pairs: Vec<(f64, f64)> = p_curve
                        .iter()
                        .zip(loc_grid_vec.iter())
                        .map(|(&p, &l)| (p, l))
                        .collect();
                    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                    let xp: Vec<f64> = pairs.iter().map(|(p, _)| *p).collect();
                    let fp: Vec<f64> = pairs.iter().map(|(_, l)| *l).collect();

                    let (xp_unique, fp_unique) = unique_pairs(&xp, &fp);

                    scale_cache.insert(s_key.clone(), (
                        Array1::from_vec(xp_unique.clone()),
                        Array1::from_vec(fp_unique.clone()),
                    ));

                    self.lookup_curves_2d_inverse.insert(
                        s_key,
                        (
                            Array1::from_vec(xp_unique).to_pyarray(py).unbind(),
                            Array1::from_vec(fp_unique).to_pyarray(py).unbind(),
                        ),
                    );
                }

                // Invert per runner using cached per-scale curves
                for i in 0..n {
                    let si = scales[i];
                    let pi_target = prices[i];
                    let sg = &per_runner_sg[i];

                    let mut loc_estimates = Vec::new();
                    for &s in sg {
                        let s_key = format!("{}", s);
                        let (xp_unique, fp_unique) = scale_cache.get(&s_key).unwrap();

                        let p_min = xp_unique.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let p_max = xp_unique.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let p_clamped = pi_target.max(p_min).min(p_max);

                        let loc_s = interp(p_clamped, xp_unique, fp_unique);
                        loc_estimates.push(loc_s);
                    }

                    let loc_estimates_arr = Array1::from_vec(loc_estimates);

                    if sg.len() == 1 {
                        locs[i] = loc_estimates_arr[0];
                    } else {
                        let sg_arr = Array1::from_vec(sg.clone());
                        locs[i] = interp(si, &sg_arr, &loc_estimates_arr);
                    }
                }
            }

            // Remove translation (median-centering)
            let mut locs_sorted = locs.clone();
            locs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = if locs_sorted.len() % 2 == 0 {
                let mid = locs_sorted.len() / 2;
                (locs_sorted[mid - 1] + locs_sorted[mid]) / 2.0
            } else {
                locs_sorted[locs_sorted.len() / 2]
            };

            let locs_centered: Vec<f64> = locs.iter().map(|&l| l - median).collect();
            Ok(Array1::from_vec(locs_centered).to_pyarray(py))
        }
    }

    /// Solve for abilities from dividends
    #[pyo3(signature = (dividends, nan_value=2000.0))]
    fn solve_from_dividends<'py>(
        &mut self,
        py: Python<'py>,
        dividends: Vec<Option<f64>>,
        nan_value: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let prices_arr = StatePricer::prices_from_dividends(py, dividends, nan_value);
        let prices: Vec<f64> = prices_arr.readonly().as_array().to_vec();
        self.solve_from_prices(py, prices, None)
    }

    /// Forward direction: abilities to state prices
    fn state_prices_from_ability<'py>(
        &self,
        py: Python<'py>,
        ability: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let base_ref = self.base.borrow(py);
        let unit = base_ref.lattice.borrow(py).unit_value();
        drop(base_ref);

        let offsets: Vec<f64> = ability.iter().map(|&a| a / unit).collect();

        let splitter = ClusterSplitter::new(3.0, 3);
        splitter.extended_state_prices(py, &self.base.borrow(py), offsets)
    }

    /// Forward direction: abilities to dividends
    #[pyo3(signature = (ability, multiplicity=1.0))]
    fn dividends_from_ability<'py>(
        &self,
        py: Python<'py>,
        ability: Vec<f64>,
        multiplicity: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let prices = self.state_prices_from_ability(py, ability)?;
        let prices_vec: Vec<f64> = prices.readonly().as_array().to_vec();

        let dividends: Vec<f64> = prices_vec
            .iter()
            .map(|&p| 1.0 / (multiplicity * p))
            .collect();

        Ok(Array1::from_vec(dividends).to_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "AbilityCalibrator(n_iter={}, scale_steps={}, loc_span={}, loc_step={})",
            self.n_iter, self.scale_steps, self.loc_span, self.loc_step
        )
    }

    /// Get lookup_curve_1d_prices (for Python access)
    #[getter]
    fn get_lookup_curve_1d_prices(&self, py: Python<'_>) -> Option<Py<LookupCurve1D>> {
        self.lookup_curve_1d_prices.as_ref().map(|c| c.clone_ref(py))
    }

    /// Get lookup_curves_2d_prices (for Python access)
    #[getter]
    fn get_lookup_curves_2d_prices(&self, py: Python<'_>) -> HashMap<String, (Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        self.lookup_curves_2d_prices.iter()
            .map(|(k, (v1, v2))| (k.clone(), (v1.clone_ref(py), v2.clone_ref(py))))
            .collect()
    }
}

impl AbilityCalibrator {
    /// Generate scale grid for a given scale value
    fn scale_grid_for(&self, si: f64) -> Vec<f64> {
        if self.scale_steps <= 1 {
            return vec![si.max(1e-6)];
        }

        let n = self.scale_steps;
        let mut sg = Vec::new();
        for i in 0..n {
            let t = -self.scale_span + (2.0 * self.scale_span * i as f64) / ((n - 1) as f64);
            sg.push((si + t).max(1e-6));
        }
        sg
    }

    /// Accessor for lookup_curve_1d_prices (for internal Rust use)
    pub fn lookup_curve_1d_prices(&self, py: Python<'_>) -> Option<Py<LookupCurve1D>> {
        self.lookup_curve_1d_prices.as_ref().map(|c| c.clone_ref(py))
    }

    /// Accessor for lookup_curves_2d_prices (for internal Rust use)
    pub fn lookup_curves_2d_prices(&self, py: Python<'_>) -> HashMap<String, (Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        self.lookup_curves_2d_prices.iter()
            .map(|(k, (v1, v2))| (k.clone(), (v1.clone_ref(py), v2.clone_ref(py))))
            .collect()
    }
}

// ---- Helper functions ----

/// Create range of values similar to numpy.arange
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let mut result = Vec::new();
    let mut current = start;
    while current < stop {
        result.push(current);
        current += step;
    }
    result
}

/// Linear interpolation (similar to numpy.interp)
fn interp(x: f64, xp: &Array1<f64>, fp: &Array1<f64>) -> f64 {
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

/// Get unique pairs (x, y) by unique x values
fn unique_pairs(xp: &[f64], fp: &[f64]) -> (Vec<f64>, Vec<f64>) {
    if xp.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut result_x = Vec::new();
    let mut result_y = Vec::new();

    result_x.push(xp[0]);
    result_y.push(fp[0]);

    for i in 1..xp.len() {
        if xp[i] != *result_x.last().unwrap() {
            result_x.push(xp[i]);
            result_y.push(fp[i]);
        }
    }

    (result_x, result_y)
}
