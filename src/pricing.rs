use crate::density::Density;
use crate::order_stats::{expected_payoff_with_multiplicity, winner_of_many};
use ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const EPS: f64 = 1e-18;
const DEL: f64 = 1e-12;

// ---- Tie handling strategies ----

/// Trait for tie handling models
#[pyclass(subclass)]
pub struct TieModel {}

#[pymethods]
impl TieModel {
    #[new]
    fn new() -> Self {
        TieModel {}
    }

    fn draw_payoff(&self, _multiplicity_rest_at_k: f64) -> PyResult<f64> {
        Err(PyValueError::new_err("draw_payoff not implemented"))
    }
}

/// Half-point tie model (default)
#[pyclass(extends=TieModel)]
pub struct HalfPointTie {}

#[pymethods]
impl HalfPointTie {
    #[new]
    fn new() -> (Self, TieModel) {
        (HalfPointTie {}, TieModel {})
    }

    fn draw_payoff(&self, _multiplicity_rest_at_k: f64) -> f64 {
        0.5
    }
}

// ---- Core pricing primitives ----

/// Convert PDF to CDF by cumulative sum with clamping to [0,1] monotone non-decreasing
fn cdf_from_pdf(pdf: &Array1<f64>) -> Array1<f64> {
    let mut c = Array1::zeros(pdf.len());
    let mut sum: f64 = 0.0;
    for i in 0..pdf.len() {
        sum += pdf[i];
        sum = sum.max(if i > 0 { c[i - 1] } else { 0.0 }).min(1.0);
        c[i] = sum;
    }
    c.mapv_inplace(|x| x.clamp(0.0, 1.0));
    c
}

/// Convert CDF to PDF via diff (prepend 0 then diff)
fn pdf_from_cdf(cdf: &Array1<f64>) -> Array1<f64> {
    let mut pdf = Array1::zeros(cdf.len());
    pdf[0] = cdf[0];
    for i in 1..cdf.len() {
        pdf[i] = cdf[i] - cdf[i - 1];
    }
    pdf
}

/// CDF of the minimum of independent contestants
#[pyfunction]
pub fn cdf_min<'py>(
    py: Python<'py>,
    densities: Vec<Py<Density>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if densities.is_empty() {
        return Err(PyValueError::new_err("cdf_min requires at least one density"));
    }

    let first = densities[0].borrow(py);
    let size = first.p_array().len();
    drop(first);

    let mut prod_s = Array1::ones(size);

    for d_py in densities.iter() {
        let d = d_py.borrow(py);
        let p = d.p_array();
        let c = cdf_from_pdf(&p);
        let s = 1.0 - &c;
        prod_s = prod_s * &s;
    }

    let result: Array1<f64> = 1.0 - &prod_s;
    Ok(result.to_pyarray(py))
}

/// Return density of the minimum (winner) over the group
#[pyfunction]
pub fn winner_of_many_simple(py: Python<'_>, densities: Vec<Py<Density>>) -> PyResult<Py<Density>> {
    if densities.is_empty() {
        return Err(PyValueError::new_err(
            "winner_of_many requires at least one density",
        ));
    }

    let first = densities[0].borrow(py);
    let lattice = first.lattice_ref(py);
    let size = first.p_array().len();
    drop(first);

    let mut prod_s = Array1::ones(size);

    for d_py in densities.iter() {
        let d = d_py.borrow(py);
        let p = d.p_array();
        let c = cdf_from_pdf(&p);
        let s = 1.0 - &c;
        prod_s = prod_s * &s;
    }

    let c = 1.0 - &prod_s;
    let p = pdf_from_cdf(&c);

    Density::new_internal(py, lattice, p)
        .and_then(|d| Py::new(py, d))
}

/// CDF of min of 'rest' contestants, computed from all vs self
#[pyfunction]
pub fn rest_min_cdf<'py>(
    py: Python<'py>,
    all_min_cdf: PyReadonlyArray1<'py, f64>,
    self_cdf: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let all_cdf = all_min_cdf.to_owned_array();
    let s_cdf = self_cdf.to_owned_array();

    let s_all = 1.0 - &all_cdf;
    let s_self = 1.0 - &s_cdf;
    let s_rest = (&s_all + EPS) / (&s_self + DEL);
    let mut c_rest = 1.0 - &s_rest;

    // Apply maximum.accumulate and minimum with 1.0
    for i in 1..c_rest.len() {
        c_rest[i] = c_rest[i].max(c_rest[i - 1]).min(1.0);
    }
    for i in 0..c_rest.len() {
        c_rest[i] = c_rest[i].min(1.0);
    }

    c_rest.to_pyarray(py)
}

/// Per-lattice probabilities of win/draw/loss against 'rest min'
#[pyfunction]
pub fn conditional_win_draw_loss<'py>(
    py: Python<'py>,
    self_d: &Density,
    rest_cdf: PyReadonlyArray1<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let pdf_a = self_d.p_array();
    let cdf_a = cdf_from_pdf(&pdf_a);
    let rest_c = rest_cdf.to_owned_array();
    let pdf_rest = pdf_from_cdf(&rest_c);

    let win = &pdf_a * &(1.0 - &rest_c);
    let draw = &pdf_a * &pdf_rest;
    let loss = &pdf_rest * &(1.0 - &cdf_a);

    Ok((
        win.to_pyarray(py),
        draw.to_pyarray(py),
        loss.to_pyarray(py),
    ))
}

/// Expected payoff vs rest
#[pyfunction]
#[pyo3(signature = (self_d, all_min_cdf, tie_model=None))]
pub fn expected_payoff_vs_rest<'py>(
    py: Python<'py>,
    self_d: &Density,
    all_min_cdf: PyReadonlyArray1<'py, f64>,
    tie_model: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let pdf_a = self_d.p_array();
    let cdf_a = cdf_from_pdf(&pdf_a);
    let cdf_a_py = cdf_a.to_pyarray(py);

    let rest_c_py = rest_min_cdf(py, all_min_cdf, cdf_a_py.readonly());
    let rest_c = rest_c_py.readonly();
    let (win_py, draw_py, _) = conditional_win_draw_loss(py, self_d, rest_c)?;

    let draw_payoff = if let Some(tm) = tie_model {
        tm.call_method1("draw_payoff", (1.0,))?.extract::<f64>()?
    } else {
        0.5
    };

    let win = win_py.readonly().to_owned_array();
    let draw = draw_py.readonly().to_owned_array();
    let result = win + draw_payoff * &draw;

    Ok(result.to_pyarray(py))
}

// ---- Race class ----

/// A race between multiple contestants
#[pyclass]
pub struct Race {
    #[pyo3(get)]
    densities: Vec<Py<Density>>,
    tie_model: Py<PyAny>,
}

#[pymethods]
impl Race {
    #[new]
    #[pyo3(signature = (densities, tie_model=None))]
    pub fn new(py: Python<'_>, densities: Vec<Py<Density>>, tie_model: Option<Py<PyAny>>) -> PyResult<Self> {
        if densities.is_empty() {
            return Err(PyValueError::new_err("Race requires at least one density"));
        }

        // Validate all densities have the same lattice
        let first = densities[0].borrow(py);
        let first_l = first.lattice_l(py);
        let first_unit = first.lattice.borrow(py).unit_value();
        drop(first);

        for d_py in densities.iter().skip(1) {
            let d = d_py.borrow(py);
            let l = d.lattice_l(py);
            let unit = d.lattice.borrow(py).unit_value();
            if l != first_l || (unit - first_unit).abs() > 1e-10 {
                return Err(PyValueError::new_err(
                    "All densities must share the same lattice",
                ));
            }
        }

        let tm = if let Some(tm) = tie_model {
            tm
        } else {
            // Create default HalfPointTie
            Py::new(py, HalfPointTie::new())?.into_any()
        };

        Ok(Race {
            densities,
            tie_model: tm,
        })
    }

    /// Get the winner density
    fn winner_density(&self, py: Python<'_>) -> PyResult<Py<Density>> {
        let densities_copy: Vec<Py<Density>> = self.densities.iter().map(|d| d.clone_ref(py)).collect();
        let (winner, _) = winner_of_many(py, densities_copy)?;
        Ok(winner)
    }

    /// Risk-neutral winning probabilities for each contestant (multiplicity-aware)
    pub fn state_prices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let densities_copy: Vec<Py<Density>> = self.densities.iter().map(|d| d.clone_ref(py)).collect();
        let (density_all, mult_all) = winner_of_many(py, densities_copy)?;
        let density_all_borrowed = density_all.borrow(py);
        let cdf_all = cdf_from_pdf(&density_all_borrowed.p_array());
        let cdf_all_py = cdf_all.to_pyarray(py);
        drop(density_all_borrowed);

        let mut prices = Vec::new();

        for d_py in self.densities.iter() {
            let d = d_py.borrow(py);
            let density_all_borrowed = density_all.borrow(py);
            let ep_py = expected_payoff_with_multiplicity(
                py,
                &d,
                &density_all_borrowed,
                mult_all.readonly(),
                None,
                Some(cdf_all_py.readonly()),
            )?;
            let ep = ep_py.readonly();
            let sum: f64 = ep.as_array().sum();
            prices.push(sum);
        }

        let mut p = Array1::from_vec(prices);
        let s = p.sum();
        if s > 0.0 {
            p = p / s;
        }

        Ok(p.to_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!("Race(densities=[{} contestants])", self.densities.len())
    }
}

// ---- StatePricer class ----

/// State pricing utilities
#[pyclass]
pub struct StatePricer {}

#[pymethods]
impl StatePricer {
    #[new]
    fn new() -> Self {
        StatePricer {}
    }

    /// Convert dividends to prices
    #[staticmethod]
    #[pyo3(signature = (dividends, nan_value=2000.0))]
    pub fn prices_from_dividends<'py>(
        py: Python<'py>,
        dividends: Vec<Option<f64>>,
        nan_value: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let mut inv = Vec::new();

        for x in dividends.iter() {
            let v = match x {
                None => nan_value,
                Some(val) => {
                    if val.is_nan() {
                        nan_value
                    } else {
                        *val
                    }
                }
            };
            inv.push(if v <= 0.0 { 0.0 } else { 1.0 / v });
        }

        let mut p = Array1::from_vec(inv);
        let s = p.sum();
        if s > 0.0 {
            p = p / s;
        }

        p.to_pyarray(py)
    }

    /// Convert prices to dividends
    #[staticmethod]
    #[pyo3(signature = (prices, multiplicity=1.0))]
    fn dividends_from_prices<'py>(
        py: Python<'py>,
        prices: PyReadonlyArray1<'py, f64>,
        multiplicity: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let prices_arr = prices.to_owned_array();
        let s = prices_arr.sum();
        let p = if s > 0.0 {
            &prices_arr / s
        } else {
            prices_arr.clone()
        };

        let mut out = Array1::from_elem(p.len(), f64::NAN);
        for (i, &p_val) in p.iter().enumerate() {
            if p_val > 0.0 {
                out[i] = 1.0 / (multiplicity * p_val);
            }
        }

        out.to_pyarray(py)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdf_from_pdf() {
        let pdf = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let cdf = cdf_from_pdf(&pdf);
        assert_eq!(cdf.len(), 4);
        assert!((cdf[0] - 0.1).abs() < 1e-10);
        assert!((cdf[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pdf_from_cdf() {
        let cdf = Array1::from_vec(vec![0.1, 0.3, 0.6, 1.0]);
        let pdf = pdf_from_cdf(&cdf);
        assert_eq!(pdf.len(), 4);
        assert!((pdf[0] - 0.1).abs() < 1e-10);
        assert!((pdf[1] - 0.2).abs() < 1e-10);
        assert!((pdf[2] - 0.3).abs() < 1e-10);
        assert!((pdf[3] - 0.4).abs() < 1e-10);
    }
}
