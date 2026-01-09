use crate::density::Density;
use ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const EPS: f64 = 1e-18;
const DEL: f64 = 1e-12;

/// Compute conditional win/draw/loss probabilities
fn conditional_win_draw_loss(
    pdf_a: &Array1<f64>,
    pdf_b: &Array1<f64>,
    cdf_a: &Array1<f64>,
    cdf_b: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let win_a = pdf_a * &(1.0 - cdf_b); // X < Y
    let draw = pdf_a * pdf_b; // X == Y
    let win_b = pdf_b * &(1.0 - cdf_a); // Y < X
    (win_a, draw, win_b)
}

/// Convert CDF to PDF via diff (prepend 0 then diff)
fn _pdf_from_cdf(cdf: &Array1<f64>) -> Array1<f64> {
    let mut pdf = Array1::zeros(cdf.len());
    pdf[0] = cdf[0];
    for i in 1..cdf.len() {
        pdf[i] = cdf[i] - cdf[i - 1];
    }
    pdf
}

/// Convert PDF to CDF by cumulative sum with clamping to [0,1] monotone non-decreasing
fn _cdf_from_pdf(pdf: &Array1<f64>) -> Array1<f64> {
    let mut c = Array1::zeros(pdf.len());
    let mut sum: f64 = 0.0;
    for i in 0..pdf.len() {
        sum += pdf[i];
        // clamp to [0,1] monotone non-decreasing
        sum = sum.max(if i > 0 { c[i - 1] } else { 0.0 }).min(1.0);
        c[i] = sum;
    }

    c.mapv_inplace(|x| x.clamp(0.0, 1.0));
    c
}

/// Internal function to compute winner of two densities
fn winner_of_two_internal(
    d_a: &Density,
    d_b: &Density,
    py: Python<'_>,
    mult_a: Option<&Array1<f64>>,
    mult_b: Option<&Array1<f64>>,
) -> PyResult<(Density, Array1<f64>)> {
    let p_a = d_a.p_array();
    let p_b = d_b.p_array();
    let c_a = _cdf_from_pdf(&p_a);
    let c_b = _cdf_from_pdf(&p_b);

    let c_min = 1.0 - &((1.0 - &c_a) * &(1.0 - &c_b));
    let pdf_min = _pdf_from_cdf(&c_min);

    let lattice = d_a.lattice_ref(py);
    let out = Density::new_internal(py, lattice, pdf_min)?;

    let l = d_a.lattice_l(py);
    let size = 2 * l + 1;

    let mult_a_arr = match mult_a {
        Some(arr) => arr.clone(),
        None => Array1::ones(size),
    };

    let mult_b_arr = match mult_b {
        Some(arr) => arr.clone(),
        None => Array1::ones(size),
    };

    let (w_a, dr, w_b) = conditional_win_draw_loss(&p_a, &p_b, &c_a, &c_b);
    let numer = &w_a * &mult_a_arr + &dr * &(&mult_a_arr + &mult_b_arr) + &w_b * &mult_b_arr + EPS;
    let denom = w_a + dr + w_b + EPS;
    let mult = numer / denom;

    Ok((out, mult))
}

/// Compute the winner distribution of many densities
#[pyfunction]
#[pyo3(signature = (densities))]
pub fn winner_of_many<'py>(
    py: Python<'py>,
    densities: Vec<Py<Density>>,
) -> PyResult<(Py<Density>, Bound<'py, PyArray1<f64>>)> {
    if densities.is_empty() {
        return Err(PyValueError::new_err(
            "winner_of_many requires at least one density.",
        ));
    }

    let first = densities[0].borrow(py);
    let l = first.lattice_l(py);
    let size = 2 * l + 1;
    drop(first);

    let mut d = densities[0].clone_ref(py);
    let mut mult = Array1::ones(size);

    for d2_py in densities.iter().skip(1) {
        let d_borrowed = d.borrow(py);
        let d2_borrowed = d2_py.borrow(py);
        let ones = Array1::ones(size);
        let (new_d, new_mult) =
            winner_of_two_internal(&d_borrowed, &d2_borrowed, py, Some(&mult), Some(&ones))?;
        drop(d_borrowed);
        drop(d2_borrowed);

        d = Py::new(py, new_d)?;
        mult = new_mult;
    }

    Ok((d, mult.to_pyarray(py)))
}

/// Compute the "rest" distribution (all except the given density)
#[pyfunction]
#[pyo3(signature = (density, density_all, multiplicity_all, cdf=None, cdf_all=None))]
pub fn get_the_rest<'py>(
    py: Python<'py>,
    density: &Density,
    density_all: Option<&Density>,
    multiplicity_all: PyReadonlyArray1<'py, f64>,
    cdf: Option<PyReadonlyArray1<'py, f64>>,
    cdf_all: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let cdf_arr = match cdf {
        Some(c) => c.to_owned_array(),
        None => {
            let p = density.p_array();
            _cdf_from_pdf(&p)
        }
    };

    let cdf_all_arr = match cdf_all {
        Some(c) => c.to_owned_array(),
        None => {
            if density_all.is_none() {
                return Err(PyValueError::new_err("Need densityAll or cdfAll."));
            }
            let p_all = density_all.unwrap().p_array();
            _cdf_from_pdf(&p_all)
        }
    };

    let pdf = _pdf_from_cdf(&cdf_arr);

    let s_all = 1.0 - &cdf_all_arr;
    let s_self = 1.0 - &cdf_arr;
    let s_rest = (&s_all + EPS) / (&s_self + DEL);
    let mut cdf_rest = 1.0 - &s_rest;

    // Apply maximum.accumulate
    for i in 1..cdf_rest.len() {
        if cdf_rest[i] < cdf_rest[i - 1] {
            cdf_rest[i] = cdf_rest[i - 1];
        }
    }

    let pdf_rest = _pdf_from_cdf(&cdf_rest);

    let m = multiplicity_all.as_array().to_owned();
    let f1 = &pdf;
    let m1 = 1.0;

    let numer =
        &m * f1 * &s_rest + &m * &(f1 + &s_self) * &pdf_rest - m1 * f1 * &(&s_rest + &pdf_rest);
    let denom = &pdf_rest * &(f1 + &s_self) + EPS;
    let mult_left = (EPS + numer) / denom;

    let t1 = (&s_self + EPS) / (f1 + DEL);
    let t_rest = (&s_rest + EPS) / (&pdf_rest + DEL);
    let mult_right = &m * &t_rest / (1.0 + &t1) + &m - m1 * &(1.0 + t_rest) / (1.0 + t1);

    // Find argmax of f1
    let mut k = 0;
    let mut max_val = f1[0];
    for (i, &val) in f1.iter().enumerate() {
        if val > max_val {
            max_val = val;
            k = i;
        }
    }

    let mut mult = mult_left.clone();
    for i in k..mult.len() {
        mult[i] = mult_right[i];
    }

    // Enforce non-negativity
    mult.mapv_inplace(|x| x.max(0.0));

    Ok((cdf_rest.to_pyarray(py), mult.to_pyarray(py)))
}

/// Compute expected payoff with multiplicity
#[pyfunction]
#[pyo3(signature = (density, density_all, multiplicity_all, cdf=None, cdfAll=None))]
pub fn expected_payoff_with_multiplicity<'py>(
    py: Python<'py>,
    density: &Density,
    density_all: &Density,
    multiplicity_all: PyReadonlyArray1<'py, f64>,
    cdf: Option<PyReadonlyArray1<'py, f64>>,
    cdfAll: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let pdf = match &cdf {
        Some(c) => _pdf_from_cdf(&c.to_owned_array()),
        None => density.p_array(),
    };

    let (c_rest_py, m_rest_py) = get_the_rest(
        py,
        density,
        Some(density_all),
        multiplicity_all,
        cdf,
        cdfAll,
    )?;

    let c_rest = c_rest_py.readonly();
    let m_rest = m_rest_py.readonly();

    let pdf_rest = _pdf_from_cdf(&c_rest.to_owned_array());
    let cdf_density = _cdf_from_pdf(&pdf);
    let c_rest_owned = c_rest.to_owned_array();

    let (win, draw, _) = conditional_win_draw_loss(&pdf, &pdf_rest, &cdf_density, &c_rest_owned);

    let m_rest_arr = m_rest.as_array();

    // Clamp small negative values to 0
    let m_rest_clamped = m_rest_arr.mapv(|x| x.max(0.0));

    // Check for non-finite values
    if !m_rest_clamped.iter().all(|&x| x.is_finite()) {
        return Err(PyValueError::new_err(
            "Multiplicity contains non-finite values.",
        ));
    }

    let result = win + &draw / &(1.0 + m_rest_clamped);

    Ok(result.to_pyarray(py))
}

impl Density {
    /// Get the PDF array as an owned Array1
    pub fn p_array(&self) -> Array1<f64> {
        self.p.clone()
    }

    /// Get the lattice reference
    pub fn lattice_ref(&self, py: Python<'_>) -> Py<crate::lattice::UniformLattice> {
        self.lattice.clone_ref(py)
    }

    /// Get the L value from the lattice
    pub fn lattice_l(&self, py: Python<'_>) -> usize {
        self.lattice.borrow(py).l()
    }

    /// Internal constructor from Array1
    pub fn new_internal(
        _py: Python<'_>,
        lattice: Py<crate::lattice::UniformLattice>,
        p: Array1<f64>,
    ) -> PyResult<Self> {
        use crate::density::normalize;
        let p_normalized = normalize(p)?;
        Ok(Density {
            lattice,
            p: p_normalized,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conditional_win_draw_loss() {
        let pdf_a = Array1::from_vec(vec![0.2, 0.5, 0.3]);
        let pdf_b = Array1::from_vec(vec![0.3, 0.4, 0.3]);
        let cdf_a = Array1::from_vec(vec![0.2, 0.7, 1.0]);
        let cdf_b = Array1::from_vec(vec![0.3, 0.7, 1.0]);

        let (win_a, draw, win_b) = conditional_win_draw_loss(&pdf_a, &pdf_b, &cdf_a, &cdf_b);

        assert_eq!(win_a.len(), 3);
        assert_eq!(draw.len(), 3);
        assert_eq!(win_b.len(), 3);
    }

    #[test]
    fn test_pdf_from_cdf() {
        let cdf = Array1::from_vec(vec![0.1, 0.3, 0.6, 1.0]);
        let pdf = _pdf_from_cdf(&cdf);
        assert_eq!(pdf.len(), 4);
        assert!((pdf[0] - 0.1).abs() < 1e-10);
        assert!((pdf[1] - 0.2).abs() < 1e-10);
        assert!((pdf[2] - 0.3).abs() < 1e-10);
        assert!((pdf[3] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_cdf_from_pdf() {
        let pdf = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let cdf = _cdf_from_pdf(&pdf);
        assert_eq!(cdf.len(), 4);
        assert!((cdf[0] - 0.1).abs() < 1e-10);
        assert!((cdf[3] - 1.0).abs() < 1e-10);
    }
}
