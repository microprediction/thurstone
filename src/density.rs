use crate::lattice::UniformLattice;
use crate::normaldist::{normcdf, normpdf};
use ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn cdf_from_pdf<'py>(
    py: Python<'py>,
    pdf: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    _cdf_from_pdf(&pdf.to_owned_array()).to_pyarray(py)
}

#[pyfunction]
pub fn pdf_from_cdf<'py>(
    py: Python<'py>,
    cdf: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    _pdf_from_cdf(&cdf.to_owned_array()).to_pyarray(py)
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

/// Convert CDF to PDF via diff (prepend 0 then diff)
fn _pdf_from_cdf(cdf: &Array1<f64>) -> Array1<f64> {
    let mut pdf = Array1::zeros(cdf.len());
    pdf[0] = cdf[0];
    for i in 1..cdf.len() {
        pdf[i] = cdf[i] - cdf[i - 1];
    }
    pdf
}

/// Normalize PDF to sum to 1.0 (or allow 0 for off-lattice case)
pub(crate) fn normalize(pdf: Array1<f64>) -> PyResult<Array1<f64>> {
    let s: f64 = pdf.sum();
    if s < 0.0 {
        return Err(PyValueError::new_err(
            "PDF has negative total mass, which is invalid.",
        ));
    }
    if s == 0.0 {
        // Allow zero-mass arrays to pass through (e.g., extreme shifts off the lattice)
        return Ok(pdf);
    }
    Ok(pdf / s)
}

/// Lattice-aligned measure. In the typical case sum(p) == 1, but we allow
/// sum(p) == 0 as a sentinel for 'off-lattice' runners in extreme offsets.
/// Negative total mass is always an error.
#[pyclass]
pub struct Density {
    #[pyo3(get)]
    pub lattice: Py<UniformLattice>,
    pub(crate) p: Array1<f64>, // shape (2L+1,), sum ~ 1
}

impl Clone for Density {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Density {
            lattice: self.lattice.clone_ref(py),
            p: self.p.clone(),
        })
    }
}

#[pymethods]
impl Density {
    #[new]
    fn new(
        py: Python<'_>,
        lattice: Py<UniformLattice>,
        p: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        let p_array = p.as_array().to_owned();

        // Assert compatibility
        let lattice_size = lattice.borrow(py).size_value();
        if p_array.len() != lattice_size {
            return Err(PyValueError::new_err(format!(
                "Array length {} incompatible with lattice size {}.",
                p_array.len(),
                lattice_size
            )));
        }

        let p_normalized = normalize(p_array)?;

        Ok(Density {
            lattice,
            p: p_normalized,
        })
    }

    /// Get the PDF array as a numpy array
    #[getter]
    fn p<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.p.to_pyarray(py)
    }

    // ---- statistics ----

    /// Compute the CDF from the PDF
    pub fn cdf<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        _cdf_from_pdf(&self.p).to_pyarray(py)
    }

    /// Compute the mean of the distribution
    fn mean(&self, py: Python<'_>) -> f64 {
        let lattice_ref = self.lattice.borrow(py);
        let grid = lattice_ref.grid_vec();
        let mut sum = 0.0;
        for (i, &p_val) in self.p.iter().enumerate() {
            sum += p_val * grid[i];
        }
        sum
    }

    /// Find approximate support (indices where p > tol)
    #[pyo3(signature = (tol=1e-12))]
    fn approx_support<'py>(&self, py: Python<'py>, tol: f64) -> Bound<'py, PyArray1<usize>> {
        let indices: Vec<usize> = self
            .p
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if val > tol { Some(i) } else { None })
            .collect();
        PyArray1::from_vec(py, indices)
    }

    /// Compute approximate support width
    #[pyo3(signature = (tol=1e-12))]
    pub fn approx_support_width(&self, tol: f64) -> usize {
        let indices: Vec<usize> = self
            .p
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if val > tol { Some(i) } else { None })
            .collect();

        if indices.is_empty() {
            0
        } else {
            indices.iter().max().unwrap() - indices.iter().min().unwrap()
        }
    }

    // ---- transforms ----

    /// Shift CDF right by k steps, then re-diff to PDF
    fn shift_integer(&self, py: Python<'_>, k: i32) -> PyResult<Density> {
        let c = _cdf_from_pdf(&self.p);
        let k_len = c.len() as i32;

        let c2 = if k <= -k_len {
            Array1::ones(c.len())
        } else if -k_len < k && k < 0 {
            let abs_k = k.abs() as usize;
            let mut result = Array1::zeros(c.len());
            let slice_len = c.len() - abs_k;
            result
                .slice_mut(ndarray::s![..slice_len])
                .assign(&c.slice(ndarray::s![abs_k..]));
            for i in slice_len..c.len() {
                result[i] = c[c.len() - 1];
            }
            result
        } else if 0 < k && k < k_len {
            let k_usize = k as usize;
            let mut result = Array1::zeros(c.len());
            let slice_len = c.len() - k_usize;
            result
                .slice_mut(ndarray::s![k_usize..])
                .assign(&c.slice(ndarray::s![..slice_len]));
            result
        } else if k >= k_len {
            Array1::zeros(c.len())
        } else {
            // k == 0
            c.clone()
        };

        let p2 = _pdf_from_cdf(&c2);
        let p2_py = p2.to_pyarray(py);
        Density::new(py, self.lattice.clone_ref(py), p2_py.readonly())
    }

    /// Linear blend of neighboring integer shifts (on the CDF)
    pub fn shift_fractional(&self, py: Python<'_>, x: f64) -> PyResult<Density> {
        let lattice_ref = self.lattice.borrow(py);
        let l_val = lattice_ref.l() as i32;

        let (l, u, lc, uc) = if -l_val + 2 < (x as i32) && (x as i32) < l_val - 2 {
            let l = x.floor() as i32;
            let u = x.ceil() as i32;
            let r = x - (l as f64);
            (l, u, 1.0 - r, r)
        } else if x >= (l_val - 2) as f64 {
            (l_val - 2, l_val - 1, 1.0, 0.0)
        } else {
            // x <= -L+2
            (-l_val + 1, -l_val + 2, 0.0, 1.0)
        };

        let c_l = self.shift_integer(py, l)?.cdf_array();
        let c_u = self.shift_integer(py, u)?.cdf_array();
        let c2 = lc * &c_l + uc * &c_u;
        let p2 = _pdf_from_cdf(&c2);
        let p2_py = p2.to_pyarray(py);

        Density::new(py, self.lattice.clone_ref(py), p2_py.readonly())
    }

    /// Center the distribution by shifting to zero mean
    fn center(&self, py: Python<'_>) -> PyResult<Density> {
        let m = self.mean(py);
        let lattice_ref = self.lattice.borrow(py);
        let steps = m / lattice_ref.unit_value();
        self.shift_fractional(py, -steps)
    }

    /// Convolve two densities
    #[pyo3(signature = (other, *, keep_L=None, pad=false))]
    fn convolve(
        &self,
        py: Python<'_>,
        other: &Density,
        keep_L: Option<usize>,
        pad: bool,
    ) -> PyResult<Density> {
        let self_lattice = self.lattice.borrow(py);
        let other_lattice = other.lattice.borrow(py);

        if (self_lattice.unit_value() - other_lattice.unit_value()).abs() > 1e-10 {
            return Err(PyValueError::new_err("Units must match for convolution."));
        }

        let l = if let Some(kl) = keep_L {
            kl
        } else {
            if self_lattice.l() != other_lattice.l() {
                return Err(PyValueError::new_err(
                    "Convolution with differing L; specify keep_L.",
                ));
            }
            self_lattice.l()
        };

        // Convolution
        let mut p = convolve_arrays(&self.p, &other.p);

        // Ensure odd length
        if p.len() % 2 == 0 {
            p = p.slice(ndarray::s![..p.len() - 1]).to_owned();
        }

        // Truncate or pad to target size
        let target_size = 2 * l + 1;
        let p_mid = if p.len() > target_size {
            let c = _cdf_from_pdf(&p);
            let n_extra = (p.len() - target_size) / 2;
            let c_slice = c.slice(ndarray::s![n_extra..c.len() - n_extra]).to_owned();
            _pdf_from_cdf(&c_slice)
        } else if p.len() < target_size {
            if pad {
                let n_extra = target_size - p.len();
                let left = n_extra / 2;
                let _right = n_extra - left;
                let mut result = Array1::zeros(target_size);
                result
                    .slice_mut(ndarray::s![left..left + p.len()])
                    .assign(&p);
                result
            } else {
                return Err(PyValueError::new_err(
                    "Resulting convolution too short; set pad=True or increase L.",
                ));
            }
        } else {
            p
        };

        // Correct mean drift via fractional shift
        let mu_self = self.mean(py);
        let mu_other = other.mean(py);

        let grid_mid: Vec<f64> = (-(l as i32)..=(l as i32))
            .map(|i| self_lattice.unit_value() * (i as f64))
            .collect();
        let mut mu_mid = 0.0;
        for (i, &p_val) in p_mid.iter().enumerate() {
            mu_mid += p_val * grid_mid[i];
        }

        let mu_diff = mu_mid - (mu_self + mu_other);

        let lattice_mid = Py::new(py, UniformLattice::new(l, self_lattice.unit_value()))?;
        let p_mid_py = p_mid.to_pyarray(py);
        let d_mid = Density::new(py, lattice_mid, p_mid_py.readonly())?;
        d_mid.shift_fractional(py, -mu_diff / self_lattice.unit_value())
    }

    /// Move mass as if unit size increased by unit_ratio (coarser lattice)
    #[pyo3(signature = (unit_ratio=2.0))]
    fn dilate(&self, py: Python<'_>, unit_ratio: f64) -> PyResult<Density> {
        let lattice_ref = self.lattice.borrow(py);
        let l = lattice_ref.l() as i32;
        let mut out = Array1::zeros(self.p.len());

        for idx in -l..=l {
            let array_idx = (idx + l) as usize;
            let prob = self.p[array_idx];
            let x = (idx as f64) / unit_ratio;

            let (l_idx, u_idx, lc, uc) = if -l + 2 < (x as i32) && (x as i32) < l - 2 {
                let floor_x = x.floor() as i32;
                let ceil_x = x.ceil() as i32;
                let r = x - (floor_x as f64);
                (floor_x, ceil_x, 1.0 - r, r)
            } else if x >= (l - 2) as f64 {
                (l - 2, l - 1, 1.0, 0.0)
            } else {
                (-l + 1, -l + 2, 0.0, 1.0)
            };

            let li = ((l_idx + l).max(0).min(2 * l)) as usize;
            let ui = ((u_idx + l).max(0).min(2 * l)) as usize;
            out[li] += prob * lc;
            out[ui] += prob * uc;
        }

        let out_py = out.to_pyarray(py);
        Density::new(py, self.lattice.clone_ref(py), out_py.readonly())
    }

    // ---- constructors ----

    /// Create density from a callable function
    #[staticmethod]
    #[pyo3(signature = (lattice, f, *, center=true))]
    fn from_callable(
        py: Python<'_>,
        lattice: Py<UniformLattice>,
        f: Py<PyAny>,
        center: bool,
    ) -> PyResult<Density> {
        let grid = {
            let lattice_ref = lattice.borrow(py);
            lattice_ref.grid_vec()
        };

        let mut p = Array1::zeros(grid.len());
        for (i, &xi) in grid.iter().enumerate() {
            let val: f64 = f.call1(py, (xi,))?.extract(py)?;
            p[i] = val.max(0.0);
        }

        let p_py = p.to_pyarray(py);
        let d = Density::new(py, lattice, p_py.readonly())?;
        if center {
            d.center(py)
        } else {
            Ok(d)
        }
    }

    /// Create a skew-normal distribution
    #[staticmethod]
    #[pyo3(signature = (lattice, loc=0.0, scale=1.0, a=0.0))]
    pub fn skew_normal(
        py: Python<'_>,
        lattice: Py<UniformLattice>,
        loc: f64,
        scale: f64,
        a: f64,
    ) -> PyResult<Density> {
        let lattice_ref = lattice.borrow(py);
        let grid = lattice_ref.grid_vec();

        let mut p = Array1::zeros(grid.len());
        for (i, &x) in grid.iter().enumerate() {
            let t = (x - loc) / scale;
            p[i] = (2.0 / scale * normpdf(t) * normcdf(a * t)).max(0.0);
        }

        let p_py = p.to_pyarray(py);
        let d = Density::new(py, lattice.clone_ref(py), p_py.readonly())?;
        let centered = d.center(py)?;
        centered.shift_fractional(py, loc / lattice_ref.unit_value())
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!("Density(lattice={:?}, p=...)", self.lattice.borrow(py))
    }
}

impl Density {
    /// Internal helper to get CDF as Array1
    fn cdf_array(&self) -> Array1<f64> {
        _cdf_from_pdf(&self.p)
    }
}

/// Convolve two 1D arrays
fn convolve_arrays(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    let result_len = a.len() + b.len() - 1;
    let mut result = Array1::zeros(result_len);

    for i in 0..a.len() {
        for j in 0..b.len() {
            result[i + j] += a[i] * b[j];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdf_from_pdf() {
        let pdf = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let cdf = _cdf_from_pdf(&pdf);
        assert_eq!(cdf.len(), 4);
        assert!((cdf[0] - 0.1).abs() < 1e-10);
        assert!((cdf[3] - 1.0).abs() < 1e-10);
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
    fn test_normalize() {
        let pdf = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let normalized = normalize(pdf).unwrap();
        let sum: f64 = normalized.sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero() {
        let pdf = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let normalized = normalize(pdf).unwrap();
        let sum: f64 = normalized.sum();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_convolve_arrays() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![0.5, 1.0]);
        let result = convolve_arrays(&a, &b);
        assert_eq!(result.len(), 4);
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 4.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
    }
}
