use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A uniform lattice with evenly spaced points centered at zero
#[pyclass]
#[derive(Clone, Debug)]
pub struct UniformLattice {
    /// Half-width (number of steps on one side)
    #[pyo3(get)]
    r#L: usize,

    /// Spacing between lattice points
    #[pyo3(get)]
    unit: f64,
}

#[pymethods]
impl UniformLattice {
    #[new]
    #[pyo3(signature = (r#L, unit))]
    pub fn new(r#L: usize, unit: f64) -> Self {
        UniformLattice { r#L, unit }
    }

    /// Get the total size of the lattice (2*L + 1)
    #[getter]
    fn size(&self) -> usize {
        2 * self.r#L + 1
    }

    /// Get the grid as a Python list of floats
    #[getter]
    fn grid(&self) -> Vec<f64> {
        let l_signed = self.r#L as i32;

        (-l_signed..=l_signed)
            .map(|i| self.unit * (i as f64))
            .collect()
    }

    /// Get the index grid as a Python list of integers
    fn index_grid(&self) -> Vec<i32> {
        let l_signed = self.r#L as i32;
        (-l_signed..=l_signed).collect()
    }

    /// Assert that an array is compatible with this lattice
    ///
    /// # Arguments
    /// * `arr` - A Python list or numpy array to check
    ///
    /// # Raises
    /// * `ValueError` if the array length doesn't match the lattice size
    fn assert_compatible(&self, arr: &Bound<'_, PyAny>) -> PyResult<()> {
        let len = arr.len()?;
        let expected_size = self.size();

        if len != expected_size {
            return Err(PyValueError::new_err(format!(
                "Array length {} incompatible with lattice size {}.",
                len, expected_size
            )));
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("UniformLattice(L={}, unit={})", self.r#L, self.unit)
    }
}

impl UniformLattice {
    /// Get L (Rust-accessible)
    pub fn l(&self) -> usize {
        self.r#L
    }

    /// Get unit (Rust-accessible)
    pub fn unit_value(&self) -> f64 {
        self.unit
    }

    /// Get size (Rust-accessible)
    pub fn size_value(&self) -> usize {
        2 * self.r#L + 1
    }

    /// Get grid (Rust-accessible)
    pub fn grid_vec(&self) -> Vec<f64> {
        let l_signed = self.r#L as i32;
        (-l_signed..=l_signed)
            .map(|i| self.unit * (i as f64))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size() {
        let lattice = UniformLattice::new(5, 1.0);
        assert_eq!(lattice.size(), 11);

        let lattice2 = UniformLattice::new(10, 0.5);
        assert_eq!(lattice2.size(), 21);
    }

    #[test]
    fn test_grid() {
        let lattice = UniformLattice::new(2, 0.5);
        let grid = lattice.grid();

        assert_eq!(grid.len(), 5);
        assert_eq!(grid, vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_index_grid() {
        let lattice = UniformLattice::new(3, 1.0);
        let indices = lattice.index_grid();

        assert_eq!(indices.len(), 7);
        assert_eq!(indices, vec![-3, -2, -1, 0, 1, 2, 3]);
    }
}
