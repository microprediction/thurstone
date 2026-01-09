use crate::density::Density;
use crate::order_stats::winner_of_many;
use crate::pricing::Race;
use ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Center offsets by integer mean of finite values
fn int_centered(offsets: &[f64]) -> Vec<f64> {
    let finite: Vec<f64> = offsets.iter().filter(|&&o| o.is_finite()).copied().collect();

    if finite.is_empty() {
        return offsets.to_vec();
    }

    let mean_val: f64 = finite.iter().sum::<f64>() / (finite.len() as f64);
    let mean_int = mean_val.round() as i32;

    offsets.iter().map(|&o| o - (mean_int as f64)).collect()
}

/// Choose a divider that maximizes the biggest gap near the center; ensures two non-empty groups.
fn divide_offsets(centered_offsets: &[f64]) -> f64 {
    let mut srt: Vec<f64> = centered_offsets.to_vec();
    srt.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if srt.len() <= 2 {
        return srt.iter().sum::<f64>() / (srt.len() as f64);
    }

    // Compute gaps
    let gaps: Vec<f64> = (0..srt.len() - 1)
        .map(|i| srt[i + 1] - srt[i])
        .collect();

    // Find index of maximum absolute gap
    let idx = gaps.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    0.5 * (srt[idx] + srt[idx + 1])
}

/// Helper to compute state prices from densities
fn state_prices_from_densities(py: Python<'_>, densities: Vec<Py<Density>>) -> PyResult<Vec<f64>> {
    let race = Race::new(py, densities, None)?;
    let prices_arr = race.state_prices(py)?;
    let prices_ro = prices_arr.readonly();
    Ok(prices_ro.as_array().to_vec())
}

/// Helper to create densities from offsets
fn densities_from_offsets(py: Python<'_>, base: &Density, offsets: &[f64]) -> PyResult<Vec<Py<Density>>> {
    let mut out = Vec::new();
    for &o in offsets {
        let shifted = base.shift_fractional(py, o)?;
        out.push(Py::new(py, shifted)?);
    }
    Ok(out)
}

/// Cluster splitter for extended state pricing
#[pyclass]
pub struct ClusterSplitter {
    #[pyo3(get, set)]
    pub unit_ratio: f64,
    #[pyo3(get, set)]
    pub max_depth: i32,
}

#[pymethods]
impl ClusterSplitter {
    #[new]
    #[pyo3(signature = (unit_ratio=3.0, max_depth=3))]
    pub fn new(unit_ratio: f64, max_depth: i32) -> Self {
        ClusterSplitter { unit_ratio, max_depth }
    }

    /// Offsets may include +/-inf; returns normalized winning probabilities.
    pub fn extended_state_prices<'py>(
        &self,
        py: Python<'py>,
        base: &Density,
        offsets: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let prices = self.extended_state_prices_internal(py, base, &offsets)?;
        Ok(Array1::from_vec(prices).to_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "ClusterSplitter(unit_ratio={}, max_depth={})",
            self.unit_ratio, self.max_depth
        )
    }
}

impl ClusterSplitter {
    fn extended_state_prices_internal(
        &self,
        py: Python<'_>,
        base: &Density,
        offsets: &[f64],
    ) -> PyResult<Vec<f64>> {
        let n = offsets.len();
        if n == 1 {
            return Ok(vec![1.0]);
        }

        // Handle +inf (no chance)
        let pos_inf_idx: Vec<usize> = offsets.iter()
            .enumerate()
            .filter(|(_, &o)| o == f64::INFINITY)
            .map(|(i, _)| i)
            .collect();

        if !pos_inf_idx.is_empty() {
            let finite_idx: Vec<usize> = (0..n)
                .filter(|i| !pos_inf_idx.contains(i))
                .collect();

            if finite_idx.is_empty() {
                return Ok(vec![1.0 / (n as f64); n]);
            }

            let finite_offsets: Vec<f64> = finite_idx.iter()
                .map(|&i| offsets[i])
                .collect();
            let centered_finite = int_centered(&finite_offsets);
            let finite_prices = self.extended_state_prices_internal(py, base, &centered_finite)?;

            let mut out = vec![0.0; n];
            for (j, &idx) in finite_idx.iter().enumerate() {
                out[idx] = finite_prices[j];
            }
            return Ok(out);
        }

        // Handle -inf (certain winners share)
        let neg_inf_idx: Vec<usize> = offsets.iter()
            .enumerate()
            .filter(|(_, &o)| o == f64::NEG_INFINITY)
            .map(|(i, _)| i)
            .collect();

        if !neg_inf_idx.is_empty() {
            let mut p = vec![0.0; n];
            let share = 1.0 / (neg_inf_idx.len() as f64);
            for &idx in &neg_inf_idx {
                p[idx] = share;
            }
            return Ok(p);
        }

        let l = base.lattice_l(py);
        let w = base.approx_support_width(1e-12);
        let centered = int_centered(offsets);

        // Boundaries
        let lower_bound = -(l as f64) + (w as f64);
        let upper_bound = (l as f64) - (w as f64);
        let hang_left: Vec<usize> = centered.iter()
            .enumerate()
            .filter(|(_, &o)| o < lower_bound)
            .map(|(i, _)| i)
            .collect();
        let hang_right: Vec<usize> = centered.iter()
            .enumerate()
            .filter(|(_, &o)| o > upper_bound)
            .map(|(i, _)| i)
            .collect();

        if hang_left.is_empty() && hang_right.is_empty() {
            let dens = densities_from_offsets(py, base, &centered)?;
            return state_prices_from_densities(py, dens);
        }

        // Stop recursion
        if self.max_depth <= 0 {
            let mut modified = centered.clone();
            for &i in &hang_right {
                modified[i] = f64::INFINITY;
            }
            for &i in &hang_left {
                modified[i] = f64::NEG_INFINITY;
            }
            return self.extended_state_prices_internal(py, base, &modified);
        }

        // Split symmetrically
        let divider = divide_offsets(&centered);
        let mut left_idx: Vec<usize> = centered.iter()
            .enumerate()
            .filter(|(_, &o)| o < divider)
            .map(|(i, _)| i)
            .collect();
        let mut right_idx: Vec<usize> = centered.iter()
            .enumerate()
            .filter(|(_, &o)| o >= divider)
            .map(|(i, _)| i)
            .collect();

        if left_idx.is_empty() || right_idx.is_empty() {
            if left_idx.is_empty() {
                let min_idx = centered.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                left_idx = vec![min_idx];
                right_idx = (0..n).filter(|&i| i != min_idx).collect();
            } else {
                let max_idx = centered.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                right_idx = vec![max_idx];
                left_idx = (0..n).filter(|&i| i != max_idx).collect();
            }
        }

        // Group representatives via first-order statistics
        let left_offsets: Vec<f64> = left_idx.iter().map(|&i| centered[i]).collect();
        let right_offsets: Vec<f64> = right_idx.iter().map(|&i| centered[i]).collect();

        let dens_left = densities_from_offsets(py, base, &left_offsets)?;
        let dens_right = densities_from_offsets(py, base, &right_offsets)?;

        if dens_left.is_empty() || dens_right.is_empty() {
            let dens = densities_from_offsets(py, base, &centered)?;
            return state_prices_from_densities(py, dens);
        }

        // Clone the vectors for winner_of_many calls
        let dens_left_copy1: Vec<Py<Density>> = dens_left.iter().map(|d| d.clone_ref(py)).collect();
        let dens_right_copy1: Vec<Py<Density>> = dens_right.iter().map(|d| d.clone_ref(py)).collect();

        let (rep_left, _) = winner_of_many(py, dens_left_copy1)?;
        let (rep_right, _) = winner_of_many(py, dens_right_copy1)?;

        let group_race = Race::new(py, vec![rep_left, rep_right], None)?;
        let group_prices_arr = group_race.state_prices(py)?;
        let group_prices_ro = group_prices_arr.readonly();
        let group_prices = group_prices_ro.as_array();
        let left_share = group_prices[0];
        let right_share = group_prices[1];

        // Refine inside groups using "weak-as-single" logic
        let (left_prices_rel, right_prices_rel) = if left_share <= right_share {
            // left group is weaker
            let left_sub = ClusterSplitter {
                unit_ratio: self.unit_ratio,
                max_depth: self.max_depth - 1,
            };
            let left_prices_rel = left_sub.extended_state_prices_internal(py, base, &left_offsets)?;

            let right_prices_rel = state_prices_from_densities(py, dens_right)?;
            let s_r: f64 = right_prices_rel.iter().sum();
            let right_prices_rel = if s_r > 0.0 {
                right_prices_rel.iter().map(|&pr| pr / s_r).collect()
            } else {
                right_prices_rel
            };
            (left_prices_rel, right_prices_rel)
        } else {
            // right group is weaker
            let right_sub = ClusterSplitter {
                unit_ratio: self.unit_ratio,
                max_depth: self.max_depth - 1,
            };
            let right_prices_rel = right_sub.extended_state_prices_internal(py, base, &right_offsets)?;

            let left_prices_rel = state_prices_from_densities(py, dens_left)?;
            let s_l: f64 = left_prices_rel.iter().sum();
            let left_prices_rel = if s_l > 0.0 {
                left_prices_rel.iter().map(|&pl| pl / s_l).collect()
            } else {
                left_prices_rel
            };
            (left_prices_rel, right_prices_rel)
        };

        let mut out = vec![0.0; n];
        for (j, &idx) in left_idx.iter().enumerate() {
            out[idx] = left_share * left_prices_rel[j];
        }
        for (j, &idx) in right_idx.iter().enumerate() {
            out[idx] = right_share * right_prices_rel[j];
        }

        let s: f64 = out.iter().sum();
        if s <= 0.0 {
            return Err(PyValueError::new_err(
                "Extended state prices have non-positive total mass."
            ));
        }

        if !(0.999..=1.001).contains(&s) {
            return Err(PyValueError::new_err(format!(
                "State prices not normalized in extended offsets; sum={}",
                s
            )));
        }

        Ok(out.iter().map(|&oi| oi / s).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_centered() {
        let offsets = vec![1.0, 2.0, 3.0, 4.0];
        let centered = int_centered(&offsets);
        // mean = 2.5, rounded to 2 or 3
        assert_eq!(centered.len(), 4);
    }

    #[test]
    fn test_int_centered_with_inf() {
        let offsets = vec![1.0, f64::INFINITY, 3.0];
        let centered = int_centered(&offsets);
        // Only finite values: 1.0, 3.0, mean = 2.0
        assert_eq!(centered.len(), 3);
        assert!((centered[0] - (-1.0)).abs() < 1e-10);
        assert!(centered[1].is_infinite());
        assert!((centered[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_divide_offsets() {
        let offsets = vec![1.0, 2.0, 5.0, 6.0];
        let div = divide_offsets(&offsets);
        // Largest gap is between 2.0 and 5.0, so divider should be 3.5
        assert!((div - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_divide_offsets_small() {
        let offsets = vec![1.0, 2.0];
        let div = divide_offsets(&offsets);
        // Average of 1.0 and 2.0
        assert!((div - 1.5).abs() < 1e-10);
    }
}
