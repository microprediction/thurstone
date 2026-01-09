use pyo3::pyfunction;

const SQRT2: f64 = 1.4142135623730951;
const SQRT2PI: f64 = 2.5066282746310002;

/// Probability density function of the standard normal distribution
#[pyfunction]
pub fn normpdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / SQRT2PI
}

/// Cumulative distribution function of the standard normal distribution
/// High-accuracy via erf (error function)
#[pyfunction]
pub fn normcdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / SQRT2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normpdf() {
        // Test at x=0 (peak of normal distribution)
        let result = normpdf(0.0);
        let expected = 1.0 / SQRT2PI;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_normcdf() {
        // Test at x=0 (should be 0.5)
        let result = normcdf(0.0);
        assert!((result - 0.5).abs() < 1e-10);

        // Test symmetry
        let x = 1.0;
        assert!((normcdf(x) + normcdf(-x) - 1.0).abs() < 1e-10);
    }
}
