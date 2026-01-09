mod clustering;
mod density;
mod dynamic;
mod global_fit;
mod global_ls;
mod inference;
mod kalman_tracker;
mod lattice;
mod normaldist;
mod order_stats;
mod pricing;
mod sim_world;

use pyo3::prelude::*;

/// A simple example class exposed to Python
#[pyclass]
struct Thurstone {
    #[pyo3(get, set)]
    value: i32,
}

#[pymethods]
impl Thurstone {
    #[new]
    fn new(value: i32) -> Self {
        Thurstone { value }
    }

    /// Doubles the current value
    fn double(&mut self) -> PyResult<i32> {
        self.value *= 2;
        Ok(self.value)
    }

    fn __repr__(&self) -> String {
        format!("Thurstone(value={})", self.value)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn thurstone(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let m_normaldist = PyModule::new(py, "normaldist")?;
    m_normaldist.add_function(pyo3::wrap_pyfunction!(normaldist::normcdf, m)?)?;
    m_normaldist.add_function(pyo3::wrap_pyfunction!(normaldist::normpdf, m)?)?;

    let m_lattice = PyModule::new(py, "lattice")?;
    m_lattice.add_class::<lattice::UniformLattice>()?;

    let m_density = PyModule::new(py, "density")?;
    m_density.add_class::<density::Density>()?;
    m_density.add_function(pyo3::wrap_pyfunction!(density::pdf_from_cdf, m)?)?;
    m_density.add_function(pyo3::wrap_pyfunction!(density::cdf_from_pdf, m)?)?;

    let m_order_stats = PyModule::new(py, "order_stats")?;
    m_order_stats.add_function(pyo3::wrap_pyfunction!(order_stats::winner_of_many, m)?)?;
    m_order_stats.add_function(pyo3::wrap_pyfunction!(order_stats::get_the_rest, m)?)?;
    m_order_stats.add_function(pyo3::wrap_pyfunction!(
        order_stats::expected_payoff_with_multiplicity,
        m
    )?)?;

    let m_pricing = PyModule::new(py, "pricing")?;
    m_pricing.add_class::<pricing::TieModel>()?;
    m_pricing.add_class::<pricing::HalfPointTie>()?;
    m_pricing.add_class::<pricing::Race>()?;
    m_pricing.add_class::<pricing::StatePricer>()?;
    m_pricing.add_function(pyo3::wrap_pyfunction!(pricing::cdf_min, m)?)?;
    m_pricing.add_function(pyo3::wrap_pyfunction!(pricing::winner_of_many_simple, m)?)?;
    m_pricing.add_function(pyo3::wrap_pyfunction!(pricing::rest_min_cdf, m)?)?;
    m_pricing.add_function(pyo3::wrap_pyfunction!(
        pricing::conditional_win_draw_loss,
        m
    )?)?;
    m_pricing.add_function(pyo3::wrap_pyfunction!(pricing::expected_payoff_vs_rest, m)?)?;

    let m_clustering = PyModule::new(py, "clustering")?;
    m_clustering.add_class::<clustering::ClusterSplitter>()?;

    let m_inference = PyModule::new(py, "inference")?;
    m_inference.add_class::<inference::AbilityCalibrator>()?;
    m_inference.add_class::<inference::LookupCurve1D>()?;
    m_inference.add_class::<inference::LookupCurve1DInverse>()?;
    m_inference.add_function(pyo3::wrap_pyfunction!(inference::implicit_state_prices, m)?)?;

    let m_dynamic = PyModule::new(py, "dynamic")?;
    m_dynamic.add_class::<dynamic::RaceObservation>()?;
    m_dynamic.add_class::<dynamic::DynamicThurstoneCalibrator>()?;
    m_dynamic.add_class::<dynamic::ParametricSigmaFn>()?;
    m_dynamic.add_class::<dynamic::PiecewiseSigmaFn>()?;

    let m_global_fit = PyModule::new(py, "global_fit")?;
    m_global_fit.add_class::<global_fit::RaceSpec>()?;
    m_global_fit.add_class::<global_fit::GlobalAbilityCalibrator>()?;

    let m_global_ls = PyModule::new(py, "global_ls")?;
    m_global_ls.add_class::<global_ls::RaceLS>()?;
    m_global_ls.add_class::<global_ls::GlobalLSCalibrator>()?;

    let m_kalman_tracker = PyModule::new(py, "kalman_tracker")?;
    m_kalman_tracker.add_class::<kalman_tracker::HorseFilterState>()?;
    m_kalman_tracker.add_class::<kalman_tracker::KalmanAbilityTracker>()?;

    let m_sim_world = PyModule::new(py, "sim_world")?;
    m_sim_world.add_function(pyo3::wrap_pyfunction!(sim_world::sigma_true, m)?)?;
    m_sim_world.add_function(pyo3::wrap_pyfunction!(sim_world::simulate_schedule, m)?)?;
    m_sim_world.add_function(pyo3::wrap_pyfunction!(sim_world::simulate_world, m)?)?;

    m.add_submodule(&m_normaldist)?;
    m.add_submodule(&m_lattice)?;
    m.add_submodule(&m_density)?;
    m.add_submodule(&m_order_stats)?;
    m.add_submodule(&m_pricing)?;
    m.add_submodule(&m_clustering)?;
    m.add_submodule(&m_inference)?;
    m.add_submodule(&m_dynamic)?;
    m.add_submodule(&m_global_fit)?;
    m.add_class::<Thurstone>()?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("thurstone.normaldist", m_normaldist)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("thurstone.lattice", m_lattice)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("thurstone.density", m_density)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("thurstone.order_stats", m_order_stats)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("thurstone.pricing", m_pricing)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("thurstone.clustering", m_clustering)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("thurstone.inference", m_inference)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("thurstone.dynamic", m_dynamic)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("thurstone.global_fit", m_global_fit)?;

    m.add_submodule(&m_global_ls)?;
    m.add_submodule(&m_kalman_tracker)?;
    m.add_submodule(&m_sim_world)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("thurstone.global_ls", m_global_ls)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("thurstone.kalman_tracker", m_kalman_tracker)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("thurstone.sim_world", m_sim_world)?;

    Ok(())
}
