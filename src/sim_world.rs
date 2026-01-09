use crate::density::Density;
use crate::dynamic::RaceObservation;
use crate::lattice::UniformLattice;
use ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::prelude::*;
use rand_pcg::Pcg64;
use std::collections::HashMap;

/// Ground-truth stickiness: random-walk std scales like sqrt(alpha * dt)
#[pyfunction]
pub fn sigma_true(dt: f64, alpha: f64) -> f64 {
    (alpha * dt.max(0.0)).max(1e-12).sqrt()
}

/// Return sorted race times and a list of participant index sets per race
#[pyfunction]
pub fn simulate_schedule(
    seed: u64,
    n_horses: usize,
    n_races: usize,
    race_size_range: (usize, usize),
    horizon_days: f64,
) -> PyResult<(Vec<f64>, Vec<Vec<usize>>)> {
    let mut rng = Pcg64::seed_from_u64(seed);

    // Generate random race times
    let mut times: Vec<f64> = (0..n_races)
        .map(|_| rng.gen::<f64>() * horizon_days)
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Generate fields
    let mut fields: Vec<Vec<usize>> = Vec::new();
    let (lo, hi) = race_size_range;

    for _ in 0..n_races {
        let m = rng.gen_range(lo..=hi).min(n_horses).max(2);
        let mut field: Vec<usize> = (0..n_horses).collect();
        field.shuffle(&mut rng);
        field.truncate(m);
        fields.push(field);
    }

    Ok((times, fields))
}

/// Simulate a dataset
#[pyfunction]
#[pyo3(signature = (seed, n_horses=60, n_races=90, race_size_range=(8, 14), horizon_days=240.0, alpha=0.04, sigma0=0.8, bookmaker_rel_tau=0.5, bookmaker_bias_tau=0.0))]
pub fn simulate_world<'py>(
    py: Python<'py>,
    seed: u64,
    n_horses: usize,
    n_races: usize,
    race_size_range: (usize, usize),
    horizon_days: f64,
    alpha: f64,
    sigma0: f64,
    bookmaker_rel_tau: f64,
    bookmaker_bias_tau: f64,
) -> PyResult<(
    Vec<Py<RaceObservation>>,
    Py<PyDict>,
    Py<PyDict>,
    Py<PyDict>,
    Py<PyDict>,
)> {
    let mut rng = Pcg64::seed_from_u64(seed);

    let (times, fields) = simulate_schedule(
        seed + 1,
        n_horses,
        n_races,
        race_size_range,
        horizon_days,
    )?;

    let horse_ids: Vec<String> = (0..n_horses).map(|i| format!("H{}", i)).collect();

    // Build per-horse race index lists
    let mut idx_per_horse: HashMap<usize, Vec<usize>> = HashMap::new();
    for h in 0..n_horses {
        idx_per_horse.insert(h, Vec::new());
    }

    for (r_i, field) in fields.iter().enumerate() {
        for &h in field {
            idx_per_horse.get_mut(&h).unwrap().push(r_i);
        }
    }

    // Sort each horse's races by time
    for (_, indices) in idx_per_horse.iter_mut() {
        indices.sort_by(|&i, &j| times[i].partial_cmp(&times[j]).unwrap());
    }

    // Simulate true abilities μ at race times
    let mut mu_at_race: HashMap<(usize, usize), f64> = HashMap::new();
    let mut true_theta: HashMap<String, Vec<f64>> = HashMap::new();
    let mut true_times: HashMap<String, Vec<f64>> = HashMap::new();
    let mut book_theta: HashMap<String, Vec<f64>> = HashMap::new();
    let mut book_times: HashMap<String, Vec<f64>> = HashMap::new();

    for h in 0..n_horses {
        let hid = &horse_ids[h];
        true_theta.insert(hid.clone(), Vec::new());
        true_times.insert(hid.clone(), Vec::new());
        book_theta.insert(hid.clone(), Vec::new());
        book_times.insert(hid.clone(), Vec::new());
    }

    for h in 0..n_horses {
        let mut mu = rng.gen::<f64>() * sigma0 * 2.0 - sigma0; // Normal(0, sigma0) approx
        let mut prev_t: Option<f64> = None;

        for &r_i in &idx_per_horse[&h] {
            let t = times[r_i];
            if let Some(pt) = prev_t {
                let dt = t - pt;
                let noise = (rng.gen::<f64>() * 2.0 - 1.0) * sigma_true(dt, alpha) * 1.732; // ~Normal(0, sigma)
                mu += noise;
            }
            prev_t = Some(t);
            mu_at_race.insert((h, r_i), mu);

            let hid = &horse_ids[h];
            true_theta.get_mut(hid).unwrap().push(mu);
            true_times.get_mut(hid).unwrap().push(t);
        }
    }

    // Forward model (also used by bookmaker)
    let lattice = Py::new(py, UniformLattice::new(400, 0.1))?;
    let base = Density::skew_normal(py, lattice.clone_ref(py), 0.0, 1.0, 0.0)?;
    let base_py = Py::new(py, base)?;

    let calibrator_class = py
        .import("thurstone.inference")?
        .getattr("AbilityCalibrator")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("n_iter", 3)?;
    let forward = calibrator_class.call((base_py.clone_ref(py),), Some(&kwargs))?;

    // Build observed bookmaker prices from noisy abilities
    let mut races: Vec<Py<RaceObservation>> = Vec::new();

    for (r_i, field) in fields.iter().enumerate() {
        let ids: Vec<String> = field.iter().map(|&h| horse_ids[h].clone()).collect();
        let mu_true: Vec<f64> = field.iter().map(|&h| mu_at_race[&(h, r_i)]).collect();

        // True probabilities (fair)
        let p_true_arr = forward.call_method1("state_prices_from_ability", (mu_true.clone(),))?;
        let p_true_np = p_true_arr.downcast::<PyArray1<f64>>()?;
        let p_true = p_true_np.to_owned_array();

        // Bookmaker noisy ability view
        let race_bias = (rng.gen::<f64>() * 2.0 - 1.0) * bookmaker_bias_tau * 1.732;
        let mu_hat: Vec<f64> = mu_true
            .iter()
            .map(|&mu| {
                let noise = (rng.gen::<f64>() * 2.0 - 1.0) * bookmaker_rel_tau * 1.732;
                mu + noise + race_bias
            })
            .collect();

        let p_obs_arr = forward.call_method1("state_prices_from_ability", (mu_hat.clone(),))?;
        let p_obs_np = p_obs_arr.downcast::<PyArray1<f64>>()?;
        let p_obs = p_obs_np.to_owned_array().to_vec();

        // Select winner based on true probabilities
        let p_sum: f64 = p_true.sum();
        let mut cumsum = 0.0;
        let rand_val = rng.gen::<f64>() * p_sum.max(1e-12);
        let mut winner_idx = 0;
        for (i, &p) in p_true.iter().enumerate() {
            cumsum += p;
            if cumsum >= rand_val {
                winner_idx = i;
                break;
            }
        }
        let winner_id = ids[winner_idx].clone();

        // Store bookmaker noisy ability samples
        for (hid, mhat) in ids.iter().zip(mu_hat.iter()) {
            book_theta.get_mut(hid).unwrap().push(*mhat);
            book_times.get_mut(hid).unwrap().push(times[r_i]);
        }

        let race = Py::new(
            py,
            RaceObservation {
                race_id: format!("R{}", r_i),
                time: times[r_i],
                horse_ids: ids,
                prices: p_obs,
                winner: Some(winner_id),
            },
        )?;
        races.push(race);
    }

    // Convert to numpy arrays
    let true_theta_dict = PyDict::new(py);
    for (h, v) in true_theta {
        true_theta_dict.set_item(h, Array1::from_vec(v).to_pyarray(py))?;
    }

    let true_times_dict = PyDict::new(py);
    for (h, v) in true_times {
        true_times_dict.set_item(h, Array1::from_vec(v).to_pyarray(py))?;
    }

    let book_theta_dict = PyDict::new(py);
    for (h, v) in book_theta {
        book_theta_dict.set_item(h, Array1::from_vec(v).to_pyarray(py))?;
    }

    let book_times_dict = PyDict::new(py);
    for (h, v) in book_times {
        book_times_dict.set_item(h, Array1::from_vec(v).to_pyarray(py))?;
    }

    Ok((
        races,
        true_theta_dict.into(),
        true_times_dict.into(),
        book_theta_dict.into(),
        book_times_dict.into(),
    ))
}
