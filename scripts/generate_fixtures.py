"""Emit JSON golden fixtures for the JS port to verify against.

Each fixture file lives in docs/fixtures/ and is loaded by docs/tests/*.test.js.
Tolerances are per-fixture (chosen to be tight but not flaky).

Run from repo root:
    python scripts/generate_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thurstone import (
    AbilityCalibrator,
    Density,
    Race,
    StatePricer,
    UniformLattice,
)
from thurstone.global_fit import GlobalAbilityCalibrator
from thurstone.global_ls import GlobalLSCalibrator
from thurstone.normaldist import normcdf, normpdf
from thurstone.order_stats import winner_of_many

OUT_DIR = ROOT / "docs" / "fixtures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(name: str, payload: dict) -> None:
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size} bytes)")


def fixture_normaldist() -> None:
    xs = [-4.0, -2.5, -1.0, -0.5, 0.0, 0.25, 1.0, 2.5, 4.0]
    payload = {
        "tolerance": 1e-6,
        "xs": xs,
        "pdf": [normpdf(x) for x in xs],
        "cdf": [normcdf(x) for x in xs],
    }
    _save("normaldist", payload)


def fixture_skew_normal() -> None:
    lat = UniformLattice(L=200, unit=0.05)
    cases = []
    for loc, scale, a in [(0.0, 1.0, 0.0), (0.5, 1.2, 1.0), (-1.0, 0.8, -2.0)]:
        d = Density.skew_normal(lat, loc=loc, scale=scale, a=a)
        cases.append(
            {
                "params": {"loc": loc, "scale": scale, "a": a},
                "p": d.p.tolist(),
                "mean": float(d.mean()),
            }
        )
    payload = {
        "tolerance": 1e-10,
        "lattice": {"L": lat.L, "unit": lat.unit},
        "cases": cases,
    }
    _save("density_skewnormal", payload)


def fixture_shift_and_convolve() -> None:
    lat = UniformLattice(L=200, unit=0.05)
    d = Density.skew_normal(lat, loc=0.0, scale=1.0, a=0.0)
    payload = {
        "tolerance": 1e-10,
        "lattice": {"L": lat.L, "unit": lat.unit},
        "shift_integer_5": d.shift_integer(5).p.tolist(),
        "shift_integer_neg7": d.shift_integer(-7).p.tolist(),
        "shift_fractional_3p25": d.shift_fractional(3.25).p.tolist(),
        "shift_fractional_neg2p7": d.shift_fractional(-2.7).p.tolist(),
        "convolve_self": d.convolve(d).p.tolist(),
        "dilate_2": d.dilate(2.0).p.tolist(),
        "mean_after_shift": float(d.shift_fractional(3.25).mean()),
    }
    _save("density_transforms", payload)


def fixture_winner_of_many() -> None:
    lat = UniformLattice(L=200, unit=0.05)
    offsets = [-2.0, -0.5, 0.0, 0.7, 1.5]
    base = Density.skew_normal(lat, loc=0.0, scale=1.0, a=0.0)
    densities = [base.shift_fractional(o) for o in offsets]
    d_all, m_all = winner_of_many(densities)
    payload = {
        "tolerance": 1e-10,
        "lattice": {"L": lat.L, "unit": lat.unit},
        "offsets": offsets,
        "winner_pdf": d_all.p.tolist(),
        "multiplicity": m_all.tolist(),
    }
    _save("winner_of_many", payload)


def fixture_race_state_prices() -> None:
    lat = UniformLattice(L=200, unit=0.05)
    base = Density.skew_normal(lat, loc=0.0, scale=1.0, a=0.0)
    cases = []
    for offsets in [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [-3.0, -1.0, 0.0, 1.0, 3.0],
        [-2.0, -0.5, 0.0, 0.5, 0.5, 0.5, 2.0],  # cluster of ties
    ]:
        densities = [base.shift_fractional(o) for o in offsets]
        prices = Race(densities).state_prices().tolist()
        cases.append({"offsets": offsets, "prices": prices})
    payload = {
        "tolerance": 1e-8,
        "lattice": {"L": lat.L, "unit": lat.unit},
        "cases": cases,
    }
    _save("race_state_prices", payload)


def fixture_inverse_roundtrip() -> None:
    lat = UniformLattice(L=200, unit=0.05)
    base = Density.skew_normal(lat, loc=0.0, scale=1.0, a=0.0)
    cal = AbilityCalibrator(base, n_iter=3)
    dividends_cases = [
        [3.2, 4.8, 12.0, 7.5, 20.0],
        [2.5, 2.7, 3.0, 12.0, 50.0],
        [4.0, 4.0, 4.0, 4.0, 4.0],
    ]
    cases = []
    for div in dividends_cases:
        ability = cal.solve_from_dividends(div)
        prices = StatePricer.prices_from_dividends(div).tolist()
        cases.append({"dividends": div, "prices": prices, "ability": ability})
    payload = {
        "tolerance": 1e-6,
        "lattice": {"L": lat.L, "unit": lat.unit},
        "cases": cases,
    }
    _save("inverse_roundtrip", payload)


def fixture_state_prices_from_ability() -> None:
    lat = UniformLattice(L=200, unit=0.05)
    base = Density.skew_normal(lat, loc=0.0, scale=1.0, a=0.0)
    cal = AbilityCalibrator(base)
    abilities_cases = [
        [-1.0, -0.4, 0.0, 0.3, 0.9],
        [0.0, 0.0, 0.0],
        [-3.5, -0.2, 0.1, 4.8],  # exercises ClusterSplitter
    ]
    cases = []
    for ab in abilities_cases:
        prices = cal.state_prices_from_ability(ab)
        cases.append({"ability": ab, "prices": list(prices)})
    payload = {
        "tolerance": 1e-6,
        "lattice": {"L": lat.L, "unit": lat.unit},
        "cases": cases,
    }
    _save("state_prices_from_ability", payload)


def fixture_global_fit() -> None:
    lat = UniformLattice(L=200, unit=0.05)
    base = Density.skew_normal(lat, loc=0.0, scale=1.0, a=0.0)
    cal = AbilityCalibrator(base, n_iter=3)
    races = [
        {"ids": ["A", "B", "C"], "div": [3.0, 4.5, 6.0]},
        {"ids": ["B", "C", "D"], "div": [3.5, 5.5, 7.0]},
        {"ids": ["A", "C", "D"], "div": [3.2, 5.0, 9.0]},
    ]
    gn = GlobalAbilityCalibrator(["A", "B", "C", "D"], l2=1e-8, step_bias=0.3, step_theta=0.3)
    races_payload = []
    for r in races:
        prices = StatePricer.prices_from_dividends(r["div"]).tolist()
        gn.add_race(cal, r["ids"], prices)
        races_payload.append({"ids": r["ids"], "prices": prices})
    gn.fit_with_rebuild(num_outer_iters=3, num_inner_iters=10)
    payload = {
        "tolerance": 1e-6,
        "lattice": {"L": lat.L, "unit": lat.unit},
        "races": races_payload,
        "theta": gn.theta,
        "biases": list(gn.biases),
    }
    _save("global_fit", payload)


def fixture_global_ls() -> None:
    lat = UniformLattice(L=200, unit=0.05)
    base = Density.skew_normal(lat, loc=0.0, scale=1.0, a=0.0)
    cal = AbilityCalibrator(base, n_iter=3)
    # 4 horses A,B,C,D across 3 overlapping races
    races = [
        {"ids": ["A", "B", "C"], "div": [3.0, 4.5, 6.0]},
        {"ids": ["B", "C", "D"], "div": [3.5, 5.5, 7.0]},
        {"ids": ["A", "C", "D"], "div": [3.2, 5.0, 9.0]},
    ]
    gl = GlobalLSCalibrator(["A", "B", "C", "D"])
    races_payload = []
    for r in races:
        prices = StatePricer.prices_from_dividends(r["div"]).tolist()
        gl.add_race(cal, r["ids"], prices)
        races_payload.append({"ids": r["ids"], "prices": prices})
    gl.fit(use_slope_weights=True, ridge=1e-6)
    payload = {
        "tolerance": 5e-4,
        "lattice": {"L": lat.L, "unit": lat.unit},
        "races": races_payload,
        "theta": gl.theta,
    }
    _save("global_ls", payload)


def main() -> None:
    fixture_normaldist()
    fixture_skew_normal()
    fixture_shift_and_convolve()
    fixture_winner_of_many()
    fixture_race_state_prices()
    fixture_inverse_roundtrip()
    fixture_state_prices_from_ability()
    fixture_global_ls()
    fixture_global_fit()


if __name__ == "__main__":
    main()
