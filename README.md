
# thurstone

An object‑oriented toolkit for **lattice‑based performance models** and the **horse‑race inverse problem** (inferring relative ability from market prices). It is inspired by lattice approaches but aims to be clearer and more modular.

## Highlights
- Uniform lattice with explicit `L` and `unit` (`UniformLattice`).
- Normalized `Density` with safe integer/fractional shifts, centering, convolution, and dilation. Negative mass is rejected; zero‑mass vectors are allowed as an explicit off‑lattice sentinel in extreme shifts.
- Clean `Race` and `StatePricer` API for risk‑neutral **state prices** (winning probabilities).
- `AbilityCalibrator` solves the **inverse problem** with per‑runner monotone interpolation (sorted price→offset tables) and returns abilities in physical units.
- `ClusterSplitter` handles offsets that hang off the lattice (both directions) with symmetric clustering and coarse‑to‑fine recursion, including walkover behavior and an equal‑share rule only when all offsets hang on one side and are tightly bunched (spread < support width).
- Tie handling via pluggable `TieModel` (default: 0.5 split).

## Quickstart

```python
from thurstone import UniformLattice, Density, AbilityCalibrator, STD_L, STD_UNIT

lat = UniformLattice(L=STD_L, unit=STD_UNIT)
base = Density.skew_normal(lat, loc=0.0, scale=1.0, a=0.0)  # symmetric base

# Inverse calibration from dividends -> abilities (physical units)
from thurstone import StatePricer
dividends = [3.2, 4.8, 12.0, 7.5, 20.0]
cal = AbilityCalibrator(base)
ability = cal.solve_from_dividends(dividends)     # abilities in same units as lattice grid
prices = cal.state_prices_from_ability(ability)   # forward map (prices sum to 1)
```

## Design notes
- Fractional shifts are performed on the CDF then differenced back to preserve mass and monotonicity.
- The inverse loop builds per‑runner `(price, offset)` tables (holding others fixed), sorts them by price, uniques on the price key, clips, and uses `np.interp`.
- `ClusterSplitter` splits at the largest gap near the center, evaluates **both sides**, and combines refined within‑group prices using coarse group shares from a dilated lattice. It exhibits:
  - Walkover: sufficiently better (left) runners take ~1, sufficiently worse (right) get ~0.
  - Equal‑share only when all runners hang on the same side and are indistinguishable at resolution (spread < support width).

## Testing
- Optional test extras: `pip install -e ".[test]"`
- Run tests: `pytest -q`
- Property tests (Hypothesis) are auto‑skipped if Hypothesis is not installed.
- A performance benchmark (pytest‑benchmark) is included and auto‑skips if the plugin is missing.

## Project conventions
- No mocking or stubbing: use real implementations; let code fail fast if deps are missing.
- No fallbacks or silent handling: no hidden catch blocks or auto‑retries; surface errors explicitly.
- DRY: extract shared logic/config into a single, reusable source of truth.

## Status
This is a minimal working prototype intended for clarity and extension. Tie multiplicity modeling and additional diagnostics can be added as strategies without changing the core API.
