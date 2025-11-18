
# thurstone

An object‑oriented toolkit for **lattice‑based performance models** and the **horse‑race inverse problem** (inferring relative ability from market prices). It is inspired by lattice approaches but aims to be clearer and more modular.

## Highlights
- Uniform lattice with explicit `L` and `unit` (`UniformLattice`).
- Immutable, normalized `Density` with safe integer/fractional shifts, centering, convolution, and dilation.
- Clean `Race` and `StatePricer` API for risk‑neutral **state prices** (winning probabilities).
- `AbilityCalibrator` solves the **inverse problem** with a monotone interpolation guard.
- `ClusterSplitter` handles offsets that hang off the lattice (both directions) with symmetric clustering and coarse‑to‑fine recursion.
- Tie handling via pluggable `TieModel` (default: 0.5 split).

## Quickstart

```python
from thurstone import UniformLattice, Density, AbilityCalibrator, STD_L, STD_UNIT

lat = UniformLattice(L=STD_L, unit=STD_UNIT)
base = Density.skew_normal(lat, loc=0.0, scale=1.0, a=0.0)  # symmetric base

# Inverse calibration from dividends
from thurstone import StatePricer
dividends = [3.2, 4.8, 12.0, 7.5, 20.0]
cal = AbilityCalibrator(base)
ability = cal.solve_from_dividends(dividends)     # offsets (scale‑free in lattice steps)
prices = cal.state_prices_from_ability([a*lat.unit for a in ability])
```

## Design notes
- Fractional shifts are performed on the CDF then differenced back to preserve mass and monotonicity.
- The inverse loop sorts `(price, offset)` pairs before `np.interp` to avoid order assumptions.
- `ClusterSplitter` splits at the largest gap near the center, evaluates **both sides**, and combines refined within‑group prices using coarse group shares from a dilated lattice.

## Status
This is a minimal working prototype intended for clarity and extension. Tie multiplicity modeling and additional diagnostics can be added as strategies without changing the core API.
