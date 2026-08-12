# thurstone

Convert winning probabilities to relative abilities using the fast ability transform. 

[![PyPI version](https://badge.fury.io/py/thurstone.svg)](https://badge.fury.io/py/thurstone)
[![CI](https://github.com/microprediction/thurstone/actions/workflows/ci.yml/badge.svg)](https://github.com/microprediction/thurstone/actions/workflows/ci.yml)

## What it does

Given market odds or winning probabilities, infer the relative abilities of competitors.

**Input**: Market odds `[3.2, 4.8, 12.0, 7.5, 20.0]`  
**Output**: Relative abilities `[1.15, 0.73, -0.88, 0.21, -1.21]`

The model assumes each competitor's performance = true ability + random noise, and the best performance wins.

## Usage

```bash
pip install thurstone
```

```python
from thurstone import UniformLattice, Density, AbilityCalibrator, STD_L, STD_UNIT

# Setup
lattice = UniformLattice(L=STD_L, unit=STD_UNIT)
base = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0)
calibrator = AbilityCalibrator(base)

# Convert odds to abilities
odds = [3.2, 4.8, 12.0, 7.5, 20.0]
abilities = calibrator.solve_from_dividends(odds)
probabilities = calibrator.state_prices_from_ability(abilities)
```

## Exact joint inversion (Laplacian Newton–CG)

The Jacobian of the map from abilities to win probabilities is minus a weighted
graph Laplacian, and it can be applied to a vector in O(nM) without ever being
formed. `invert_outright_probabilities` exploits this for exact joint Newton–CG
inversion — all runners move together, no per-runner curve approximation:

```python
from thurstone import Density, UniformLattice, invert_outright_probabilities

lattice = UniformLattice(L=400, unit=0.05)
bases = [Density.skew_normal(lattice, loc=0.0, scale=s, a=0.0) for s in (0.8, 1.0, 1.2)]

result = invert_outright_probabilities(bases, [0.5, 0.3, 0.2])  # normalized prices
result.abilities  # mean-zero abilities reproducing the price ratios
result.converged  # honest flag; result.message explains any failure
```

Targets are matched in ratio (a single multiplicative renormalization absorbs
the lattice tie mass, so longshot probabilities keep their meaning), per-runner
distributions may be heterogeneous, and non-convergence is diagnosed — e.g. a
target requiring more ability spread than the lattice represents. The building
blocks `laplacian_weights`, `laplacian_matvec`, and `LaplacianOperator` are
exported for direct use.

## Applications

- E-commerce product ranking
- Search result relevance scoring  
- Financial instrument comparison
- Sports betting analysis
- Any competitive scenario with market-implied rankings

## Examples

```bash
python examples/global_calibration_demo.py      # 500 competitors
python examples/dynamic_calibration_demo.py     # Time-varying abilities
python examples/diffeomorphism_demo.py          # Advanced mappings
python examples/laplacian_newton_demo.py        # Laplacian Jacobian + Newton-CG inversion
```

## Documentation

**📖 [Full Documentation & Interactive Demos](https://thurstone.microprediction.org/)**

## Citation

Cotton, Peter. "Inferring Relative Ability from Winning Probability in Multientrant Contests." *SIAM Journal on Financial Mathematics* 12.1 (2021): 295-317.

## Development

```bash
pip install -e ".[test,viz]"
python scripts/format-code.py
pytest
```
