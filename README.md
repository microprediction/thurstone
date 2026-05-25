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
