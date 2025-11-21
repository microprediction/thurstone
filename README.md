
# thurstone 

An object‑oriented toolkit for **lattice‑based performance models** and the **horse‑race inverse problem** (inferring relative ability from market prices, a.k.a. the ability transform). This package seeks to revitalize Thurstone Class V models which have tended to play a poor cousin to Plackett-Luce and other alternatives due to lack of tractability. It implements a relatively recent fast algorithm. 

## Uses 
(Beyond horseracing)

See the [notebook](https://github.com/microprediction/winning/blob/main/Ability_Transforms_Updated.ipynb) for examples of the use of this ability transform. 

See the [paper](https://github.com/microprediction/winning/blob/main/docs/Horse_Race_Problem__SIAM_.pdf) for why this is useful in lots of places, according to a wise man. For instance, the algorithm may also find use anywhere winning probabilities or frequencies are apparent, such as with e-commerce product placement, in web search, or, as is shown in the paper: addressing a fundamental problem of trade. 


## Highlights

This is the successor to the `winning` package. Essentially a clean up and mild generalization. New and inherited functionality includes:

- Uniform lattice with explicit `L` and `unit` (`UniformLattice`).
- Normalized `Density` with safe integer/fractional shifts, centering, convolution, and dilation. Negative mass is rejected; zero‑mass vectors are allowed as an explicit off‑lattice sentinel in extreme shifts.
- Clean `Race` and `StatePricer` API for risk‑neutral **state prices** (winning probabilities), now multiplicity‑aware.
- `AbilityCalibrator` solves the **inverse problem** with per‑runner monotone interpolation (sorted price→offset tables) and returns abilities in physical units.
- `ClusterSplitter` handles offsets that hang off the lattice (both directions) with symmetric clustering and coarse‑to‑fine recursion, including walkover behavior and an equal‑share rule only when all offsets hang on one side and are tightly bunched (spread < support width).
- `order_stats` module: multiplicity‑aware `winner_of_many` and `expected_payoff_with_multiplicity` (draws split by 1/(1 + multiplicity)).
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
- Forward pricing and inverse calibration use multiplicity‑aware payoffs: `win + draw/(1 + multiplicity_rest)`. This improves accuracy in dense/tie regimes.

## Testing
- Optional test extras: `pip install -e ".[test]"`
- Run tests: `pytest -q`
- Property tests (Hypothesis) are auto‑skipped if Hypothesis is not installed.
- A performance benchmark (pytest‑benchmark) is included and auto‑skips if the plugin is missing.
- Multiplicity tests ensure clone ties scale multiplicity correctly and that multiplicity‑aware payoffs differ from naive half‑point splits.

## Examples
- Plot offset densities for 150 runners:
  - Requires matplotlib: `pip install matplotlib`
  - Run: `python examples/plot_offset_densities_150.py`
- 2D calibration with per‑runner scales (150 runners, scales 15→20):
  - Run: `python examples/calibrate_with_scales_150.py`

## Cite
See 

- Cotton, Peter. “Inferring Relative Ability from Winning Probability in Multientrant Contests,” SIAM Journal on Financial Mathematics, 12(1), 295–317 (2021). DOI: `https://doi.org/10.1137/19M1276261`
- Original reference implementation and additional context: `https://github.com/microprediction/winning`

BibTeX:

```bibtex
@article{doi:10.1137/19M1276261,
  author = {Cotton, Peter},
  title = {Inferring Relative Ability from Winning Probability in Multientrant Contests},
  journal = {SIAM Journal on Financial Mathematics},
  volume = {12},
  number = {1},
  pages = {295-317},
  year = {2021},
  doi = {10.1137/19M1276261},
  URL = {https://doi.org/10.1137/19M1276261}
}
```

## Contribute

The most obvious improvement would be a rust implementation. 
