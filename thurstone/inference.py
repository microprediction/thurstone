
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple
import numpy as np
from .density import Density
from .lattice import UniformLattice
from .pricing import Race, StatePricer
from .clustering import ClusterSplitter

# ---- Densities from offsets ----

def densities_from_offsets(base: Density, offsets: Sequence[float]) -> List[Density]:
    out = []
    for o in offsets:
        out.append(base.shift_fractional(o))
    return out

def state_prices_from_densities(densities: Sequence[Density]) -> List[float]:
    race = Race(list(densities))
    return list(race.state_prices())

# ---- Implicit map: offset -> price ----

def implicit_state_prices(base: Density, offsets: Sequence[float]) -> List[float]:
    dens = densities_from_offsets(base, offsets)
    return state_prices_from_densities(dens)

# ---- Calibration (inverse) ----

@dataclass
class AbilityCalibrator:
    base: Density
    offset_grid: Optional[Sequence[float]] = None
    n_iter: int = 3

    def __post_init__(self):
        if self.offset_grid is None:
            L = self.base.lattice.L
            self.offset_grid = list(range(int(-L/2), int(L/2)))[::-1]

    def solve_from_prices(self, prices: Sequence[float], *, initial_offsets: Optional[Sequence[float]]=None) -> List[float]:
        if initial_offsets is None:
            initial_offsets = [0.0 for _ in prices]
        unit = self.base.lattice.unit
        # Work internally in lattice steps
        offsets = np.array(initial_offsets, dtype=float) / unit

        for _ in range(self.n_iter):
            n = len(prices)
            new_offsets = offsets.copy()
            grid = list(self.offset_grid)
            # Update each runner independently via monotone interpolation,
            # conditioning on current offsets of others.
            for i in range(n):
                # Tabulate the implicit curve p_i(g) with others fixed
                implied_prices_i = []
                cur = offsets.copy()
                for g in grid:
                    cur[i] = g
                    pvec = implicit_state_prices(self.base, cur)
                    implied_prices_i.append(pvec[i])
                implied_prices_i = np.array(implied_prices_i, dtype=float)
                # Enforce monotone xp by sorting and uniquing keys
                pairs = sorted(zip(implied_prices_i, grid), key=lambda t: t[0])
                xp = np.array([t[0] for t in pairs], dtype=float)  # prices
                fp = np.array([t[1] for t in pairs], dtype=float)  # offsets
                # Unique xp to satisfy numpy.interp strictness; keep first occurrence
                xp_unique, idx = np.unique(xp, return_index=True)
                fp_unique = fp[idx]
                # Clip target into bracket and interpolate
                target = float(prices[i])
                target = float(np.clip(target, xp_unique.min(), xp_unique.max()))
                gi = float(np.interp(target, xp_unique, fp_unique))
                new_offsets[i] = gi
            offsets = new_offsets

        # Convert lattice-step offsets back to ability units
        return list(np.array(offsets, dtype=float) * unit)

    def solve_from_dividends(self, dividends: Sequence[float], nan_value: float=2000.0) -> List[float]:
        prices = StatePricer.prices_from_dividends(dividends, nan_value=nan_value)
        return self.solve_from_prices(prices)

    # forward direction (ability -> state prices/dividends), with clustering for extended offsets
    def state_prices_from_ability(self, ability: Sequence[float]) -> List[float]:
        offsets = [a / self.base.lattice.unit for a in ability]
        splitter = ClusterSplitter()
        return splitter.extended_state_prices(self.base, offsets)

    def dividends_from_ability(self, ability: Sequence[float], multiplicity: float=1.0) -> List[float]:
        prices = self.state_prices_from_ability(ability)
        return list(1.0/(multiplicity*np.array(prices)))
