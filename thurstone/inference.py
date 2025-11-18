
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Sequence, Optional, Tuple
import numpy as np
from .density import Density
from .lattice import UniformLattice
from .pricing import Race, StatePricer
from .order_stats import winner_of_many, expected_payoff_with_multiplicity
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
    # Optional per-runner scale handling for 2D calibration (loc, scale)
    scales: Optional[np.ndarray] = field(default=None, repr=False)
    scale_span: float = 0.5          # +/- around each scale value (physical units of scale)
    scale_steps: int = 3             # number of scale samples per runner
    loc_span: float = 5.0            # +/- location range (physical units) for 2D path
    loc_step: float = 0.25           # step for location grid (physical units)
    skew_a: float = 0.0              # skew-normal 'a' used in density_for

    def __post_init__(self):
        if self.offset_grid is None:
            L = self.base.lattice.L
            self.offset_grid = list(range(int(-L/2), int(L/2)))[::-1]

    def set_scales(self, scales: Sequence[float]) -> None:
        self.scales = np.asarray(scales, dtype=float)

    def density_for(self, loc: float, scale: float) -> Density:
        lat = self.base.lattice
        return Density.skew_normal(lat, loc=loc, scale=scale, a=self.skew_a)

    def solve_from_prices(self, prices: Sequence[float], *, initial_offsets: Optional[Sequence[float]]=None) -> List[float]:
        prices_arr = np.asarray(prices, dtype=float)
        n = len(prices_arr)
        # 2D path if scales provided and length matches; else 1D fallback
        if self.scales is None or len(self.scales) != n:
            if initial_offsets is None:
                initial_offsets = [0.0 for _ in prices_arr]
            unit = self.base.lattice.unit
            # Work internally in lattice steps
            offsets = np.array(initial_offsets, dtype=float) / unit

            for _ in range(self.n_iter):
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
                    xp_unique, idx = np.unique(xp, return_index=True)
                    fp_unique = fp[idx]
                    target = float(prices_arr[i])
                    target = float(np.clip(target, xp_unique.min(), xp_unique.max()))
                    gi = float(np.interp(target, xp_unique, fp_unique))
                    new_offsets[i] = gi
                offsets = new_offsets

            # Convert lattice-step offsets back to ability units
            return list(np.array(offsets, dtype=float) * unit)

        # ---- 2D calibration in (loc, scale) per runner ----
        scales = np.asarray(self.scales, dtype=float)
        unit = self.base.lattice.unit
        # Initialize locs (abilities) in physical units
        if initial_offsets is None:
            locs = np.zeros(n, dtype=float)
        else:
            locs = np.asarray(initial_offsets, dtype=float) * unit

        # Location grid (physical units) around 0
        loc_grid = np.arange(-self.loc_span, self.loc_span + self.loc_step, self.loc_step, dtype=float)

        def scale_grid_for(si: float) -> np.ndarray:
            if self.scale_steps <= 1:
                return np.array([max(1e-6, si)], dtype=float)
            offsets_scale = np.linspace(-self.scale_span, self.scale_span, self.scale_steps, dtype=float)
            sg = si + offsets_scale
            return np.maximum(1e-6, sg.astype(float))

        for _ in range(self.n_iter):
            # Precompute field distribution once per iteration (current locs/scales), multiplicity-aware
            current_densities = [self.density_for(float(locs[j]), float(scales[j])) for j in range(n)]
            densityAll, multAll = winner_of_many(current_densities)
            cdfAll = densityAll.cdf()
            for i in range(n):
                si = float(scales[i])
                pi_target = float(prices_arr[i])
                sg = scale_grid_for(si)
                loc_estimates = []
                for s in sg:
                    # Build p_i(loc | scale=s) curve with others fixed
                    p_curve = []
                    for loc_candidate in loc_grid:
                        d_i_candidate = self.density_for(float(loc_candidate), float(s))
                        payoff_vec = expected_payoff_with_multiplicity(d_i_candidate, densityAll, multAll, cdf=None, cdfAll=cdfAll)
                        p_i = float(np.sum(payoff_vec))
                        p_curve.append(p_i)
                    p_curve = np.array(p_curve, dtype=float)
                    pairs = sorted(zip(p_curve, loc_grid), key=lambda t: t[0])
                    xp = np.array([pp for pp, _ in pairs], dtype=float)
                    fp = np.array([ll for _, ll in pairs], dtype=float)
                    p_clamped = float(np.clip(pi_target, xp.min(), xp.max()))
                    loc_s = float(np.interp(p_clamped, xp, fp))
                    loc_estimates.append(loc_s)
                loc_estimates = np.array(loc_estimates, dtype=float)
                if len(sg) == 1:
                    locs[i] = loc_estimates[0]
                else:
                    locs[i] = float(np.interp(si, sg, loc_estimates))

        # Remove translation (identifiability) by median-centering
        locs = locs - np.median(locs)
        return list(locs)

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
