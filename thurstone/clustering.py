
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np
from .density import Density
from .lattice import UniformLattice

def _int_centered(offsets: Sequence[float]) -> List[float]:
    finite = [o for o in offsets if np.isfinite(o)]
    if not finite:
        return list(offsets)
    mean_int = int(np.mean(finite))
    return [o - mean_int for o in offsets]

def _divide_offsets(centered_offsets: Sequence[float]) -> float:
    """Choose a divider that maximizes the biggest gap near the center; ensures two non-empty groups."""
    srt = sorted(centered_offsets)
    if len(srt) <= 2:
        return float(np.mean(srt))
    gaps = np.diff(srt)
    idx = int(np.argmax(np.abs(gaps)))
    return float(0.5*(srt[idx] + srt[idx+1]))

@dataclass
class ClusterSplitter:
    unit_ratio: float = 3.0
    max_depth: int = 3

    def extended_state_prices(self, base: Density, offsets: Sequence[float]) -> List[float]:
        """Offsets may include +/-inf; returns normalized winning probabilities."""
        n = len(offsets)
        if n == 1:
            return [1.0]
        # handle +inf (no chance)
        pos_inf_idx = [i for i,o in enumerate(offsets) if o == float('inf')]
        if pos_inf_idx:
            finite_idx = [i for i in range(n) if i not in pos_inf_idx]
            if not finite_idx:
                return [1.0/n for _ in range(n)]
            finite_prices = self.extended_state_prices(base, _int_centered([offsets[i] for i in finite_idx]))
            out = [0.0 for _ in range(n)]
            for j, idx in enumerate(finite_idx):
                out[idx] = finite_prices[j]
            return out
        # handle -inf (certain winners share)
        neg_inf_idx = [i for i,o in enumerate(offsets) if o == float('-inf')]
        if neg_inf_idx:
            p = [0.0 for _ in range(n)]
            share = 1.0/len(neg_inf_idx)
            for idx in neg_inf_idx:
                p[idx] = share
            return p

        L = base.lattice.L
        W = int(base.approx_support_width())
        centered = _int_centered(offsets)

        # boundaries
        lower_bound = -L + W
        upper_bound = L - W
        hang_left = [i for i,o in enumerate(centered) if o < lower_bound]
        hang_right = [i for i,o in enumerate(centered) if o > upper_bound]
        # If all contestants hang on the same side and are tightly bunched relative to support,
        # treat them as indistinguishable at this resolution (equal share).
        if len(hang_left) == n or len(hang_right) == n:
            spread = float(max(centered) - min(centered)) if n > 0 else 0.0
            if spread < W:
                return [1.0/n for _ in range(n)]
        if (not hang_left) and (not hang_right):
            from .inference import densities_from_offsets, state_prices_from_densities
            dens = densities_from_offsets(base, centered)
            return state_prices_from_densities(dens)

        # stop recursion
        if self.max_depth <= 0:
            for i in hang_right:
                centered[i] = float('inf')
            for i in hang_left:
                centered[i] = float('-inf')
            return self.extended_state_prices(base, centered)

        # split symmetrically
        divider = _divide_offsets(centered)
        left_idx  = [i for i,o in enumerate(centered) if o < divider]
        right_idx = [i for i,o in enumerate(centered) if o >= divider]
        if len(left_idx)==0 or len(right_idx)==0:
            if len(left_idx)==0:
                left_idx = [int(np.argmin(centered))]
                right_idx = [i for i in range(n) if i not in left_idx]
            else:
                right_idx = [int(np.argmax(centered))]
                left_idx = [i for i in range(n) if i not in right_idx]

        # coarse group shares
        dilated = base.dilate(unit_ratio=self.unit_ratio)
        dil_offsets = [o / self.unit_ratio for o in centered]
        from .inference import densities_from_offsets, state_prices_from_densities
        coarse_prices = state_prices_from_densities(densities_from_offsets(dilated, dil_offsets))
        left_share  = float(sum(coarse_prices[i] for i in left_idx))
        right_share = 1.0 - left_share

        # refine inside groups
        left_prices_rel  = self.__class__(self.unit_ratio, self.max_depth - 1).extended_state_prices(base, [centered[i] for i in left_idx])
        right_prices_rel = self.__class__(self.unit_ratio, self.max_depth - 1).extended_state_prices(base, [centered[i] for i in right_idx])

        out = [0.0 for _ in range(n)]
        for j, idx in enumerate(left_idx):
            out[idx] = left_share * left_prices_rel[j]
        for j, idx in enumerate(right_idx):
            out[idx] = right_share * right_prices_rel[j]

        S = sum(out)
        if S <= 0:
            raise ValueError("Extended state prices have non-positive total mass.")
        assert 0.999 <= S <= 1.001, f"State prices not normalized in extended offsets; sum={S}"
        return [oi / S for oi in out]
