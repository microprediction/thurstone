"""
Example: 150 runners – visualize offset-shifted densities as a heatmap and a few overlays.

Setup:
    pip install matplotlib

Run:
    python examples/plot_offset_densities_150.py
"""
import numpy as np
import matplotlib.pyplot as plt
from thurstone import UniformLattice, Density


def main():
    # Lattice and symmetric base density
    lattice = UniformLattice(L=500, unit=0.1)
    base = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0)

    # 150 abilities across a reasonable range (physical units)
    num_runners = 150
    abilities = np.linspace(-4.0, 4.0, num_runners)

    # Shift base by ability / unit to get per-runner densities
    densities = [base.shift_fractional(a / lattice.unit) for a in abilities]
    P = np.vstack([d.p for d in densities])  # shape (N, 2L+1)
    x = lattice.grid

    # Heatmap of all offset densities
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(
        P,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], 0, num_runners - 1],
        cmap="viridis",
    )
    ax.set_title("Offset densities for 150 runners (heatmap)")
    ax.set_xlabel("Performance (lattice units)")
    ax.set_ylabel("Runner index (sorted by ability)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Probability density")
    plt.tight_layout()

    # Overlay a few representative densities
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    sample_idx = [0, 30, 60, 90, 120, 149]
    for idx in sample_idx:
        ax2.plot(x, densities[idx].p, lw=1.5, label=f"runner {idx} (a={abilities[idx]:.2f})")
    ax2.set_title("Representative offset densities (subset overlays)")
    ax2.set_xlabel("Performance (lattice units)")
    ax2.set_ylabel("Density")
    ax2.legend(ncols=2, fontsize=8)
    ax2.grid(True, alpha=0.25)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()


