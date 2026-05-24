"""
Generate publication-quality figures for the Thurstone diffeomorphism paper.

This script creates all the figures needed for the research paper,
with publication-ready styling and clear visualizations.
"""

import os
import sys
from typing import List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thurstone import (CubeToSimplexMapping, SigmoidParams,
                       comprehensive_quality_assessment,
                       optimize_diffeomorphism)

# Set publication-quality matplotlib parameters
plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "font.serif": ["Times"],
        "text.usetex": False,  # Set to True if LaTeX is available
        "axes.linewidth": 1.2,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.figsize": [8, 6],
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def create_mapping_examples() -> List[Tuple[str, CubeToSimplexMapping]]:
    """Create example mappings with different characteristics."""
    mappings = []

    # 1. Symmetric mapping
    symmetric_params = [
        SigmoidParams(alpha=1.0, beta=4.0, gamma=0.5),
        SigmoidParams(alpha=1.0, beta=4.0, gamma=0.5),
    ]
    mappings.append(
        (
            "Symmetric",
            CubeToSimplexMapping(
                sigmoid_params=symmetric_params, special_horse_ability=0.0
            ),
        )
    )

    # 2. Asymmetric mapping
    asymmetric_params = [
        SigmoidParams(alpha=1.8, beta=6.0, gamma=0.3),
        SigmoidParams(alpha=0.9, beta=3.0, gamma=0.7),
    ]
    mappings.append(
        (
            "Asymmetric",
            CubeToSimplexMapping(
                sigmoid_params=asymmetric_params, special_horse_ability=0.4
            ),
        )
    )

    # 3. Sharp mapping (high beta)
    sharp_params = [
        SigmoidParams(alpha=1.2, beta=8.0, gamma=0.4),
        SigmoidParams(alpha=1.3, beta=9.0, gamma=0.6),
    ]
    mappings.append(
        (
            "Sharp",
            CubeToSimplexMapping(
                sigmoid_params=sharp_params, special_horse_ability=-0.2
            ),
        )
    )

    # 4. Smooth mapping (low beta)
    smooth_params = [
        SigmoidParams(alpha=1.1, beta=2.5, gamma=0.5),
        SigmoidParams(alpha=1.0, beta=2.0, gamma=0.5),
    ]
    mappings.append(
        (
            "Smooth",
            CubeToSimplexMapping(
                sigmoid_params=smooth_params, special_horse_ability=0.1
            ),
        )
    )

    return mappings


def figure_1_lattice_mappings():
    """Figure 1: Lattice point mappings for different configurations."""
    mappings = create_mapping_examples()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        "Figure 1: Cube-to-Simplex Lattice Point Mappings", fontsize=18, y=0.95
    )

    resolution = 21  # 21x21 grid for clear visualization
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    cube_points = np.column_stack([X.ravel(), Y.ravel()])

    for col, (name, mapping) in enumerate(mappings):
        # Top row: Cube lattice
        ax_cube = axes[0, col]
        ax_cube.scatter(X.ravel(), Y.ravel(), c="navy", alpha=0.6, s=15)
        ax_cube.set_xlim(-0.05, 1.05)
        ax_cube.set_ylim(-0.05, 1.05)
        ax_cube.set_xlabel("$x_1$")
        ax_cube.set_ylabel("$x_2$")
        ax_cube.set_title(f"{name}\nUnit Square $[0,1]^2$")
        ax_cube.grid(True, alpha=0.3)
        ax_cube.set_aspect("equal")

        # Bottom row: Simplex image
        ax_simplex = axes[1, col]

        simplex_points = mapping.batch_forward(cube_points)
        p1, p2, p3 = simplex_points[:, 0], simplex_points[:, 1], simplex_points[:, 2]

        # Color by third coordinate
        scatter = ax_simplex.scatter(
            p1, p2, c=p3, cmap="plasma", alpha=0.7, s=15, vmin=0, vmax=1
        )

        # Draw simplex boundary
        triangle = patches.Polygon(
            [(0, 0), (1, 0), (0, 1)], fill=False, edgecolor="black", linewidth=2
        )
        ax_simplex.add_patch(triangle)

        # Shade simplex interior
        triangle_fill = patches.Polygon(
            [(0, 0), (1, 0), (0, 1)], alpha=0.1, facecolor="gray"
        )
        ax_simplex.add_patch(triangle_fill)

        # Label vertices
        ax_simplex.annotate("$(1,0,0)$", xy=(1, 0), xytext=(1.05, -0.05), fontsize=10)
        ax_simplex.annotate("$(0,1,0)$", xy=(0, 1), xytext=(-0.1, 1.05), fontsize=10)
        ax_simplex.annotate("$(0,0,1)$", xy=(0, 0), xytext=(-0.1, -0.05), fontsize=10)

        ax_simplex.set_xlim(-0.15, 1.15)
        ax_simplex.set_ylim(-0.15, 1.15)
        ax_simplex.set_xlabel("$p_1$ (Horse 1 probability)")
        ax_simplex.set_ylabel("$p_2$ (Horse 2 probability)")
        ax_simplex.set_title("2-Simplex Image")
        ax_simplex.set_aspect("equal")

        # Add colorbar for last subplot
        if col == len(mappings) - 1:
            cbar = plt.colorbar(scatter, ax=ax_simplex, shrink=0.8)
            cbar.set_label("$p_3$ (Horse 3 probability)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig


def figure_2_3d_visualization():
    """Figure 2: 3D visualization of the mapping process."""
    # Use the asymmetric mapping for visual interest
    params = [
        SigmoidParams(alpha=1.5, beta=5.0, gamma=0.35),
        SigmoidParams(alpha=1.2, beta=4.5, gamma=0.65),
    ]
    mapping = CubeToSimplexMapping(params, special_horse_ability=0.2)

    fig = plt.figure(figsize=(15, 5))

    # Create grid
    resolution = 15
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    cube_points = np.column_stack([X.ravel(), Y.ravel()])
    simplex_points = mapping.batch_forward(cube_points)

    # Plot 1: Cube points
    ax1 = fig.add_subplot(131, projection="3d")
    scatter1 = ax1.scatter(
        X.ravel(),
        Y.ravel(),
        np.zeros_like(X.ravel()),
        c=range(len(X.ravel())),
        cmap="viridis",
        alpha=0.7,
        s=25,
    )
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_zlabel("$z = 0$")
    ax1.set_title("(a) Cube Points $[0,1]^2$")
    ax1.view_init(elev=20, azim=45)

    # Plot 2: Simplex points
    ax2 = fig.add_subplot(132, projection="3d")
    p1, p2, p3 = simplex_points[:, 0], simplex_points[:, 1], simplex_points[:, 2]
    scatter2 = ax2.scatter(
        p1, p2, p3, c=range(len(p1)), cmap="viridis", alpha=0.7, s=25
    )

    # Draw simplex vertices and edges
    vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ax2.scatter(*vertices.T, c="red", s=100, marker="o", alpha=0.8)

    # Draw simplex edges
    edges = [(0, 1), (0, 2), (1, 2)]
    for edge in edges:
        ax2.plot3D(*vertices[list(edge)].T, "k-", alpha=0.6, linewidth=2)

    ax2.set_xlabel("$p_1$")
    ax2.set_ylabel("$p_2$")
    ax2.set_zlabel("$p_3$")
    ax2.set_title("(b) Simplex Points $\\Delta^2$")
    ax2.view_init(elev=20, azim=45)

    # Plot 3: Combined view with flow
    ax3 = fig.add_subplot(133, projection="3d")

    # Show cube points at z=0
    ax3.scatter(
        X.ravel(),
        Y.ravel(),
        np.zeros_like(X.ravel()),
        c="blue",
        alpha=0.5,
        s=20,
        label="Cube",
    )

    # Show simplex points at z=1
    ax3.scatter(p1, p2, p3 + 1, c="red", alpha=0.5, s=20, label="Simplex")

    # Draw flow lines for sample of points
    n_flow = 30
    indices = np.linspace(0, len(cube_points) - 1, n_flow, dtype=int)
    for i in indices:
        x_start, y_start = cube_points[i]
        x_end, y_end, z_end = simplex_points[i]
        ax3.plot3D(
            [x_start, x_end],
            [y_start, y_end],
            [0, z_end + 1],
            "gray",
            alpha=0.4,
            linewidth=0.8,
        )

    ax3.set_xlabel("$x/p_1$")
    ax3.set_ylabel("$y/p_2$")
    ax3.set_zlabel("Level")
    ax3.set_title("(c) Mapping Flow")
    ax3.legend()
    ax3.view_init(elev=25, azim=60)

    plt.tight_layout()
    fig.suptitle(
        "Figure 2: Three-Dimensional Visualization of Cube-to-Simplex Mapping",
        fontsize=16,
        y=1.02,
    )

    return fig


def figure_3_quality_analysis():
    """Figure 3: Quality measure analysis and parameter sensitivity."""
    mappings = create_mapping_examples()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Figure 3: Quality Assessment and Parameter Analysis", fontsize=18)

    # Compute quality metrics for all mappings
    quality_data = {}
    for name, mapping in mappings:
        print(f"Computing quality for {name} mapping...")
        metrics = comprehensive_quality_assessment(
            mapping,
            symmetry_samples=3000,
            volume_samples=200,
            smoothness_samples=200,
            coverage_samples=2000,
            invertibility_samples=30,
            random_seed=42,
        )
        quality_data[name] = {
            "Symmetry": metrics.symmetry_score,
            "Volume Pres.": metrics.volume_preservation_score or 0,
            "Smoothness": metrics.smoothness_score or 0,
            "Coverage": metrics.coverage_score or 0,
            "Invertibility": metrics.invertibility_score or 0,
            "Overall": metrics.overall_score(),
        }

    # Plot 1: Quality comparison radar chart
    ax1 = axes[0, 0]
    categories = ["Symmetry", "Volume Pres.", "Smoothness", "Coverage", "Invertibility"]

    # Create radar chart data
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    colors = ["blue", "red", "green", "orange"]
    for i, (name, mapping) in enumerate(mappings):
        values = [quality_data[name][cat] for cat in categories]
        values += values[:1]

        ax1.plot(angles, values, "o-", linewidth=2, label=name, color=colors[i])
        ax1.fill(angles, values, alpha=0.1, color=colors[i])

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories)
    ax1.set_ylim(0, 1)
    ax1.set_title("(a) Quality Profiles")
    ax1.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
    ax1.grid(True)

    # Plot 2: Overall quality scores
    ax2 = axes[0, 1]
    names = [name for name, _ in mappings]
    scores = [quality_data[name]["Overall"] for name in names]
    colors_bar = ["blue", "red", "green", "orange"]

    bars = ax2.bar(names, scores, color=colors_bar, alpha=0.7)
    ax2.set_ylabel("Overall Quality Score")
    ax2.set_title("(b) Overall Quality Comparison")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )

    # Plot 3: Parameter visualization
    ax3 = axes[0, 2]
    param_names = []
    param_matrix = []

    for name, mapping in mappings:
        row = []
        for i, param in enumerate(mapping.sigmoid_params):
            row.extend([param.alpha, param.beta, param.gamma])
        row.append(mapping.special_horse_ability)
        param_matrix.append(row)

    param_names = ["α₁", "β₁", "γ₁", "α₂", "β₂", "γ₂", "a*"]
    param_matrix = np.array(param_matrix)

    im = ax3.imshow(param_matrix, cmap="RdYlBu", aspect="auto")
    ax3.set_xticks(range(len(param_names)))
    ax3.set_xticklabels(param_names)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names)
    ax3.set_title("(c) Parameter Values")

    # Add text annotations
    for i in range(len(names)):
        for j in range(len(param_names)):
            ax3.text(
                j,
                i,
                f"{param_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(param_matrix[i, j]) > 1 else "black",
            )

    plt.colorbar(im, ax=ax3, shrink=0.6)

    # Plot 4: Jacobian determinant heatmap for asymmetric mapping
    ax4 = axes[1, 0]
    asymmetric_mapping = mappings[1][1]  # Use asymmetric mapping

    print("Computing Jacobian heatmap...")
    resolution = 25
    margin = 0.05
    x = np.linspace(margin, 1 - margin, resolution)
    y = np.linspace(margin, 1 - margin, resolution)
    X, Y = np.meshgrid(x, y)

    jacobian_dets = np.zeros_like(X)
    epsilon = 1e-4

    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], Y[i, j]])

            # Compute Jacobian via finite differences
            base_output = asymmetric_mapping(point)
            jacobian = np.zeros((3, 2))

            for dim in range(2):
                point_plus = point.copy()
                point_plus[dim] += epsilon
                point_minus = point.copy()
                point_minus[dim] -= epsilon

                output_plus = asymmetric_mapping(point_plus)
                output_minus = asymmetric_mapping(point_minus)
                jacobian[:, dim] = (output_plus - output_minus) / (2 * epsilon)

            # Use 2x2 submatrix determinant
            det = np.linalg.det(jacobian[:2, :])
            jacobian_dets[i, j] = abs(det)

    im4 = ax4.imshow(
        jacobian_dets,
        extent=[margin, 1 - margin, margin, 1 - margin],
        origin="lower",
        cmap="plasma",
        aspect="equal",
    )
    ax4.set_xlabel("$x_1$")
    ax4.set_ylabel("$x_2$")
    ax4.set_title("(d) Jacobian Determinant |det(J)|")
    plt.colorbar(im4, ax=ax4, shrink=0.8)

    # Plot 5: Coverage visualization for asymmetric mapping
    ax5 = axes[1, 1]

    # Sample points and show distribution
    n_sample = 2000
    cube_samples = np.random.uniform(0, 1, (n_sample, 2))
    simplex_samples = asymmetric_mapping.batch_forward(cube_samples)

    # Create hexbin plot for density
    hexplot = ax5.hexbin(
        simplex_samples[:, 0],
        simplex_samples[:, 1],
        C=simplex_samples[:, 2],
        gridsize=20,
        cmap="viridis",
        extent=[0, 1, 0, 1],
    )

    # Draw simplex boundary
    triangle = patches.Polygon(
        [(0, 0), (1, 0), (0, 1)], fill=False, edgecolor="white", linewidth=2
    )
    ax5.add_patch(triangle)

    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_xlabel("$p_1$")
    ax5.set_ylabel("$p_2$")
    ax5.set_title("(e) Coverage Density")
    ax5.set_aspect("equal")
    plt.colorbar(hexplot, ax=ax5, shrink=0.8, label="$p_3$")

    # Plot 6: Parameter sensitivity
    ax6 = axes[1, 2]

    # Show how quality changes with one parameter
    base_params = [SigmoidParams(1.0, 4.0, 0.5), SigmoidParams(1.0, 4.0, 0.5)]
    base_mapping = CubeToSimplexMapping(base_params, special_horse_ability=0.0)

    alpha_values = np.linspace(0.5, 2.5, 8)
    alpha_scores = []

    print("Computing parameter sensitivity...")
    for alpha in alpha_values:
        test_params = [SigmoidParams(alpha, 4.0, 0.5), SigmoidParams(1.0, 4.0, 0.5)]
        test_mapping = CubeToSimplexMapping(test_params, special_horse_ability=0.0)
        metrics = comprehensive_quality_assessment(
            test_mapping,
            symmetry_samples=1500,
            volume_samples=100,
            smoothness_samples=100,
            coverage_samples=1000,
            invertibility_samples=15,
            random_seed=42,
        )
        alpha_scores.append(metrics.overall_score())

    ax6.plot(alpha_values, alpha_scores, "bo-", linewidth=2, markersize=6)
    ax6.set_xlabel("$\\alpha_1$ (Scale Parameter)")
    ax6.set_ylabel("Overall Quality Score")
    ax6.set_title("(f) Parameter Sensitivity")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    return fig


def figure_4_optimization_results():
    """Figure 4: Optimization algorithm comparison and results."""
    print("Running optimization comparison (this may take a few minutes)...")

    # Run multiple optimization algorithms
    algorithms = ["random", "evolutionary"]
    results = {}

    for alg in algorithms:
        print(f"Running {alg} optimization...")
        result = optimize_diffeomorphism(
            k=2,
            optimizer=alg,
            max_evaluations=30,  # Reduced for demo
            quality_weights={"symmetry": 2.0, "coverage": 1.5, "smoothness": 1.0},
            random_seed=42,
        )
        results[alg] = result

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Figure 4: Optimization Algorithm Comparison and Results", fontsize=16)

    # Plot 1: Convergence comparison
    ax1 = axes[0, 0]

    colors = {"random": "blue", "evolutionary": "red"}
    for alg, result in results.items():
        history = result.optimization_history
        evaluations = [h["evaluation"] for h in history]
        scores = [h["score"] for h in history]

        # Compute running maximum for convergence
        running_max = []
        current_max = 0
        for score in scores:
            current_max = max(current_max, score)
            running_max.append(current_max)

        ax1.plot(
            evaluations,
            running_max,
            color=colors[alg],
            linewidth=2,
            label=f"{alg.title()} Search",
        )

    ax1.set_xlabel("Evaluation Number")
    ax1.set_ylabel("Best Quality Score")
    ax1.set_title("(a) Convergence Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final score comparison
    ax2 = axes[0, 1]

    alg_names = list(results.keys())
    final_scores = [results[alg].best_score for alg in alg_names]
    colors_bar = [colors[alg] for alg in alg_names]

    bars = ax2.bar(
        [alg.title() for alg in alg_names], final_scores, color=colors_bar, alpha=0.7
    )
    ax2.set_ylabel("Best Quality Score")
    ax2.set_title("(b) Final Performance")
    ax2.grid(True, alpha=0.3)

    for bar, score in zip(bars, final_scores):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )

    # Plot 3: Best mapping visualization
    ax3 = axes[1, 0]

    best_result = max(results.values(), key=lambda r: r.best_score)
    best_mapping = best_result.best_mapping

    # Show lattice mapping for best result
    resolution = 15
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    cube_points = np.column_stack([X.ravel(), Y.ravel()])
    simplex_points = best_mapping.batch_forward(cube_points)

    scatter = ax3.scatter(
        simplex_points[:, 0],
        simplex_points[:, 1],
        c=simplex_points[:, 2],
        cmap="plasma",
        alpha=0.7,
        s=20,
    )

    # Draw simplex
    triangle = patches.Polygon(
        [(0, 0), (1, 0), (0, 1)], fill=False, edgecolor="black", linewidth=2
    )
    ax3.add_patch(triangle)

    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_xlabel("$p_1$")
    ax3.set_ylabel("$p_2$")
    ax3.set_title("(c) Optimized Mapping")
    ax3.set_aspect("equal")
    plt.colorbar(scatter, ax=ax3, shrink=0.8, label="$p_3$")

    # Plot 4: Quality breakdown of best mapping
    ax4 = axes[1, 1]

    best_metrics = best_result.best_metrics
    quality_names = ["Symmetry", "Volume", "Smoothness", "Coverage", "Invertibility"]
    quality_values = [
        best_metrics.symmetry_score,
        best_metrics.volume_preservation_score or 0,
        best_metrics.smoothness_score or 0,
        best_metrics.coverage_score or 0,
        best_metrics.invertibility_score or 0,
    ]

    bars = ax4.bar(
        quality_names,
        quality_values,
        color=["blue", "green", "orange", "red", "purple"],
        alpha=0.7,
    )
    ax4.set_ylabel("Quality Score")
    ax4.set_title("(d) Optimized Quality Profile")
    ax4.set_ylim(0, 1)
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, alpha=0.3)

    for bar, value in zip(bars, quality_values):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    return fig


def main():
    """Generate all paper figures."""
    print(
        "Generating publication-quality figures for Thurstone diffeomorphism paper..."
    )
    print("This may take several minutes due to quality assessments and optimizations.")

    # Create output directory
    output_dir = "paper_figures"
    os.makedirs(output_dir, exist_ok=True)

    figures = []

    # Generate Figure 1: Lattice mappings
    print("\nGenerating Figure 1: Lattice point mappings...")
    fig1 = figure_1_lattice_mappings()
    fig1.savefig(
        f"{output_dir}/figure_1_lattice_mappings.png",
        bbox_inches="tight",
        facecolor="white",
    )
    figures.append(("Figure 1", fig1))

    # Generate Figure 2: 3D visualization
    print("\nGenerating Figure 2: 3D visualization...")
    fig2 = figure_2_3d_visualization()
    fig2.savefig(
        f"{output_dir}/figure_2_3d_visualization.png",
        bbox_inches="tight",
        facecolor="white",
    )
    figures.append(("Figure 2", fig2))

    # Generate Figure 3: Quality analysis
    print("\nGenerating Figure 3: Quality analysis...")
    fig3 = figure_3_quality_analysis()
    fig3.savefig(
        f"{output_dir}/figure_3_quality_analysis.png",
        bbox_inches="tight",
        facecolor="white",
    )
    figures.append(("Figure 3", fig3))

    # Generate Figure 4: Optimization results
    print("\nGenerating Figure 4: Optimization results...")
    fig4 = figure_4_optimization_results()
    fig4.savefig(
        f"{output_dir}/figure_4_optimization_results.png",
        bbox_inches="tight",
        facecolor="white",
    )
    figures.append(("Figure 4", fig4))

    print(f"\n✅ All figures generated successfully!")
    print(f"📁 Saved to directory: {output_dir}/")
    print(f"🖼️  Generated {len(figures)} publication-quality figures:")

    for name, fig in figures:
        print(f"   • {name}: {fig.get_figwidth():.1f}x{fig.get_figheight():.1f} inches")

    print(f"\n📊 Figure Summary:")
    print(
        f"   • Figure 1: Demonstrates lattice mappings for 4 different parameter configurations"
    )
    print(f"   • Figure 2: Shows 3D visualization of the mapping process")
    print(
        f"   • Figure 3: Comprehensive quality analysis including radar charts and heatmaps"
    )
    print(f"   • Figure 4: Optimization algorithm comparison and best results")

    print(f"\n📝 These figures are ready for inclusion in the research paper!")
    print(f"   • High resolution (300 DPI) for publication quality")
    print(f"   • Clean typography using serif fonts")
    print(f"   • Clear labels and legends for academic presentation")
    print(f"   • Consistent color schemes across related visualizations")

    return figures


if __name__ == "__main__":
    main()
