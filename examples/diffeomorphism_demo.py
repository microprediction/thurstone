"""
Comprehensive demonstration of Thurstone-based cube-to-simplex diffeomorphisms.

This example showcases the complete framework:
1. Creating diffeomorphisms with different parameter choices
2. Assessing quality using multiple metrics
3. Optimizing parameters for better performance
4. Visualizing mappings and results

Run this script to see the framework in action!
"""

import os
import sys

import numpy as np

# Add the parent directory to Python path to import thurstone modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thurstone.cube_to_simplex import (CubeToSimplexMapping,  # noqa: E402
                                       SigmoidParams)
from thurstone.optimization import ParameterBounds  # noqa: E402
from thurstone.optimization import optimize_diffeomorphism
from thurstone.quality_assessment import QualityMetrics  # noqa: E402
from thurstone.quality_assessment import comprehensive_quality_assessment


def create_example_mappings() -> dict:
    """Create several example mappings with different characteristics."""
    mappings = {}

    # 1. Symmetric mapping (all parameters similar)
    symmetric_params = [
        SigmoidParams(alpha=1.0, beta=4.0, gamma=0.5),
        SigmoidParams(alpha=1.0, beta=4.0, gamma=0.5),
    ]
    mappings["symmetric"] = CubeToSimplexMapping(
        sigmoid_params=symmetric_params, special_horse_ability=0.0, noise_scale=1.0
    )

    # 2. Asymmetric mapping (different parameters)
    asymmetric_params = [
        SigmoidParams(alpha=2.0, beta=6.0, gamma=0.2),
        SigmoidParams(alpha=0.8, beta=3.0, gamma=0.8),
    ]
    mappings["asymmetric"] = CubeToSimplexMapping(
        sigmoid_params=asymmetric_params, special_horse_ability=0.5, noise_scale=1.0
    )

    # 3. Steep mapping (high beta values)
    steep_params = [
        SigmoidParams(alpha=1.5, beta=10.0, gamma=0.3),
        SigmoidParams(alpha=1.2, beta=8.0, gamma=0.7),
    ]
    mappings["steep"] = CubeToSimplexMapping(
        sigmoid_params=steep_params, special_horse_ability=-0.3, noise_scale=1.0
    )

    # 4. Gentle mapping (low beta values)
    gentle_params = [
        SigmoidParams(alpha=1.0, beta=2.0, gamma=0.5),
        SigmoidParams(alpha=1.0, beta=2.0, gamma=0.5),
    ]
    mappings["gentle"] = CubeToSimplexMapping(
        sigmoid_params=gentle_params, special_horse_ability=0.0, noise_scale=1.0
    )

    return mappings


def demonstrate_basic_usage():
    """Demonstrate basic usage of the cube-to-simplex mapping."""
    print("=" * 60)
    print("BASIC USAGE DEMONSTRATION")
    print("=" * 60)

    # Create a simple mapping
    sigmoid_params = [
        SigmoidParams(alpha=1.5, beta=4.0, gamma=0.3),
        SigmoidParams(alpha=1.2, beta=5.0, gamma=0.7),
    ]
    mapping = CubeToSimplexMapping(
        sigmoid_params=sigmoid_params, special_horse_ability=0.2, noise_scale=1.0
    )

    print(f"Created mapping for k={mapping.k} (triangle)")
    print(f"Sigmoid parameters: {mapping.sigmoid_params}")
    print(f"Special horse ability: {mapping.special_horse_ability}")

    # Test some specific points
    test_points = [
        [0.0, 0.0],  # Bottom-left corner
        [1.0, 1.0],  # Top-right corner
        [0.5, 0.5],  # Center
        [0.2, 0.8],  # Asymmetric point
    ]

    print("\nTesting specific points:")
    print("Cube Point      → Simplex Point (p₁, p₂, p₃)")
    print("-" * 50)
    for point in test_points:
        simplex_point = mapping(point)
        print(
            f"{point!s:15} → ({simplex_point[0]:.3f}, {simplex_point[1]:.3f}, {simplex_point[2]:.3f})"
        )

    # Verify probabilities sum to 1
    print("\nVerifying probability constraints:")
    for point in test_points:
        simplex_point = mapping(point)
        total = np.sum(simplex_point)
        print(f"Sum for {point}: {total:.6f} {'✓' if abs(total - 1.0) < 1e-6 else '✗'}")

    return mapping


def compare_mappings():
    """Compare different mapping configurations."""
    print("\n" + "=" * 60)
    print("MAPPING COMPARISON")
    print("=" * 60)

    mappings = create_example_mappings()

    print("Assessing quality for different mapping configurations...")
    results = {}

    for name, mapping in mappings.items():
        print(f"\nEvaluating '{name}' mapping...")

        # Use smaller sample sizes for faster demo
        metrics = comprehensive_quality_assessment(
            mapping,
            symmetry_samples=3000,
            volume_samples=200,
            smoothness_samples=200,
            coverage_samples=2000,
            invertibility_samples=30,
            random_seed=42,
        )

        results[name] = metrics
        print(f"Overall score: {metrics.overall_score():.4f}")

    # Display comparison table
    print("\n" + "=" * 80)
    print("QUALITY COMPARISON TABLE")
    print("=" * 80)
    print(
        f"{'Mapping':<12} {'Overall':<8} {'Symmetry':<9} {'Volume':<8} {'Smooth':<8} {'Coverage':<9} {'Invert':<8}"
    )
    print("-" * 80)

    for name, metrics in results.items():
        print(
            f"{name:<12} "
            f"{metrics.overall_score():<8.4f} "
            f"{metrics.symmetry_score:<9.4f} "
            f"{(metrics.volume_preservation_score or 0):<8.4f} "
            f"{(metrics.smoothness_score or 0):<8.4f} "
            f"{(metrics.coverage_score or 0):<9.4f} "
            f"{(metrics.invertibility_score or 0):<8.4f}"
        )

    # Find best mapping
    best_name = max(results.keys(), key=lambda k: results[k].overall_score())
    best_score = results[best_name].overall_score()

    print(f"\nBest mapping: '{best_name}' with overall score {best_score:.4f}")

    return results


def demonstrate_optimization():
    """Demonstrate parameter optimization."""
    print("\n" + "=" * 60)
    print("PARAMETER OPTIMIZATION DEMONSTRATION")
    print("=" * 60)

    print("Optimizing parameters for k=2 triangle mapping...")

    # Use smaller evaluation budget for demo
    result = optimize_diffeomorphism(
        k=2,
        optimizer="random",
        max_evaluations=15,  # Small for demo
        quality_weights={"symmetry": 2.0, "coverage": 1.5, "smoothness": 1.0},
        random_seed=42,
    )

    print(f"\nOptimization Results:")
    print(f"Best score achieved: {result.best_score:.4f}")
    print(f"Total evaluations: {result.total_evaluations}")

    print(f"\nBest parameters found:")
    for key, value in result.best_params.items():
        print(f"  {key}: {value:.4f}")

    print(f"\nQuality breakdown of best mapping:")
    metrics = result.best_metrics
    print(f"  Symmetry: {metrics.symmetry_score:.4f}")
    print(f"  Volume preservation: {(metrics.volume_preservation_score or 0):.4f}")
    print(f"  Smoothness: {(metrics.smoothness_score or 0):.4f}")
    print(f"  Coverage: {(metrics.coverage_score or 0):.4f}")
    print(f"  Invertibility: {(metrics.invertibility_score or 0):.4f}")

    # Compare with a manual configuration
    manual_mapping = CubeToSimplexMapping(
        sigmoid_params=[
            SigmoidParams(alpha=1.0, beta=4.0, gamma=0.5),
            SigmoidParams(alpha=1.0, beta=4.0, gamma=0.5),
        ],
        special_horse_ability=0.0,
        noise_scale=1.0,
    )

    print(f"\nComparing with manual symmetric configuration:")
    manual_metrics = comprehensive_quality_assessment(
        manual_mapping,
        symmetry_samples=2000,
        volume_samples=150,
        smoothness_samples=150,
        coverage_samples=1500,
        invertibility_samples=25,
        random_seed=42,
    )

    print(f"Manual mapping score: {manual_metrics.overall_score():.4f}")
    print(f"Optimized mapping score: {result.best_score:.4f}")
    improvement = result.best_score - manual_metrics.overall_score()
    print(f"Improvement: {improvement:+.4f}")

    return result


def analyze_lattice_behavior():
    """Analyze how lattice points behave under the mapping."""
    print("\n" + "=" * 60)
    print("LATTICE POINT ANALYSIS")
    print("=" * 60)

    # Create a well-behaved mapping
    sigmoid_params = [
        SigmoidParams(alpha=1.2, beta=4.0, gamma=0.4),
        SigmoidParams(alpha=1.3, beta=4.5, gamma=0.6),
    ]
    mapping = CubeToSimplexMapping(
        sigmoid_params=sigmoid_params, special_horse_ability=0.1, noise_scale=1.0
    )

    # Create regular lattice
    resolution = 11  # 11x11 grid
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    cube_points = np.column_stack([X.ravel(), Y.ravel()])

    # Map to simplex
    simplex_points = mapping.batch_forward(cube_points)

    print(f"Analyzing {len(cube_points)} lattice points...")

    # Analyze distribution of simplex coordinates
    p1_coords = simplex_points[:, 0]
    p2_coords = simplex_points[:, 1]
    p3_coords = simplex_points[:, 2]

    print(f"\nSimplex coordinate statistics:")
    print(
        f"p₁: mean={np.mean(p1_coords):.3f}, std={np.std(p1_coords):.3f}, "
        f"range=[{np.min(p1_coords):.3f}, {np.max(p1_coords):.3f}]"
    )
    print(
        f"p₂: mean={np.mean(p2_coords):.3f}, std={np.std(p2_coords):.3f}, "
        f"range=[{np.min(p2_coords):.3f}, {np.max(p2_coords):.3f}]"
    )
    print(
        f"p₃: mean={np.mean(p3_coords):.3f}, std={np.std(p3_coords):.3f}, "
        f"range=[{np.min(p3_coords):.3f}, {np.max(p3_coords):.3f}]"
    )

    # Check if we're covering the simplex well
    # Distance from simplex vertices
    vertex1 = np.array([1, 0, 0])
    vertex2 = np.array([0, 1, 0])
    vertex3 = np.array([0, 0, 1])

    min_dist_v1 = np.min([np.linalg.norm(p - vertex1) for p in simplex_points])
    min_dist_v2 = np.min([np.linalg.norm(p - vertex2) for p in simplex_points])
    min_dist_v3 = np.min([np.linalg.norm(p - vertex3) for p in simplex_points])

    print(f"\nSimplex coverage analysis:")
    print(f"Closest approach to vertex (1,0,0): {min_dist_v1:.3f}")
    print(f"Closest approach to vertex (0,1,0): {min_dist_v2:.3f}")
    print(f"Closest approach to vertex (0,0,1): {min_dist_v3:.3f}")

    # Check symmetry by examining corner mappings
    corners = [[0, 0], [0, 1], [1, 0], [1, 1]]
    print(f"\nCorner point mappings:")
    for corner in corners:
        simplex_point = mapping(corner)
        winner = np.argmax(simplex_point)
        print(
            f"  {corner} → ({simplex_point[0]:.3f}, {simplex_point[1]:.3f}, {simplex_point[2]:.3f}) [Horse {winner+1} wins]"
        )

    return cube_points, simplex_points


def demonstrate_parameter_sensitivity():
    """Demonstrate how parameter changes affect the mapping."""
    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)

    base_params = [
        SigmoidParams(alpha=1.0, beta=4.0, gamma=0.5),
        SigmoidParams(alpha=1.0, beta=4.0, gamma=0.5),
    ]
    base_mapping = CubeToSimplexMapping(
        sigmoid_params=base_params, special_horse_ability=0.0, noise_scale=1.0
    )

    print("Base configuration:")
    base_metrics = comprehensive_quality_assessment(
        base_mapping,
        symmetry_samples=2000,
        volume_samples=100,
        smoothness_samples=100,
        coverage_samples=1000,
        invertibility_samples=20,
        random_seed=42,
    )
    print(f"Base score: {base_metrics.overall_score():.4f}")

    # Test parameter variations
    variations = [
        (
            "Higher alpha",
            [
                SigmoidParams(alpha=2.0, beta=4.0, gamma=0.5),
                SigmoidParams(alpha=2.0, beta=4.0, gamma=0.5),
            ],
            0.0,
        ),
        (
            "Higher beta",
            [
                SigmoidParams(alpha=1.0, beta=8.0, gamma=0.5),
                SigmoidParams(alpha=1.0, beta=8.0, gamma=0.5),
            ],
            0.0,
        ),
        (
            "Shifted gamma",
            [
                SigmoidParams(alpha=1.0, beta=4.0, gamma=0.3),
                SigmoidParams(alpha=1.0, beta=4.0, gamma=0.7),
            ],
            0.0,
        ),
        (
            "Strong special horse",
            [
                SigmoidParams(alpha=1.0, beta=4.0, gamma=0.5),
                SigmoidParams(alpha=1.0, beta=4.0, gamma=0.5),
            ],
            1.0,
        ),
        (
            "Weak special horse",
            [
                SigmoidParams(alpha=1.0, beta=4.0, gamma=0.5),
                SigmoidParams(alpha=1.0, beta=4.0, gamma=0.5),
            ],
            -1.0,
        ),
    ]

    print(f"\nParameter sensitivity results:")
    print(f"{'Variation':<20} {'Score':<8} {'Change':<8} {'Symmetry':<9} {'Smooth':<8}")
    print("-" * 65)

    for name, params, special_ability in variations:
        mapping = CubeToSimplexMapping(
            sigmoid_params=params,
            special_horse_ability=special_ability,
            noise_scale=1.0,
        )

        metrics = comprehensive_quality_assessment(
            mapping,
            symmetry_samples=1500,
            volume_samples=80,
            smoothness_samples=80,
            coverage_samples=800,
            invertibility_samples=15,
            random_seed=42,
        )

        score = metrics.overall_score()
        change = score - base_metrics.overall_score()

        print(
            f"{name:<20} "
            f"{score:<8.4f} "
            f"{change:+8.4f} "
            f"{metrics.symmetry_score:<9.4f} "
            f"{(metrics.smoothness_score or 0):<8.4f}"
        )


def main():
    """Run the comprehensive demonstration."""
    print("THURSTONE DIFFEOMORPHISM FRAMEWORK DEMONSTRATION")
    print("=" * 60)
    print(
        "This demo showcases cube-to-simplex diffeomorphisms using Thurstone racing models."
    )
    print("The framework creates smooth mappings from [0,1]^k to the k-simplex using")
    print(
        "parametric sigmoid functions and optimizes for quality measures like symmetry,"
    )
    print("volume preservation, smoothness, coverage, and invertibility.")

    try:
        # 1. Basic usage
        example_mapping = demonstrate_basic_usage()

        # 2. Compare different configurations
        comparison_results = compare_mappings()

        # 3. Demonstrate optimization
        optimization_result = demonstrate_optimization()

        # 4. Analyze lattice behavior
        cube_points, simplex_points = analyze_lattice_behavior()

        # 5. Parameter sensitivity
        demonstrate_parameter_sensitivity()

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("✅ Successfully demonstrated:")
        print("   • Basic diffeomorphism creation and usage")
        print("   • Quality assessment with multiple metrics")
        print("   • Parameter optimization with random search")
        print("   • Lattice point analysis and coverage")
        print("   • Parameter sensitivity analysis")

        print("\nKey Findings:")
        print("   • Mappings preserve probability constraints (sum to 1)")
        print("   • Parameter optimization can improve quality scores")
        print("   • Different configurations show various trade-offs")
        print("   • The framework handles k=2 (triangle) mappings robustly")

        print("\n🔧 Framework Components:")
        print("   • CubeToSimplexMapping: Core mapping class")
        print("   • Quality assessment: 5 different quality measures")
        print("   • Optimization: Random search and evolutionary algorithms")
        print("   • Visualization: Comprehensive plotting tools")

        print("\nNext Steps:")
        print("   • Try higher dimensions (k=3, k=4)")
        print("   • Experiment with evolutionary optimization")
        print("   • Use visualization tools to explore mappings")
        print("   • Apply to specific use cases requiring cube-simplex mappings")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("This may be due to missing dependencies or numerical issues.")
        raise


if __name__ == "__main__":
    main()
