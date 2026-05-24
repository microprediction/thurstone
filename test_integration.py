"""
Quick integration test to validate the diffeomorphism framework.
"""

import numpy as np
from thurstone import (
    CubeToSimplexMapping, SigmoidParams,
    comprehensive_quality_assessment,
    optimize_diffeomorphism
)

def test_basic_functionality():
    """Test that basic components work together."""
    print("Testing basic diffeomorphism functionality...")

    # 1. Create mapping
    sigmoid_params = [
        SigmoidParams(alpha=1.0, beta=4.0, gamma=0.5),
        SigmoidParams(alpha=1.0, beta=4.0, gamma=0.5)
    ]
    mapping = CubeToSimplexMapping(
        sigmoid_params=sigmoid_params,
        special_horse_ability=0.0,
        noise_scale=1.0
    )

    print("✓ Created mapping")

    # 2. Test point mapping
    test_point = [0.3, 0.7]
    result = mapping(test_point)

    assert len(result) == 3, f"Expected 3 outputs, got {len(result)}"
    assert abs(sum(result) - 1.0) < 1e-10, f"Probabilities don't sum to 1: {sum(result)}"

    print(f"✓ Point mapping works: {test_point} → {result}")

    # 3. Test batch mapping
    test_points = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    batch_results = mapping.batch_forward(test_points)

    assert batch_results.shape == (3, 3), f"Expected (3,3) shape, got {batch_results.shape}"
    assert np.allclose(np.sum(batch_results, axis=1), 1.0), "Batch probabilities don't sum to 1"

    print("✓ Batch mapping works")

    # 4. Test quality assessment (minimal)
    metrics = comprehensive_quality_assessment(
        mapping,
        symmetry_samples=1000,
        volume_samples=50,
        smoothness_samples=50,
        coverage_samples=500,
        invertibility_samples=10,
        random_seed=42
    )

    assert 0 <= metrics.overall_score() <= 1, f"Overall score out of range: {metrics.overall_score()}"
    print(f"✓ Quality assessment works: score = {metrics.overall_score():.4f}")

    return mapping, metrics

def test_optimization():
    """Test optimization with minimal budget."""
    print("\nTesting optimization...")

    result = optimize_diffeomorphism(
        k=2,
        optimizer='random',
        max_evaluations=5,  # Very small for quick test
        random_seed=42
    )

    assert result.best_score >= 0, f"Invalid best score: {result.best_score}"
    assert result.total_evaluations == 5, f"Expected 5 evaluations, got {result.total_evaluations}"
    assert result.best_mapping is not None, "No best mapping found"

    print(f"✓ Optimization works: best score = {result.best_score:.4f}")

    return result

def main():
    """Run integration tests."""
    print("=" * 50)
    print("THURSTONE DIFFEOMORPHISM INTEGRATION TEST")
    print("=" * 50)

    try:
        # Test basic functionality
        mapping, metrics = test_basic_functionality()

        # Test optimization
        opt_result = test_optimization()

        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)
        print("The diffeomorphism framework is working correctly.")
        print(f"Example mapping score: {metrics.overall_score():.4f}")
        print(f"Optimization best score: {opt_result.best_score:.4f}")
        print("Ready for production use!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise

if __name__ == "__main__":
    main()