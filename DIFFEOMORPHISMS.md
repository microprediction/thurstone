# Thurstone Diffeomorphisms: k-cube to k-simplex

This document describes the diffeomorphism functionality added to the Thurstone package, which creates smooth mappings from the k-dimensional unit cube [0,1]^k to the k-simplex using Thurstone racing models.

## Overview

The diffeomorphism framework creates smooth, invertible mappings by:

1. **Parametric Sigmoids**: Each cube coordinate x_i ∈ [0,1] is mapped to ability a_i using f_i(x) = α_i × sigmoid(β_i × (x - γ_i))
2. **Thurstone Racing**: k+1 horses (k regular + 1 special) compete with abilities and normal noise
3. **Winning Probabilities**: Race outcomes give probability vectors on the k-simplex

## Core Components

### 1. CubeToSimplexMapping

The main class that implements the diffeomorphism.

```python
from thurstone import CubeToSimplexMapping, SigmoidParams

# Define sigmoid parameters for each dimension
sigmoid_params = [
    SigmoidParams(alpha=1.5, beta=4.0, gamma=0.3),  # Dimension 1
    SigmoidParams(alpha=1.2, beta=5.0, gamma=0.7)   # Dimension 2
]

# Create mapping
mapping = CubeToSimplexMapping(
    sigmoid_params=sigmoid_params,
    special_horse_ability=0.2,
    noise_scale=1.0
)

# Map points
cube_point = [0.3, 0.7]
simplex_point = mapping(cube_point)  # Returns [p1, p2, p3] on triangle
```

### 2. Quality Assessment

Comprehensive quality measures to evaluate diffeomorphisms:

- **Symmetry**: How well each horse wins ~1/(k+1) of the time
- **Volume Preservation**: Consistency of Jacobian determinant
- **Smoothness**: Gradient magnitude analysis
- **Coverage**: How uniformly the simplex is covered
- **Invertibility**: Numerical inversion success rate

```python
from thurstone import comprehensive_quality_assessment

metrics = comprehensive_quality_assessment(mapping)
print(f"Overall score: {metrics.overall_score():.4f}")
print(f"Symmetry: {metrics.symmetry_score:.4f}")
print(f"Smoothness: {metrics.smoothness_score:.4f}")
```

### 3. Parameter Optimization

Automated optimization to find good parameter sets:

```python
from thurstone import optimize_diffeomorphism

result = optimize_diffeomorphism(
    k=2,  # Triangle
    optimizer='random',  # or 'evolutionary'
    max_evaluations=100,
    quality_weights={'symmetry': 2.0, 'coverage': 1.5}
)

best_mapping = result.best_mapping
print(f"Best score: {result.best_score:.4f}")
```

### 4. Visualization Tools

Comprehensive plotting for analysis and presentation:

```python
from thurstone.visualization import visualize_mapping_comprehensive

figures = visualize_mapping_comprehensive(mapping, resolution=20)
# Creates: lattice plots, 3D visualization, Jacobian heatmaps, quality summary
```

## Example Usage

### Quick Start

```python
from thurstone import CubeToSimplexMapping, SigmoidParams

# Create a 2D → triangle mapping
params = [
    SigmoidParams(1.0, 4.0, 0.4),  # alpha, beta, gamma
    SigmoidParams(1.2, 4.5, 0.6)
]
mapping = CubeToSimplexMapping(params, special_horse_ability=0.0)

# Test the mapping
test_points = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
for point in test_points:
    result = mapping(point)
    print(f"{point} → {result} (sum: {sum(result):.6f})")
```

### Complete Workflow

```python
# 1. Create and assess mapping
mapping = CubeToSimplexMapping(params, special_horse_ability=0.1)
metrics = comprehensive_quality_assessment(mapping)

# 2. Optimize if needed
if metrics.overall_score() < 0.7:
    result = optimize_diffeomorphism(k=2, max_evaluations=50)
    mapping = result.best_mapping

# 3. Visualize results
figures = visualize_mapping_comprehensive(mapping)
```

## Key Parameters

### Sigmoid Parameters

Each `SigmoidParams(alpha, beta, gamma)` controls one dimension:

- **alpha**: Scale (typical range: 0.5-3.0)
- **beta**: Steepness (typical range: 2.0-10.0)  
- **gamma**: Inflection point (typical range: 0.1-0.9)

### Special Horse Ability

The (k+1)-st horse's fixed ability:
- **Positive values**: Special horse is stronger
- **Negative values**: Special horse is weaker
- **Zero**: Neutral special horse

### Quality Weights

Customize optimization objectives:
```python
weights = {
    'symmetry': 2.0,        # Emphasize balanced probabilities
    'volume_preservation': 1.0,  # Jacobian consistency
    'smoothness': 1.0,      # Low gradient variation
    'coverage': 1.5,        # Uniform simplex coverage
    'invertibility': 1.0    # Numerical inversion quality
}
```

## Mathematical Details

### The Mapping

For cube point x = (x₁, ..., xₖ) ∈ [0,1]^k:

1. **Ability Computation**: aᵢ = αᵢ × sigmoid(βᵢ × (xᵢ - γᵢ))
2. **Performance Model**: Xᵢ = aᵢ + εᵢ, εᵢ ~ N(0,σ²)
3. **Winning Probabilities**: pᵢ = P(Xᵢ = max(X₁, ..., Xₖ₊₁))

### Quality Metrics

- **Symmetry Score**: 1 - (avg_deviation / max_possible_deviation)
- **Volume Score**: 1 - min(CV(|det(J)|), 1) where CV is coefficient of variation
- **Smoothness Score**: 1 / (1 + normalized_gradient_magnitude)
- **Coverage Score**: (occupancy_ratio + uniformity_score) / 2
- **Invertibility Score**: (success_rate + error_score + condition_score) / 3

## Performance Tips

1. **Sample Sizes**: Balance accuracy vs speed
   - Symmetry: 5000-10000 samples
   - Volume/Smoothness: 200-500 samples  
   - Coverage: 2000-5000 samples
   - Invertibility: 50-100 samples

2. **Optimization**: 
   - Start with random search (faster)
   - Use evolutionary for refinement
   - 50-200 evaluations usually sufficient

3. **Parameter Bounds**:
   - Keep beta > 1 for meaningful sigmoids
   - Keep gamma ∈ [0.1, 0.9] to avoid boundary issues
   - Special horse ability ∈ [-2, 2] for reasonable influence

## Examples and Demos

- **Basic Demo**: `examples/diffeomorphism_demo.py`
- **Interactive Tutorial**: See Jupyter notebooks in `examples/`
- **Optimization Examples**: Parameter search demonstrations

## Extensions

The framework supports:
- **Higher Dimensions**: k=3,4,... (visualization limited to k=2)
- **Custom Quality Metrics**: Extend QualityMetrics class
- **Alternative Optimizers**: Implement Optimizer interface
- **Different Noise Models**: Modify underlying Thurstone infrastructure

## References

This implementation builds on:
- Thurstone's Class V models for choice behavior
- The existing `thurstone` package infrastructure
- Differential geometry concepts for smooth mappings
- Multi-objective optimization for parameter tuning

For theoretical background, see the main package documentation and academic literature on Thurstone models.