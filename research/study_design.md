# Research Study: Optimal Thurstone-Based Diffeomorphisms from k-Cube to k-Simplex

## Research Objectives

**Primary Question**: What are the optimal parameters and design choices for creating high-quality diffeomorphisms from [0,1]^k to the k-simplex using Thurstone racing models with parametric sigmoid mappings?

**Secondary Questions**:
1. How do different quality measures trade off against each other?
2. Which optimization algorithms are most effective for this problem?
3. How does mapping quality scale with dimension k?
4. What is the relationship between sigmoid parameters and mapping properties?
5. How sensitive are optimal solutions to hyperparameter choices?

## Study Design Overview

### **Phase 1: Parameter Space Exploration (k=2)**
- **Objective**: Establish baseline understanding and identify promising parameter regions
- **Approach**: Systematic grid search + random sampling
- **Scope**: Triangle mappings (k=2) with full quality assessment

### **Phase 2: Optimization Algorithm Comparison**
- **Objective**: Determine best optimization strategy
- **Approaches**: Random Search, Evolutionary Algorithm, Bayesian Optimization
- **Metrics**: Convergence speed, final quality, consistency

### **Phase 3: Multi-Dimensional Analysis**
- **Objective**: Study scalability to higher dimensions
- **Scope**: k=2,3,4,5 with adapted quality measures
- **Focus**: Computational complexity and quality degradation

### **Phase 4: Application-Specific Studies**
- **Objective**: Optimize for specific use cases
- **Scenarios**: Volume preservation, maximum symmetry, coverage optimization
- **Deliverable**: Specialized parameter recommendations

## Experimental Protocol

### **Evaluation Framework**

**Quality Measures** (standardized 0-1 scores):
- **Symmetry (S)**: Deviation from uniform winning probabilities
- **Volume Preservation (V)**: Jacobian determinant consistency  
- **Smoothness (M)**: Gradient magnitude uniformity
- **Coverage (C)**: Simplex space utilization
- **Invertibility (I)**: Numerical inversion success

**Composite Scores**:
- **Balanced**: w_S=1, w_V=1, w_M=1, w_C=1, w_I=1
- **Symmetry-First**: w_S=3, w_V=1, w_M=1, w_C=2, w_I=1  
- **Volume-First**: w_S=1, w_V=3, w_M=2, w_C=1, w_I=1
- **Coverage-First**: w_S=2, w_V=1, w_M=1, w_C=3, w_I=1

**Statistical Protocol**:
- Multiple random seeds (n=10) for each configuration
- Bootstrap confidence intervals for quality measures
- Statistical significance testing for algorithm comparisons

### **Phase 1: Parameter Space Exploration**

**Grid Search Design** (k=2):
```
Alpha (α): [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  (6 values)
Beta (β):  [1.0, 2.0, 4.0, 6.0, 8.0, 10.0] (6 values)  
Gamma (γ): [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] (7 values)
Special Horse Ability: [-1.0, -0.5, 0.0, 0.5, 1.0] (5 values)

For k=2: 2 sigmoids × (6×6×7) + 5 special = 1,512 + 5 = 7,560 combinations per quality weighting
Total: 4 weightings × 7,560 = 30,240 evaluations
```

**Random Sampling**: Additional 10,000 random parameter sets for coverage

**Analysis Plan**:
- Parameter correlation analysis
- Quality measure trade-off visualization
- Identification of high-performance regions
- Sensitivity analysis around optimal points

### **Phase 2: Optimization Algorithm Comparison**

**Algorithms**:
1. **Random Search**: Baseline method
2. **Evolutionary Algorithm**: Population-based optimization
3. **Bayesian Optimization**: Gaussian Process surrogate model
4. **Particle Swarm**: Swarm intelligence approach

**Experimental Setup**:
- Budget: 200 evaluations per run
- Replications: 20 runs per algorithm × quality weighting
- Initialization: Same random starting points for fairness
- Termination: Fixed budget (no early stopping)

**Comparison Metrics**:
- **Convergence Rate**: Quality score vs. evaluation number
- **Final Performance**: Best score achieved
- **Reliability**: Coefficient of variation across runs
- **Efficiency**: Area under convergence curve

### **Phase 3: Multi-Dimensional Analysis**

**Dimension Studies**: k ∈ {2, 3, 4, 5}

**Adapted Quality Measures**:
- Symmetry: Same definition scales naturally
- Volume Preservation: k×k Jacobian determinants
- Smoothness: Gradient norms in k dimensions
- Coverage: Adaptive binning strategy for higher dimensions
- Invertibility: Multi-dimensional numerical inversion

**Computational Considerations**:
- Sample size scaling: Adjust for dimensionality
- Quality assessment budget: Fixed computational time per dimension
- Parameter space growth: Systematic subsampling

### **Phase 4: Application-Specific Optimization**

**Use Case Scenarios**:

1. **Maximum Symmetry**: w_S=5, others=0
   - Goal: Perfect balance between all competitors
   - Applications: Fair allocation, unbiased sampling

2. **Volume Preservation**: w_V=5, others=0
   - Goal: Preserve measure relationships
   - Applications: Probability density transformations

3. **Maximum Coverage**: w_C=5, others=0
   - Goal: Full simplex utilization
   - Applications: Space-filling designs

4. **Smooth Mapping**: w_M=5, others=0
   - Goal: Minimal gradient variation
   - Applications: Interpolation, function approximation

## Expected Deliverables

### **Quantitative Results**
- **Parameter Optima**: Best configurations for each quality weighting
- **Performance Bounds**: Theoretical and empirical limits
- **Scaling Laws**: How quality changes with dimension k
- **Trade-off Surfaces**: Pareto frontiers between quality measures

### **Methodological Insights**
- **Algorithm Rankings**: Which optimizers work best
- **Convergence Patterns**: Typical optimization trajectories  
- **Parameter Sensitivity**: Which parameters matter most
- **Design Guidelines**: Rules of thumb for practitioners

### **Implementation Artifacts**
- **Optimal Parameter Sets**: Production-ready configurations
- **Benchmark Suite**: Standard test problems and expected results
- **Performance Profiles**: Quality vs. computational cost trade-offs
- **Software Tools**: Enhanced optimization and analysis capabilities

## Timeline and Resource Requirements

**Phase 1**: 2-3 weeks (extensive parameter exploration)
**Phase 2**: 1-2 weeks (algorithm comparison)  
**Phase 3**: 2-3 weeks (multi-dimensional scaling)
**Phase 4**: 1-2 weeks (application-specific tuning)
**Analysis & Writing**: 2-3 weeks (paper preparation)

**Total Duration**: 8-13 weeks for comprehensive study

**Computational Requirements**:
- ~50,000+ quality assessments
- Parallelizable across parameter sets
- Est. 100-200 hours compute time (depending on sample sizes)

## Success Criteria

**Minimum Viable Study**:
- [ ] Complete Phase 1 parameter exploration for k=2
- [ ] Identify top 10% parameter configurations  
- [ ] Compare at least 2 optimization algorithms
- [ ] Demonstrate quality measure trade-offs

**Full Success**:
- [ ] All 4 phases completed
- [ ] Statistical significance in algorithm comparisons
- [ ] Scaling laws established for k=2,3,4,5
- [ ] Production-ready optimal parameter sets
- [ ] Peer-review ready research paper

This systematic study will definitively answer what constitutes the "best" mapping under various criteria and provide the research community with optimal parameter sets and design guidelines.