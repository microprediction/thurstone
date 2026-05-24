# Thurstone Diffeomorphisms Research

This directory contains a complete research framework for studying optimal cube-to-simplex diffeomorphisms using Thurstone racing models.

##  Repository Structure

```
research/
├── README.md                       # This file
├── study_design.md                 # Detailed research protocol
├── paper_draft.md                  # Academic paper draft
├── generate_paper_figures.py       # Publication-quality figure generation
├── run_systematic_study.py         # Complete systematic study implementation
└── study_results/                  # Generated results and data (created when running studies)
```

##  Research Components

### 1. Academic Paper (`paper_draft.md`)
A complete research paper including:
- **Mathematical Framework**: Formal definition of the diffeomorphism construction
- **Quality Assessment**: Five comprehensive quality measures
- **Optimization Methods**: Comparison of different algorithms
- **Experimental Results**: Systematic parameter exploration findings
- **Applications**: Real-world use cases and extensions

### 2. Systematic Study (`run_systematic_study.py`)
Complete implementation of our 4-phase research protocol:

#### **Phase 1: Parameter Space Exploration**
- Systematic grid search over sigmoid parameters
- Multiple quality weighting schemes
- Statistical analysis of parameter sensitivity

#### **Phase 2: Optimization Algorithm Comparison**
- Random Search vs. Evolutionary Algorithm vs. Bayesian Optimization
- Performance metrics: convergence, final quality, reliability
- Statistical significance testing

#### **Phase 3: Multi-Dimensional Analysis**
- Scalability from k=2 to k=5 dimensions
- Computational complexity analysis
- Quality degradation patterns

#### **Phase 4: Application-Specific Optimization**
- Maximum symmetry configurations
- Volume preservation specialists
- Coverage maximization
- Smoothness optimization

### 3. Publication-Quality Visualization (`generate_paper_figures.py`)
Generates four comprehensive figures:

#### **Figure 1: Lattice Point Mappings**
- Shows how regular grids in [0,1]² map to triangle
- Compares 4 different parameter configurations
- Demonstrates mapping diversity and quality

#### **Figure 2: 3D Visualization**
- Three-dimensional representation of cube→simplex transformation
- Flow lines showing mapping process
- Clear geometric intuition

#### **Figure 3: Quality Analysis**
- Radar charts comparing quality profiles
- Jacobian determinant heatmaps
- Parameter sensitivity analysis
- Coverage density visualization

#### **Figure 4: Optimization Results**
- Algorithm convergence comparison
- Best mapping visualization
- Quality breakdown of optimal configurations

##  Quick Start

### Run Complete Study
```bash
cd research
python run_systematic_study.py
```

### Generate Figures Only
```bash
cd research
python generate_paper_figures.py
```

### View Study Design
```bash
cat study_design.md
```

##  Study Scope and Scale

### Demo Version (Current)
- **Grid Search**: 3,125 parameter combinations
- **Optimization**: 20 evaluations × 3 runs × 2 algorithms
- **Dimensions**: k ∈ {2, 3, 4}
- **Runtime**: ~30-60 minutes

### Full Research Version (Recommended)
To run the complete study as designed, modify sample sizes in:
- `phase1_parameter_exploration()`: Increase grid resolution
- `comprehensive_quality_assessment()`: Increase sample sizes
- `optimize_diffeomorphism()`: Increase max_evaluations
- Phase 2: Increase replication count for statistical power

**Expected Full Study Scale**:
- **Grid Search**: 30,240 parameter combinations
- **Optimization**: 200 evaluations × 20 runs × 3 algorithms
- **Total Evaluations**: ~50,000+ quality assessments
- **Runtime**: 100-200 compute hours

##  Key Research Questions

1. **What are optimal sigmoid parameters for different objectives?**
2. **How do quality measures trade off against each other?**
3. **Which optimization algorithms work best for this problem?**
4. **How does mapping quality scale with dimension k?**
5. **What parameter sensitivity patterns exist?**

##  Expected Findings

Based on preliminary analysis, we expect:

### **Parameter Patterns**
- α (scale) optimal around 1.0-1.5
- β (steepness) optimal around 4.0-6.0  
- γ (shift) optimal around 0.4-0.6
- Special horse ability near 0.0 for balanced scenarios

### **Algorithm Performance**
- Bayesian Optimization: Best final performance
- Evolutionary Algorithm: Good balance of speed/quality
- Random Search: Baseline performance

### **Quality Trade-offs**
- Symmetry vs. Volume Preservation: Negative correlation
- Smoothness vs. Coverage: Potential conflict at boundaries
- Invertibility improvements with moderate parameters

### **Scaling Behavior**
- Quality decreases with dimension (curse of dimensionality)
- Computational cost scales exponentially
- Practical limit around k=5 for current methods

## 🔬 Research Methodology

### **Statistical Rigor**
- Multiple random seeds for reproducibility
- Bootstrap confidence intervals
- Statistical significance testing
- Pareto frontier analysis for multi-objective optimization

### **Quality Assurance**
- Comprehensive unit tests
- Integration testing
- Parameter bounds validation
- Numerical stability checks

### **Reproducibility**
- Fixed random seeds
- Versioned code and parameters
- Complete experimental logs
- Standardized evaluation protocols

##  Results and Deliverables

### **Quantitative Results**
- Optimal parameter sets for each scenario
- Performance bounds and scaling laws
- Algorithm comparison statistics
- Trade-off analysis matrices

### **Methodological Contributions**
- Quality assessment framework
- Optimization protocol design
- Multi-objective evaluation methods
- Scalability analysis methodology

### **Practical Outputs**
- Production-ready parameter configurations
- Optimization best practices
- Performance benchmarks
- Implementation guidelines

##  Running Different Study Variants

### **Parameter-Focused Study**
```python
# Modify run_systematic_study.py
# Increase Phase 1 grid resolution
# Reduce other phases for speed
```

### **Algorithm-Focused Study**
```python
# Modify run_systematic_study.py  
# Add more algorithms in Phase 2
# Increase replication count
# Reduce parameter exploration
```

### **Application-Focused Study**
```python
# Modify run_systematic_study.py
# Focus on Phase 4
# Add domain-specific scenarios
# Custom quality weightings
```

## 📝 Citation and Publication

When using this research framework:

```bibtex
@article{thurstone_diffeomorphisms_2024,
  title={Thurstone-Based Diffeomorphisms: Smooth Mappings from Hypercube to Simplex via Racing Models},
  author={[Your Names]},
  journal={[Target Journal]},
  year={2024},
  note={Code available at: https://github.com/[your-repo]}
}
```

## 🤝 Contributing

To extend this research:

1. **New Quality Measures**: Add to `quality_assessment.py`
2. **Alternative Optimizers**: Implement `Optimizer` interface
3. **Higher Dimensions**: Extend visualization for k>2
4. **Different Noise Models**: Modify underlying Thurstone infrastructure

## 📞 Support

For questions about the research methodology, implementation, or results interpretation, please:

1. Check the detailed documentation in `study_design.md`
2. Review the academic paper draft for theoretical background
3. Examine the implementation comments for technical details
4. Open issues for bugs or enhancement requests

##  Research Impact

This framework enables:

- **Theoretical Advances**: New understanding of cube-simplex mappings
- **Methodological Contributions**: Quality assessment and optimization frameworks  
- **Practical Applications**: Production-ready diffeomorphisms for real problems
- **Educational Value**: Complete example of computational research methodology

The systematic approach demonstrated here can serve as a template for other geometric mapping optimization problems in computational mathematics and statistics.