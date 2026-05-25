# thurstone

A comprehensive **object-oriented toolkit** for lattice-based performance models, the horse-race inverse problem, and **advanced diffeomorphism optimization**. This package modernizes Thurstone Class V models with fast algorithms, clean APIs, and cutting-edge mathematical frameworks for mapping between probability spaces.

[![PyPI version](https://badge.fury.io/py/thurstone.svg)](https://badge.fury.io/py/thurstone)
[![CI](https://github.com/microprediction/thurstone/actions/workflows/ci.yml/badge.svg)](https://github.com/microprediction/thurstone/actions/workflows/ci.yml)

## ✨ Major Features

### 🏆 Core Racing Models
- **Uniform lattices** with explicit `L` and `unit` parameters
- **Normalized densities** with safe shifts, convolution, and dilation
- **Multiplicity-aware state pricing** for realistic tie handling
- **Fast inverse calibration** from market prices to relative abilities

### 🎯 Diffeomorphism Framework *(New!)*
- **Cube-to-simplex mappings** using parametric sigmoid functions
- **Quality assessment** with symmetry, volume preservation, and smoothness metrics  
- **Enhanced optimization** with 22+ pure Python algorithms (genetic, particle swarm, etc.)
- **Adaptive special horses** with configurable probability distributions
- **Interactive visualizations** for exploring parameter spaces

### 🔬 Research Tools
- **Systematic parameter studies** with comprehensive quality analysis
- **Paper-quality figure generation** for academic publications
- **Performance benchmarking** and statistical validation
- **Monte Carlo validation** frameworks

### 🛠️ Developer Experience  
- **Modern tooling** with ruff formatting and comprehensive CI/CD
- **Clean APIs** with full type hints and documentation
- **Extensive test suite** with property-based testing (Hypothesis)
- **Automated PyPI deployment** with semantic versioning

## 🚀 Quick Start

```bash
pip install thurstone
```

### Basic Racing Model
```python
from thurstone import UniformLattice, Density, AbilityCalibrator, STD_L, STD_UNIT

# Set up lattice and base density
lattice = UniformLattice(L=STD_L, unit=STD_UNIT)
base = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0)

# Inverse calibration: market prices → relative abilities
dividends = [3.2, 4.8, 12.0, 7.5, 20.0]
calibrator = AbilityCalibrator(base)
abilities = calibrator.solve_from_dividends(dividends)
probabilities = calibrator.state_prices_from_ability(abilities)
```

### Advanced Diffeomorphism Optimization
```python
from thurstone import CubeToSimplexMapping, SigmoidParams, optimize_diffeomorphism

# Create cube-to-simplex mapping with sigmoid parameters
params = [
    SigmoidParams(alpha=1.2, beta=4.0, gamma=0.5),
    SigmoidParams(alpha=0.8, beta=3.5, gamma=0.3)
]
mapping = CubeToSimplexMapping(params)

# Optimize mapping quality
result = optimize_diffeomorphism(
    mapping,
    max_evaluations=100,
    quality_weights={"symmetry": 0.4, "smoothness": 0.3, "coverage": 0.3}
)

print(f"Best quality score: {result.best_score:.4f}")
```

## 📊 Interactive Visualizations

Explore Thurstone models directly in your browser:

- **[Diffeomorphism Explorer](https://thurstone.microprediction.org/interactive/diffeomorphism-explorer.html)** - Real-time parameter tuning
- **[Multi-dimensional Analysis](https://thurstone.microprediction.org/interactive/multi-dimensional.html)** - Advanced visualization tools
- **[All Interactive Demos](https://thurstone.microprediction.org/interactive/)** - Complete collection

```bash
# Run locally
python -m http.server 8000 --directory docs
# Visit http://localhost:8000/interactive/
```

## 🧪 Examples & Research

### Core Examples
```bash
# Comprehensive diffeomorphism demo
python examples/diffeomorphism_demo.py

# Global ability calibration with 500 horses
python examples/global_calibration_demo.py

# Dynamic calibration with time-varying abilities  
python examples/dynamic_calibration_demo.py

# Multi-ray synthetic data analysis
python examples/multiray_synthetic.py
```

### Research Scripts
```bash
# Generate publication-quality figures
python research/generate_paper_figures.py

# Run systematic parameter study
python research/run_systematic_study.py

# Analyze adaptive special horse performance
python research/special_horse_study.py
```

## 🔧 Development

### Setup
```bash
git clone https://github.com/microprediction/thurstone.git
cd thurstone
pip install -e ".[test,viz]"
```

### Code Quality
```bash
# Format code (matches CI exactly)
python scripts/format-code.py

# Check formatting without changes
python scripts/format-code.py --check

# Run tests
pytest

# Full development workflow
python scripts/review.py  # rebase → lint → test → PR
```

## 📈 What's New in v0.1.0

- ✅ **Complete ruff migration** - unified formatting and linting
- ✅ **Diffeomorphism framework** - advanced probability space mappings  
- ✅ **22 optimization algorithms** - pure Python implementations
- ✅ **Quality assessment metrics** - comprehensive mapping evaluation
- ✅ **Interactive visualizations** - browser-based exploration tools
- ✅ **Research infrastructure** - systematic studies and figure generation
- ✅ **Modern CI/CD** - automated testing and PyPI deployment
- ✅ **Enhanced documentation** - comprehensive examples and guides

## 🎯 Applications

Beyond horse racing, this toolkit enables:

- **E-commerce recommendation** - product ranking and placement optimization
- **Web search algorithms** - relevance scoring and result ranking  
- **Financial modeling** - relative performance analysis and risk assessment
- **Game theory** - multi-agent competition and strategy analysis
- **Machine learning** - probability calibration and uncertainty quantification

See the [research paper](https://github.com/microprediction/winning/blob/main/docs/Horse_Race_Problem__SIAM_.pdf) for theoretical foundations and [notebook examples](https://github.com/microprediction/winning/blob/main/Ability_Transforms_Updated.ipynb) for detailed applications.

## 🏗️ Architecture

### Core Components
- **`UniformLattice`** - Discrete probability space foundation
- **`Density`** - Mass-preserving probability distributions  
- **`AbilityCalibrator`** - Inverse problem solver with monotone interpolation
- **`Race` & `StatePricer`** - Forward pricing with multiplicity awareness

### Advanced Framework  
- **`CubeToSimplexMapping`** - Diffeomorphism between probability spaces
- **`SigmoidParams`** - Parametric transformation functions
- **`comprehensive_quality_assessment`** - Multi-metric evaluation suite
- **`optimize_diffeomorphism`** - Meta-optimization with multiple algorithms

### Research Tools
- **`StudyManager`** - Systematic parameter exploration
- **`AdaptiveSpecialHorse`** - Advanced racing model variants
- **Visualization suite** - Publication-quality plotting and analysis

## 📚 Citation

```bibtex
@article{cotton2021inferring,
  title={Inferring Relative Ability from Winning Probability in Multientrant Contests},
  author={Cotton, Peter},
  journal={SIAM Journal on Financial Mathematics},
  volume={12},
  number={1},  
  pages={295--317},
  year={2021},
  publisher={SIAM},
  doi={10.1137/19M1276261}
}
```

## 🤝 Contributing

We welcome contributions! The codebase uses modern Python practices:

- **Code formatting**: `ruff` with 100-character lines
- **Type hints**: Full typing coverage
- **Testing**: `pytest` with `hypothesis` property-based tests  
- **CI/CD**: Comprehensive GitHub Actions workflows

Priority areas for contribution:
- 🦀 **Rust implementation** for performance-critical components
- 🎨 **Additional visualizations** for complex probability spaces  
- 🧮 **Extended optimization algorithms** and quality metrics
- 📖 **Documentation** and tutorial content

---

**thurstone v0.1.0** - Modernizing probabilistic inference with clean code and cutting-edge mathematics. 🏆