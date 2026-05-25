#!/usr/bin/env python3
"""
Apply exact isort formatting that matches CI expectations.
This script applies the precise formatting changes shown in CI diffs.
"""

import re
from pathlib import Path

# Define the exact formatting fixes based on CI diff
fixes = {
    "thurstone/export_data.py": [
        {
            "old": "from .adaptive_special_horse import AdaptiveSpecialHorse, SpecialHorseConfig",
            "new": "from .adaptive_special_horse import (AdaptiveSpecialHorse,\n                                         SpecialHorseConfig)"
        },
        {
            "old": "from .adaptive_special_horse import (\n        AdaptiveSpecialHorse,\n        DistributionType,\n        SpecialHorseConfig,\n    )",
            "new": "from .adaptive_special_horse import (AdaptiveSpecialHorse,\n                                             DistributionType,\n                                             SpecialHorseConfig)"
        },
        {
            "old": "from .adaptive_special_horse import (\n            AdaptiveSpecialHorse,\n            DistributionType,\n            SpecialHorseConfig,\n        )",
            "new": "from .adaptive_special_horse import (AdaptiveSpecialHorse,\n                                             DistributionType,\n                                             SpecialHorseConfig)"
        }
    ],
    "thurstone/optimization.py": [
        {
            "old": "from .quality_assessment import QualityMetrics, comprehensive_quality_assessment",
            "new": "from .quality_assessment import (QualityMetrics,\n                                 comprehensive_quality_assessment)"
        }
    ],
    "thurstone/enhanced_optimization.py": [
        {
            "old": "from .adaptive_special_horse import (\n    AdaptiveSpecialHorse,\n    DistributionType,\n    SpecialHorseConfig,\n)",
            "new": "from .adaptive_special_horse import (AdaptiveSpecialHorse, DistributionType,\n                                     SpecialHorseConfig)"
        },
        {
            "old": "from .quality_assessment import QualityMetrics, comprehensive_quality_assessment",
            "new": "from .quality_assessment import (QualityMetrics,\n                                 comprehensive_quality_assessment)"
        }
    ],
    "thurstone/__init__.py": [
        {
            "old": "from .conventions import (\n    ALT_A,\n    ALT_L,\n    ALT_SCALE,\n    ALT_UNIT,\n    NAN_DIVIDEND,\n    STD_A,\n    STD_L,\n    STD_SCALE,\n    STD_UNIT,\n)",
            "new": "from .conventions import (ALT_A, ALT_L, ALT_SCALE, ALT_UNIT, NAN_DIVIDEND,\n                          STD_A, STD_L, STD_SCALE, STD_UNIT)"
        },
        {
            "old": "from .optimization import OptimizationResult, ParameterBounds, optimize_diffeomorphism",
            "new": "from .optimization import (OptimizationResult, ParameterBounds,\n                           optimize_diffeomorphism)"
        },
        {
            "old": "from .quality_assessment import QualityMetrics, comprehensive_quality_assessment",
            "new": "from .quality_assessment import (QualityMetrics,\n                                 comprehensive_quality_assessment)"
        }
    ],
    "thurstone/visualization.py": [
        {
            "old": "from .quality_assessment import (\n    assess_invertibility,\n    assess_smoothness,\n    assess_symmetry,\n    assess_uniform_coverage,\n    assess_volume_preservation,\n)",
            "new": "from .quality_assessment import (assess_invertibility, assess_smoothness,\n                                 assess_symmetry, assess_uniform_coverage,\n                                 assess_volume_preservation)"
        }
    ],
    "research/special_horse_study.py": [
        {
            "old": "from thurstone.adaptive_special_horse import (\n    AdaptiveSpecialHorse,\n    DistributionType,\n    SpecialHorseConfig,\n)",
            "new": "from thurstone.adaptive_special_horse import (AdaptiveSpecialHorse,\n                                              DistributionType,\n                                              SpecialHorseConfig)"
        }
    ],
    "research/generate_paper_figures.py": [
        {
            "old": "from thurstone import (\n    CubeToSimplexMapping,\n    comprehensive_quality_assessment,\n    optimize_diffeomorphism,\n)",
            "new": "from thurstone import (CubeToSimplexMapping, comprehensive_quality_assessment,\n                       optimize_diffeomorphism)"
        }
    ],
    "research/run_systematic_study.py": [
        {
            "old": "from thurstone import (\n    CubeToSimplexMapping,\n    ParameterBounds,\n    SigmoidParams,\n    comprehensive_quality_assessment,\n    optimize_diffeomorphism,\n)",
            "new": "from thurstone import (CubeToSimplexMapping, ParameterBounds, SigmoidParams,\n                       comprehensive_quality_assessment,\n                       optimize_diffeomorphism)"
        }
    ],
    "tests/test_multiplicity.py": [
        {
            "old": "from thurstone.order_stats import expected_payoff_with_multiplicity, winner_of_many",
            "new": "from thurstone.order_stats import (expected_payoff_with_multiplicity,\n                                   winner_of_many)"
        }
    ],
    "scripts/generate_fixtures.py": [
        {
            "old": "from thurstone import (\n    STD_A,\n    STD_L,\n    STD_SCALE,\n    STD_UNIT,\n    AbilityCalibrator,\n    Density,\n    Race,\n    StatePricer,\n    UniformLattice,\n)",
            "new": "from thurstone import (STD_A, STD_L, STD_SCALE, STD_UNIT, AbilityCalibrator,\n                       Density, Race, StatePricer, UniformLattice)"
        },
        {
            "old": "from thurstone.order_stats import expected_payoff_with_multiplicity, winner_of_many",
            "new": "from thurstone.order_stats import (expected_payoff_with_multiplicity,\n                                   winner_of_many)"
        }
    ],
    "examples/multiray_synthetic.py": [
        {
            "old": "from thurstone import (\n    AbilityCalibrator,\n    Density,\n    MultiRayGlobalCalibrator,\n    UniformLattice,\n)",
            "new": "from thurstone import (AbilityCalibrator, Density, MultiRayGlobalCalibrator,\n                       UniformLattice)"
        },
        {
            "old": "from thurstone.multiray import _interp_price_and_slope_1d, _interp_price_and_slope_2d",
            "new": "from thurstone.multiray import (_interp_price_and_slope_1d,\n                                _interp_price_and_slope_2d)"
        }
    ],
    "examples/diffeomorphism_demo.py": [
        {
            "old": "from thurstone.cube_to_simplex import CubeToSimplexMapping, SigmoidParams  # noqa: E402",
            "new": "from thurstone.cube_to_simplex import (CubeToSimplexMapping,  # noqa: E402\n                                       SigmoidParams)"
        }
    ],
    "examples/global_calibration_compare.py": [
        {
            "old": "from thurstone import (\n    AbilityCalibrator,\n    Density,\n    GlobalAbilityCalibrator,\n    GlobalLSCalibrator,\n    UniformLattice,\n)",
            "new": "from thurstone import (AbilityCalibrator, Density, GlobalAbilityCalibrator,\n                       GlobalLSCalibrator, UniformLattice)"
        }
    ],
    "examples/global_calibration_ls_demo.py": [
        {
            "old": "from thurstone import AbilityCalibrator, Density, GlobalLSCalibrator, UniformLattice",
            "new": "from thurstone import (AbilityCalibrator, Density, GlobalLSCalibrator,\n                       UniformLattice)"
        }
    ]
}

def apply_fixes():
    """Apply all the fixes to match CI formatting."""
    for file_path, file_fixes in fixes.items():
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue

        content = path.read_text()
        original_content = content

        for fix in file_fixes:
            old_pattern = fix["old"]
            new_replacement = fix["new"]

            if old_pattern in content:
                content = content.replace(old_pattern, new_replacement)
                print(f"✅ Fixed import in {file_path}")
            else:
                # Try with flexible whitespace matching for multi-line imports
                old_flexible = re.sub(r'\s+', r'\\s+', re.escape(old_pattern))
                old_flexible = old_flexible.replace(r'\\s+', r'\s+')
                if re.search(old_flexible, content, re.MULTILINE):
                    content = re.sub(old_flexible, new_replacement, content, flags=re.MULTILINE)
                    print(f"✅ Fixed import in {file_path} (flexible match)")
                else:
                    print(f"⚠️  Pattern not found in {file_path}: {old_pattern[:50]}...")

        if content != original_content:
            path.write_text(content)
            print(f"📝 Updated {file_path}")
        else:
            print(f"ℹ️  No changes needed in {file_path}")

if __name__ == "__main__":
    print("🔧 Applying exact CI isort formatting...")
    apply_fixes()
    print("✅ Done!")