"""
Special Horse Focused Study: Finding Optimal (k+1)-th Horse Configurations

This study focuses specifically on optimizing the special horse while keeping
sigmoid parameters fixed to symmetric configurations that we know work well.
"""

import numpy as np
import pandas as pd
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
import itertools
from dataclasses import asdict
import sys

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thurstone import SigmoidParams, comprehensive_quality_assessment
from thurstone.enhanced_cube_to_simplex import EnhancedCubeToSimplexMapping
from thurstone.adaptive_special_horse import (
    AdaptiveSpecialHorse,
    SpecialHorseConfig,
    DistributionType,
)


class SpecialHorseStudyManager:
    """Manages the special horse focused study."""

    def __init__(self, output_dir: str = "special_horse_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Quality weightings (same as before)
        self.quality_weightings = {
            "balanced": {
                "symmetry": 1.0,
                "volume_preservation": 1.0,
                "smoothness": 1.0,
                "coverage": 1.0,
                "invertibility": 1.0,
            },
            "symmetry_coverage": {
                "symmetry": 2.0,
                "volume_preservation": 1.0,
                "smoothness": 1.0,
                "coverage": 2.0,
                "invertibility": 1.0,
            },
            "coverage_first": {
                "symmetry": 1.0,
                "volume_preservation": 1.0,
                "smoothness": 1.0,
                "coverage": 3.0,
                "invertibility": 1.0,
            },
        }

        # Results storage
        self.results = []

    def save_results(self, phase: str, data: List[Dict]):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"{phase}_{timestamp}.json")

        # Convert numpy types to native Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        data_clean = convert_numpy(data)

        with open(filepath, "w") as f:
            json.dump(data_clean, f, indent=2)

        print(f" Saved {phase} results to {filepath}")


def run_special_horse_study(k: int = 2):
    """Run systematic study of special horse configurations."""
    print(f" SPECIAL HORSE SYSTEMATIC STUDY")
    print(f"{'='*60}")
    print(f"Focus: Optimize (k+1)-th horse with fixed symmetric sigmoids")
    print(f"Dimension: k={k}")

    study = SpecialHorseStudyManager()

    # Fixed symmetric sigmoid parameters (we know these work well)
    sigmoid_params = []
    for i in range(k):
        sigmoid_params.append(SigmoidParams(alpha=1.2, beta=4.0, gamma=0.5))

    print(f" Using fixed symmetric sigmoids: α=1.2, β=4.0, γ=0.5")

    # Define special horse parameter space
    distributions = [
        DistributionType.NORMAL,
        DistributionType.STUDENT_T,
        DistributionType.LAPLACE,
        DistributionType.UNIFORM,
    ]

    base_abilities = [-0.5, 0.0, 0.5]
    locations = [-0.2, 0.0, 0.2]
    scales = [0.5, 1.0, 1.5]
    adaptive_modes = ["fixed", "mean_adaptive", "position_adaptive"]

    # Calculate total combinations
    total_combinations = (
        len(distributions)
        * len(base_abilities)
        * len(locations)
        * len(scales)
        * len(adaptive_modes)
    )
    total_evaluations = total_combinations * len(study.quality_weightings)

    print(f" Special horse configurations: {total_combinations:,}")
    print(f" Total evaluations: {total_evaluations:,}")
    print(f"⏱️ Estimated time: {total_evaluations * 3 / 60:.1f} minutes")

    results = []
    start_time = time.time()
    eval_count = 0

    print(f"\n Starting evaluations...")

    # Test each special horse configuration
    for dist in distributions:
        for base_ability in base_abilities:
            for location in locations:
                for scale in scales:
                    for adaptive_mode in adaptive_modes:
                        eval_count += 1

                        # Progress reporting
                        if eval_count % 10 == 0:
                            elapsed = time.time() - start_time
                            rate = eval_count / elapsed if elapsed > 0 else 0
                            remaining = (
                                (total_combinations - eval_count) / rate / 60
                                if rate > 0
                                else 0
                            )
                            print(
                                f"   Progress: {eval_count}/{total_combinations} ({100*eval_count/total_combinations:.1f}%) "
                                f"Est. {remaining:.1f}m remaining"
                            )

                        # Create special horse config
                        special_config = SpecialHorseConfig(
                            distribution=dist,
                            base_ability=base_ability,
                            location=location,
                            scale=scale,
                            adaptive_mode=adaptive_mode,
                        )

                        special_horse = AdaptiveSpecialHorse(special_config)

                        # Create enhanced mapping
                        mapping = EnhancedCubeToSimplexMapping(
                            sigmoid_params=sigmoid_params,
                            special_horse=special_horse,
                            noise_scale=1.0,
                        )

                        # Test with each quality weighting
                        for weighting_name, weights in study.quality_weightings.items():

                            # Assess quality (minimal samples for systematic study speed)
                            metrics = comprehensive_quality_assessment(
                                mapping,
                                symmetry_samples=50,
                                volume_samples=10,
                                smoothness_samples=10,
                                coverage_samples=50,
                                invertibility_samples=5,
                                random_seed=42,
                            )

                            # Record result
                            result = {
                                "weighting": weighting_name,
                                "special_horse_config": {
                                    "distribution": dist.value,
                                    "base_ability": base_ability,
                                    "location": location,
                                    "scale": scale,
                                    "adaptive_mode": adaptive_mode,
                                },
                                "quality_scores": {
                                    "symmetry": metrics.symmetry_score,
                                    "volume_preservation": metrics.volume_preservation_score
                                    or 0,
                                    "smoothness": metrics.smoothness_score or 0,
                                    "coverage": metrics.coverage_score or 0,
                                    "invertibility": metrics.invertibility_score or 0,
                                    "overall": metrics.overall_score(weights),
                                },
                                "timestamp": datetime.now().isoformat(),
                            }

                            results.append(result)

    # Save results
    study.save_results("special_horse_systematic", results)

    # Analyze results
    print(f"\n SPECIAL HORSE ANALYSIS")

    for weighting_name in study.quality_weightings.keys():
        weighting_results = [r for r in results if r["weighting"] == weighting_name]
        weighting_results.sort(
            key=lambda x: x["quality_scores"]["overall"], reverse=True
        )

        top_10 = weighting_results[:10]
        print(f"\n Top 10 for {weighting_name} weighting:")
        for i, result in enumerate(top_10, 1):
            config = result["special_horse_config"]
            score = result["quality_scores"]["overall"]
            sym = result["quality_scores"]["symmetry"]
            cov = result["quality_scores"]["coverage"]
            print(
                f"   {i:2d}. {config['distribution']:<10} {config['adaptive_mode']:<15} "
                f"Base:{config['base_ability']:+.1f} Scale:{config['scale']:.1f} "
                f"→ Score:{score:.4f} (S:{sym:.3f} C:{cov:.3f})"
            )

    # Find overall best
    all_results = sorted(
        results, key=lambda x: x["quality_scores"]["overall"], reverse=True
    )
    best_result = all_results[0]
    best_config = best_result["special_horse_config"]

    print(f"\n OVERALL BEST SPECIAL HORSE:")
    print(f"   Distribution: {best_config['distribution']}")
    print(f"   Base ability: {best_config['base_ability']}")
    print(f"   Location: {best_config['location']}")
    print(f"   Scale: {best_config['scale']}")
    print(f"   Adaptive mode: {best_config['adaptive_mode']}")
    print(f"   Best score: {best_result['quality_scores']['overall']:.4f}")

    total_time = time.time() - start_time
    print(f"\n Special horse study completed in {total_time/60:.1f} minutes!")

    return results


if __name__ == "__main__":
    print(" STARTING SPECIAL HORSE FOCUSED STUDY")
    print("=" * 60)

    results = run_special_horse_study(k=2)

    print(
        f"\n🎊 Study complete! Check special_horse_results/ directory for detailed results."
    )
