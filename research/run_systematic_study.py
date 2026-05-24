"""
Systematic Study Implementation: Finding Optimal Thurstone Diffeomorphisms

This script implements the comprehensive research study designed to find
the best cube-to-simplex mappings across different scenarios and dimensions.
"""

import itertools
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thurstone import (CubeToSimplexMapping, ParameterBounds, SigmoidParams,
                       comprehensive_quality_assessment,
                       optimize_diffeomorphism)


class StudyManager:
    """Manages the systematic study execution and results."""

    def __init__(self, output_dir: str = "study_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Study configuration
        self.quality_weightings = {
            "balanced": {
                "symmetry": 1.0,
                "volume_preservation": 1.0,
                "smoothness": 1.0,
                "coverage": 1.0,
                "invertibility": 1.0,
            },
            "symmetry_first": {
                "symmetry": 3.0,
                "volume_preservation": 1.0,
                "smoothness": 1.0,
                "coverage": 2.0,
                "invertibility": 1.0,
            },
            "volume_first": {
                "symmetry": 1.0,
                "volume_preservation": 3.0,
                "smoothness": 2.0,
                "coverage": 1.0,
                "invertibility": 1.0,
            },
            "coverage_first": {
                "symmetry": 2.0,
                "volume_preservation": 1.0,
                "smoothness": 1.0,
                "coverage": 3.0,
                "invertibility": 1.0,
            },
        }

        # Initialize results storage
        self.results = {
            "phase1_grid_search": [],
            "phase1_random_search": [],
            "phase2_optimization_comparison": [],
            "phase3_dimensional_scaling": [],
            "phase4_application_specific": [],
        }

    def save_results(self, phase: str, data: Any, filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{phase}_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)

        # Convert numpy types to native Python for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
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


def phase1_parameter_exploration(study: StudyManager, k: int = 2):
    """Phase 1: Systematic parameter space exploration."""
    print(f"\n{'='*60}")
    print(f"PHASE 1: PARAMETER SPACE EXPLORATION (k={k})")
    print(f"{'='*60}")

    # Define parameter grids (optimized for reasonable runtime)
    alpha_values = [1.0, 1.5, 2.0]  # 3 values
    beta_values = [2.0, 4.0, 6.0]  # 3 values
    gamma_values = [0.3, 0.5, 0.7]  # 3 values
    special_abilities = [0.0, 0.5, 1.0]  # 3 values

    total_combinations = (
        len(alpha_values) ** k
        * len(beta_values) ** k
        * len(gamma_values) ** k
        * len(special_abilities)
    )
    print(f" Grid search: {total_combinations:,} parameter combinations")

    grid_results = []

    # Generate all parameter combinations
    param_combinations = []
    for special_ability in special_abilities:
        for alpha_combo in itertools.product(alpha_values, repeat=k):
            for beta_combo in itertools.product(beta_values, repeat=k):
                for gamma_combo in itertools.product(gamma_values, repeat=k):
                    params = []
                    for i in range(k):
                        params.append(
                            SigmoidParams(
                                alpha=alpha_combo[i],
                                beta=beta_combo[i],
                                gamma=gamma_combo[i],
                            )
                        )
                    param_combinations.append((params, special_ability))

    print(f" Evaluating {len(param_combinations):,} combinations...")
    start_time = time.time()

    # Evaluate each combination for each quality weighting
    for weighting_name, weights in study.quality_weightings.items():
        print(f"\n📏 Evaluating {weighting_name} weighting...")

        for i, (sigmoid_params, special_ability) in enumerate(param_combinations):
            if i % 500 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (len(param_combinations) - i) / rate / 60
                print(
                    f"   Progress: {i:,}/{len(param_combinations):,} ({100*i/len(param_combinations):.1f}%) "
                    f"Est. {remaining:.1f}m remaining"
                )

            # Create mapping
            mapping = CubeToSimplexMapping(
                sigmoid_params=sigmoid_params,
                special_horse_ability=special_ability,
                noise_scale=1.0,
            )

            # Assess quality with reduced sample sizes for speed
            metrics = comprehensive_quality_assessment(
                mapping,
                symmetry_samples=1000,  # Reduced
                volume_samples=50,  # Reduced
                smoothness_samples=50,  # Reduced
                coverage_samples=500,  # Reduced
                invertibility_samples=10,  # Reduced
                random_seed=42,
            )

            # Record results
            result = {
                "weighting": weighting_name,
                "parameters": {
                    "sigmoid_params": [asdict(p) for p in sigmoid_params],
                    "special_ability": special_ability,
                },
                "quality_scores": {
                    "symmetry": metrics.symmetry_score,
                    "volume_preservation": metrics.volume_preservation_score or 0,
                    "smoothness": metrics.smoothness_score or 0,
                    "coverage": metrics.coverage_score or 0,
                    "invertibility": metrics.invertibility_score or 0,
                    "overall": metrics.overall_score(weights),
                },
                "timestamp": datetime.now().isoformat(),
            }

            grid_results.append(result)

    # Save grid search results
    study.results["phase1_grid_search"] = grid_results
    study.save_results("phase1_grid_search", grid_results)

    # Analyze top performers
    print(f"\n GRID SEARCH ANALYSIS")
    for weighting_name in study.quality_weightings.keys():
        weighting_results = [
            r for r in grid_results if r["weighting"] == weighting_name
        ]
        weighting_results.sort(
            key=lambda x: x["quality_scores"]["overall"], reverse=True
        )

        top_10 = weighting_results[:10]
        print(f"\n Top 10 for {weighting_name}:")
        for i, result in enumerate(top_10, 1):
            score = result["quality_scores"]["overall"]
            sym = result["quality_scores"]["symmetry"]
            cov = result["quality_scores"]["coverage"]
            print(f"   {i:2d}. Score: {score:.4f} (Sym: {sym:.3f}, Cov: {cov:.3f})")

    return grid_results


def phase2_optimization_comparison(study: StudyManager, k: int = 2):
    """Phase 2: Compare optimization algorithms."""
    print(f"\n{'='*60}")
    print(f"PHASE 2: OPTIMIZATION ALGORITHM COMPARISON (k={k})")
    print(f"{'='*60}")

    algorithms = ["random", "evolutionary"]
    optimization_results = []

    for weighting_name, weights in study.quality_weightings.items():
        print(f"\n Testing {weighting_name} weighting...")

        for algorithm in algorithms:
            print(f"    Running {algorithm} optimization...")

            # Run optimization multiple times for statistical significance
            runs = 3  # Reduced from 20 for demo
            alg_results = []

            for run in range(runs):
                start_time = time.time()

                result = optimize_diffeomorphism(
                    k=k,
                    optimizer=algorithm,
                    max_evaluations=20,  # Reduced for demo
                    quality_weights=weights,
                    random_seed=42 + run,
                )

                runtime = time.time() - start_time

                run_result = {
                    "weighting": weighting_name,
                    "algorithm": algorithm,
                    "run": run + 1,
                    "best_score": result.best_score,
                    "total_evaluations": result.total_evaluations,
                    "runtime_seconds": runtime,
                    "best_params": result.best_params,
                    "quality_breakdown": {
                        "symmetry": result.best_metrics.symmetry_score,
                        "volume_preservation": result.best_metrics.volume_preservation_score
                        or 0,
                        "smoothness": result.best_metrics.smoothness_score or 0,
                        "coverage": result.best_metrics.coverage_score or 0,
                        "invertibility": result.best_metrics.invertibility_score or 0,
                    },
                    "timestamp": datetime.now().isoformat(),
                }

                alg_results.append(run_result)
                optimization_results.append(run_result)

                print(f"      Run {run+1}: {result.best_score:.4f} ({runtime:.1f}s)")

            # Compute statistics for this algorithm/weighting combination
            scores = [r["best_score"] for r in alg_results]
            runtimes = [r["runtime_seconds"] for r in alg_results]

            print(
                f"    {algorithm} results: "
                f"Mean={np.mean(scores):.4f}±{np.std(scores):.4f}, "
                f"Time={np.mean(runtimes):.1f}±{np.std(runtimes):.1f}s"
            )

    # Save optimization results
    study.results["phase2_optimization_comparison"] = optimization_results
    study.save_results("phase2_optimization_comparison", optimization_results)

    # Statistical analysis
    print(f"\n OPTIMIZATION ALGORITHM COMPARISON")
    df = pd.DataFrame(optimization_results)

    for weighting_name in study.quality_weightings.keys():
        print(f"\n {weighting_name} weighting:")
        weighting_df = df[df["weighting"] == weighting_name]

        summary = weighting_df.groupby("algorithm")["best_score"].agg(
            ["mean", "std", "count"]
        )
        print(summary)

        # Find best algorithm for this weighting
        best_alg = summary["mean"].idxmax()
        best_score = summary.loc[best_alg, "mean"]
        print(f"    Winner: {best_alg} (mean score: {best_score:.4f})")

    return optimization_results


def phase3_dimensional_scaling(study: StudyManager):
    """Phase 3: Multi-dimensional analysis."""
    print(f"\n{'='*60}")
    print(f"PHASE 3: MULTI-DIMENSIONAL SCALING ANALYSIS")
    print(f"{'='*60}")

    dimensions = [2, 3, 4]  # Reduced from [2,3,4,5] for demo
    scaling_results = []

    for k in dimensions:
        print(f"\n📐 Analyzing k={k} dimensions...")

        # Use balanced weighting for scaling analysis
        weights = study.quality_weightings["balanced"]

        # Run optimization for this dimension
        print(f"    Optimizing k={k} mapping...")
        start_time = time.time()

        result = optimize_diffeomorphism(
            k=k,
            optimizer="evolutionary",  # Use best algorithm from Phase 2
            max_evaluations=15,  # Reduced for demo
            quality_weights=weights,
            random_seed=42,
        )

        runtime = time.time() - start_time

        # Estimate computational complexity
        param_dimension = k * 3 + 1
        quality_assessments = result.total_evaluations

        scaling_result = {
            "dimension": k,
            "parameter_dimension": param_dimension,
            "best_score": result.best_score,
            "quality_breakdown": {
                "symmetry": result.best_metrics.symmetry_score,
                "volume_preservation": result.best_metrics.volume_preservation_score
                or 0,
                "smoothness": result.best_metrics.smoothness_score or 0,
                "coverage": result.best_metrics.coverage_score or 0,
                "invertibility": result.best_metrics.invertibility_score or 0,
            },
            "runtime_seconds": runtime,
            "evaluations": quality_assessments,
            "time_per_evaluation": runtime / quality_assessments,
            "best_params": result.best_params,
            "timestamp": datetime.now().isoformat(),
        }

        scaling_results.append(scaling_result)

        print(f"    k={k} results: Score={result.best_score:.4f}, Time={runtime:.1f}s")

    # Save scaling results
    study.results["phase3_dimensional_scaling"] = scaling_results
    study.save_results("phase3_dimensional_scaling", scaling_results)

    # Analyze scaling patterns
    print(f"\n DIMENSIONAL SCALING ANALYSIS")
    print(f"{'Dimension':<10} {'Score':<8} {'Runtime':<10} {'Time/Eval':<12}")
    print("-" * 45)

    for result in scaling_results:
        k = result["dimension"]
        score = result["best_score"]
        runtime = result["runtime_seconds"]
        time_per_eval = result["time_per_evaluation"]

        print(f"{k:<10} {score:<8.4f} {runtime:<10.1f} {time_per_eval:<12.3f}")

    return scaling_results


def phase4_application_specific(study: StudyManager, k: int = 2):
    """Phase 4: Application-specific optimization scenarios."""
    print(f"\n{'='*60}")
    print(f"PHASE 4: APPLICATION-SPECIFIC OPTIMIZATION (k={k})")
    print(f"{'='*60}")

    # Define specialized scenarios
    scenarios = {
        "max_symmetry": {
            "symmetry": 5.0,
            "volume_preservation": 0.0,
            "smoothness": 0.0,
            "coverage": 0.0,
            "invertibility": 0.0,
        },
        "max_volume_preservation": {
            "symmetry": 0.0,
            "volume_preservation": 5.0,
            "smoothness": 0.0,
            "coverage": 0.0,
            "invertibility": 0.0,
        },
        "max_coverage": {
            "symmetry": 0.0,
            "volume_preservation": 0.0,
            "smoothness": 0.0,
            "coverage": 5.0,
            "invertibility": 0.0,
        },
        "max_smoothness": {
            "symmetry": 0.0,
            "volume_preservation": 0.0,
            "smoothness": 5.0,
            "coverage": 0.0,
            "invertibility": 0.0,
        },
    }

    application_results = []

    for scenario_name, weights in scenarios.items():
        print(f"\n Optimizing for {scenario_name}...")

        result = optimize_diffeomorphism(
            k=k,
            optimizer="evolutionary",
            max_evaluations=15,  # Reduced for demo
            quality_weights=weights,
            random_seed=42,
        )

        scenario_result = {
            "scenario": scenario_name,
            "weights": weights,
            "best_score": result.best_score,
            "quality_breakdown": {
                "symmetry": result.best_metrics.symmetry_score,
                "volume_preservation": result.best_metrics.volume_preservation_score
                or 0,
                "smoothness": result.best_metrics.smoothness_score or 0,
                "coverage": result.best_metrics.coverage_score or 0,
                "invertibility": result.best_metrics.invertibility_score or 0,
            },
            "best_params": result.best_params,
            "timestamp": datetime.now().isoformat(),
        }

        application_results.append(scenario_result)

        # Show the specialized metric we optimized for
        target_metric = max(weights.keys(), key=lambda k: weights[k])
        target_value = scenario_result["quality_breakdown"][target_metric]

        print(f"    Target metric ({target_metric}): {target_value:.4f}")
        print(f"    Overall score: {result.best_score:.4f}")

    # Save application results
    study.results["phase4_application_specific"] = application_results
    study.save_results("phase4_application_specific", application_results)

    # Create application-specific recommendations
    print(f"\n APPLICATION-SPECIFIC RECOMMENDATIONS")
    for result in application_results:
        scenario = result["scenario"]
        params = result["best_params"]

        print(f"\n🔧 {scenario} optimal parameters:")
        for param_name, value in params.items():
            print(f"   {param_name}: {value:.4f}")

    return application_results


def generate_study_report(study: StudyManager):
    """Generate comprehensive study report."""
    print(f"\n{'='*60}")
    print(f"GENERATING COMPREHENSIVE STUDY REPORT")
    print(f"{'='*60}")

    # Compile all results
    report = {
        "study_metadata": {
            "timestamp": datetime.now().isoformat(),
            "phases_completed": len([k for k, v in study.results.items() if v]),
            "total_evaluations": sum(
                [
                    len(study.results.get("phase1_grid_search", [])),
                    len(study.results.get("phase2_optimization_comparison", [])),
                    len(study.results.get("phase3_dimensional_scaling", [])),
                    len(study.results.get("phase4_application_specific", [])),
                ]
            ),
        },
        "results": study.results,
    }

    # Save comprehensive report
    study.save_results("comprehensive_report", report, "complete_study_report.json")

    # Generate summary statistics
    print(f"\n STUDY SUMMARY STATISTICS")
    print(
        f"   • Total parameter configurations evaluated: {len(study.results.get('phase1_grid_search', [])):,}"
    )
    print(
        f"   • Optimization algorithm runs: {len(study.results.get('phase2_optimization_comparison', [])):,}"
    )
    print(
        f"   • Dimensions analyzed: {len(study.results.get('phase3_dimensional_scaling', []))}"
    )
    print(
        f"   • Application scenarios: {len(study.results.get('phase4_application_specific', []))}"
    )

    # Find overall best configurations
    if study.results.get("phase1_grid_search"):
        all_grid = study.results["phase1_grid_search"]
        best_overall = max(all_grid, key=lambda x: x["quality_scores"]["overall"])

        print(f"\n BEST CONFIGURATION FOUND (Grid Search):")
        print(f"   • Overall score: {best_overall['quality_scores']['overall']:.4f}")
        print(f"   • Weighting: {best_overall['weighting']}")
        print(
            f"   • Parameters: {best_overall['parameters']['special_ability']:.3f} special ability"
        )

    if study.results.get("phase2_optimization_comparison"):
        opt_results = study.results["phase2_optimization_comparison"]
        best_opt = max(opt_results, key=lambda x: x["best_score"])

        print(f"\n BEST OPTIMIZATION RESULT:")
        print(f"   • Best score: {best_opt['best_score']:.4f}")
        print(f"   • Algorithm: {best_opt['algorithm']}")
        print(f"   • Weighting: {best_opt['weighting']}")

    print(f"\n Study complete! All results saved to {study.output_dir}/")
    return report


def main():
    """Run the complete systematic study."""
    print("🔬 SYSTEMATIC STUDY: OPTIMAL THURSTONE DIFFEOMORPHISMS")
    print("=" * 80)
    print("This study systematically explores parameter space and optimization")
    print("strategies to find the best cube-to-simplex diffeomorphisms.")
    print("\n⚠️  Note: This is a DEMO version with reduced sample sizes.")
    print("For full research study, increase evaluation counts in each phase.")

    # Initialize study manager
    study = StudyManager(output_dir="study_results")

    try:
        # Phase 1: Parameter space exploration
        phase1_results = phase1_parameter_exploration(study, k=2)

        # Phase 2: Optimization algorithm comparison
        phase2_results = phase2_optimization_comparison(study, k=2)

        # Phase 3: Dimensional scaling
        phase3_results = phase3_dimensional_scaling(study)

        # Phase 4: Application-specific optimization
        phase4_results = phase4_application_specific(study, k=2)

        # Generate comprehensive report
        final_report = generate_study_report(study)

        print(f"\n🎉 SYSTEMATIC STUDY COMPLETED SUCCESSFULLY!")
        print(f" Generated comprehensive research data for Thurstone diffeomorphisms")
        print(f" All results saved to: {study.output_dir}/")

    except KeyboardInterrupt:
        print(f"\n⏸️  Study interrupted by user.")
        print(f" Partial results saved to: {study.output_dir}/")
    except Exception as e:
        print(f"\n❌ Error during study execution: {e}")
        raise


if __name__ == "__main__":
    main()
