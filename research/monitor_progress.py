"""
Monitor the progress of the systematic study and prepare for next phases.
"""

import json
import os
import time
from datetime import datetime


def check_study_progress():
    """Check the current progress of the systematic study."""
    results_dir = "study_results"

    if not os.path.exists(results_dir):
        print(" Study not yet started - no results directory found")
        return

    # Check for completed phases
    phase_files = {
        "Phase 1 (Grid Search)": "phase1_grid_search_*.json",
        "Phase 2 (Optimization)": "phase2_optimization_comparison_*.json",
        "Phase 3 (Dimensional)": "phase3_dimensional_scaling_*.json",
        "Phase 4 (Application)": "phase4_application_specific_*.json",
        "Final Report": "complete_study_report.json",
    }

    print(f" STUDY PROGRESS CHECK - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    completed_phases = []

    for phase, pattern in phase_files.items():
        # Check for files matching pattern
        import glob

        matches = glob.glob(os.path.join(results_dir, pattern))

        if matches:
            latest_file = max(matches, key=os.path.getctime)
            file_size = os.path.getsize(latest_file)
            mod_time = datetime.fromtimestamp(os.path.getmtime(latest_file))

            print(
                f"✅ {phase}: {os.path.basename(latest_file)} ({file_size:,} bytes, {mod_time.strftime('%H:%M:%S')})"
            )
            completed_phases.append(phase)

            # Load and show summary if it's a results file
            if file_size > 100 and phase != "Final Report":
                try:
                    with open(latest_file, "r") as f:
                        data = json.load(f)

                    if isinstance(data, list) and len(data) > 0:
                        print(f"    Contains {len(data):,} evaluation records")

                        # Show some key stats
                        if "quality_scores" in data[0]:
                            scores = [
                                item["quality_scores"]["overall"]
                                for item in data
                                if "quality_scores" in item
                            ]
                            if scores:
                                print(
                                    f"    Score range: {min(scores):.4f} - {max(scores):.4f}"
                                )

                except Exception as e:
                    print(f"   ⚠️ Could not parse file: {e}")
        else:
            print(f"⏳ {phase}: In progress or not started")

    print(f"\n Progress: {len(completed_phases)}/{len(phase_files)} phases complete")

    return len(completed_phases), len(phase_files)


def prepare_next_phase():
    """Prepare commands for the next phase of research."""
    completed, total = check_study_progress()

    print(f"\n NEXT STEPS:")

    if completed == 0:
        print("⏳ Phase 1 (Systematic Study) is running...")
        print("   Command: python research/run_systematic_study.py")
        print("   Expected: 1-3 hours for demo version")

    elif completed >= 1 and completed < total:
        print(" Phase 1 completed! Ready for Phase 2 (Generate Figures)")
        print("   Command: python research/generate_paper_figures.py")
        print("   Expected: 30-60 minutes")

    elif completed == total:
        print("🎉 All phases completed! Ready for analysis and paper finalization")
        print("   Next: Manual analysis of results")
        print("   Then: Update paper_draft.md with findings")

    else:
        print(" Study in progress - check back later")


def show_key_findings():
    """Show key findings from completed phases."""
    results_dir = "study_results"

    # Check for Phase 1 results (parameter exploration)
    import glob

    phase1_files = glob.glob(os.path.join(results_dir, "phase1_grid_search_*.json"))

    if phase1_files:
        print(f"\n KEY FINDINGS FROM PHASE 1:")
        print("=" * 40)

        latest_file = max(phase1_files, key=os.path.getctime)
        try:
            with open(latest_file, "r") as f:
                data = json.load(f)

            if isinstance(data, list) and len(data) > 10:
                # Find best overall configurations
                best_configs = sorted(
                    data, key=lambda x: x["quality_scores"]["overall"], reverse=True
                )[:5]

                print(" TOP 5 PARAMETER CONFIGURATIONS:")
                for i, config in enumerate(best_configs, 1):
                    score = config["quality_scores"]["overall"]
                    weighting = config["weighting"]
                    special_ability = config["parameters"]["special_ability"]

                    print(
                        f"   {i}. Score: {score:.4f} ({weighting}, special: {special_ability:.2f})"
                    )

                # Analyze by weighting scheme
                weightings = set(item["weighting"] for item in data)
                print(f"\n BEST BY WEIGHTING SCHEME:")
                for weight in weightings:
                    weight_data = [item for item in data if item["weighting"] == weight]
                    best = max(
                        weight_data, key=lambda x: x["quality_scores"]["overall"]
                    )
                    print(f"   {weight}: {best['quality_scores']['overall']:.4f}")

        except Exception as e:
            print(f"❌ Error analyzing Phase 1 results: {e}")

    # Check for Phase 2 results (optimization comparison)
    phase2_files = glob.glob(os.path.join(results_dir, "phase2_optimization_*.json"))

    if phase2_files:
        print(f"\n KEY FINDINGS FROM PHASE 2:")
        print("=" * 40)

        latest_file = max(phase2_files, key=os.path.getctime)
        try:
            with open(latest_file, "r") as f:
                data = json.load(f)

            if isinstance(data, list) and len(data) > 0:
                # Group by algorithm
                algorithms = set(item["algorithm"] for item in data)
                print(" ALGORITHM PERFORMANCE:")

                for alg in algorithms:
                    alg_data = [item for item in data if item["algorithm"] == alg]
                    scores = [item["best_score"] for item in alg_data]

                    if scores:
                        mean_score = sum(scores) / len(scores)
                        max_score = max(scores)
                        print(
                            f"   {alg}: Mean={mean_score:.4f}, Best={max_score:.4f} ({len(scores)} runs)"
                        )

        except Exception as e:
            print(f"❌ Error analyzing Phase 2 results: {e}")


def main():
    """Main monitoring function."""
    print("🔬 THURSTONE DIFFEOMORPHISMS RESEARCH MONITOR")
    print("=" * 60)

    # Check progress
    check_study_progress()

    # Show findings if available
    show_key_findings()

    # Suggest next steps
    prepare_next_phase()

    print(f"\n TIP: Run this monitor script periodically to track progress:")
    print(f"   python research/monitor_progress.py")


if __name__ == "__main__":
    main()
