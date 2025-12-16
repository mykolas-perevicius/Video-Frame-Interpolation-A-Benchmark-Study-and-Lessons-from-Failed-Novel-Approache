#!/usr/bin/env python3
"""Generate a quick summary of current benchmark status."""

import json
from pathlib import Path
from collections import defaultdict

OUTPUTS_DIR = Path(__file__).parent.parent / 'outputs'
CLIPS_DIR = Path(__file__).parent.parent / 'data' / 'clips'


def main():
    print("\n" + "=" * 70)
    print("VFI BENCHMARK STATUS SUMMARY")
    print("=" * 70)

    # Check clips and intervals
    registry_file = CLIPS_DIR / 'clips_registry.json'
    if registry_file.exists():
        with open(registry_file) as f:
            registry = json.load(f)
        clips = registry.get('clips', [])
        print(f"\n📁 CLIPS: {len(clips)} registered")
        for clip in clips:
            intervals_dir = CLIPS_DIR / clip['clip_id'] / 'intervals'
            interval_count = len(list(intervals_dir.glob('interval_*'))) if intervals_dir.exists() else 0
            print(f"   - {clip['clip_id']}: {interval_count} intervals, {clip['duration_s']:.1f}s")
    else:
        print("\n📁 CLIPS: None registered")

    # Check experiment results
    results_file = OUTPUTS_DIR / 'experiment_results.json'
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        experiments = results.get('experiments', [])

        # Group by method
        by_method = defaultdict(list)
        for exp in experiments:
            by_method[exp['name']].append(exp)

        print(f"\n📊 EXPERIMENTS: {len(experiments)} total runs")
        print(f"   Methods: {len(by_method)}")

        # Count by n
        n_counts = defaultdict(int)
        for method, runs in by_method.items():
            n_counts[len(runs)] += 1

        print(f"   Distribution: ", end="")
        for n, count in sorted(n_counts.items()):
            print(f"n={n}: {count} methods, ", end="")
        print()

        # Top 5 methods
        print(f"\n🏆 TOP 5 BY VFI PSNR:")
        ranked = sorted(by_method.items(),
                       key=lambda x: max(e.get('vfi_psnr_db', 0) or 0 for e in x[1]),
                       reverse=True)
        for i, (method, runs) in enumerate(ranked[:5], 1):
            best_psnr = max(e.get('vfi_psnr_db', 0) or 0 for e in runs)
            print(f"   {i}. {method}: {best_psnr:.2f} dB (n={len(runs)})")

        # Statistical readiness
        print(f"\n📈 STATISTICAL READINESS:")
        ready = sum(1 for runs in by_method.values() if len(runs) >= 5)
        partial = sum(1 for runs in by_method.values() if 2 <= len(runs) < 5)
        single = sum(1 for runs in by_method.values() if len(runs) == 1)
        print(f"   Ready (n≥5): {ready} methods")
        print(f"   Partial (n=2-4): {partial} methods")
        print(f"   Single run: {single} methods")

    else:
        print("\n📊 EXPERIMENTS: No results yet")

    # Check generated outputs
    print(f"\n📄 OUTPUTS:")
    outputs = [
        ('experiment_results.json', 'Raw experiment data'),
        ('statistical_report.json', 'Statistical analysis'),
        ('statistical_summary.md', 'Human-readable stats'),
        ('figures/', 'Publication figures'),
    ]
    for name, desc in outputs:
        path = OUTPUTS_DIR / name
        exists = "✅" if path.exists() else "❌"
        print(f"   {exists} {name}: {desc}")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)

    # Suggest next steps
    if not registry_file.exists():
        print("1. Register a clip: python scripts/register_clip.py data/raw/clip1.mp4")
    elif not results_file.exists():
        print("1. Run experiments: python scripts/run_experiments.py --clip arc_raiders_001 --intervals all")
    elif single > ready:
        print("1. Run more intervals for statistical power:")
        print("   python scripts/run_experiments.py --clip arc_raiders_001 --intervals all")
    else:
        print("1. Generate final report figures:")
        print("   python scripts/generate_report_figures.py")

    print()


if __name__ == '__main__':
    main()
