#!/usr/bin/env python3
"""Run full benchmark suite on all pre-computed intervals.

This script runs all experiments on all available intervals to generate
statistically rigorous results.

Usage:
    python scripts/run_full_benchmark.py                    # Run all methods on all intervals
    python scripts/run_full_benchmark.py --light-only       # CPU-only methods
    python scripts/run_full_benchmark.py --methods rife,lanczos_blend  # Specific methods
    python scripts/run_full_benchmark.py --clip arc_raiders_001 --intervals 3  # First 3 intervals
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
CLIPS_DIR = SCRIPTS_DIR.parent / 'data' / 'clips'
OUTPUTS_DIR = SCRIPTS_DIR.parent / 'outputs'


def get_available_clips():
    """Get list of registered clips."""
    registry_file = CLIPS_DIR / 'clips_registry.json'
    if not registry_file.exists():
        return []
    with open(registry_file) as f:
        data = json.load(f)
    return [c['clip_id'] for c in data.get('clips', [])]


def get_clip_intervals(clip_id):
    """Get list of intervals for a clip."""
    intervals_dir = CLIPS_DIR / clip_id / 'intervals'
    if not intervals_dir.exists():
        return []

    intervals = []
    for interval_path in sorted(intervals_dir.glob('interval_*')):
        meta_file = interval_path / 'meta.json'
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            intervals.append(meta)
    return intervals


def run_experiment_batch(clip_id, intervals, light_only=False, heavy_only=False, methods=None):
    """Run experiments on specified intervals."""

    # Build command
    cmd = [
        sys.executable, str(SCRIPTS_DIR / 'run_experiments.py'),
        '--clip', clip_id,
        '--intervals', 'all' if not intervals else ','.join([i['interval_id'] for i in intervals])
    ]

    if light_only:
        cmd.append('--light-only')
    elif heavy_only:
        cmd.append('--heavy-only')

    if methods:
        cmd.extend(['--experiment', methods[0]])  # TODO: support multiple methods

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Run subprocess
    result = subprocess.run(cmd, cwd=str(SCRIPTS_DIR.parent))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run full benchmark suite')
    parser.add_argument('--clip', type=str, help='Specific clip to benchmark (default: all)')
    parser.add_argument('--intervals', type=int, default=None, help='Number of intervals to use (default: all)')
    parser.add_argument('--light-only', action='store_true', help='Run only CPU-based experiments')
    parser.add_argument('--heavy-only', action='store_true', help='Run only GPU-based experiments')
    parser.add_argument('--methods', type=str, help='Comma-separated list of methods to run')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run without executing')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("FULL BENCHMARK SUITE")
    print(f"{'='*60}")
    print(f"Started: {datetime.now().isoformat()}")

    # Get clips to benchmark
    if args.clip:
        clips = [args.clip]
    else:
        clips = get_available_clips()

    if not clips:
        print("\n[ERROR] No clips registered. Run register_clip.py first.")
        sys.exit(1)

    print(f"\nClips: {', '.join(clips)}")

    # Process each clip
    total_intervals = 0
    for clip_id in clips:
        intervals = get_clip_intervals(clip_id)

        if not intervals:
            print(f"\n[WARNING] No intervals for '{clip_id}'. Run extract_intervals.py first.")
            continue

        # Limit intervals if specified
        if args.intervals:
            intervals = intervals[:args.intervals]

        total_intervals += len(intervals)

        print(f"\n{clip_id}: {len(intervals)} intervals")
        for i in intervals:
            print(f"  - {i['interval_id']}: {i['start_s']:.1f}-{i['end_s']:.1f}s ({i.get('difficulty', 'N/A')})")

        if args.dry_run:
            continue

        # Run experiments
        success = run_experiment_batch(
            clip_id, intervals,
            light_only=args.light_only,
            heavy_only=args.heavy_only,
            methods=args.methods.split(',') if args.methods else None
        )

        if not success:
            print(f"\n[WARNING] Some experiments failed for '{clip_id}'")

    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Clips: {len(clips)}")
    print(f"Total intervals: {total_intervals}")
    print(f"Finished: {datetime.now().isoformat()}")

    if not args.dry_run:
        print(f"\nRun statistical analysis:")
        print(f"  python scripts/analyze_statistics.py")


if __name__ == '__main__':
    main()
