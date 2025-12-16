#!/usr/bin/env python3
"""List all registered clips and their extracted intervals."""

import argparse
import json
from pathlib import Path

CLIPS_DIR = Path(__file__).parent.parent / 'data' / 'clips'
REGISTRY_FILE = CLIPS_DIR / 'clips_registry.json'


def load_registry() -> dict:
    """Load clips registry."""
    if not REGISTRY_FILE.exists():
        return {'clips': []}
    with open(REGISTRY_FILE) as f:
        return json.load(f)


def list_clips(verbose: bool = False, clip_filter: str = None):
    """List all registered clips and their intervals."""

    registry = load_registry()

    if not registry['clips']:
        print("No clips registered.")
        print("\nRegister a clip with:")
        print("  python scripts/register_clip.py data/raw/your_video.mp4")
        return

    print(f"\n{'='*70}")
    print("REGISTERED CLIPS AND INTERVALS")
    print(f"{'='*70}")

    for clip in registry['clips']:
        clip_id = clip['clip_id']

        if clip_filter and clip_filter not in clip_id:
            continue

        print(f"\n{clip_id}:")
        print(f"  Source: {clip['source']}")
        print(f"  Resolution: {clip['resolution']} @ {clip['fps']}fps")
        print(f"  Duration: {clip['duration_s']:.1f}s ({clip.get('frame_count', 'N/A')} frames)")
        print(f"  Intervals: {clip['intervals_count']}")

        # List intervals
        clip_dir = CLIPS_DIR / clip_id / 'intervals'
        if clip_dir.exists():
            intervals = sorted(clip_dir.glob('interval_*'))

            if intervals:
                print(f"\n  {'ID':<16} {'Time':<16} {'Difficulty':<10} {'Frames':<8} {'Mean Flow'}")
                print(f"  {'-'*16} {'-'*16} {'-'*10} {'-'*8} {'-'*10}")

                for interval_path in intervals:
                    meta_file = interval_path / 'meta.json'
                    if meta_file.exists():
                        with open(meta_file) as f:
                            meta = json.load(f)

                        interval_id = meta['interval_id']
                        time_range = f"{meta['start_s']:.1f}-{meta['end_s']:.1f}s"
                        difficulty = meta.get('difficulty', 'N/A')
                        frames = meta.get('keyframe_count', 'N/A')
                        mean_flow = meta.get('motion_stats', {}).get('mean_flow', 0)

                        print(f"  {interval_id:<16} {time_range:<16} {difficulty:<10} {frames:<8} {mean_flow:.2f}")

                        if verbose:
                            stats = meta.get('motion_stats', {})
                            print(f"    Motion breakdown: "
                                  f"STATIC={stats.get('static_pct', 0):.0f}% "
                                  f"EASY={stats.get('easy_pct', 0):.0f}% "
                                  f"MEDIUM={stats.get('medium_pct', 0):.0f}% "
                                  f"HARD={stats.get('hard_pct', 0):.0f}% "
                                  f"EXTREME={stats.get('extreme_pct', 0):.0f}%")
            else:
                print("  (no intervals extracted)")

    # Summary
    total_intervals = sum(c['intervals_count'] for c in registry['clips'])
    print(f"\n{'='*70}")
    print(f"Total: {len(registry['clips'])} clips, {total_intervals} intervals")

    if total_intervals == 0:
        print("\nExtract intervals with:")
        print(f"  python scripts/extract_intervals.py {registry['clips'][0]['clip_id']}")


def main():
    parser = argparse.ArgumentParser(description='List registered clips and intervals')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed motion stats')
    parser.add_argument('--clip', type=str, help='Filter by clip ID')
    args = parser.parse_args()

    list_clips(verbose=args.verbose, clip_filter=args.clip)


if __name__ == '__main__':
    main()
