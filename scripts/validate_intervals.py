#!/usr/bin/env python3
"""Validate that cached intervals match on-the-fly extraction."""

import argparse
import cv2
import numpy as np
import sys
from pathlib import Path

# Import from run_experiments
sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import generate_reference, crop16_9, load_interval, CLIPS_DIR

PROJECT_ROOT = Path(__file__).parent.parent


def validate_interval(clip_id: str, interval_id: str, raw_video: str = None):
    """Validate cached interval frames match on-the-fly extraction.

    Args:
        clip_id: Clip identifier
        interval_id: Interval identifier
        raw_video: Path to raw video (auto-detected if not provided)
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING INTERVAL: {clip_id}/{interval_id}")
    print(f"{'='*60}")

    # Load cached interval
    cached = load_interval(clip_id, interval_id)
    if not cached:
        print(f"[ERROR] Could not load cached interval")
        return False

    print(f"Cached: {len(cached['keyframes'])} keyframes, {len(cached['midpoints'])} midpoints")
    print(f"Meta: {cached['meta']['start_s']:.1f}s - {cached['meta']['end_s']:.1f}s")

    # Get raw video path
    if raw_video is None:
        registry_file = CLIPS_DIR / 'clips_registry.json'
        if registry_file.exists():
            import json
            with open(registry_file) as f:
                registry = json.load(f)
            clip = next((c for c in registry['clips'] if c['clip_id'] == clip_id), None)
            if clip:
                raw_video = clip['source']

    if not raw_video or not Path(raw_video).exists():
        print(f"[ERROR] Could not find raw video")
        return False

    print(f"Raw video: {raw_video}")

    # Get start frame from metadata
    start_s = cached['meta']['start_s']
    cap = cv2.VideoCapture(raw_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_s * fps)
    cap.release()

    # Extract on-the-fly for comparison
    print(f"\nExtracting on-the-fly from frame {start_frame}...")

    # Use cached frame resolution
    cached_h, cached_w = cached['keyframes'][0].shape[:2]

    # Generate reference frames
    from run_experiments import OUT_W, OUT_H, DURATION
    # Temporarily set globals
    import run_experiments
    orig_w, orig_h, orig_dur = run_experiments.OUT_W, run_experiments.OUT_H, run_experiments.DURATION
    run_experiments.OUT_W, run_experiments.OUT_H = cached_w, cached_h
    run_experiments.DURATION = cached['meta']['end_s'] - cached['meta']['start_s']

    try:
        ref_frames, gt_midpoints = generate_reference(raw_video, start_frame, extract_midpoints=True)
    finally:
        run_experiments.OUT_W, run_experiments.OUT_H, run_experiments.DURATION = orig_w, orig_h, orig_dur

    print(f"On-the-fly: {len(ref_frames)} keyframes, {len(gt_midpoints)} midpoints")

    # Compare
    n_compare = min(len(cached['keyframes']), len(ref_frames), 10)
    print(f"\nComparing first {n_compare} keyframes...")

    psnr_values = []
    for i in range(n_compare):
        cached_frame = cached['keyframes'][i]
        ref_frame = ref_frames[i]

        # Calculate PSNR
        mse = np.mean((cached_frame.astype(float) - ref_frame.astype(float)) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(255**2 / mse)
        psnr_values.append(psnr)

        match = "✅ MATCH" if psnr > 50 else ("⚠️ CLOSE" if psnr > 30 else "❌ MISMATCH")
        print(f"  Frame {i}: PSNR = {psnr:.1f} dB {match}")

    # Summary
    avg_psnr = np.mean([p for p in psnr_values if p != float('inf')])
    perfect = sum(1 for p in psnr_values if p == float('inf'))

    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Perfect matches: {perfect}/{n_compare}")
    print(f"Average PSNR: {avg_psnr:.1f} dB")

    if perfect == n_compare or avg_psnr > 50:
        print(f"\n✅ VALIDATION PASSED - Cached frames match on-the-fly extraction")
        return True
    elif avg_psnr > 30:
        print(f"\n⚠️ VALIDATION WARNING - Minor differences detected")
        return True
    else:
        print(f"\n❌ VALIDATION FAILED - Cached frames do NOT match")
        return False


def main():
    parser = argparse.ArgumentParser(description='Validate cached intervals')
    parser.add_argument('clip_id', type=str, help='Clip ID to validate')
    parser.add_argument('--interval', type=str, help='Specific interval to validate (default: first)')
    parser.add_argument('--raw', type=str, help='Path to raw video (auto-detected if not provided)')
    args = parser.parse_args()

    # Get interval
    if args.interval:
        interval_id = args.interval
    else:
        intervals_dir = CLIPS_DIR / args.clip_id / 'intervals'
        intervals = sorted(intervals_dir.glob('interval_*'))
        if not intervals:
            print(f"[ERROR] No intervals found for {args.clip_id}")
            sys.exit(1)
        interval_id = intervals[0].name

    success = validate_interval(args.clip_id, interval_id, args.raw)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
