#!/usr/bin/env python3
"""Extract pre-computed intervals from a registered video clip."""

import argparse
import json
import cv2
import numpy as np
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

CLIPS_DIR = Path(__file__).parent.parent / 'data' / 'clips'
REGISTRY_FILE = CLIPS_DIR / 'clips_registry.json'

# Motion difficulty thresholds (pixels of mean optical flow)
DIFFICULTY_THRESHOLDS = {
    'STATIC': 1.0,
    'EASY': 5.0,
    'MEDIUM': 15.0,
    'HARD': 30.0,
    'EXTREME': float('inf')
}


def crop16_9(frame):
    """Crop frame to 16:9 aspect ratio (matching run_experiments.py)."""
    h, w = frame.shape[:2]
    target_ratio = 16 / 9

    if w / h > target_ratio:
        # Too wide - crop width
        new_w = int(h * target_ratio)
        start_x = (w - new_w) // 2
        return frame[:, start_x:start_x + new_w]
    elif w / h < target_ratio:
        # Too tall - crop height
        new_h = int(w / target_ratio)
        start_y = (h - new_h) // 2
        return frame[start_y:start_y + new_h, :]
    return frame


def load_registry() -> dict:
    """Load clips registry."""
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError("No clips registered. Run register_clip.py first.")
    with open(REGISTRY_FILE) as f:
        return json.load(f)


def save_registry(registry: dict):
    """Save clips registry."""
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)


def get_clip_info(clip_id: str) -> dict:
    """Get clip metadata from registry."""
    registry = load_registry()
    clip = next((c for c in registry['clips'] if c['clip_id'] == clip_id), None)
    if not clip:
        raise ValueError(f"Clip '{clip_id}' not found. Run register_clip.py first.")
    return clip


def compute_motion_stats(frame1: np.ndarray, frame2: np.ndarray) -> dict:
    """Compute optical flow statistics between two frames."""
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute at half resolution for speed
    h, w = gray1.shape
    small1 = cv2.resize(gray1, (w // 2, h // 2))
    small2 = cv2.resize(gray2, (w // 2, h // 2))

    # Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(
        small1, small2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Compute flow magnitude
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2) * 2  # Scale back up

    return {
        'mean_flow': float(np.mean(mag)),
        'max_flow': float(np.max(mag)),
        'std_flow': float(np.std(mag)),
        'median_flow': float(np.median(mag))
    }


def classify_difficulty(mean_flow: float) -> str:
    """Classify motion difficulty based on mean flow."""
    if mean_flow < DIFFICULTY_THRESHOLDS['STATIC']:
        return 'STATIC'
    elif mean_flow < DIFFICULTY_THRESHOLDS['EASY']:
        return 'EASY'
    elif mean_flow < DIFFICULTY_THRESHOLDS['MEDIUM']:
        return 'MEDIUM'
    elif mean_flow < DIFFICULTY_THRESHOLDS['HARD']:
        return 'HARD'
    else:
        return 'EXTREME'


def extract_single_interval(
    video_path: str,
    clip_id: str,
    interval_id: str,
    start_s: float,
    duration_s: float,
    output_resolution: Tuple[int, int] = (1920, 1080)
) -> dict:
    """Extract a single interval with keyframes, midpoints, and motion stats."""

    clip_dir = CLIPS_DIR / clip_id
    interval_dir = clip_dir / 'intervals' / interval_id

    # Create directories
    keyframes_dir = interval_dir / 'keyframes'
    midpoints_dir = interval_dir / 'midpoints'
    motion_dir = interval_dir / 'motion'

    keyframes_dir.mkdir(parents=True, exist_ok=True)
    midpoints_dir.mkdir(parents=True, exist_ok=True)
    motion_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_s * fps)
    end_frame = int((start_s + duration_s) * fps)
    end_frame = min(end_frame, total_frames)

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    keyframes = []
    midpoints = []
    motion_data = []
    difficulty_counts = {k: 0 for k in DIFFICULTY_THRESHOLDS.keys()}

    prev_keyframe = None
    frame_idx = 0
    kf_idx = 0

    print(f"  Extracting {interval_id}: {start_s:.1f}s - {start_s + duration_s:.1f}s")

    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop to 16:9 and resize to output resolution
        # (matching the processing in run_experiments.py generate_reference)
        frame = crop16_9(frame)
        if frame.shape[1] != output_resolution[0] or frame.shape[0] != output_resolution[1]:
            frame = cv2.resize(frame, output_resolution, interpolation=cv2.INTER_LANCZOS4)

        # Even frames = keyframes, Odd frames = midpoints (ground truth)
        if frame_idx % 2 == 0:
            # Keyframe
            kf_path = keyframes_dir / f'kf_{kf_idx:04d}.png'
            cv2.imwrite(str(kf_path), frame)
            keyframes.append(f'kf_{kf_idx:04d}.png')

            # Compute motion stats if we have previous keyframe
            if prev_keyframe is not None:
                stats = compute_motion_stats(prev_keyframe, frame)
                motion_data.append({
                    'pair_idx': kf_idx - 1,
                    **stats
                })
                difficulty = classify_difficulty(stats['mean_flow'])
                difficulty_counts[difficulty] += 1

            prev_keyframe = frame.copy()
            kf_idx += 1
        else:
            # Midpoint (ground truth)
            gt_path = midpoints_dir / f'gt_{len(midpoints):04d}.png'
            cv2.imwrite(str(gt_path), frame)
            midpoints.append(f'gt_{len(midpoints):04d}.png')

        frame_idx += 1

        # Progress indicator
        if frame_idx % 100 == 0:
            print(f"    Processed {frame_idx} frames...")

    cap.release()

    # Compute aggregate motion stats
    if motion_data:
        all_mean = [m['mean_flow'] for m in motion_data]
        all_max = [m['max_flow'] for m in motion_data]
        aggregate_stats = {
            'mean_flow': float(np.mean(all_mean)),
            'max_flow': float(np.max(all_max)),
            'variance': float(np.var(all_mean)),
            'static_pct': round(100 * difficulty_counts['STATIC'] / len(motion_data), 1),
            'easy_pct': round(100 * difficulty_counts['EASY'] / len(motion_data), 1),
            'medium_pct': round(100 * difficulty_counts['MEDIUM'] / len(motion_data), 1),
            'hard_pct': round(100 * difficulty_counts['HARD'] / len(motion_data), 1),
            'extreme_pct': round(100 * difficulty_counts['EXTREME'] / len(motion_data), 1)
        }
        overall_difficulty = classify_difficulty(aggregate_stats['mean_flow'])
    else:
        aggregate_stats = {}
        overall_difficulty = 'UNKNOWN'

    # Save motion data
    with open(motion_dir / 'complexity.json', 'w') as f:
        json.dump(motion_data, f, indent=2)
    with open(motion_dir / 'flow_stats.json', 'w') as f:
        json.dump(aggregate_stats, f, indent=2)

    # Create interval metadata
    meta = {
        'interval_id': interval_id,
        'clip_id': clip_id,
        'start_s': start_s,
        'end_s': start_s + duration_s,
        'start_frame': start_frame,
        'end_frame': start_frame + frame_idx,
        'keyframe_count': len(keyframes),
        'midpoint_count': len(midpoints),
        'resolution': f'{output_resolution[0]}x{output_resolution[1]}',
        'difficulty': overall_difficulty,
        'motion_stats': aggregate_stats,
        'extracted': datetime.now().isoformat()
    }

    with open(interval_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"    Done: {len(keyframes)} keyframes, {len(midpoints)} midpoints, difficulty={overall_difficulty}")

    return meta


def extract_intervals(
    clip_id: str,
    count: int = 10,
    duration: float = 10.0,
    start: Optional[float] = None,
    auto_difficulty: bool = False,
    output_resolution: Tuple[int, int] = (1920, 1080)
) -> List[dict]:
    """Extract multiple intervals from a clip."""

    clip_info = get_clip_info(clip_id)
    video_path = clip_info['source']
    video_duration = clip_info['duration_s']

    # Calculate interval positions
    if start is not None:
        # Single interval at specific position
        starts = [start]
    elif auto_difficulty:
        # Spread intervals to cover different parts of the video
        # Sample more from middle where action typically is
        positions = np.linspace(0.1, 0.9, count)
        starts = [p * (video_duration - duration) for p in positions]
    else:
        # Evenly spaced intervals
        max_start = video_duration - duration
        if count == 1:
            starts = [max_start / 2]  # Middle of video
        else:
            starts = np.linspace(0, max_start, count).tolist()

    # Validate intervals fit
    valid_starts = [s for s in starts if s >= 0 and s + duration <= video_duration]
    if len(valid_starts) < len(starts):
        print(f"Warning: {len(starts) - len(valid_starts)} intervals skipped (out of bounds)")

    # Find existing intervals to determine next index
    clip_dir = CLIPS_DIR / clip_id / 'intervals'
    existing = list(clip_dir.glob('interval_*')) if clip_dir.exists() else []
    next_idx = len(existing)

    # Extract each interval
    results = []
    for i, start_s in enumerate(valid_starts):
        interval_id = f'interval_{next_idx + i:04d}'
        meta = extract_single_interval(
            video_path, clip_id, interval_id,
            start_s, duration, output_resolution
        )
        results.append(meta)

    # Update registry
    registry = load_registry()
    for clip in registry['clips']:
        if clip['clip_id'] == clip_id:
            clip['intervals_count'] = next_idx + len(results)
            clip['last_processed'] = datetime.now().isoformat()
            break
    save_registry(registry)

    # Update clip metadata
    clip_meta_path = CLIPS_DIR / clip_id / 'clip_meta.json'
    if clip_meta_path.exists():
        with open(clip_meta_path) as f:
            clip_meta = json.load(f)
        clip_meta['intervals'] = [r['interval_id'] for r in results]
        clip_meta['intervals_count'] = next_idx + len(results)
        clip_meta['last_processed'] = datetime.now().isoformat()
        with open(clip_meta_path, 'w') as f:
            json.dump(clip_meta, f, indent=2)

    # Update difficulty index
    difficulty_index = {}
    for r in results:
        difficulty_index[r['interval_id']] = {
            'difficulty': r['difficulty'],
            'mean_flow': r['motion_stats'].get('mean_flow', 0),
            'start_s': r['start_s'],
            'end_s': r['end_s']
        }
    with open(CLIPS_DIR / clip_id / 'difficulty_index.json', 'w') as f:
        json.dump(difficulty_index, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description='Extract intervals from a registered clip')
    parser.add_argument('clip_id', type=str, help='Clip ID to extract from')
    parser.add_argument('--count', type=int, default=10, help='Number of intervals (default: 10)')
    parser.add_argument('--duration', type=float, default=10.0, help='Duration per interval in seconds (default: 10)')
    parser.add_argument('--start', type=float, help='Extract single interval at this start time')
    parser.add_argument('--auto-difficulty', action='store_true', help='Spread intervals to cover difficulty range')
    parser.add_argument('--resolution', type=str, default='1920x1080', help='Output resolution (default: 1920x1080)')
    args = parser.parse_args()

    # Parse resolution
    try:
        w, h = map(int, args.resolution.split('x'))
        resolution = (w, h)
    except ValueError:
        print(f"Invalid resolution format: {args.resolution}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print("EXTRACT INTERVALS")
    print(f"{'='*60}")
    print(f"Clip: {args.clip_id}")
    print(f"Count: {args.count if args.start is None else 1}")
    print(f"Duration: {args.duration}s per interval")
    print(f"Resolution: {args.resolution}")

    try:
        results = extract_intervals(
            args.clip_id,
            count=args.count,
            duration=args.duration,
            start=args.start,
            auto_difficulty=args.auto_difficulty,
            output_resolution=resolution
        )

        print(f"\n{'='*60}")
        print(f"Extracted {len(results)} intervals:")
        for r in results:
            print(f"  {r['interval_id']}: {r['start_s']:.1f}-{r['end_s']:.1f}s, {r['difficulty']}, {r['keyframe_count']} frames")

        print(f"\nView all intervals with:")
        print(f"  python scripts/list_intervals.py")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
