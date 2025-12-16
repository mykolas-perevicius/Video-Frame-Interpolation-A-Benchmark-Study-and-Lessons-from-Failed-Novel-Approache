#!/usr/bin/env python3
"""Register a new source video clip for the VFI benchmark pipeline."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CLIPS_DIR = Path(__file__).parent.parent / 'data' / 'clips'
REGISTRY_FILE = CLIPS_DIR / 'clips_registry.json'


def get_video_info(video_path: Path) -> dict:
    """Extract video metadata using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)

    # Find video stream
    video_stream = None
    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'video':
            video_stream = stream
            break

    if not video_stream:
        raise RuntimeError("No video stream found")

    # Parse fps (handle fractional fps like "60000/1001")
    fps_str = video_stream.get('r_frame_rate', '30/1')
    if '/' in fps_str:
        num, den = map(int, fps_str.split('/'))
        fps = num / den
    else:
        fps = float(fps_str)

    return {
        'width': int(video_stream.get('width', 0)),
        'height': int(video_stream.get('height', 0)),
        'fps': round(fps, 2),
        'duration_s': float(data.get('format', {}).get('duration', 0)),
        'codec': video_stream.get('codec_name', 'unknown'),
        'frame_count': int(video_stream.get('nb_frames', 0)) or int(fps * float(data.get('format', {}).get('duration', 0)))
    }


def load_registry() -> dict:
    """Load the clips registry, creating if needed."""
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return {'clips': [], 'version': '1.0'}


def save_registry(registry: dict):
    """Save the clips registry."""
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)


def register_clip(video_path: Path, clip_id: str, force: bool = False) -> dict:
    """Register a video clip in the registry."""

    # Validate video exists
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Get video info
    info = get_video_info(video_path)

    # Validate requirements
    min_height = 1080
    min_fps = 60

    if info['height'] < min_height:
        raise ValueError(f"Video height {info['height']}p is below minimum {min_height}p")

    if info['fps'] < min_fps - 1:  # Allow small tolerance
        raise ValueError(f"Video fps {info['fps']} is below minimum {min_fps}")

    # Load registry
    registry = load_registry()

    # Check if clip_id already exists
    existing = next((c for c in registry['clips'] if c['clip_id'] == clip_id), None)
    if existing and not force:
        raise ValueError(f"Clip '{clip_id}' already registered. Use --force to overwrite.")

    # Create clip entry
    clip_entry = {
        'clip_id': clip_id,
        'source': str(video_path.resolve()),
        'duration_s': round(info['duration_s'], 2),
        'fps': info['fps'],
        'resolution': f"{info['width']}x{info['height']}",
        'codec': info['codec'],
        'frame_count': info['frame_count'],
        'intervals_count': 0,
        'registered': datetime.now().isoformat(),
        'last_processed': None
    }

    # Update registry
    if existing:
        idx = registry['clips'].index(existing)
        registry['clips'][idx] = clip_entry
    else:
        registry['clips'].append(clip_entry)

    save_registry(registry)

    # Create clip directory structure
    clip_dir = CLIPS_DIR / clip_id
    clip_dir.mkdir(exist_ok=True)
    (clip_dir / 'intervals').mkdir(exist_ok=True)

    # Save clip metadata
    clip_meta = {
        **clip_entry,
        'intervals': []
    }
    with open(clip_dir / 'clip_meta.json', 'w') as f:
        json.dump(clip_meta, f, indent=2)

    return clip_entry


def main():
    parser = argparse.ArgumentParser(description='Register a source video for VFI benchmarks')
    parser.add_argument('video_path', type=Path, help='Path to source video (MP4)')
    parser.add_argument('--clip-id', type=str, help='Unique identifier for this clip (default: filename)')
    parser.add_argument('--force', action='store_true', help='Overwrite existing registration')
    args = parser.parse_args()

    # Generate clip_id from filename if not provided
    clip_id = args.clip_id or args.video_path.stem

    print(f"\n{'='*60}")
    print("REGISTER VIDEO CLIP")
    print(f"{'='*60}")
    print(f"Source: {args.video_path}")
    print(f"Clip ID: {clip_id}")

    try:
        entry = register_clip(args.video_path, clip_id, args.force)

        print(f"\nVideo Info:")
        print(f"  Resolution: {entry['resolution']}")
        print(f"  FPS: {entry['fps']}")
        print(f"  Duration: {entry['duration_s']:.1f}s")
        print(f"  Frames: {entry['frame_count']}")
        print(f"  Codec: {entry['codec']}")

        print(f"\nClip registered successfully!")
        print(f"Directory: data/clips/{clip_id}/")
        print(f"\nNext step: Extract intervals with:")
        print(f"  python scripts/extract_intervals.py {clip_id}")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
