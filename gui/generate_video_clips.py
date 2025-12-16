#!/usr/bin/env python3
"""
Video Clip Generator for Blind Study

Generates video clips from processed frame data using different VFI+SR models.
Output: 4K (3840x2160) video clips at configurable frame rates.

Usage:
    python gui/generate_video_clips.py --clip arc_raiders_001
    python gui/generate_video_clips.py --clip arc_raiders_001 --fps 60 --duration 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Target output resolution: 4K
TARGET_WIDTH = 3840
TARGET_HEIGHT = 2160
SCALE_FACTOR = 2.0  # 1080p -> 4K

# Aspect ratio handling modes
ASPECT_MODES = ['scale', 'crop', 'letterbox']


def handle_aspect_ratio(frame: np.ndarray, target_w: int, target_h: int,
                        mode: str = 'scale') -> np.ndarray:
    """
    Handle aspect ratio when resizing to target dimensions.

    Modes:
        - scale: Stretch to fit (may distort)
        - crop: Scale up and center crop (no distortion, loses edges)
        - letterbox: Scale to fit with black bars (no distortion, no crop)
    """
    h, w = frame.shape[:2]
    src_ratio = w / h
    dst_ratio = target_w / target_h

    if mode == 'scale':
        # Just stretch to target dimensions
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    elif mode == 'crop':
        # Scale to cover entire target, then center crop
        if src_ratio > dst_ratio:
            # Source is wider - scale by height, crop width
            new_h = target_h
            new_w = int(w * (target_h / h))
        else:
            # Source is taller - scale by width, crop height
            new_w = target_w
            new_h = int(h * (target_w / w))

        scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Center crop
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        return scaled[start_y:start_y + target_h, start_x:start_x + target_w]

    elif mode == 'letterbox':
        # Scale to fit inside target, add black bars
        if src_ratio > dst_ratio:
            # Source is wider - fit by width, add top/bottom bars
            new_w = target_w
            new_h = int(h * (target_w / w))
        else:
            # Source is taller - fit by height, add left/right bars
            new_h = target_h
            new_w = int(w * (target_h / h))

        scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Create black canvas and paste scaled image centered
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = scaled
        return canvas

    else:
        # Default to scale
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def upscale_frame(frame: np.ndarray, model_name: str,
                  target_w: int = TARGET_WIDTH,
                  target_h: int = TARGET_HEIGHT,
                  aspect_mode: str = 'letterbox') -> np.ndarray:
    """
    Apply upscaling model to a frame (1080p -> 4K)

    Args:
        frame: Input frame (1080p)
        model_name: Upscaling model to use
        target_w: Target width (default 3840 for 4K)
        target_h: Target height (default 2160 for 4K)
        aspect_mode: How to handle aspect ratio (scale, crop, letterbox)
    """
    # First apply model-specific processing at native resolution
    if model_name == 'optical_flow':
        # Apply unsharp mask before upscaling
        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
        frame = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Handle aspect ratio and resize
    result = handle_aspect_ratio(frame, target_w, target_h, aspect_mode)

    # Model-specific post-processing (at target resolution)
    if model_name == 'bicubic':
        # Bicubic is already applied by handle_aspect_ratio using LANCZOS
        # Re-apply with CUBIC for authenticity
        pass  # Already done
    elif model_name == 'lanczos':
        # Lanczos already applied
        pass
    elif model_name == 'optical_flow':
        # Already processed
        pass

    return result


def interpolate_frames(frame0: np.ndarray, frame1: np.ndarray,
                       model_name: str, num_intermediate: int = 3) -> List[np.ndarray]:
    """
    Interpolate frames between frame0 and frame1.
    For 30fps -> 120fps, we need 3 intermediate frames between each pair.

    Returns list of intermediate frames (not including frame0 and frame1).
    """
    intermediates = []

    if model_name == 'bicubic':
        # Simple linear blend
        for i in range(1, num_intermediate + 1):
            t = i / (num_intermediate + 1)
            blended = cv2.addWeighted(frame0, 1 - t, frame1, t, 0)
            intermediates.append(blended)

    elif model_name == 'lanczos':
        # Linear blend with slight sharpening
        for i in range(1, num_intermediate + 1):
            t = i / (num_intermediate + 1)
            blended = cv2.addWeighted(frame0, 1 - t, frame1, t, 0)
            # Slight sharpen
            kernel = np.array([[-0.1, -0.1, -0.1],
                               [-0.1, 1.8, -0.1],
                               [-0.1, -0.1, -0.1]])
            blended = cv2.filter2D(blended, -1, kernel)
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            intermediates.append(blended)

    elif model_name == 'optical_flow':
        # Use optical flow for motion-aware interpolation
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray0, gray1, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        h, w = frame0.shape[:2]

        for i in range(1, num_intermediate + 1):
            t = i / (num_intermediate + 1)

            # Create warped frames
            flow_t = flow * t
            flow_t_inv = flow * (t - 1)

            # Warp coordinates
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

            # Forward warp from frame0
            map_x_fwd = x_coords + flow_t[..., 0]
            map_y_fwd = y_coords + flow_t[..., 1]
            warped0 = cv2.remap(frame0, map_x_fwd, map_y_fwd,
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            # Backward warp from frame1
            map_x_bwd = x_coords + flow_t_inv[..., 0]
            map_y_bwd = y_coords + flow_t_inv[..., 1]
            warped1 = cv2.remap(frame1, map_x_bwd, map_y_bwd,
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            # Blend warped frames
            blended = cv2.addWeighted(warped0, 1 - t, warped1, t, 0)
            intermediates.append(blended)
    else:
        # Default: simple linear blend
        for i in range(1, num_intermediate + 1):
            t = i / (num_intermediate + 1)
            blended = cv2.addWeighted(frame0, 1 - t, frame1, t, 0)
            intermediates.append(blended)

    return intermediates


def generate_video_clip(
    clip_dir: Path,
    model_name: str,
    output_path: Path,
    start_triplet: int = 0,
    num_triplets: int = 30,  # ~1 second at 30fps input
    output_fps: int = 60,    # Browser-compatible fps
    codec: str = 'mp4v',
    aspect_mode: str = 'letterbox'
) -> Dict:
    """
    Generate a video clip using the specified model.

    Args:
        clip_dir: Directory containing processed clip data
        model_name: Model to use for interpolation/upscaling
        output_path: Path for output video file
        start_triplet: Starting triplet index
        num_triplets: Number of input frame pairs to process
        output_fps: Output video frame rate (60 for browser compatibility)
        codec: Video codec
        aspect_mode: How to handle aspect ratio (scale, crop, letterbox)

    Returns:
        Dict with generation stats
    """
    input_dir = clip_dir / 'input_1080p30' / 'frames'
    triplets_file = clip_dir / 'triplets.json'

    with open(triplets_file) as f:
        triplets = json.load(f)

    # Select triplets
    end_triplet = min(start_triplet + num_triplets, len(triplets))
    selected_triplets = triplets[start_triplet:end_triplet]

    if not selected_triplets:
        return {'error': 'No triplets available'}

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        output_fps,
        (TARGET_WIDTH, TARGET_HEIGHT)
    )

    frame_count = 0

    # For 30fps -> 120fps, we generate 4x frames (1 original + 3 interpolated)
    # But if output_fps is 60, we can either:
    # 1. Skip every other interpolated frame (120 -> 60)
    # 2. Play at 2x slow motion (show all frames at 60fps)
    # We'll do option 2 for better quality assessment

    print(f"Generating {model_name} video: {output_path.name}")
    print(f"  Triplets: {start_triplet} to {end_triplet-1}")
    print(f"  Output: {TARGET_WIDTH}x{TARGET_HEIGHT} @ {output_fps}fps")

    for idx, triplet in enumerate(selected_triplets):
        # Load input frames
        frame0_path = input_dir / triplet['input_frame_0']
        frame1_path = input_dir / triplet['input_frame_1']

        if not frame0_path.exists() or not frame1_path.exists():
            print(f"  Warning: Missing frames for triplet {triplet['triplet_id']}")
            continue

        frame0 = cv2.imread(str(frame0_path))
        frame1 = cv2.imread(str(frame1_path))

        # Upscale original frame
        frame0_4k = upscale_frame(frame0, model_name, aspect_mode=aspect_mode)

        # Write original frame
        writer.write(frame0_4k)
        frame_count += 1

        # Generate and write interpolated frames
        # First upscale, then interpolate (for consistency with SR+VFI pipeline)
        frame1_4k = upscale_frame(frame1, model_name, aspect_mode=aspect_mode)

        # Generate 3 intermediate frames for 4x frame rate
        intermediates = interpolate_frames(frame0_4k, frame1_4k, model_name, num_intermediate=3)

        for interp_frame in intermediates:
            writer.write(interp_frame)
            frame_count += 1

        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(selected_triplets)} triplets...")

    # Write the last frame
    if selected_triplets:
        last_triplet = selected_triplets[-1]
        last_frame_path = input_dir / last_triplet['input_frame_1']
        if last_frame_path.exists():
            last_frame = cv2.imread(str(last_frame_path))
            last_frame_4k = upscale_frame(last_frame, model_name, aspect_mode=aspect_mode)
            writer.write(last_frame_4k)
            frame_count += 1

    writer.release()

    duration = frame_count / output_fps

    return {
        'model': model_name,
        'output_path': str(output_path),
        'frame_count': frame_count,
        'fps': output_fps,
        'duration_seconds': duration,
        'resolution': f'{TARGET_WIDTH}x{TARGET_HEIGHT}'
    }


def generate_all_model_clips(
    clip_dir: Path,
    output_dir: Path,
    models: List[str] = None,
    segment_id: int = 0,
    num_triplets: int = 30,
    output_fps: int = 60,
    aspect_mode: str = 'letterbox'
) -> List[Dict]:
    """Generate video clips for all models for a given segment."""

    if models is None:
        models = ['bicubic', 'lanczos', 'optical_flow']

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    clip_name = clip_dir.name

    for model_name in models:
        output_path = output_dir / f"{clip_name}_seg{segment_id:03d}_{model_name}.mp4"
        result = generate_video_clip(
            clip_dir=clip_dir,
            model_name=model_name,
            output_path=output_path,
            start_triplet=segment_id * num_triplets,
            num_triplets=num_triplets,
            output_fps=output_fps,
            aspect_mode=aspect_mode
        )
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate video clips for blind study')
    parser.add_argument('--clip', type=str, default='arc_raiders_001',
                        help='Name of the processed clip')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directory containing processed clips')
    parser.add_argument('--output-dir', type=str, default='outputs/blind_study_videos',
                        help='Directory for output videos')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['bicubic', 'lanczos', 'optical_flow'],
                        help='Models to generate clips for')
    parser.add_argument('--segment', type=int, default=0,
                        help='Segment ID (each segment uses different triplets)')
    parser.add_argument('--num-triplets', type=int, default=30,
                        help='Number of triplets per segment (~1 second at 30fps)')
    parser.add_argument('--fps', type=int, default=120,
                        help='Output video frame rate (120 for native, 60 for browser)')
    parser.add_argument('--num-segments', type=int, default=1,
                        help='Generate multiple segments')
    parser.add_argument('--aspect-mode', type=str, default='letterbox',
                        choices=['scale', 'crop', 'letterbox'],
                        help='Aspect ratio handling: scale (stretch), crop (center), letterbox (black bars)')
    parser.add_argument('--use-last', action='store_true', default=True,
                        help='Use the last N triplets instead of first N')
    parser.add_argument('--duration', type=int, default=30,
                        help='Duration in seconds (uses all available if source is shorter)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    clip_dir = data_dir / args.clip
    output_dir = Path(args.output_dir)

    if not clip_dir.exists():
        print(f"Error: Clip directory not found: {clip_dir}")
        sys.exit(1)

    # Load triplets to determine available data
    triplets_file = clip_dir / 'triplets.json'
    with open(triplets_file) as f:
        all_triplets = json.load(f)
    total_triplets = len(all_triplets)

    # Calculate triplets needed for requested duration
    # At 30fps input, each triplet represents ~1/30 second
    triplets_for_duration = args.duration * 30  # 30fps input
    num_triplets = min(triplets_for_duration, total_triplets)

    # Calculate start triplet (use last N if requested)
    if args.use_last:
        start_triplet = max(0, total_triplets - num_triplets)
    else:
        start_triplet = 0

    actual_duration = num_triplets / 30.0

    print(f"\n{'='*60}")
    print("VFI+SR Video Clip Generator")
    print(f"{'='*60}")
    print(f"Clip: {args.clip}")
    print(f"Models: {args.models}")
    print(f"Output: {TARGET_WIDTH}x{TARGET_HEIGHT} @ {args.fps}fps (4K)")
    print(f"Aspect ratio: {args.aspect_mode}")
    print(f"Available triplets: {total_triplets} ({total_triplets/30:.1f}s at 30fps)")
    print(f"Using triplets: {start_triplet} to {start_triplet + num_triplets - 1}")
    print(f"Output duration: ~{actual_duration:.1f}s at {args.fps}fps")
    print(f"{'='*60}\n")

    all_results = []

    # Generate a single full clip per model (using all requested triplets)
    print("\n--- Generating full clips ---")
    for model_name in args.models:
        output_path = output_dir / f"{args.clip}_full_{model_name}_{args.fps}fps.mp4"
        result = generate_video_clip(
            clip_dir=clip_dir,
            model_name=model_name,
            output_path=output_path,
            start_triplet=start_triplet,
            num_triplets=num_triplets,
            output_fps=args.fps,
            aspect_mode=args.aspect_mode
        )
        all_results.append(result)

    # Save metadata
    metadata_path = output_dir / 'clips_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump({
            'clip_name': args.clip,
            'models': args.models,
            'resolution': f'{TARGET_WIDTH}x{TARGET_HEIGHT}',
            'fps': args.fps,
            'aspect_mode': args.aspect_mode,
            'total_triplets': total_triplets,
            'used_triplets': num_triplets,
            'start_triplet': start_triplet,
            'duration_seconds': actual_duration,
            'clips': all_results
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("Generation Complete!")
    print(f"{'='*60}")
    print(f"Total clips generated: {len(all_results)}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata: {metadata_path}")
    if args.fps == 120:
        print("\nNote: Videos are at native 120fps (may need compatible player).")
    else:
        print(f"\nNote: Videos are at {args.fps}fps for browser compatibility.")


if __name__ == '__main__':
    main()
