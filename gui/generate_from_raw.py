#!/usr/bin/env python3
"""
Generate video clips directly from raw ultrawide source.
Pipeline: Raw → Center Crop (16:9) → 1080p → Upscale to 4K

Center crops ultrawide source to 16:9, scales to 1080p, then upscales to 4K.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Intermediate resolution: 1080p (simulated input)
INPUT_WIDTH = 1920
INPUT_HEIGHT = 1080

# Target output: 4K (3840x2160)
TARGET_WIDTH = 3840
TARGET_HEIGHT = 2160


def center_crop_16_9(frame: np.ndarray) -> np.ndarray:
    """Center crop frame to 16:9 aspect ratio"""
    h, w = frame.shape[:2]

    # Calculate 16:9 crop dimensions
    target_ratio = 16 / 9
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Too wide - crop width
        new_w = int(h * target_ratio)
        start_x = (w - new_w) // 2
        return frame[:, start_x:start_x + new_w]
    else:
        # Too tall - crop height
        new_h = int(w / target_ratio)
        start_y = (h - new_h) // 2
        return frame[start_y:start_y + new_h, :]


def upscale_no_stretch(frame: np.ndarray, scale: float, model: str) -> np.ndarray:
    """Upscale frame maintaining aspect ratio (no stretch)"""
    h, w = frame.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)

    if model == 'bicubic':
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif model == 'lanczos':
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    elif model == 'optical_flow':
        # Lanczos + sharpening
        result = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        blurred = cv2.GaussianBlur(result, (0, 0), 3)
        result = cv2.addWeighted(result, 1.5, blurred, -0.5, 0)
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def interpolate_frames(frame0: np.ndarray, frame1: np.ndarray,
                       model: str, num_intermediate: int = 3) -> list:
    """Generate intermediate frames for VFI"""
    intermediates = []

    if model == 'optical_flow':
        # Motion-aware interpolation
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        h, w = frame0.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        for i in range(1, num_intermediate + 1):
            t = i / (num_intermediate + 1)

            # Warp both directions and blend
            flow_t = flow * t
            flow_t_inv = flow * (t - 1)

            map_x_fwd = x_coords + flow_t[..., 0]
            map_y_fwd = y_coords + flow_t[..., 1]
            warped0 = cv2.remap(frame0, map_x_fwd, map_y_fwd, cv2.INTER_LINEAR)

            map_x_bwd = x_coords + flow_t_inv[..., 0]
            map_y_bwd = y_coords + flow_t_inv[..., 1]
            warped1 = cv2.remap(frame1, map_x_bwd, map_y_bwd, cv2.INTER_LINEAR)

            blended = cv2.addWeighted(warped0, 1-t, warped1, t, 0)
            intermediates.append(blended)
    else:
        # Simple linear blend
        for i in range(1, num_intermediate + 1):
            t = i / (num_intermediate + 1)
            blended = cv2.addWeighted(frame0, 1-t, frame1, t, 0)
            intermediates.append(blended)

    return intermediates


def process_raw_video(raw_path: str, output_dir: Path, model: str,
                      duration: float = 10.0, output_fps: int = 120):
    """
    Process raw video: Center Crop → 1080p → Upscale to 4K

    Pipeline:
    1. Read raw frame (e.g., 3840x1080 ultrawide)
    2. Center crop to 16:9
    3. Scale to 1080p (1920x1080) - simulated degraded input
    4. Upscale to 4K (3840x2160)
    """
    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {raw_path}")
        return None

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Intermediate: 1080p
    input_w, input_h = INPUT_WIDTH, INPUT_HEIGHT

    # Output: 4K
    out_w, out_h = TARGET_WIDTH, TARGET_HEIGHT

    print(f"Raw: {raw_w}x{raw_h} @ {raw_fps:.0f}fps")
    print(f"Pipeline: {raw_w}x{raw_h} → crop 16:9 → 1080p ({input_w}x{input_h}) → 4K ({out_w}x{out_h})")

    # Calculate frames to process
    input_fps = 30  # Simulated input fps
    frames_needed = int(duration * input_fps)
    frame_skip = max(1, int(raw_fps / input_fps))

    # Start from end of video (last N seconds)
    start_frame = max(0, total_frames - int(duration * raw_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Output path - pipe to ffmpeg for H.264 encoding
    output_path = output_dir / f"{model}_h264.mp4"

    # Use ffmpeg for H.264 encoding via pipe
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{out_w}x{out_h}',
        '-pix_fmt', 'bgr24',
        '-r', str(output_fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL)

    frame_count = 0
    prev_frame = None
    frames_processed = 0

    print(f"Generating {model} video...")
    start_time = time.time()

    while frames_processed < frames_needed:
        ret, raw_frame = cap.read()
        if not ret:
            break

        # Skip frames to simulate 30fps input
        if frames_processed % frame_skip != 0:
            frames_processed += 1
            continue

        # 1. Center crop to 16:9
        cropped = center_crop_16_9(raw_frame)

        # 2. Scale to 1080p (simulated input)
        input_frame = cv2.resize(cropped, (input_w, input_h), interpolation=cv2.INTER_AREA)

        # 3. Upscale to 4K
        upscaled = cv2.resize(input_frame, (out_w, out_h),
                              interpolation=cv2.INTER_CUBIC if model == 'bicubic'
                              else cv2.INTER_LANCZOS4)

        # Write frame to ffmpeg pipe
        ffmpeg_proc.stdin.write(upscaled.tobytes())
        frame_count += 1

        # Generate interpolated frames (30fps → 120fps = 3 intermediate frames)
        if prev_frame is not None:
            # Process previous frame same way
            prev_crop = center_crop_16_9(prev_frame)
            prev_input = cv2.resize(prev_crop, (input_w, input_h), interpolation=cv2.INTER_AREA)
            prev_up = cv2.resize(prev_input, (out_w, out_h),
                                 interpolation=cv2.INTER_CUBIC if model == 'bicubic'
                                 else cv2.INTER_LANCZOS4)

            # Interpolate between previous and current upscaled frames
            intermediates = interpolate_frames(prev_up, upscaled, model, num_intermediate=3)
            for interp in intermediates:
                ffmpeg_proc.stdin.write(interp.tobytes())
                frame_count += 1

        prev_frame = raw_frame.copy()
        frames_processed += 1

        if frames_processed % 30 == 0:
            print(f"  Processed {frames_processed}/{frames_needed} frames...")

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    cap.release()

    elapsed = time.time() - start_time
    output_duration = frame_count / output_fps
    fps_achieved = frame_count / elapsed if elapsed > 0 else 0

    print(f"  Output: {output_path.name} ({frame_count} frames)")
    print(f"  Time: {elapsed:.1f}s | Speed: {fps_achieved:.1f} fps | Realtime: {output_duration/elapsed:.2f}x")

    return {
        'model': model,
        'output_path': str(output_path),
        'frame_count': frame_count,
        'fps': output_fps,
        'duration_seconds': output_duration,
        'resolution': f'{TARGET_WIDTH}x{TARGET_HEIGHT}',
        'processing_time': elapsed,
        'processing_fps': fps_achieved,
        'realtime_factor': output_duration / elapsed if elapsed > 0 else 0,
        'vfi_method': 'linear_blend',
        'sr_method': model
    }


def process_control_video(raw_path: str, output_dir: Path,
                          duration: float = 10.0, output_fps: int = 120):
    """
    Generate control video - center cropped source upscaled to 4K (no degradation).
    This is the "ground truth" reference.
    """
    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {raw_path}")
        return None

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output is 4K
    out_w, out_h = TARGET_WIDTH, TARGET_HEIGHT

    print(f"Control: {raw_w}x{raw_h} → center crop → 4K {out_w}x{out_h} (no degradation)")

    # Calculate frames to process
    input_fps = 30
    frames_needed = int(duration * input_fps)
    frame_skip = max(1, int(raw_fps / input_fps))

    # Start from end of video
    start_frame = max(0, total_frames - int(duration * raw_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_path = output_dir / "control_h264.mp4"

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{out_w}x{out_h}',
        '-pix_fmt', 'bgr24',
        '-r', str(output_fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL)

    frame_count = 0
    prev_frame = None
    frames_processed = 0

    print("Generating control video (center crop + upscale to 4K)...")
    start_time = time.time()

    while frames_processed < frames_needed:
        ret, raw_frame = cap.read()
        if not ret:
            break

        if frames_processed % frame_skip != 0:
            frames_processed += 1
            continue

        # Center crop to 16:9 and upscale to 4K (no degradation)
        cropped = center_crop_16_9(raw_frame)
        upscaled = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        # Write current frame
        ffmpeg_proc.stdin.write(upscaled.tobytes())
        frame_count += 1

        # Simple VFI (30fps → 120fps)
        if prev_frame is not None:
            prev_crop = center_crop_16_9(prev_frame)
            prev_up = cv2.resize(prev_crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

            for i in range(1, 4):  # 3 intermediate frames
                t = i / 4
                blended = cv2.addWeighted(prev_up, 1-t, upscaled, t, 0)
                ffmpeg_proc.stdin.write(blended.tobytes())
                frame_count += 1

        prev_frame = raw_frame.copy()
        frames_processed += 1

        if frames_processed % 30 == 0:
            print(f"  Control: {frames_processed}/{frames_needed} frames...")

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    cap.release()

    elapsed = time.time() - start_time
    output_duration = frame_count / output_fps

    print(f"  Output: {output_path.name} ({frame_count} frames)")
    print(f"  Time: {elapsed:.1f}s")

    return {
        'model': 'control',
        'output_path': str(output_path),
        'frame_count': frame_count,
        'fps': output_fps,
        'duration_seconds': output_duration,
        'resolution': f'{TARGET_WIDTH}x{TARGET_HEIGHT}',
        'processing_time': elapsed,
        'is_control': True,
        'vfi_method': 'linear_blend',
        'sr_method': 'lanczos (no degrade)'
    }


def main():
    parser = argparse.ArgumentParser(description='Generate from raw ultrawide video')
    parser.add_argument('--raw', type=str, default='data/raw/arc_raiders.mp4',
                        help='Raw video path')
    parser.add_argument('--output-dir', type=str, default='outputs/blind_study_videos',
                        help='Output directory')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['bicubic', 'lanczos'],
                        help='Models to test')
    parser.add_argument('--control', action='store_true',
                        help='Also generate control video (raw reference)')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Duration in seconds')
    parser.add_argument('--fps', type=int, default=120,
                        help='Output fps')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear old files
    for f in output_dir.glob('*.mp4'):
        f.unlink()

    # Get source info
    cap_temp = cv2.VideoCapture(args.raw)
    raw_w = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = int(cap_temp.get(cv2.CAP_PROP_FPS))
    cap_temp.release()

    print(f"\n{'='*60}")
    print("VFI+SR Video Processing Pipeline")
    print(f"{'='*60}")
    print(f"Source: {args.raw} ({raw_w}x{raw_h} @ {raw_fps}fps)")
    print(f"Pipeline: {raw_w}x{raw_h} → crop 16:9 → 1080p ({INPUT_WIDTH}x{INPUT_HEIGHT}) → 4K ({TARGET_WIDTH}x{TARGET_HEIGHT})")
    print(f"Frame rates: {raw_fps}fps → 30fps → {args.fps}fps")
    print(f"Output: {TARGET_WIDTH}x{TARGET_HEIGHT} @ {args.fps}fps")
    print(f"Duration: {args.duration}s")
    print(f"{'='*60}\n")

    results = []

    # Generate control video first if requested
    if args.control:
        print("\n--- Generating Control (Reference) ---")
        control_result = process_control_video(
            args.raw, output_dir,
            duration=args.duration,
            output_fps=args.fps
        )
        if control_result:
            results.append(control_result)

    # Generate test videos
    for model in args.models:
        result = process_raw_video(
            args.raw, output_dir, model,
            duration=args.duration,
            output_fps=args.fps
        )
        if result:
            results.append(result)

    # Save metadata
    all_models = (['control'] if args.control else []) + args.models
    metadata = {
        'clip_name': 'vfisr_4k',
        'models': all_models,
        'resolution': f'{TARGET_WIDTH}x{TARGET_HEIGHT}',
        'fps': args.fps,
        'aspect_mode': '16:9 center crop',
        'duration_seconds': args.duration,
        'pipeline': f'1080p_to_4K',
        'description': 'Source → center crop 16:9 → 1080p → upscale to 4K',
        'clips': results
    }

    with open(output_dir / 'clips_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Generated {len(results)} clips")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
