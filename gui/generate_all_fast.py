#!/usr/bin/env python3
"""
Fast parallel video generation using all CPU cores + GPU.

Model Categories:
1. Traditional: bicubic, lanczos (simple interpolation)
2. Best CPU: optical_flow (motion-compensated)
3. Best GPU: RIFE neural VFI
4. Innovative: RIFE + enhanced sharpening

Uses multiprocessing for CPU tasks, GPU for RIFE.
"""

import argparse
import json
import subprocess
import sys
import time
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count

import cv2
import numpy as np

# Use all available cores
NUM_WORKERS = cpu_count()
os.environ['OMP_NUM_THREADS'] = str(NUM_WORKERS)
os.environ['OPENBLAS_NUM_THREADS'] = str(NUM_WORKERS)
cv2.setNumThreads(NUM_WORKERS)

# Intermediate resolution: 1080p
INPUT_WIDTH = 1920
INPUT_HEIGHT = 1080

# Target output: 4K
TARGET_WIDTH = 3840
TARGET_HEIGHT = 2160


def center_crop_16_9(frame: np.ndarray) -> np.ndarray:
    """Center crop frame to 16:9 aspect ratio"""
    h, w = frame.shape[:2]
    target_ratio = 16 / 9
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        start_x = (w - new_w) // 2
        return frame[:, start_x:start_x + new_w]
    else:
        new_h = int(w / target_ratio)
        start_y = (h - new_h) // 2
        return frame[start_y:start_y + new_h, :]


def interpolate_linear(frame0, frame1, num_intermediate=3):
    """Simple linear blend interpolation"""
    intermediates = []
    for i in range(1, num_intermediate + 1):
        t = i / (num_intermediate + 1)
        blended = cv2.addWeighted(frame0, 1-t, frame1, t, 0)
        intermediates.append(blended)
    return intermediates


def interpolate_optical_flow(frame0, frame1, num_intermediate=3):
    """Motion-compensated interpolation using optical flow"""
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    h, w = frame0.shape[:2]
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    intermediates = []
    for i in range(1, num_intermediate + 1):
        t = i / (num_intermediate + 1)

        # Bidirectional warping
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

    return intermediates


def process_video_cpu(raw_path: str, output_dir: Path, model: str,
                      duration: float = 10.0, output_fps: int = 120):
    """Process video using CPU-based methods"""
    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        return None

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    input_w, input_h = INPUT_WIDTH, INPUT_HEIGHT
    out_w, out_h = TARGET_WIDTH, TARGET_HEIGHT

    input_fps = 30
    frames_needed = int(duration * input_fps)
    frame_skip = max(1, int(raw_fps / input_fps))
    start_frame = max(0, total_frames - int(duration * raw_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_path = output_dir / f"{model}_h264.mp4"

    # Select interpolation and upscaling method
    if model == 'bicubic':
        interp_fn = interpolate_linear
        upscale_interp = cv2.INTER_CUBIC
    elif model == 'lanczos':
        interp_fn = interpolate_linear
        upscale_interp = cv2.INTER_LANCZOS4
    elif model == 'optical_flow':
        interp_fn = interpolate_optical_flow
        upscale_interp = cv2.INTER_LANCZOS4
    else:
        interp_fn = interpolate_linear
        upscale_interp = cv2.INTER_LINEAR

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{out_w}x{out_h}', '-pix_fmt', 'bgr24', '-r', str(output_fps),
        '-i', '-', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
        '-threads', str(NUM_WORKERS), '-pix_fmt', 'yuv420p', str(output_path)
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL)

    frame_count = 0
    prev_frame = None
    frames_processed = 0
    start_time = time.time()

    print(f"[{model}] Starting... ({NUM_WORKERS} threads)")

    while frames_processed < frames_needed:
        ret, raw_frame = cap.read()
        if not ret:
            break

        if frames_processed % frame_skip != 0:
            frames_processed += 1
            continue

        # Pipeline: crop → 1080p → 4K
        cropped = center_crop_16_9(raw_frame)
        input_frame = cv2.resize(cropped, (input_w, input_h), interpolation=cv2.INTER_AREA)
        upscaled = cv2.resize(input_frame, (out_w, out_h), interpolation=upscale_interp)

        ffmpeg_proc.stdin.write(upscaled.tobytes())
        frame_count += 1

        # VFI: 30fps → 120fps
        if prev_frame is not None:
            prev_crop = center_crop_16_9(prev_frame)
            prev_input = cv2.resize(prev_crop, (input_w, input_h), interpolation=cv2.INTER_AREA)
            prev_up = cv2.resize(prev_input, (out_w, out_h), interpolation=upscale_interp)

            intermediates = interp_fn(prev_up, upscaled, num_intermediate=3)
            for interp in intermediates:
                ffmpeg_proc.stdin.write(interp.tobytes())
                frame_count += 1

        prev_frame = raw_frame.copy()
        frames_processed += 1

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    cap.release()

    elapsed = time.time() - start_time
    output_duration = frame_count / output_fps
    fps_achieved = frame_count / elapsed if elapsed > 0 else 0

    print(f"[{model}] Done: {frame_count} frames in {elapsed:.1f}s ({fps_achieved:.1f} fps)")

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
        'vfi_method': 'optical_flow' if model == 'optical_flow' else 'linear_blend',
        'sr_method': model
    }


def process_control(raw_path: str, output_dir: Path,
                    duration: float = 10.0, output_fps: int = 120):
    """Generate control video (no degradation)"""
    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        return None

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w, out_h = TARGET_WIDTH, TARGET_HEIGHT

    input_fps = 30
    frames_needed = int(duration * input_fps)
    frame_skip = max(1, int(raw_fps / input_fps))
    start_frame = max(0, total_frames - int(duration * raw_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_path = output_dir / "control_h264.mp4"

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{out_w}x{out_h}', '-pix_fmt', 'bgr24', '-r', str(output_fps),
        '-i', '-', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
        '-threads', str(NUM_WORKERS), '-pix_fmt', 'yuv420p', str(output_path)
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL)

    frame_count = 0
    prev_frame = None
    frames_processed = 0
    start_time = time.time()

    print(f"[control] Starting...")

    while frames_processed < frames_needed:
        ret, raw_frame = cap.read()
        if not ret:
            break

        if frames_processed % frame_skip != 0:
            frames_processed += 1
            continue

        cropped = center_crop_16_9(raw_frame)
        upscaled = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        ffmpeg_proc.stdin.write(upscaled.tobytes())
        frame_count += 1

        if prev_frame is not None:
            prev_crop = center_crop_16_9(prev_frame)
            prev_up = cv2.resize(prev_crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

            for i in range(1, 4):
                t = i / 4
                blended = cv2.addWeighted(prev_up, 1-t, upscaled, t, 0)
                ffmpeg_proc.stdin.write(blended.tobytes())
                frame_count += 1

        prev_frame = raw_frame.copy()
        frames_processed += 1

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    cap.release()

    elapsed = time.time() - start_time
    print(f"[control] Done: {frame_count} frames in {elapsed:.1f}s")

    return {
        'model': 'control',
        'output_path': str(output_path),
        'frame_count': frame_count,
        'fps': output_fps,
        'duration_seconds': frame_count / output_fps,
        'resolution': f'{TARGET_WIDTH}x{TARGET_HEIGHT}',
        'processing_time': elapsed,
        'is_control': True,
        'vfi_method': 'linear_blend',
        'sr_method': 'lanczos (no degrade)'
    }


def process_rife(raw_path: str, output_dir: Path, model: str,
                 duration: float = 10.0, output_fps: int = 120):
    """Process video using RIFE GPU VFI"""
    import torch

    # Setup RIFE
    EXTERNAL_DIR = Path(__file__).parent.parent / 'external'
    RIFE_DIR = EXTERNAL_DIR / 'Practical-RIFE'
    sys.path.insert(0, str(RIFE_DIR))

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    from train_log.RIFE_HDv3 import Model
    rife_model = Model()
    rife_model.load_model(str(RIFE_DIR / 'train_log'), -1)
    rife_model.eval()
    rife_model.device()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[{model}] RIFE loaded on {device}")

    def to_tensor(img):
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img.to(device)

    def to_numpy(tensor):
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (img * 255).clip(0, 255).astype(np.uint8)

    def pad_image(img, padding=32):
        h, w = img.shape[:2]
        ph = ((h - 1) // padding + 1) * padding
        pw = ((w - 1) // padding + 1) * padding
        padded = np.zeros((ph, pw, 3), dtype=img.dtype)
        padded[:h, :w] = img
        return padded, (h, w)

    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        return None

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    input_w, input_h = INPUT_WIDTH, INPUT_HEIGHT
    out_w, out_h = TARGET_WIDTH, TARGET_HEIGHT

    upscale_interp = cv2.INTER_CUBIC if 'bicubic' in model else cv2.INTER_LANCZOS4
    apply_sharpen = 'enhanced' in model

    input_fps = 30
    frames_needed = int(duration * input_fps)
    frame_skip = max(1, int(raw_fps / input_fps))
    start_frame = max(0, total_frames - int(duration * raw_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_path = output_dir / f"{model}_h264.mp4"

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{out_w}x{out_h}', '-pix_fmt', 'bgr24', '-r', str(output_fps),
        '-i', '-', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-threads', str(NUM_WORKERS), '-pix_fmt', 'yuv420p', str(output_path)
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL)

    frame_count = 0
    prev_upscaled = None
    frames_processed = 0
    start_time = time.time()

    while frames_processed < frames_needed:
        ret, raw_frame = cap.read()
        if not ret:
            break

        if frames_processed % frame_skip != 0:
            frames_processed += 1
            continue

        # Pipeline: crop → 1080p → 4K
        cropped = center_crop_16_9(raw_frame)
        input_frame = cv2.resize(cropped, (input_w, input_h), interpolation=cv2.INTER_AREA)
        upscaled = cv2.resize(input_frame, (out_w, out_h), interpolation=upscale_interp)

        # Apply sharpening for enhanced version
        if apply_sharpen:
            blurred = cv2.GaussianBlur(upscaled, (0, 0), 2)
            upscaled = cv2.addWeighted(upscaled, 1.3, blurred, -0.3, 0)
            upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)

        # RIFE interpolation
        if prev_upscaled is not None:
            # Pad frames
            frame0_pad, (h, w) = pad_image(prev_upscaled)
            frame1_pad, _ = pad_image(upscaled)

            # Convert BGR to RGB and to tensor
            img0 = to_tensor(frame0_pad[:, :, ::-1])
            img1 = to_tensor(frame1_pad[:, :, ::-1])

            # Generate 3 intermediate frames
            for i in range(1, 4):
                t = i / 4
                with torch.no_grad():
                    mid = rife_model.inference(img0, img1, timestep=t, scale=0.5)
                mid_np = to_numpy(mid)[:h, :w, ::-1]  # RGB to BGR
                ffmpeg_proc.stdin.write(mid_np.tobytes())
                frame_count += 1

        ffmpeg_proc.stdin.write(upscaled.tobytes())
        frame_count += 1
        prev_upscaled = upscaled.copy()
        frames_processed += 1

        if frames_processed % 30 == 0:
            elapsed = time.time() - start_time
            print(f"[{model}] {frames_processed}/{frames_needed} ({elapsed:.1f}s)")

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    cap.release()

    elapsed = time.time() - start_time
    output_duration = frame_count / output_fps
    fps_achieved = frame_count / elapsed if elapsed > 0 else 0

    print(f"[{model}] Done: {frame_count} frames in {elapsed:.1f}s ({fps_achieved:.1f} fps)")

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
        'vfi_method': 'RIFE_v4.25',
        'sr_method': ('lanczos+sharpen' if apply_sharpen else
                     ('bicubic' if 'bicubic' in model else 'lanczos'))
    }


def main():
    parser = argparse.ArgumentParser(description='Fast parallel video generation')
    parser.add_argument('--raw', type=str, default='data/raw/arc_raiders.mp4')
    parser.add_argument('--output-dir', type=str, default='outputs/blind_study_videos')
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--fps', type=int, default=120)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear old files
    for f in output_dir.glob('*.mp4'):
        f.unlink()

    print(f"\n{'='*60}")
    print("FAST PARALLEL VIDEO GENERATION")
    print(f"{'='*60}")
    print(f"CPU cores: {NUM_WORKERS}")
    print(f"GPU: RTX 3090 (CUDA)")
    print(f"Pipeline: Source → crop 16:9 → 1080p → 4K")
    print(f"Duration: {args.duration}s @ {args.fps}fps")
    print(f"{'='*60}\n")

    results = []

    # PHASE 1: CPU tasks in parallel (control + traditional + optical_flow)
    print("PHASE 1: CPU models (parallel)")
    print("-" * 40)

    cpu_models = ['bicubic', 'lanczos', 'optical_flow']

    with ProcessPoolExecutor(max_workers=3) as executor:
        # Submit control
        control_future = executor.submit(
            process_control, args.raw, output_dir, args.duration, args.fps
        )

        # Submit CPU models
        cpu_futures = {
            executor.submit(process_video_cpu, args.raw, output_dir, m,
                          args.duration, args.fps): m
            for m in cpu_models
        }

        # Collect control result
        control_result = control_future.result()
        if control_result:
            results.append(control_result)

        # Collect CPU results
        for future in cpu_futures:
            result = future.result()
            if result:
                results.append(result)

    # PHASE 2: GPU tasks (RIFE models) - sequential on GPU
    print("\nPHASE 2: GPU models (RIFE)")
    print("-" * 40)

    gpu_models = ['rife_bicubic', 'rife_lanczos', 'rife_enhanced']

    for model in gpu_models:
        result = process_rife(args.raw, output_dir, model, args.duration, args.fps)
        if result:
            results.append(result)

    # Save metadata
    all_models = ['control'] + cpu_models + gpu_models

    # Categorize models
    categories = {
        'traditional': ['bicubic', 'lanczos'],
        'best_cpu': ['optical_flow'],
        'best_gpu': ['rife_bicubic', 'rife_lanczos'],
        'innovative': ['rife_enhanced']
    }

    metadata = {
        'clip_name': 'vfisr_4k_complete',
        'models': all_models,
        'categories': categories,
        'resolution': f'{TARGET_WIDTH}x{TARGET_HEIGHT}',
        'fps': args.fps,
        'aspect_mode': '16:9 center crop',
        'duration_seconds': args.duration,
        'pipeline': '1080p_to_4K',
        'description': 'Source → center crop 16:9 → 1080p → upscale to 4K',
        'clips': results
    }

    with open(output_dir / 'clips_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE! Generated {len(results)} clips")
    print(f"{'='*60}")

    # Summary
    total_time = sum(r['processing_time'] for r in results)
    print(f"\nTotal processing time: {total_time:.1f}s")
    print("\nModels by category:")
    for cat, models in categories.items():
        print(f"  {cat}: {', '.join(models)}")


if __name__ == '__main__':
    main()
