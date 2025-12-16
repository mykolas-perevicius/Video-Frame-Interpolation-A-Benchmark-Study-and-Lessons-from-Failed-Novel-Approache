#!/usr/bin/env python3
"""
FAST video generation - optimized for speed.
Output at 1080p (browser upscales to 4K), 5 second clips.
"""

import json
import subprocess
import sys
import time
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import cv2
import numpy as np

NUM_WORKERS = cpu_count()
cv2.setNumThreads(NUM_WORKERS)

# Output 1080p (fast!) - browser will upscale
OUT_W, OUT_H = 1920, 1080
DURATION = 5.0  # 5 seconds
FPS = 120


def center_crop_16_9(frame):
    h, w = frame.shape[:2]
    new_w = int(h * 16 / 9)
    start_x = (w - new_w) // 2
    return frame[:, start_x:start_x + new_w]


def process_model(args):
    raw_path, output_dir, model = args

    cap = cv2.VideoCapture(raw_path)
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_needed = int(DURATION * 30)  # 30fps input
    frame_skip = max(1, int(raw_fps / 30))
    start_frame = max(0, total_frames - int(DURATION * raw_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_path = output_dir / f"{model}_h264.mp4"

    # Ultra fast encoding
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUT_W}x{OUT_H}', '-pix_fmt', 'bgr24', '-r', str(FPS),
        '-i', '-', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
        '-tune', 'fastdecode', '-pix_fmt', 'yuv420p', str(output_path)
    ]
    ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Select method
    is_control = model == 'control'
    use_flow = model == 'optical_flow'
    interp = cv2.INTER_CUBIC if 'bicubic' in model else cv2.INTER_LANCZOS4

    frame_count = 0
    prev = None
    processed = 0
    start = time.time()

    while processed < frames_needed:
        ret, frame = cap.read()
        if not ret:
            break
        if processed % frame_skip != 0:
            processed += 1
            continue

        # Crop and resize to 1080p
        cropped = center_crop_16_9(frame)

        if is_control:
            # No degradation - direct resize
            out = cv2.resize(cropped, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
        else:
            # Degrade to 540p then upscale
            small = cv2.resize(cropped, (960, 540), interpolation=cv2.INTER_AREA)
            out = cv2.resize(small, (OUT_W, OUT_H), interpolation=interp)

        ffmpeg.stdin.write(out.tobytes())
        frame_count += 1

        # VFI: 3 intermediate frames
        if prev is not None:
            if use_flow:
                # Optical flow interpolation
                gray0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                gray1 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                h, w = prev.shape[:2]
                y, x = np.mgrid[0:h, 0:w].astype(np.float32)

                for i in range(1, 4):
                    t = i / 4
                    map_x = x + flow[..., 0] * t
                    map_y = y + flow[..., 1] * t
                    warped = cv2.remap(prev, map_x, map_y, cv2.INTER_LINEAR)
                    blended = cv2.addWeighted(warped, 1-t, out, t, 0)
                    ffmpeg.stdin.write(blended.tobytes())
                    frame_count += 1
            else:
                # Linear blend
                for i in range(1, 4):
                    t = i / 4
                    blended = cv2.addWeighted(prev, 1-t, out, t, 0)
                    ffmpeg.stdin.write(blended.tobytes())
                    frame_count += 1

        prev = out.copy()
        processed += 1

    ffmpeg.stdin.close()
    ffmpeg.wait()
    cap.release()

    elapsed = time.time() - start
    print(f"[{model}] {frame_count} frames in {elapsed:.1f}s")

    return {
        'model': model,
        'output_path': str(output_path),
        'frame_count': frame_count,
        'fps': FPS,
        'duration_seconds': frame_count / FPS,
        'resolution': f'{OUT_W}x{OUT_H}',
        'processing_time': elapsed,
        'vfi_method': 'optical_flow' if use_flow else 'linear_blend',
        'sr_method': model,
        'is_control': is_control
    }


def process_rife(raw_path, output_dir, model):
    """RIFE GPU processing"""
    import torch

    RIFE_DIR = Path(__file__).parent.parent / 'external' / 'Practical-RIFE'
    sys.path.insert(0, str(RIFE_DIR))

    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    from train_log.RIFE_HDv3 import Model
    rife = Model()
    rife.load_model(str(RIFE_DIR / 'train_log'), -1)
    rife.eval()
    rife.device()

    def to_tensor(img):
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()

    def to_numpy(t):
        return (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    cap = cv2.VideoCapture(raw_path)
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_needed = int(DURATION * 30)
    frame_skip = max(1, int(raw_fps / 30))
    start_frame = max(0, total_frames - int(DURATION * raw_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_path = output_dir / f"{model}_h264.mp4"
    interp = cv2.INTER_CUBIC if 'bicubic' in model else cv2.INTER_LANCZOS4

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUT_W}x{OUT_H}', '-pix_fmt', 'bgr24', '-r', str(FPS),
        '-i', '-', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
        '-pix_fmt', 'yuv420p', str(output_path)
    ]
    ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_count = 0
    prev = None
    processed = 0
    start = time.time()

    print(f"[{model}] RIFE on GPU...")

    while processed < frames_needed:
        ret, frame = cap.read()
        if not ret:
            break
        if processed % frame_skip != 0:
            processed += 1
            continue

        cropped = center_crop_16_9(frame)
        small = cv2.resize(cropped, (960, 540), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (OUT_W, OUT_H), interpolation=interp)

        if prev is not None:
            # Pad to 32
            h, w = OUT_H, OUT_W
            ph = ((h - 1) // 32 + 1) * 32
            pw = ((w - 1) // 32 + 1) * 32

            prev_pad = np.zeros((ph, pw, 3), dtype=np.uint8)
            prev_pad[:h, :w] = prev
            out_pad = np.zeros((ph, pw, 3), dtype=np.uint8)
            out_pad[:h, :w] = out

            img0 = to_tensor(prev_pad[:, :, ::-1])
            img1 = to_tensor(out_pad[:, :, ::-1])

            for i in range(1, 4):
                t = i / 4
                mid = rife.inference(img0, img1, timestep=t, scale=1.0)
                mid_np = to_numpy(mid)[:h, :w, ::-1]
                ffmpeg.stdin.write(mid_np.tobytes())
                frame_count += 1

        ffmpeg.stdin.write(out.tobytes())
        frame_count += 1
        prev = out.copy()
        processed += 1

    ffmpeg.stdin.close()
    ffmpeg.wait()
    cap.release()

    elapsed = time.time() - start
    print(f"[{model}] {frame_count} frames in {elapsed:.1f}s")

    return {
        'model': model,
        'output_path': str(output_path),
        'frame_count': frame_count,
        'fps': FPS,
        'duration_seconds': frame_count / FPS,
        'resolution': f'{OUT_W}x{OUT_H}',
        'processing_time': elapsed,
        'vfi_method': 'RIFE_v4.25',
        'sr_method': 'bicubic' if 'bicubic' in model else 'lanczos'
    }


def main():
    raw_path = 'data/raw/arc_raiders.mp4'
    output_dir = Path('outputs/blind_study_videos')
    output_dir.mkdir(parents=True, exist_ok=True)

    for f in output_dir.glob('*.mp4'):
        f.unlink()

    print(f"\n{'='*50}")
    print("FAST GENERATION (1080p, 5s clips)")
    print(f"{'='*50}\n")

    start_total = time.time()
    results = []

    # CPU models in parallel
    cpu_models = ['control', 'bicubic', 'lanczos', 'optical_flow']
    print("CPU models (parallel)...")

    with ProcessPoolExecutor(max_workers=4) as ex:
        args_list = [(raw_path, output_dir, m) for m in cpu_models]
        for r in ex.map(process_model, args_list):
            if r:
                results.append(r)

    # GPU models
    print("\nGPU models (RIFE)...")
    for model in ['rife_bicubic', 'rife_lanczos']:
        r = process_rife(raw_path, output_dir, model)
        if r:
            results.append(r)

    # Save metadata
    metadata = {
        'models': [r['model'] for r in results],
        'categories': {
            'traditional': ['bicubic', 'lanczos'],
            'best_cpu': ['optical_flow'],
            'best_gpu': ['rife_bicubic', 'rife_lanczos'],
            'reference': ['control']
        },
        'resolution': f'{OUT_W}x{OUT_H}',
        'fps': FPS,
        'duration_seconds': DURATION,
        'clips': results
    }

    with open(output_dir / 'clips_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    total = time.time() - start_total
    print(f"\n{'='*50}")
    print(f"DONE! {len(results)} clips in {total:.1f}s")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
