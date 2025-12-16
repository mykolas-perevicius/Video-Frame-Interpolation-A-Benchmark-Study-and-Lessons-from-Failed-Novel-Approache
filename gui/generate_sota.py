#!/usr/bin/env python3
"""
Generate video clips using SOTA models (RIFE for VFI).
GPU-accelerated pipeline: Raw → Center Crop → 1080p → 4K → RIFE VFI → Output
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Add external paths
EXTERNAL_DIR = Path(__file__).parent.parent / 'external'
RIFE_DIR = EXTERNAL_DIR / 'Practical-RIFE'

# Intermediate resolution: 1080p (simulated input)
INPUT_WIDTH = 1920
INPUT_HEIGHT = 1080

# Target output: 4K (3840x2160)
TARGET_WIDTH = 3840
TARGET_HEIGHT = 2160


def center_crop_16_9(frame: np.ndarray) -> np.ndarray:
    """Center crop frame to 16:9 aspect ratio"""
    h, w = frame.shape[:2]
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


class RIFEInterpolator:
    """RIFE-based frame interpolation (GPU accelerated)"""

    def __init__(self, device='cuda', scale=1.0):
        self.device = device
        self.scale = scale
        self.model = None

    def load(self):
        """Load RIFE model"""
        sys.path.insert(0, str(RIFE_DIR))

        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        # Import from train_log directory where model code lives
        from train_log.RIFE_HDv3 import Model
        self.model = Model()
        self.model.load_model(str(RIFE_DIR / 'train_log'), -1)
        self.model.eval()
        self.model.device()
        print(f"RIFE loaded on GPU (scale={self.scale})")

    def _pad_image(self, img, padding=32):
        """Pad image to multiple of padding"""
        h, w = img.shape[:2]
        ph = ((h - 1) // padding + 1) * padding
        pw = ((w - 1) // padding + 1) * padding
        padded = np.zeros((ph, pw, 3), dtype=img.dtype)
        padded[:h, :w] = img
        return padded, (h, w)

    def _to_tensor(self, img):
        """Convert numpy image to torch tensor"""
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img.to(self.device)

    def _to_numpy(self, tensor):
        """Convert torch tensor to numpy image"""
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (img * 255).clip(0, 255).astype(np.uint8)

    def interpolate(self, frame0, frame1, num_intermediate=3):
        """Generate intermediate frames between frame0 and frame1"""
        if self.model is None:
            self.load()

        # Pad to multiple of 32
        frame0_pad, (h, w) = self._pad_image(frame0)
        frame1_pad, _ = self._pad_image(frame1)

        # Convert to tensors (RGB)
        img0 = self._to_tensor(frame0_pad[:, :, ::-1])  # BGR to RGB
        img1 = self._to_tensor(frame1_pad[:, :, ::-1])

        intermediates = []

        for i in range(1, num_intermediate + 1):
            t = i / (num_intermediate + 1)

            # RIFE inference
            mid = self.model.inference(img0, img1, timestep=t, scale=self.scale)

            # Convert back to numpy BGR
            mid_np = self._to_numpy(mid)[:h, :w, ::-1]  # RGB to BGR, remove padding
            intermediates.append(mid_np)

        return intermediates


def upscale_bicubic(frame, scale):
    """Simple bicubic upscaling"""
    h, w = frame.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def upscale_lanczos(frame, scale):
    """Lanczos upscaling"""
    h, w = frame.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def process_video_rife(raw_path: str, output_dir: Path, model: str,
                       duration: float = 10.0, output_fps: int = 120):
    """
    Process video using RIFE for VFI (GPU accelerated).

    Pipeline:
    1. Read raw frame
    2. Center crop to 16:9
    3. Scale to 1080p (1920x1080)
    4. Upscale to 4K (bicubic/lanczos)
    5. RIFE for frame interpolation (30fps → 120fps)
    """
    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {raw_path}")
        return None

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fixed dimensions
    input_w, input_h = INPUT_WIDTH, INPUT_HEIGHT
    out_w, out_h = TARGET_WIDTH, TARGET_HEIGHT

    print(f"Raw: {raw_w}x{raw_h} @ {raw_fps:.0f}fps")
    print(f"Pipeline: {raw_w}x{raw_h} → crop 16:9 → 1080p ({input_w}x{input_h}) → 4K ({out_w}x{out_h}) + RIFE VFI")

    # Calculate frames to process
    input_fps = 30  # Simulated input fps
    frames_needed = int(duration * input_fps)
    frame_skip = max(1, int(raw_fps / input_fps))

    # Start from end of video (last N seconds)
    start_frame = max(0, total_frames - int(duration * raw_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Output path
    output_path = output_dir / f"{model}_h264.mp4"

    # Initialize RIFE interpolator
    rife = RIFEInterpolator(scale=0.5 if out_h > 1080 else 1.0)

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
    prev_upscaled = None
    frames_processed = 0

    print(f"Generating {model} video with RIFE VFI...")
    start_time = time.time()

    while frames_processed < frames_needed:
        ret, raw_frame = cap.read()
        if not ret:
            break

        # Skip frames to simulate 30fps input
        frame_count_raw = frames_processed
        if frame_count_raw % frame_skip != 0:
            frames_processed += 1
            continue

        # 1. Center crop to 16:9
        cropped = center_crop_16_9(raw_frame)

        # 2. Scale to 1080p
        input_frame = cv2.resize(cropped, (input_w, input_h), interpolation=cv2.INTER_AREA)

        # 3. Upscale to 4K
        upscaled = cv2.resize(input_frame, (out_w, out_h),
                              interpolation=cv2.INTER_CUBIC if 'bicubic' in model
                              else cv2.INTER_LANCZOS4)

        # 3. Generate interpolated frames using RIFE (30fps → 120fps = 3 intermediate)
        if prev_upscaled is not None:
            # Write interpolated frames BEFORE current frame
            intermediates = rife.interpolate(prev_upscaled, upscaled, num_intermediate=3)
            for interp in intermediates:
                ffmpeg_proc.stdin.write(interp.tobytes())
                frame_count += 1

        # Write current frame
        ffmpeg_proc.stdin.write(upscaled.tobytes())
        frame_count += 1

        prev_upscaled = upscaled.copy()
        frames_processed += 1

        if frames_processed % 30 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {frames_processed}/{frames_needed} frames ({elapsed:.1f}s)...")

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
        'vfi_method': 'RIFE_v4.25',
        'sr_method': 'bicubic' if 'bicubic' in model else 'lanczos'
    }


def main():
    parser = argparse.ArgumentParser(description='Generate video clips with SOTA models')
    parser.add_argument('--raw', type=str, default='data/raw/arc_raiders.mp4',
                        help='Raw video path')
    parser.add_argument('--output-dir', type=str, default='outputs/blind_study_videos',
                        help='Output directory')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['rife_bicubic', 'rife_lanczos'],
                        help='Models to test')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Duration in seconds')
    parser.add_argument('--fps', type=int, default=120,
                        help='Output fps')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get source info
    cap = cv2.VideoCapture(args.raw)
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    print(f"\n{'='*60}")
    print("SOTA Video Processing Pipeline (RIFE VFI)")
    print(f"{'='*60}")
    print(f"Source: {args.raw} ({raw_w}x{raw_h} @ {raw_fps}fps)")
    print(f"Pipeline: {raw_w}x{raw_h} → crop 16:9 → 1080p ({INPUT_WIDTH}x{INPUT_HEIGHT}) → 4K ({TARGET_WIDTH}x{TARGET_HEIGHT}) + RIFE VFI")
    print(f"Frame rates: {raw_fps}fps → 30fps → {args.fps}fps")
    print(f"Output: {TARGET_WIDTH}x{TARGET_HEIGHT} @ {args.fps}fps")
    print(f"Duration: {args.duration}s")
    print(f"{'='*60}\n")

    results = []
    for model in args.models:
        result = process_video_rife(
            args.raw, output_dir, model,
            duration=args.duration,
            output_fps=args.fps
        )
        if result:
            results.append(result)

    # Save metadata (append to existing if present)
    metadata_path = output_dir / 'clips_metadata.json'
    if metadata_path.exists():
        import json
        with open(metadata_path) as f:
            existing = json.load(f)
        # Append RIFE results to existing clips
        existing['models'].extend(args.models)
        existing['clips'].extend(results)
        with open(metadata_path, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"Updated {metadata_path} with RIFE results")
    else:
        metadata = {
            'clip_name': 'vfisr_4k',
            'models': args.models,
            'resolution': f'{TARGET_WIDTH}x{TARGET_HEIGHT}',
            'fps': args.fps,
            'aspect_mode': '16:9 center crop',
            'duration_seconds': args.duration,
            'pipeline': '1080p_to_4K',
            'description': 'Source → center crop 16:9 → 1080p → upscale to 4K + RIFE VFI',
            'clips': results
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("Done! Generated SOTA clips with RIFE VFI")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
