#!/usr/bin/env python3
"""
Proper VFI+SR Benchmark - 10 second clips with audio

Requirements:
- Control video (no degradation) FIRST
- Random 10-second interval from full video
- Order: control → least processed → most processed → innovative
- Include audio track
"""

import json, subprocess, sys, time, os, random
from pathlib import Path
from multiprocessing import cpu_count
import cv2
import numpy as np

cv2.setNumThreads(cpu_count())

# 540p → 4K (4x upscale)
INPUT_W, INPUT_H = 960, 540
OUT_W, OUT_H = 3840, 2160
DURATION = 10.0
FPS = 120

def crop16_9(f):
    h, w = f.shape[:2]
    nw = int(h * 16 / 9)
    return f[:, (w - nw) // 2:(w + nw) // 2]

def calc_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0: return 100.0
    return min(100.0, 20 * np.log10(255.0 / np.sqrt(mse)))

def calc_ssim(img1, img2):
    C1, C2 = (0.01 * 255)**2, (0.03 * 255)**2
    img1, img2 = img1.astype(float), img2.astype(float)
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return max(0.0, min(1.0, ssim))

def edge_enhance(img, strength=1.3):
    blur = cv2.GaussianBlur(img, (0, 0), 2)
    sharp = cv2.addWeighted(img, strength, blur, 1-strength, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def calc_motion(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 1, 15, 1, 5, 1.1, 0)
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return np.mean(mag)

class RIFEModel:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            import torch
            RIFE = Path(__file__).parent.parent / 'external/Practical-RIFE'
            sys.path.insert(0, str(RIFE))
            torch.set_grad_enabled(False)
            torch.backends.cudnn.enabled = torch.backends.cudnn.benchmark = True
            from train_log.RIFE_HDv3 import Model
            m = Model()
            m.load_model(str(RIFE/'train_log'), -1)
            m.eval()
            m.device()
            cls._instance = m
            print("[RIFE] Model loaded on GPU")
        return cls._instance

    @classmethod
    def unload(cls):
        """Explicitly unload RIFE model from GPU memory"""
        if cls._instance is not None:
            import torch
            import gc
            del cls._instance
            cls._instance = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            time.sleep(1)  # Let GPU memory settle
            print("[RIFE] Model unloaded from GPU")

def rife_interpolate(prev, curr, t):
    import torch
    m = RIFEModel.get()
    h, w = prev.shape[:2]
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    p0, p1 = np.zeros((ph, pw, 3), np.uint8), np.zeros((ph, pw, 3), np.uint8)
    p0[:h, :w], p1[:h, :w] = prev, curr
    i0 = torch.from_numpy(p0[:,:,::-1].astype(np.float32)/255).permute(2,0,1).unsqueeze(0).cuda()
    i1 = torch.from_numpy(p1[:,:,::-1].astype(np.float32)/255).permute(2,0,1).unsqueeze(0).cuda()
    mid = m.inference(i0, i1, timestep=t, scale=0.5)
    return (mid.squeeze(0).permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)[:h,:w,::-1]

def get_random_start(raw, duration):
    """Get random start time for a clip"""
    cap = cv2.VideoCapture(raw)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / fps
    cap.release()

    # Leave buffer at start and end
    max_start = max(0, total_duration - duration - 2)
    start_time = random.uniform(2, max_start) if max_start > 2 else 0
    return start_time

def extract_audio(raw, outdir, start_time, duration):
    """Extract audio for the clip interval"""
    audio_path = outdir / 'audio_segment.aac'
    cmd = [
        'ffmpeg', '-y', '-ss', str(start_time), '-i', raw,
        '-t', str(duration), '-vn', '-acodec', 'copy', str(audio_path)
    ]
    subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    return audio_path if audio_path.exists() else None

def add_audio_to_video(video_path, audio_path, output_path):
    """Mux audio into video"""
    if audio_path and audio_path.exists():
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path), '-i', str(audio_path),
            '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(output_path)
        ]
        subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        video_path.unlink()  # Remove video without audio
        return output_path
    return video_path

def extract_clip(full_video, raw_video, start_time, duration, output):
    """Extract a segment from pre-generated video with matching audio (fast)"""
    # Extract video segment
    temp_video = output.parent / f"{output.stem}_temp.mp4"
    cmd = [
        'ffmpeg', '-y', '-ss', str(start_time), '-i', str(full_video),
        '-t', str(duration), '-c', 'copy', str(temp_video)
    ]
    subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    # Extract audio from raw source
    audio_path = output.parent / f"{output.stem}_audio.aac"
    audio_cmd = [
        'ffmpeg', '-y', '-ss', str(start_time), '-i', str(raw_video),
        '-t', str(duration), '-vn', '-acodec', 'copy', str(audio_path)
    ]
    subprocess.run(audio_cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    # Mux them together
    if audio_path.exists():
        add_audio_to_video(temp_video, audio_path, output)
        if audio_path.exists():
            audio_path.unlink()
    else:
        temp_video.rename(output)

    return output.exists()


def get_full_video_duration(video_path):
    """Get duration of a video file in seconds"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps > 0 else 0

def benchmark_method(raw, outdir, method, start_frame, ref_frames, audio_path):
    """Benchmark a single method"""
    cap = cv2.VideoCapture(raw)
    if not cap.isOpened():
        print(f"[{method}] ERROR: Cannot open video")
        return None

    rfps = cap.get(cv2.CAP_PROP_FPS)
    skip = max(1, round(rfps / 30))  # round() not int() to handle ~60fps correctly
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Temp path without audio
    temp_path = outdir / f"{method}_temp.mp4"
    final_path = outdir / f"{method}.mp4"

    ff = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-s', f'{OUT_W}x{OUT_H}',
        '-pix_fmt', 'bgr24', '-r', str(FPS), '-i', '-',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '15', '-pix_fmt', 'yuv420p',
        str(temp_path)
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Method config
    is_control = method == 'control'
    is_degraded = method == 'degraded'  # Degraded baseline: no VFI, just frame duplication
    use_rife = 'rife' in method
    use_flow = method == 'optical_flow'
    use_adaptive = method == 'adaptive_vfi'
    use_edge = 'edge' in method or use_adaptive
    interp = cv2.INTER_CUBIC if ('bicubic' in method or is_degraded) else cv2.INTER_LANCZOS4

    cnt, prev, prev_gray, proc = 0, None, None, 0
    needed = int(DURATION * rfps)  # Total source frames to read (not base frames)
    psnr_sum, ssim_sum, metric_cnt = 0.0, 0.0, 0
    t0 = time.time()

    print(f"[{method}] Processing...")

    while proc < needed:
        ret, fr = cap.read()
        if not ret: break
        if proc % skip != 0:
            proc += 1
            continue

        cr = crop16_9(fr)

        if is_control:
            # Control: direct high-quality resize (NO degradation)
            out = cv2.resize(cr, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
        else:
            # Degrade: crop → 540p → 4K
            degraded = cv2.resize(cr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
            out = cv2.resize(degraded, (OUT_W, OUT_H), interpolation=interp)
            if use_edge:
                out = edge_enhance(out, 1.3)

        curr_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if (use_flow or use_adaptive) else None

        # VFI: generate 3 intermediate frames
        if prev is not None:
            if is_degraded:
                # Degraded: NO VFI - just duplicate previous frame 3 times (shows judder)
                for i in range(3):
                    ff.stdin.write(prev.tobytes()); cnt += 1
            elif use_adaptive and prev_gray is not None:
                motion = calc_motion(
                    cv2.resize(prev_gray, (480, 270)),
                    cv2.resize(curr_gray, (480, 270))
                )
                if motion > 3.0:
                    for i in range(1, 4):
                        mid = rife_interpolate(prev, out, i/4)
                        ff.stdin.write(mid.tobytes()); cnt += 1
                else:
                    for i in range(1, 4):
                        t = i / 4
                        ff.stdin.write(cv2.addWeighted(prev, 1-t, out, t, 0).tobytes()); cnt += 1
            elif use_rife:
                for i in range(1, 4):
                    mid = rife_interpolate(prev, out, i/4)
                    ff.stdin.write(mid.tobytes()); cnt += 1
            elif use_flow:
                g0_sm = cv2.resize(prev_gray, (OUT_W//4, OUT_H//4))
                g1_sm = cv2.resize(curr_gray, (OUT_W//4, OUT_H//4))
                fl = cv2.resize(cv2.calcOpticalFlowFarneback(g0_sm, g1_sm, None, 0.5, 3, 15, 3, 5, 1.2, 0),
                               (OUT_W, OUT_H)) * 4
                h, w = prev.shape[:2]
                y, x = np.mgrid[0:h, 0:w].astype(np.float32)
                for i in range(1, 4):
                    t = i / 4
                    wr = cv2.remap(prev, x + fl[...,0]*t, y + fl[...,1]*t, cv2.INTER_LINEAR)
                    ff.stdin.write(cv2.addWeighted(wr, 1-t, out, t, 0).tobytes()); cnt += 1
            else:
                for i in range(1, 4):
                    t = i / 4
                    ff.stdin.write(cv2.addWeighted(prev, 1-t, out, t, 0).tobytes()); cnt += 1

        ff.stdin.write(out.tobytes()); cnt += 1

        # Metrics vs reference
        if ref_frames and proc < len(ref_frames):
            out_sm = cv2.resize(out, (480, 270))
            ref_sm = cv2.resize(ref_frames[proc], (480, 270))
            psnr_sum += calc_psnr(out_sm, ref_sm)
            ssim_sum += calc_ssim(cv2.cvtColor(out_sm, cv2.COLOR_BGR2GRAY),
                                   cv2.cvtColor(ref_sm, cv2.COLOR_BGR2GRAY))
            metric_cnt += 1

        prev, prev_gray = out.copy(), curr_gray
        proc += 1

    ff.stdin.close()
    ff.wait(timeout=60)
    elapsed = time.time() - t0
    cap.release()

    # Add audio (if provided) or just rename temp to final
    if audio_path:
        add_audio_to_video(temp_path, audio_path, final_path)
    # else: temp file stays as-is, will be processed later

    avg_psnr = psnr_sum / metric_cnt if metric_cnt > 0 else 0
    avg_ssim = ssim_sum / metric_cnt if metric_cnt > 0 else 0

    print(f"[{method}] {cnt} frames in {elapsed:.1f}s | PSNR: {avg_psnr:.2f}dB SSIM: {avg_ssim:.4f}")

    # Cleanup GPU
    if 'rife' in method or 'adaptive' in method:
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass

    return {
        'method': method,
        'frames': cnt,
        'time_s': round(elapsed, 1),
        'fps': round(cnt / elapsed, 1) if elapsed > 0 else 0,
        'psnr_db': round(avg_psnr, 2),
        'ssim': round(avg_ssim, 4),
        'realtime_x': round((cnt/FPS) / elapsed, 2) if elapsed > 0 else 0,
        'is_control': is_control,
        'is_degraded': is_degraded
    }

def generate_reference(raw, start_frame):
    """Generate reference frames"""
    cap = cv2.VideoCapture(raw)
    rfps = cap.get(cv2.CAP_PROP_FPS)
    skip = max(1, round(rfps / 30))  # round() not int() to handle ~60fps correctly
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    proc, needed = 0, int(DURATION * rfps)  # Total source frames to read
    while proc < needed:
        ret, fr = cap.read()
        if not ret: break
        if proc % skip != 0:
            proc += 1
            continue
        frames.append(cv2.resize(crop16_9(fr), (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4))
        proc += 1
    cap.release()
    return frames

def main():
    raw = 'data/raw/clip1.mp4'
    outdir = Path('outputs/benchmark')
    outdir.mkdir(parents=True, exist_ok=True)

    # Clean old files
    for f in outdir.glob('*.mp4'):
        f.unlink()
    for f in outdir.glob('*.aac'):
        f.unlink()

    print(f"\n{'='*70}")
    print("VFI+SR BENCHMARK - 10 SECOND CLIPS WITH AUDIO")
    print(f"{'='*70}")
    print(f"Pipeline: 540p → 4K (4x upscale) | {DURATION}s @ {FPS}fps")
    print(f"{'='*70}\n")

    # Get random start time
    start_time = get_random_start(raw, DURATION)
    cap = cv2.VideoCapture(raw)
    rfps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * rfps)
    cap.release()

    print(f"Random interval: {start_time:.1f}s - {start_time + DURATION:.1f}s")

    # Generate reference frames
    print("Generating reference frames...")
    ref = generate_reference(raw, start_frame)
    print(f"Reference: {len(ref)} frames\n")

    # Methods: control, degraded baseline, best traditional, best SOTA, innovative
    methods = [
        'control',          # Reference (no degradation) - SOURCE
        'degraded',         # Degraded baseline: 540p → 4K, NO VFI (frame duplication)
        'lanczos',          # Best traditional SR with linear VFI
        'rife_lanczos',     # Best SOTA (GPU VFI + lanczos SR)
        'adaptive_vfi',     # INNOVATIVE: Motion-aware (always last)
    ]

    # Generate videos WITHOUT audio first
    results = []
    for m in methods:
        try:
            r = benchmark_method(raw, outdir, m, start_frame, ref, None)  # No audio yet
            if r:
                results.append(r)
        except Exception as e:
            print(f"[{m}] FAILED: {e}")

    # Now extract audio based on ACTUAL video duration and mux into each video
    if results:
        actual_duration = results[0]['frames'] / FPS  # All videos have same frame count
        print(f"\nAdding audio (actual duration: {actual_duration:.2f}s)...")
        audio_path = extract_audio(raw, outdir, start_time, actual_duration)

        for r in results:
            temp_path = outdir / f"{r['method']}_temp.mp4"
            final_path = outdir / f"{r['method']}.mp4"
            if temp_path.exists():
                add_audio_to_video(temp_path, audio_path, final_path)
                print(f"  [{r['method']}] Audio added")

        # Clean up audio file
        if audio_path and audio_path.exists():
            audio_path.unlink()

    # Save results with proper ordering
    meta = {
        'models': [r['method'] for r in results],
        'resolution': f'{OUT_W}x{OUT_H}',
        'input_resolution': f'{INPUT_W}x{INPUT_H}',
        'fps': FPS,
        'duration_seconds': DURATION,
        'start_time': round(start_time, 1),
        'clips': []
    }

    for r in results:
        # Calculate file size and bitrate
        video_path = outdir / f"{r['method']}.mp4"
        file_size_bytes = video_path.stat().st_size if video_path.exists() else 0
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
        duration = r['frames'] / FPS
        bitrate_mbps = round((file_size_bytes * 8) / (duration * 1_000_000), 2) if duration > 0 else 0

        meta['clips'].append({
            'model': r['method'],
            'output_path': f"outputs/benchmark/{r['method']}.mp4",
            'frame_count': r['frames'],
            'fps': FPS,
            'duration_seconds': round(duration, 2),
            'resolution': f'{OUT_W}x{OUT_H}',
            'processing_time': r['time_s'],
            'file_size_mb': file_size_mb,
            'bitrate_mbps': bitrate_mbps,
            'vfi_method': 'none' if r['method'] == 'control' else
                         'frame_dup' if r['method'] == 'degraded' else
                         'RIFE' if 'rife' in r['method'] else
                         'adaptive' if r['method'] == 'adaptive_vfi' else
                         'optical_flow' if r['method'] == 'optical_flow' else 'linear_blend',
            'sr_method': r['method'],
            'is_control': r.get('is_control', False),
            'is_degraded': r.get('is_degraded', False),
            'psnr_db': r['psnr_db'],
            'ssim': r['ssim']
        })

    with open(outdir / 'clips_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Time':>8} {'PSNR':>10} {'SSIM':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['method']:<20} {r['time_s']:>7}s {r['psnr_db']:>9.2f}dB {r['ssim']:>9.4f}")

    print(f"\nVideos saved to: {outdir}")
    print(f"Interval: {start_time:.1f}s - {start_time + DURATION:.1f}s")

if __name__ == '__main__':
    main()
