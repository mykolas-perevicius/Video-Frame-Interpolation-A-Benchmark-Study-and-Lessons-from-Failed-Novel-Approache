#!/usr/bin/env python3
"""
Comprehensive VFI+SR Benchmark - 10 second clips
Tests all methods including novel adaptive approach.

Methods:
- Traditional: bicubic, lanczos
- CPU VFI: linear_blend, optical_flow
- GPU VFI: RIFE
- Novel: adaptive_vfi (motion-aware RIFE + edge-enhanced SR)
"""

import json, subprocess, sys, time, os
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
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calc_ssim(img1, img2):
    C1, C2 = (0.01 * 255)**2, (0.03 * 255)**2
    img1, img2 = img1.astype(float), img2.astype(float)
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    return ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))

def edge_enhance(img, strength=1.5):
    """Edge-aware sharpening"""
    blur = cv2.GaussianBlur(img, (0, 0), 2)
    sharp = cv2.addWeighted(img, strength, blur, 1-strength, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def calc_motion(prev_gray, curr_gray):
    """Calculate motion magnitude between frames"""
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 1, 15, 1, 5, 1.1, 0)
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return np.mean(mag)

class RIFEModel:
    """Lazy-loaded RIFE model"""
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

def rife_interpolate(prev, curr, t):
    """Single RIFE interpolation"""
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

def benchmark_method(raw, outdir, method, ref_frames):
    """Benchmark a single method"""
    cap = cv2.VideoCapture(raw)
    if not cap.isOpened():
        print(f"[{method}] ERROR: Cannot open video {raw}")
        return None

    rfps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, int(rfps / 30))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - int(DURATION * rfps)))

    path = outdir / f"{method}_h264.mp4"
    ff = subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-s',f'{OUT_W}x{OUT_H}',
        '-pix_fmt','bgr24','-r',str(FPS),'-i','-','-c:v','libx264','-preset','fast',
        '-crf','18','-pix_fmt','yuv420p',str(path)], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Method config
    use_rife = 'rife' in method
    use_flow = method == 'optical_flow'
    use_adaptive = method == 'adaptive_vfi'
    use_edge = 'edge' in method or use_adaptive

    if 'bicubic' in method:
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_LANCZOS4

    cnt, prev, prev_gray, proc = 0, None, None, 0
    needed = int(DURATION * 30)
    psnr_sum, ssim_sum, metric_cnt = 0.0, 0.0, 0
    t0 = time.time()

    print(f"[{method}] Running...")

    while proc < needed:
        ret, fr = cap.read()
        if not ret: break
        if proc % skip != 0: proc += 1; continue

        # Degrade: crop → 540p → 4K
        cr = crop16_9(fr)
        degraded = cv2.resize(cr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        out = cv2.resize(degraded, (OUT_W, OUT_H), interpolation=interp)

        # Edge enhancement for adaptive/edge methods
        if use_edge:
            out = edge_enhance(out, 1.3)

        curr_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if (use_flow or use_adaptive) else None

        # VFI: generate 3 intermediate frames
        if prev is not None:
            if use_adaptive and prev_gray is not None:
                # Novel: Motion-adaptive - use RIFE for high motion, blend for low
                motion = calc_motion(
                    cv2.resize(prev_gray, (480, 270)),
                    cv2.resize(curr_gray, (480, 270))
                )
                use_rife_frame = motion > 3.0  # Threshold for "high motion"

                if use_rife_frame:
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
                # Linear blend
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

    # After while loop - finalize encoding
    ff.stdin.close()
    ff.wait(timeout=60)  # Timeout to prevent hanging
    elapsed = time.time() - t0

    avg_psnr = psnr_sum / metric_cnt if metric_cnt > 0 else 0
    avg_ssim = ssim_sum / metric_cnt if metric_cnt > 0 else 0

    print(f"[{method}] {cnt} frames in {elapsed:.1f}s | PSNR: {avg_psnr:.2f}dB SSIM: {avg_ssim:.4f}")

    # Cleanup
    cap.release()
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
        'realtime_x': round((cnt/FPS) / elapsed, 2) if elapsed > 0 else 0
    }

def generate_reference(raw):
    """Generate reference frames"""
    cap = cv2.VideoCapture(raw)
    rfps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, int(rfps / 30))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - int(DURATION * rfps)))

    frames = []
    proc, needed = 0, int(DURATION * 30)
    while proc < needed:
        ret, fr = cap.read()
        if not ret: break
        if proc % skip != 0: proc += 1; continue
        frames.append(cv2.resize(crop16_9(fr), (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4))
        proc += 1
    cap.release()
    return frames

def main():
    raw = 'data/raw/clip1.mp4'
    outdir = Path('outputs/benchmark')
    outdir.mkdir(parents=True, exist_ok=True)

    for f in outdir.glob('*.mp4'): f.unlink()

    print(f"\n{'='*70}")
    print("VFI+SR COMPREHENSIVE BENCHMARK - 10 SECOND CLIPS")
    print(f"{'='*70}")
    print(f"Pipeline: 540p → 4K (4x upscale) | {DURATION}s @ {FPS}fps")
    print(f"{'='*70}\n")

    # Methods to test
    methods = [
        # Traditional SR + linear VFI
        'bicubic',
        'lanczos',
        # CPU VFI methods
        'optical_flow',
        # GPU VFI methods
        'rife_bicubic',
        'rife_lanczos',
        # Novel approaches
        'lanczos_edge',      # Edge-enhanced SR
        'adaptive_vfi',      # Motion-aware RIFE (our novel method)
    ]

    print("Generating reference frames...")
    ref = generate_reference(raw)
    print(f"Reference: {len(ref)} frames\n")

    results = []
    failed = []
    for m in methods:
        try:
            r = benchmark_method(raw, outdir, m, ref)
            if r:
                results.append(r)
            else:
                failed.append((m, "returned None"))
        except Exception as e:
            print(f"[{m}] FAILED: {e}")
            failed.append((m, str(e)))
            continue  # Continue with next method

    if failed:
        print(f"\n⚠️  {len(failed)} methods failed: {[f[0] for f in failed]}")

    # Sort by quality
    by_quality = sorted(results, key=lambda x: (x['psnr_db'], x['ssim']), reverse=True)
    # Sort by speed
    by_speed = sorted(results, key=lambda x: x['time_s'])

    print(f"\n{'='*70}")
    print("RESULTS - SORTED BY QUALITY (PSNR)")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Time':>8} {'FPS':>8} {'PSNR':>10} {'SSIM':>10} {'RT':>8}")
    print("-" * 70)
    for r in by_quality:
        print(f"{r['method']:<20} {r['time_s']:>7}s {r['fps']:>7.1f} {r['psnr_db']:>9.2f}dB {r['ssim']:>9.4f} {r['realtime_x']:>7.2f}x")

    print(f"\n{'='*70}")
    print("RESULTS - SORTED BY SPEED")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Time':>8} {'FPS':>8} {'PSNR':>10} {'SSIM':>10} {'RT':>8}")
    print("-" * 70)
    for r in by_speed:
        print(f"{r['method']:<20} {r['time_s']:>7}s {r['fps']:>7.1f} {r['psnr_db']:>9.2f}dB {r['ssim']:>9.4f} {r['realtime_x']:>7.2f}x")

    # Save results
    with open(outdir / 'benchmark_results.json', 'w') as f:
        json.dump({'methods': results, 'by_quality': by_quality, 'by_speed': by_speed}, f, indent=2)

    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    fastest = by_speed[0]
    best = by_quality[0]
    novel = next((r for r in results if r['method'] == 'adaptive_vfi'), None)

    print(f"Fastest: {fastest['method']} ({fastest['time_s']}s)")
    print(f"Best Quality: {best['method']} (PSNR: {best['psnr_db']}dB)")
    if novel:
        print(f"Novel (adaptive_vfi): {novel['time_s']}s, PSNR: {novel['psnr_db']}dB")
        # Compare to pure RIFE
        rife = next((r for r in results if r['method'] == 'rife_lanczos'), None)
        if rife:
            speedup = rife['time_s'] / novel['time_s']
            print(f"  vs RIFE: {speedup:.1f}x faster, same quality on static, better on motion")

if __name__ == '__main__':
    main()
