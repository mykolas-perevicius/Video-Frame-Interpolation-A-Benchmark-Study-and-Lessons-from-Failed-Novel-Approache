#!/usr/bin/env python3
"""
Scientific blind study - visible quality differences.
Aggressive degradation: 540p → 4K (4x upscale) to show real differences.
Includes PSNR/SSIM metrics for each model.
"""

import json, subprocess, sys, time, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import cv2
import numpy as np

cv2.setNumThreads(cpu_count())

# Aggressive degradation: 540p input → 4K output (4x upscale)
INPUT_W, INPUT_H = 960, 540  # 540p - makes differences VISIBLE
OUT_W, OUT_H = 3840, 2160    # 4K output
DURATION, FPS = 5.0, 120

def crop16_9(f):
    h, w = f.shape[:2]
    nw = int(h * 16 / 9)
    return f[:, (w - nw) // 2:(w + nw) // 2]

def calc_psnr(img1, img2):
    """Calculate PSNR - returns 100.0 for identical images (JSON-safe)"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0: return 100.0  # Cap at 100dB for JSON compatibility
    return min(100.0, 20 * np.log10(255.0 / np.sqrt(mse)))

def calc_ssim(img1, img2):
    """Simplified SSIM - clamped to [0, 1] for JSON safety"""
    C1, C2 = (0.01 * 255)**2, (0.03 * 255)**2
    img1, img2 = img1.astype(float), img2.astype(float)
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return max(0.0, min(1.0, ssim))  # Clamp to valid range

def cpu_model(args):
    raw, outdir, model, ref_frames = args

    cap = cv2.VideoCapture(raw)
    rfps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, int(rfps / 30))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - int(DURATION * rfps)))

    path = outdir / f"{model}_h264.mp4"
    ff = subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-s',f'{OUT_W}x{OUT_H}',
        '-pix_fmt','bgr24','-r',str(FPS),'-i','-','-c:v','libx264','-preset','ultrafast',
        '-crf','18','-pix_fmt','yuv420p',str(path)], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    ctrl = model == 'control'
    flow = model == 'optical_flow'

    # Upscaling method
    if 'bicubic' in model:
        interp = cv2.INTER_CUBIC
    elif 'lanczos' in model:
        interp = cv2.INTER_LANCZOS4
    else:
        interp = cv2.INTER_LANCZOS4

    cnt, prev, proc, t0 = 0, None, 0, time.time()
    needed = int(DURATION * 30)
    psnr_sum, ssim_sum, metric_cnt = 0.0, 0.0, 0

    while proc < needed:
        ret, fr = cap.read()
        if not ret: break
        if proc % skip != 0: proc += 1; continue

        cr = crop16_9(fr)

        if ctrl:
            # Control: direct high-quality resize (no degradation)
            out = cv2.resize(cr, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
        else:
            # Degrade to 540p then upscale 4x to 4K - THIS IS WHERE QUALITY DIFFERS
            degraded = cv2.resize(cr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
            out = cv2.resize(degraded, (OUT_W, OUT_H), interpolation=interp)

        ff.stdin.write(out.tobytes()); cnt += 1

        # Calculate metrics vs reference
        if ref_frames and proc < len(ref_frames):
            ref = ref_frames[proc]
            # Downsample for faster metric calculation
            out_sm = cv2.resize(out, (480, 270))
            ref_sm = cv2.resize(ref, (480, 270))
            psnr_sum += calc_psnr(out_sm, ref_sm)
            ssim_sum += calc_ssim(cv2.cvtColor(out_sm, cv2.COLOR_BGR2GRAY),
                                   cv2.cvtColor(ref_sm, cv2.COLOR_BGR2GRAY))
            metric_cnt += 1

        # VFI: 3 intermediate frames (30fps → 120fps)
        if prev is not None:
            if flow:
                # Optical flow at output resolution for quality
                g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                g1 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                # Compute flow at lower res for speed, then scale
                g0_sm = cv2.resize(g0, (OUT_W//4, OUT_H//4))
                g1_sm = cv2.resize(g1, (OUT_W//4, OUT_H//4))
                fl_sm = cv2.calcOpticalFlowFarneback(g0_sm, g1_sm, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                fl = cv2.resize(fl_sm, (OUT_W, OUT_H)) * 4  # Scale flow vectors

                h, w = prev.shape[:2]
                y, x = np.mgrid[0:h, 0:w].astype(np.float32)
                for i in range(1, 4):
                    t = i / 4
                    mx, my = x + fl[..., 0] * t, y + fl[..., 1] * t
                    wr = cv2.remap(prev, mx, my, cv2.INTER_LINEAR)
                    ff.stdin.write(cv2.addWeighted(wr, 1-t, out, t, 0).tobytes()); cnt += 1
            else:
                for i in range(1, 4):
                    t = i / 4
                    ff.stdin.write(cv2.addWeighted(prev, 1-t, out, t, 0).tobytes()); cnt += 1

        prev = out.copy(); proc += 1

    ff.stdin.close(); ff.wait(); cap.release()
    el = time.time() - t0

    avg_psnr = psnr_sum / metric_cnt if metric_cnt > 0 else 0
    avg_ssim = ssim_sum / metric_cnt if metric_cnt > 0 else 0

    print(f"[{model}] {cnt} frames, {el:.1f}s | PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

    return {'model': model, 'output_path': str(path), 'frame_count': cnt, 'fps': FPS,
            'duration_seconds': cnt/FPS, 'resolution': f'{OUT_W}x{OUT_H}', 'processing_time': el,
            'vfi_method': 'optical_flow' if flow else 'linear_blend', 'sr_method': model,
            'is_control': ctrl, 'psnr_db': round(avg_psnr, 2), 'ssim': round(avg_ssim, 4),
            'degradation': 'none' if ctrl else f'{INPUT_W}x{INPUT_H} -> {OUT_W}x{OUT_H}'}

def generate_reference_frames(raw):
    """Generate reference frames for metric calculation"""
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
        cr = crop16_9(fr)
        out = cv2.resize(cr, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
        frames.append(out)
        proc += 1
    cap.release()
    return frames

def rife_model(raw, outdir, model, ref_frames):
    """RIFE GPU processing with metrics"""
    import torch
    RIFE = Path(__file__).parent.parent / 'external/Practical-RIFE'
    sys.path.insert(0, str(RIFE))
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = torch.backends.cudnn.benchmark = True
    from train_log.RIFE_HDv3 import Model
    m = Model(); m.load_model(str(RIFE/'train_log'), -1); m.eval(); m.device()

    def t2np(t): return (t.squeeze(0).permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
    def np2t(i): return torch.from_numpy(i.astype(np.float32)/255).permute(2,0,1).unsqueeze(0).cuda()

    cap = cv2.VideoCapture(raw)
    rfps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, int(rfps/30))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - int(DURATION*rfps)))

    path = outdir / f"{model}_h264.mp4"
    interp = cv2.INTER_CUBIC if 'bicubic' in model else cv2.INTER_LANCZOS4

    ff = subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-s',f'{OUT_W}x{OUT_H}',
        '-pix_fmt','bgr24','-r',str(FPS),'-i','-','-c:v','libx264','-preset','fast',
        '-crf','18','-pix_fmt','yuv420p',str(path)], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    cnt, prev, proc, t0 = 0, None, 0, time.time()
    needed = int(DURATION * 30)
    ph, pw = ((OUT_H-1)//32+1)*32, ((OUT_W-1)//32+1)*32
    psnr_sum, ssim_sum, metric_cnt = 0.0, 0.0, 0

    print(f"[{model}] RIFE GPU...")
    while proc < needed:
        ret, fr = cap.read()
        if not ret: break
        if proc % skip != 0: proc += 1; continue

        cr = crop16_9(fr)
        # Degrade to 540p then upscale
        degraded = cv2.resize(cr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        out = cv2.resize(degraded, (OUT_W, OUT_H), interpolation=interp)

        # Metrics
        if ref_frames and proc < len(ref_frames):
            ref = ref_frames[proc]
            out_sm = cv2.resize(out, (480, 270))
            ref_sm = cv2.resize(ref, (480, 270))
            psnr_sum += calc_psnr(out_sm, ref_sm)
            ssim_sum += calc_ssim(cv2.cvtColor(out_sm, cv2.COLOR_BGR2GRAY),
                                   cv2.cvtColor(ref_sm, cv2.COLOR_BGR2GRAY))
            metric_cnt += 1

        if prev is not None:
            p0, p1 = np.zeros((ph,pw,3),np.uint8), np.zeros((ph,pw,3),np.uint8)
            p0[:OUT_H,:OUT_W], p1[:OUT_H,:OUT_W] = prev, out
            i0, i1 = np2t(p0[:,:,::-1]), np2t(p1[:,:,::-1])
            for i in range(1, 4):
                mid = m.inference(i0, i1, timestep=i/4, scale=0.5)
                ff.stdin.write(t2np(mid)[:OUT_H,:OUT_W,::-1].tobytes()); cnt += 1

        ff.stdin.write(out.tobytes()); cnt += 1
        prev = out.copy(); proc += 1

    ff.stdin.close(); ff.wait(); cap.release()
    el = time.time() - t0

    avg_psnr = psnr_sum / metric_cnt if metric_cnt > 0 else 0
    avg_ssim = ssim_sum / metric_cnt if metric_cnt > 0 else 0

    print(f"[{model}] {cnt} frames, {el:.1f}s | PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

    return {'model': model, 'output_path': str(path), 'frame_count': cnt, 'fps': FPS,
            'duration_seconds': cnt/FPS, 'resolution': f'{OUT_W}x{OUT_H}', 'processing_time': el,
            'vfi_method': 'RIFE_v4.25', 'sr_method': 'bicubic' if 'bicubic' in model else 'lanczos',
            'psnr_db': round(avg_psnr, 2), 'ssim': round(avg_ssim, 4),
            'degradation': f'{INPUT_W}x{INPUT_H} -> {OUT_W}x{OUT_H}'}

def main():
    raw = 'data/raw/clip1.mp4'
    outdir = Path('outputs/blind_study_videos')
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.glob('*.mp4'): f.unlink()

    print(f"\n{'='*60}")
    print("SCIENTIFIC BLIND STUDY - Visible Quality Differences")
    print(f"{'='*60}")
    print(f"Degradation: 540p ({INPUT_W}x{INPUT_H}) → 4K ({OUT_W}x{OUT_H}) = 4x upscale")
    print(f"Duration: {DURATION}s @ {FPS}fps")
    print(f"{'='*60}\n")

    t0 = time.time()

    # Generate reference frames first
    print("Generating reference frames for metrics...")
    ref_frames = generate_reference_frames(raw)
    print(f"Reference: {len(ref_frames)} frames\n")

    results = []

    # CPU models - run sequentially to share ref_frames (can't pickle easily)
    print("CPU models...")
    for model in ['control', 'bicubic', 'lanczos', 'optical_flow']:
        r = cpu_model((raw, outdir, model, ref_frames))
        if r: results.append(r)

    # GPU models
    print("\nGPU models...")
    for model in ['rife_bicubic', 'rife_lanczos']:
        r = rife_model(raw, outdir, model, ref_frames)
        if r: results.append(r)

    # Sort by quality (PSNR)
    results_sorted = sorted([r for r in results if not r.get('is_control')],
                           key=lambda x: x.get('psnr_db', 0), reverse=True)

    meta = {
        'models': [r['model'] for r in results],
        'categories': {
            'reference': ['control'],
            'traditional': ['bicubic', 'lanczos'],
            'best_cpu': ['optical_flow'],
            'best_gpu': ['rife_bicubic', 'rife_lanczos']
        },
        'quality_ranking': [r['model'] for r in results_sorted],
        'resolution': f'{OUT_W}x{OUT_H}',
        'input_resolution': f'{INPUT_W}x{INPUT_H}',
        'upscale_factor': '4x',
        'fps': FPS,
        'duration_seconds': DURATION,
        'clips': results
    }

    with open(outdir/'clips_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE! {len(results)} clips in {total:.1f}s")
    print(f"{'='*60}")
    print("\nQuality Ranking (by PSNR):")
    for i, r in enumerate(results_sorted, 1):
        print(f"  {i}. {r['model']}: PSNR={r['psnr_db']:.2f}dB, SSIM={r['ssim']:.4f}")

if __name__ == '__main__': main()
