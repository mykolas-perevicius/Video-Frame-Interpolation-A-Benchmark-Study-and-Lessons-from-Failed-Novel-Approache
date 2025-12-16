#!/usr/bin/env python3
"""All models, 4K output, 5 second clips, parallel CPU + GPU RIFE"""

import json, subprocess, sys, time, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import cv2
import numpy as np

cv2.setNumThreads(cpu_count())
OUT_W, OUT_H = 3840, 2160
DURATION, FPS = 5.0, 120

def crop16_9(f):
    h, w = f.shape[:2]
    nw = int(h * 16 / 9)
    return f[:, (w - nw) // 2:(w + nw) // 2]

def cpu_model(args):
    raw, outdir, model = args
    cap = cv2.VideoCapture(raw)
    rfps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, int(rfps / 30))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - int(DURATION * rfps)))

    path = outdir / f"{model}_h264.mp4"
    ff = subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-s',f'{OUT_W}x{OUT_H}',
        '-pix_fmt','bgr24','-r',str(FPS),'-i','-','-c:v','libx264','-preset','ultrafast',
        '-crf','20','-pix_fmt','yuv420p',str(path)], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    ctrl = model == 'control'
    flow = model == 'optical_flow'
    interp = cv2.INTER_CUBIC if 'bicubic' in model else cv2.INTER_LANCZOS4

    cnt, prev, proc, t0 = 0, None, 0, time.time()
    needed = int(DURATION * 30)

    while proc < needed:
        ret, fr = cap.read()
        if not ret: break
        if proc % skip != 0: proc += 1; continue

        cr = crop16_9(fr)
        if ctrl:
            out = cv2.resize(cr, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
        else:
            sm = cv2.resize(cr, (1920, 1080), interpolation=cv2.INTER_AREA)
            out = cv2.resize(sm, (OUT_W, OUT_H), interpolation=interp)

        ff.stdin.write(out.tobytes()); cnt += 1

        if prev is not None:
            if flow:
                g0, g1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                fl = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
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
    print(f"[{model}] {cnt} frames, {el:.1f}s")
    return {'model': model, 'output_path': str(path), 'frame_count': cnt, 'fps': FPS,
            'duration_seconds': cnt/FPS, 'resolution': f'{OUT_W}x{OUT_H}', 'processing_time': el,
            'vfi_method': 'optical_flow' if flow else 'linear_blend', 'sr_method': model, 'is_control': ctrl}

def rife_model(raw, outdir, model):
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
        '-crf','20','-pix_fmt','yuv420p',str(path)], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    cnt, prev, proc, t0 = 0, None, 0, time.time()
    needed = int(DURATION * 30)
    ph, pw = ((OUT_H-1)//32+1)*32, ((OUT_W-1)//32+1)*32

    print(f"[{model}] RIFE GPU...")
    while proc < needed:
        ret, fr = cap.read()
        if not ret: break
        if proc % skip != 0: proc += 1; continue

        cr = crop16_9(fr)
        sm = cv2.resize(cr, (1920, 1080), interpolation=cv2.INTER_AREA)
        out = cv2.resize(sm, (OUT_W, OUT_H), interpolation=interp)

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
    print(f"[{model}] {cnt} frames, {el:.1f}s")
    return {'model': model, 'output_path': str(path), 'frame_count': cnt, 'fps': FPS,
            'duration_seconds': cnt/FPS, 'resolution': f'{OUT_W}x{OUT_H}', 'processing_time': el,
            'vfi_method': 'RIFE_v4.25', 'sr_method': 'bicubic' if 'bicubic' in model else 'lanczos'}

def main():
    raw = 'data/raw/clip1.mp4'
    outdir = Path('outputs/blind_study_videos')
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.glob('*.mp4'): f.unlink()

    print(f"\n{'='*50}\nALL MODELS - 4K, 5s clips\n{'='*50}\n")
    t0 = time.time()
    results = []

    # CPU parallel
    print("CPU models (parallel)...")
    with ProcessPoolExecutor(max_workers=4) as ex:
        for r in ex.map(cpu_model, [(raw, outdir, m) for m in ['control','bicubic','lanczos','optical_flow']]):
            if r: results.append(r)

    # GPU sequential
    print("\nGPU models...")
    for m in ['rife_bicubic', 'rife_lanczos']:
        r = rife_model(raw, outdir, m)
        if r: results.append(r)

    meta = {'models': [r['model'] for r in results],
            'categories': {'reference': ['control'], 'traditional': ['bicubic','lanczos'],
                          'best_cpu': ['optical_flow'], 'best_gpu': ['rife_bicubic','rife_lanczos']},
            'resolution': f'{OUT_W}x{OUT_H}', 'fps': FPS, 'duration_seconds': DURATION, 'clips': results}

    with open(outdir/'clips_metadata.json', 'w') as f: json.dump(meta, f, indent=2)
    print(f"\n{'='*50}\nDONE! {len(results)} clips in {time.time()-t0:.1f}s\n{'='*50}")

if __name__ == '__main__': main()
