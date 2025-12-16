#!/usr/bin/env python3
"""Quick analysis script for VFI+SR experiment results."""

import json
from pathlib import Path

def analyze_results():
    results_file = Path(__file__).parent.parent / 'outputs' / 'experiment_results.json'

    if not results_file.exists():
        print("No results file found. Run experiments first.")
        return

    with open(results_file) as f:
        data = json.load(f)

    experiments = data.get('experiments', [])
    print(f"\n{'='*70}")
    print(f"VFI+SR EXPERIMENT ANALYSIS")
    print(f"{'='*70}")
    print(f"Total experiments: {len(experiments)}\n")

    # Sort by VFI PSNR (skip control which has 0)
    vfi_sorted = sorted(
        [e for e in experiments if e.get('vfi_psnr_db', 0) > 0],
        key=lambda x: x.get('vfi_psnr_db', 0),
        reverse=True
    )

    print("=" * 70)
    print("VFI QUALITY RANKING (sorted by interpolation quality)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Method':<25} {'VFI_PSNR':>10} {'KF_PSNR':>10} {'Time':>8} {'RIFE%':>7}")
    print("-" * 70)

    for i, exp in enumerate(vfi_sorted, 1):
        name = exp['name']
        vfi_psnr = exp.get('vfi_psnr_db', 0)
        kf_psnr = exp.get('keyframe_psnr_db', exp.get('psnr_db', 0))
        time_s = exp.get('time_s', 0)
        rife_pct = exp.get('rife_frames_pct', 0)
        print(f"{i:<5} {name:<25} {vfi_psnr:>9.2f}dB {kf_psnr:>9.2f}dB {time_s:>7.1f}s {rife_pct:>6.0f}%")

    # Find control
    control = [e for e in experiments if e['name'] == 'control']
    if control:
        c = control[0]
        print(f"\n{'Control (reference):':<30} KF_PSNR={c.get('keyframe_psnr_db', 0):.2f}dB, VFI_PSNR={c.get('vfi_psnr_db', 0):.2f}dB (skipped)")

    # Find degraded baseline
    degraded = [e for e in experiments if e['name'] == 'degraded']
    if degraded:
        d = degraded[0]
        print(f"{'Degraded (baseline):':<30} VFI_PSNR={d.get('vfi_psnr_db', 0):.2f}dB")

    # Key comparisons
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if vfi_sorted:
        best = vfi_sorted[0]
        worst = [e for e in vfi_sorted if 'degraded' not in e['name']][-1] if len(vfi_sorted) > 1 else best

        print(f"Best VFI quality:  {best['name']} ({best.get('vfi_psnr_db', 0):.2f}dB)")

        if degraded:
            improvement = best.get('vfi_psnr_db', 0) - degraded[0].get('vfi_psnr_db', 0)
            print(f"Improvement over baseline: +{improvement:.2f}dB")

        # Find linear blend for comparison
        blend = [e for e in experiments if e['name'] == 'lanczos_blend']
        if blend:
            b = blend[0]
            print(f"Linear blend quality: {b.get('vfi_psnr_db', 0):.2f}dB")

        # Find optical flow
        flow = [e for e in experiments if e['name'] == 'optical_flow_basic']
        if flow:
            f = flow[0]
            print(f"Optical flow quality: {f.get('vfi_psnr_db', 0):.2f}dB")
            if degraded:
                flow_improvement = f.get('vfi_psnr_db', 0) - degraded[0].get('vfi_psnr_db', 0)
                print(f"Optical flow vs baseline: {'+' if flow_improvement > 0 else ''}{flow_improvement:.2f}dB")

if __name__ == '__main__':
    analyze_results()
