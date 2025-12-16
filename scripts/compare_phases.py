#!/usr/bin/env python3
"""Compare Phase 1 vs Phase 2 results to validate bug fixes."""

# Phase 1 results (before fixes)
PHASE1 = {
    'control': {'vfi_psnr': 24.06, 'note': 'Bug: Was doing linear blend'},
    'degraded': {'vfi_psnr': 22.28, 'note': 'Baseline'},
    'lanczos_blend': {'vfi_psnr': 24.01, 'note': ''},
    'optical_flow_basic': {'vfi_psnr': 22.28, 'note': 'Bug: 1/4 resolution destroyed quality'},
    'rife_default': {'vfi_psnr': 25.80, 'note': ''},
    'adaptive_aggressive': {'vfi_psnr': 25.82, 'note': ''},
}

import json
from pathlib import Path

def compare():
    results_file = Path(__file__).parent.parent / 'outputs' / 'experiment_results.json'

    if not results_file.exists():
        print("No Phase 2 results yet")
        return

    with open(results_file) as f:
        data = json.load(f)

    phase2 = {e['name']: e.get('vfi_psnr_db', 0) for e in data.get('experiments', [])}

    print(f"\n{'='*70}")
    print("PHASE 1 vs PHASE 2 COMPARISON")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Phase1':>10} {'Phase2':>10} {'Change':>10} {'Note'}")
    print("-" * 70)

    for name, p1 in sorted(PHASE1.items()):
        p2 = phase2.get(name, None)
        if p2 is not None:
            change = p2 - p1['vfi_psnr']
            change_str = f"{'+' if change > 0 else ''}{change:.2f}dB"
            print(f"{name:<25} {p1['vfi_psnr']:>9.2f}dB {p2:>9.2f}dB {change_str:>10} {p1['note']}")
        else:
            print(f"{name:<25} {p1['vfi_psnr']:>9.2f}dB {'N/A':>10} {'':>10} {p1['note']}")

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # Check control fix
    if phase2.get('control', -1) == 0:
        print("✅ Control fix verified: VFI_PSNR = 0 (correctly skipped)")
    else:
        print(f"⚠️ Control: VFI_PSNR = {phase2.get('control', 'N/A')}")

    # Check optical flow fix
    of_old = PHASE1.get('optical_flow_basic', {}).get('vfi_psnr', 0)
    of_new = phase2.get('optical_flow_basic', 0)
    if of_new > of_old:
        print(f"✅ Optical flow fix verified: {of_old:.2f} → {of_new:.2f}dB (+{of_new-of_old:.2f}dB)")
    else:
        print(f"⚠️ Optical flow: {of_old:.2f} → {of_new:.2f}dB")

if __name__ == '__main__':
    compare()
