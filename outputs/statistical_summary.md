# VFI Benchmark Statistical Analysis

Generated: 2025-12-16T03:01:07.825931

## Method Rankings (by VFI PSNR)

| Rank | Method | VFI PSNR | 95% CI | Time (s) | n |
|------|--------|----------|--------|----------|---|
| 1 | adaptive_aggressive | 25.82 dB | [25.82, 25.82] | 66.7 | 1 |
| 2 | rife_default | 25.80 dB | [25.80, 25.80] | 67.9 | 1 |
| 3 | ughi_default | 24.02 dB | [24.02, 24.02] | 378.0 | 1 |
| 4 | lanczos_blend | 24.01 dB | [24.01, 24.01] | 20.7 | 1 |
| 5 | bicubic_blend | 24.00 dB | [24.00, 24.00] | 20.8 | 1 |
| 6 | adaptive_default | 23.92 dB | [23.92, 23.92] | 25.0 | 1 |
| 7 | bicubic_blend_edge | 23.80 dB | [23.80, 23.80] | 19.9 | 1 |
| 8 | lanczos_blend_edge | 23.79 dB | [23.79, 23.79] | 23.4 | 1 |
| 9 | adaptive_conservative | 23.79 dB | [23.79, 23.79] | 21.7 | 1 |
| 10 | mcar_default | 23.79 dB | [23.79, 23.79] | 60.5 | 1 |
| 11 | mcar_aggressive | 23.79 dB | [23.79, 23.79] | 59.2 | 1 |
| 12 | lanczos_blend_sharp | 23.61 dB | [23.61, 23.61] | 22.6 | 1 |
| 13 | uafi_default | 22.71 dB | [22.71, 22.71] | 244.8 | 1 |
| 14 | optical_flow_basic | 22.45 dB | [22.45, 22.45] | 40.9 | 1 |
| 15 | degraded | 22.28 dB | [22.28, 22.28] | 24.3 | 1 |
| 16 | optical_flow_edge | 22.27 dB | [22.27, 22.27] | 61.6 | 1 |
| 17 | optical_flow_sharp | 22.13 dB | [22.13, 22.13] | 75.9 | 1 |

## Statistical Comparisons vs Baseline (degraded)

| Method | PSNR Diff | Cohen's d | Effect | p-value | Sig. |
|--------|-----------|-----------|--------|---------|------|
| adaptive_aggressive | +3.54 dB | N/A | N/A | N/A |  |
| rife_default | +3.52 dB | N/A | N/A | N/A |  |
| ughi_default | +1.74 dB | N/A | N/A | N/A |  |
| lanczos_blend | +1.73 dB | N/A | N/A | N/A |  |
| bicubic_blend | +1.72 dB | N/A | N/A | N/A |  |
| adaptive_default | +1.64 dB | N/A | N/A | N/A |  |
| bicubic_blend_edge | +1.52 dB | N/A | N/A | N/A |  |
| lanczos_blend_edge | +1.51 dB | N/A | N/A | N/A |  |
| adaptive_conservative | +1.51 dB | N/A | N/A | N/A |  |
| mcar_default | +1.51 dB | N/A | N/A | N/A |  |
| mcar_aggressive | +1.51 dB | N/A | N/A | N/A |  |
| lanczos_blend_sharp | +1.33 dB | N/A | N/A | N/A |  |
| uafi_default | +0.43 dB | N/A | N/A | N/A |  |
| optical_flow_basic | +0.17 dB | N/A | N/A | N/A |  |
| optical_flow_edge | +-0.01 dB | N/A | N/A | N/A |  |
| optical_flow_sharp | +-0.15 dB | N/A | N/A | N/A |  |

## Interpretation

- **Cohen's d**: < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large
- **Significance**: * p < 0.05, *** p < 0.01
- **95% CI**: If intervals don't overlap, difference is likely significant

## Notes

- Results may have limited statistical power with small sample sizes
- Run experiments on more intervals for robust conclusions
- Best visual = highest VFI PSNR run (use for demos)