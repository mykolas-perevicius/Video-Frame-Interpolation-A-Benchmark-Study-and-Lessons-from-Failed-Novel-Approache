# VFI Methods Quick Reference

## Method Categories

### Reference/Baseline
| Method | Description | VFI | SR | Expected Quality |
|--------|-------------|-----|-----|-----------------|
| `control` | No degradation, no VFI | frame_dup | lanczos | 100dB (reference) |
| `degraded` | Frame duplication baseline | frame_dup | bicubic | Worst case |

### Traditional Methods
| Method | VFI Technique | Edge Enhancement | Notes |
|--------|--------------|------------------|-------|
| `lanczos_blend` | Linear alpha blend | None | Simple but effective |
| `lanczos_blend_edge` | Linear blend | 1.3x | Perceptual sharpening |
| `lanczos_blend_sharp` | Linear blend | 1.5x | Strong sharpening |
| `bicubic_blend` | Linear blend | None | Bicubic upscaling |
| `bicubic_blend_edge` | Linear blend | 1.3x | Bicubic + edge |
| `optical_flow_basic` | Farneback flow | None | Motion-compensated |
| `optical_flow_edge` | Farneback flow | 1.3x | Flow + edge |
| `optical_flow_sharp` | Farneback flow | 1.5x | Flow + strong edge |

### Neural Methods
| Method | VFI Technique | RIFE Scale | Notes |
|--------|--------------|------------|-------|
| `rife_fast` | RIFE | 0.25 | Fastest, lower quality |
| `rife_default` | RIFE | 0.5 | Balanced speed/quality |

### Adaptive Methods
| Method | Routing | RIFE Threshold | Notes |
|--------|---------|----------------|-------|
| `adaptive_conservative` | Motion-based | 5.0 | Minimal RIFE usage |
| `adaptive_default` | Motion-based | 3.0 | Balanced routing |
| `adaptive_aggressive` | Motion-based | 1.5 | Maximum RIFE usage |

### Novel Methods (Research Contributions)
| Method | Innovation | Notes |
|--------|-----------|-------|
| `mcar_default` | Motion-Complexity Adaptive Routing | 3-tier: linear/flow/RIFE |
| `mcar_aggressive` | MCAR with lower thresholds | More RIFE usage |
| `uafi_default` | UI-Aware Frame Interpolation | Preserves HUD elements |
| `ughi_default` | Uncertainty-Guided Hybrid | Bidirectional flow consistency |

---

## Metric Definitions

| Metric | Description | Range |
|--------|-------------|-------|
| `keyframe_psnr_db` | PSNR of keyframes vs reference | Higher = better |
| `vfi_psnr_db` | PSNR of t=0.5 interpolated frames vs ground truth | Higher = better |
| `overall_psnr_db` | Weighted: 0.25×KF + 0.75×VFI | Higher = better |
| `rife_frames_pct` | % of frames processed by RIFE | 0-100% |
| `time_s` | Processing time for 589 frames | Lower = faster |

---

## Expected Results Hierarchy

```
Best VFI Quality:
  1. RIFE / adaptive_aggressive (~26dB)
  2. Linear blend (~24dB)
  3. Optical flow (~22-24dB)
  4. Frame duplication (~22dB)

Best Efficiency (quality/time):
  1. Linear blend (fast, decent quality)
  2. Adaptive methods (smart routing)
  3. RIFE (slower but highest quality)
```
