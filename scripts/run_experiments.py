#!/usr/bin/env python3
"""
Automated Experiment Runner for VFI+SR Research

Runs ALL experiments with different parameter combinations.
Includes PSNR, SSIM, and LPIPS perceptual metrics.

Usage:
    python scripts/run_experiments.py                            # Run all experiments
    python scripts/run_experiments.py --clip ID --intervals all  # Use cached intervals
    python scripts/run_experiments.py --resume                   # Resume from checkpoint
    python scripts/run_experiments.py --experiment NAME          # Run single experiment
"""

import argparse
import gc
import json
import random
import signal
import sys
import time
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

import cv2
import numpy as np
import torch

cv2.setNumThreads(cpu_count())

# LPIPS perceptual metric (lazy loaded)
_lpips_model = None
_lpips_device = None

def get_lpips_model():
    """Lazy-load LPIPS model for perceptual quality measurement."""
    global _lpips_model, _lpips_device
    if _lpips_model is None:
        import lpips
        _lpips_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _lpips_model = lpips.LPIPS(net='alex', verbose=False).to(_lpips_device)
        _lpips_model.eval()
    return _lpips_model, _lpips_device

# ============================================================
# PRE-COMPUTED INTERVAL SUPPORT
# ============================================================
CLIPS_DIR = Path(__file__).parent.parent / 'data' / 'clips'


def load_interval(clip_id: str, interval_id: str):
    """Load pre-computed interval frames from cache.

    Args:
        clip_id: Clip identifier (e.g., 'arc_raiders_001')
        interval_id: Interval identifier (e.g., 'interval_0001')

    Returns:
        dict with: keyframes, midpoints, meta, motion_stats
        or None if interval not found
    """
    interval_dir = CLIPS_DIR / clip_id / 'intervals' / interval_id

    if not interval_dir.exists():
        return None

    meta_file = interval_dir / 'meta.json'
    if not meta_file.exists():
        return None

    with open(meta_file) as f:
        meta = json.load(f)

    # Load keyframes
    keyframes_dir = interval_dir / 'keyframes'
    keyframes = []
    for kf_file in sorted(keyframes_dir.glob('kf_*.png')):
        frame = cv2.imread(str(kf_file))
        if frame is not None:
            keyframes.append(frame)

    # Load midpoints (ground truth)
    midpoints_dir = interval_dir / 'midpoints'
    midpoints = []
    for gt_file in sorted(midpoints_dir.glob('gt_*.png')):
        frame = cv2.imread(str(gt_file))
        if frame is not None:
            midpoints.append(frame)

    # Load motion stats
    flow_stats_file = interval_dir / 'motion' / 'flow_stats.json'
    if flow_stats_file.exists():
        with open(flow_stats_file) as f:
            motion_stats = json.load(f)
    else:
        motion_stats = {}

    return {
        'keyframes': keyframes,
        'midpoints': midpoints,
        'meta': meta,
        'motion_stats': motion_stats,
        'clip_id': clip_id,
        'interval_id': interval_id
    }


def list_available_intervals(clip_id: str):
    """List all available intervals for a clip."""
    intervals_dir = CLIPS_DIR / clip_id / 'intervals'
    if not intervals_dir.exists():
        return []

    intervals = []
    for interval_path in sorted(intervals_dir.glob('interval_*')):
        meta_file = interval_path / 'meta.json'
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            intervals.append(meta)
    return intervals


# ============================================================
# GRACEFUL SHUTDOWN HANDLING
# ============================================================
shutdown_requested = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully - finish current experiment then exit."""
    global shutdown_requested
    if shutdown_requested:
        print("\n[!!] Force quit - exiting immediately!")
        sys.exit(1)
    print("\n[!] Shutdown requested - will exit after current experiment...")
    print("[!] Press Ctrl+C again to force quit (may lose current experiment)")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_completed_experiments(results):
    """Get set of (name, interval_idx) tuples for completed experiments."""
    return {(e['name'], e.get('interval_idx', 0)) for e in results.get('experiments', [])}


def check_gpu_memory(min_free_mb=1500):
    """Check if enough GPU memory is available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            free_mb = free / 1024 / 1024
            return free_mb > min_free_mb, free_mb
    except Exception:
        pass
    return True, 0  # Assume OK if can't check

# Configuration (defaults - can be overridden by quality level)
INPUT_W, INPUT_H = 960, 540
DURATION = 5.0  # Reduced from 10s for faster experiments

# Quality levels - each experiment will run at each quality level
# to test if users notice the difference
QUALITY_LEVELS = {
    'high': {'fps': 120, 'resolution': (3840, 2160), 'label': '4K@120'},
    'medium': {'fps': 90, 'resolution': (2560, 1440), 'label': '1440p@90'},
    'low': {'fps': 60, 'resolution': (1920, 1080), 'label': '1080p@60'},
}

# Default quality for quick tests
DEFAULT_QUALITY = 'medium'
OUT_W, OUT_H = QUALITY_LEVELS[DEFAULT_QUALITY]['resolution']
FPS = QUALITY_LEVELS[DEFAULT_QUALITY]['fps']

# Experiment definitions - GROUPED BY INTENSITY
# LIGHT = CPU only, safe to run together
# HEAVY = Uses RIFE/GPU, run one at a time

# Core methods to test (reduced set for quality level testing)
CORE_LIGHT = [
    {'name': 'control', 'vfi': 'none', 'sr': 'lanczos', 'edge': 0, 'description': 'Reference (no degradation)'},
    {'name': 'degraded', 'vfi': 'frame_dup', 'sr': 'bicubic', 'edge': 0, 'description': 'Worst case baseline'},
    {'name': 'lanczos_blend_edge', 'vfi': 'linear_blend', 'sr': 'lanczos', 'edge': 1.3, 'description': 'Blend + edge enhance'},
    {'name': 'optical_flow_edge', 'vfi': 'optical_flow', 'sr': 'lanczos', 'edge': 1.3, 'description': 'Optical flow + edge'},
]

CORE_HEAVY = [
    {'name': 'rife_default', 'vfi': 'rife', 'sr': 'lanczos', 'edge': 1.3, 'rife_scale': 0.5, 'description': 'RIFE default'},
    {'name': 'adaptive_default', 'vfi': 'adaptive', 'sr': 'lanczos', 'edge': 1.3, 'motion_thresh': 3.0, 'description': 'Adaptive VFI'},
]

# Full experiment sets (when not testing quality levels)
LIGHT_EXPERIMENTS = [
    {'name': 'control', 'vfi': 'none', 'sr': 'lanczos', 'edge': 0, 'motion_thresh': 0, 'description': 'Reference (no degradation)'},
    {'name': 'degraded', 'vfi': 'frame_dup', 'sr': 'bicubic', 'edge': 0, 'motion_thresh': 0, 'description': 'Worst case baseline'},
    {'name': 'lanczos_blend', 'vfi': 'linear_blend', 'sr': 'lanczos', 'edge': 0, 'motion_thresh': 0, 'description': 'Simple interpolation'},
    {'name': 'lanczos_blend_edge', 'vfi': 'linear_blend', 'sr': 'lanczos', 'edge': 1.3, 'motion_thresh': 0, 'description': 'Blend + edge enhance'},
    {'name': 'lanczos_blend_sharp', 'vfi': 'linear_blend', 'sr': 'lanczos', 'edge': 1.5, 'motion_thresh': 0, 'description': 'Blend + strong sharpen'},
    {'name': 'optical_flow_basic', 'vfi': 'optical_flow', 'sr': 'lanczos', 'edge': 0, 'description': 'Optical flow VFI'},
    {'name': 'optical_flow_edge', 'vfi': 'optical_flow', 'sr': 'lanczos', 'edge': 1.3, 'description': 'Optical flow + edge'},
    {'name': 'optical_flow_sharp', 'vfi': 'optical_flow', 'sr': 'lanczos', 'edge': 1.5, 'description': 'Optical flow + strong sharpen'},
    {'name': 'bicubic_blend', 'vfi': 'linear_blend', 'sr': 'bicubic', 'edge': 0, 'description': 'Bicubic SR + blend'},
    {'name': 'bicubic_blend_edge', 'vfi': 'linear_blend', 'sr': 'bicubic', 'edge': 1.3, 'description': 'Bicubic + edge'},
    # INNOVATIVE METHODS (Research Contributions)
    {'name': 'uafi_default', 'vfi': 'ui_aware', 'sr': 'lanczos', 'edge': 1.3, 'description': 'UI-Aware Frame Interpolation'},
    {'name': 'ughi_default', 'vfi': 'ughi', 'sr': 'lanczos', 'edge': 1.3, 'description': 'Uncertainty-Guided Hybrid'},
]

HEAVY_EXPERIMENTS = [
    {'name': 'rife_fast', 'vfi': 'rife', 'sr': 'lanczos', 'edge': 1.3, 'rife_scale': 0.25, 'description': 'RIFE fast (scale 0.25)'},
    {'name': 'rife_default', 'vfi': 'rife', 'sr': 'lanczos', 'edge': 1.3, 'rife_scale': 0.5, 'description': 'RIFE default'},
    {'name': 'adaptive_conservative', 'vfi': 'adaptive', 'sr': 'lanczos', 'edge': 1.3, 'motion_thresh': 5.0, 'description': 'Adaptive (conservative)'},
    {'name': 'adaptive_default', 'vfi': 'adaptive', 'sr': 'lanczos', 'edge': 1.3, 'motion_thresh': 3.0, 'description': 'Adaptive (default)'},
    {'name': 'adaptive_aggressive', 'vfi': 'adaptive', 'sr': 'lanczos', 'edge': 1.3, 'motion_thresh': 1.5, 'description': 'Adaptive (aggressive)'},
    # INNOVATIVE: Motion-Complexity Adaptive Routing (uses RIFE for hard cases)
    {'name': 'mcar_default', 'vfi': 'mcar', 'sr': 'lanczos', 'edge': 1.3, 'mcar_low': 0.25, 'mcar_high': 0.6, 'description': 'Motion-Complexity Adaptive Routing'},
    {'name': 'mcar_aggressive', 'vfi': 'mcar', 'sr': 'lanczos', 'edge': 1.3, 'mcar_low': 0.15, 'mcar_high': 0.4, 'description': 'MCAR (more RIFE usage)'},
]

# Combined for backwards compatibility
EXPERIMENTS = LIGHT_EXPERIMENTS + HEAVY_EXPERIMENTS


def generate_quality_experiments(base_experiments, quality_levels=None):
    """Generate experiment variants for each quality level.

    Returns experiments with quality_level field added.
    Total experiments = len(base_experiments) * len(quality_levels)
    """
    if quality_levels is None:
        quality_levels = list(QUALITY_LEVELS.keys())

    all_experiments = []
    for quality in quality_levels:
        q = QUALITY_LEVELS[quality]
        for exp in base_experiments:
            new_exp = exp.copy()
            new_exp['quality_level'] = quality
            new_exp['quality_fps'] = q['fps']
            new_exp['quality_resolution'] = q['resolution']
            new_exp['name'] = f"{exp['name']}_{q['label']}"
            new_exp['description'] = f"{exp.get('description', '')} @ {q['label']}"
            all_experiments.append(new_exp)

    return all_experiments


class RIFEModel:
    """Singleton RIFE model loader."""
    _instance = None

    @classmethod
    def get(cls, scale=0.5):
        if cls._instance is None:
            import torch
            RIFE = Path(__file__).parent.parent / 'external/Practical-RIFE'
            sys.path.insert(0, str(RIFE))
            torch.set_grad_enabled(False)
            torch.backends.cudnn.enabled = torch.backends.cudnn.benchmark = True
            from train_log.RIFE_HDv3 import Model
            m = Model()
            m.load_model(str(RIFE / 'train_log'), -1)
            m.eval()
            m.device()
            cls._instance = m
            print("[RIFE] Model loaded on GPU")
        return cls._instance


def crop16_9(f):
    h, w = f.shape[:2]
    nw = int(h * 16 / 9)
    return f[:, (w - nw) // 2:(w + nw) // 2]


def get_source_properties(video_path):
    """Get source video's native resolution and FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps


def calc_psnr(img1, img2):
    """Calculate PSNR between two images.

    Returns float('inf') for identical images, otherwise the PSNR in dB.
    Note: Values above ~60dB typically indicate near-identical images.
    """
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')  # Identical images
    return 20 * np.log10(255.0 / np.sqrt(mse))  # No artificial cap


def calc_ssim(img1, img2):
    C1, C2 = (0.01 * 255)**2, (0.03 * 255)**2
    img1, img2 = img1.astype(float), img2.astype(float)
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return max(0.0, min(1.0, ssim))


def calc_lpips(img1, img2):
    """Calculate LPIPS perceptual distance between two images.

    Lower is better (0 = identical, 1 = very different).
    Uses AlexNet backbone for efficiency.
    """
    try:
        model, device = get_lpips_model()

        # Convert BGR uint8 to RGB float tensor [-1, 1]
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1

        # Convert to NCHW tensor
        t1 = torch.from_numpy(img1_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        t2 = torch.from_numpy(img2_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            dist = model(t1, t2)

        return float(dist.item())
    except Exception as e:
        print(f"  [WARN] LPIPS calculation failed: {e}")
        return None


def safe_round(value, decimals=2):
    """Round a value safely, handling infinity and NaN for JSON serialization."""
    import math
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return None
    return round(value, decimals)


def edge_enhance(img, strength=1.3):
    if strength <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), 2)
    sharp = cv2.addWeighted(img, strength, blur, 1-strength, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def calc_motion(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 1, 15, 1, 5, 1.1, 0)
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return np.mean(mag)


def rife_interpolate(prev, curr, t, scale=0.5):
    """RIFE frame interpolation with proper padding for any resolution.

    RIFE's internal architecture requires dimensions to be multiples of 128
    when using scale=0.5 (the scale affects internal downsampling).
    """
    import torch
    m = RIFEModel.get()
    h, w = prev.shape[:2]
    # Pad to multiples of 128 for RIFE compatibility at all scales
    # 1440 -> 1536, 1080 -> 1152, 2160 -> 2176
    ph = ((h - 1) // 128 + 1) * 128
    pw = ((w - 1) // 128 + 1) * 128
    # Use reflection padding instead of black padding to avoid edge artifacts
    pad_h = ph - h
    pad_w = pw - w
    p0 = cv2.copyMakeBorder(prev, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    p1 = cv2.copyMakeBorder(curr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    i0 = torch.from_numpy(p0[:, :, ::-1].astype(np.float32)/255).permute(2, 0, 1).unsqueeze(0).cuda()
    i1 = torch.from_numpy(p1[:, :, ::-1].astype(np.float32)/255).permute(2, 0, 1).unsqueeze(0).cuda()
    mid = m.inference(i0, i1, timestep=t, scale=scale)
    return (mid.squeeze(0).permute(1, 2, 0).cpu().numpy()*255).clip(0, 255).astype(np.uint8)[:h, :w, ::-1]


# ============================================================
# INNOVATIVE VFI METHODS (Research Contributions)
# ============================================================

def detect_ui_mask(frame_0, frame_1, flow):
    """Detect UI regions via temporal inconsistency.

    UI regions have high reconstruction error but low motion.
    This is a gaming-specific innovation - no prior work handles UI in VFI.
    """
    h, w = frame_0.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Warp frame_0 to middle position
    warped = cv2.remap(frame_0, x + flow[..., 0] * 0.5, y + flow[..., 1] * 0.5, cv2.INTER_LINEAR)

    # High reconstruction error = potential UI
    recon_error = np.abs(warped.astype(float) - frame_1.astype(float)).mean(axis=2)

    # Low motion = static element
    flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    # UI mask: high error AND low motion
    ui_mask = (recon_error > 30) & (flow_mag < 2)

    # Clean up with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    ui_mask = cv2.morphologyEx(ui_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    ui_mask = cv2.morphologyEx(ui_mask, cv2.MORPH_OPEN, kernel)

    return ui_mask


def ui_aware_interpolate(prev, curr, t):
    """UI-Aware Frame Interpolation (UAFI).

    Innovation: First method to explicitly handle UI overlays in VFI,
    eliminating text ghosting and HUD warping artifacts.
    """
    g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Detect UI regions
    ui_mask = detect_ui_mask(prev, curr, flow)

    # Interpolate gameplay regions using optical flow
    h, w = prev.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = cv2.remap(prev, x + flow[..., 0] * t, y + flow[..., 1] * t, cv2.INTER_LINEAR)
    blended = cv2.addWeighted(warped, 1-t, curr, t, 0)

    # Composite: use interpolated for gameplay, source for UI
    ui_source = prev if t < 0.5 else curr
    ui_mask_3ch = np.stack([ui_mask] * 3, axis=2)
    result = np.where(ui_mask_3ch, ui_source, blended)

    return result.astype(np.uint8)


def estimate_motion_complexity(prev_gray, curr_gray):
    """Estimate motion complexity score 0-1 for adaptive routing.

    FIX: Adjusted normalization factors to be more sensitive to motion.
    With gaming footage averaging ~13px flow, we want:
    - <5px (easy): ~0.2 complexity
    - 5-15px (medium): ~0.3-0.5 complexity
    - 15-30px (hard): ~0.5-0.8 complexity
    - >30px (extreme): ~0.8+ complexity
    """
    # Optical flow magnitude
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 1, 15, 1, 5, 1.1, 0)
    flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    # Complexity factors (adjusted for gaming footage):
    # 1. Mean motion - now normalized to 20px instead of 50px
    mean_motion = min(1.0, np.mean(flow_mag) / 20.0)

    # 2. Motion variance - normalized to 15px instead of 25px
    motion_var = min(1.0, np.std(flow_mag) / 15.0)

    # 3. Edge density (more edges = more detail to preserve)
    edges = cv2.Canny(curr_gray, 100, 200)
    edge_density = np.mean(edges > 0) * 2.0  # Boost edge contribution

    # Combined score - more weight on motion
    complexity = min(1.0, 0.5 * mean_motion + 0.3 * motion_var + 0.2 * edge_density)
    return complexity, flow


def mcar_interpolate(prev, curr, t, rife_scale=0.5, threshold_low=0.3, threshold_high=0.7):
    """Motion-Complexity Adaptive Routing (MCAR).

    Innovation: Routes frames to different interpolation tiers based on complexity.
    - Tier 1 (Easy): Simple linear blend (~2ms)
    - Tier 2 (Medium): Flow-based warp (~15ms)
    - Tier 3 (Hard): RIFE neural network (~50ms)

    Returns: (interpolated_frame, tier_used, complexity_score)
    """
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    complexity, flow = estimate_motion_complexity(prev_gray, curr_gray)

    if complexity < threshold_low:
        # Tier 1: EASY - Simple linear blend (fastest)
        result = cv2.addWeighted(prev, 1-t, curr, t, 0)
        return result, 'linear', complexity
    elif complexity < threshold_high:
        # Tier 2: MEDIUM - Flow-based warp
        h, w = prev.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        warped = cv2.remap(prev, x + flow[..., 0] * t, y + flow[..., 1] * t, cv2.INTER_LINEAR)
        result = cv2.addWeighted(warped, 1-t, curr, t, 0)
        return result, 'flow', complexity
    else:
        # Tier 3: HARD - Use RIFE for complex motion
        result = rife_interpolate(prev, curr, t, rife_scale)
        return result, 'rife', complexity


def mcar_interpolate_cached(prev, curr, t, complexity, flow, rife_scale=0.5, threshold_low=0.3, threshold_high=0.7):
    """MCAR with precomputed complexity and flow.

    PERFORMANCE FIX: Use this when interpolating multiple t values per frame pair.
    Avoids recomputing optical flow for each t value (3x speedup).

    Returns: (interpolated_frame, tier_used)
    """
    if complexity < threshold_low:
        # Tier 1: EASY - Simple linear blend (fastest)
        result = cv2.addWeighted(prev, 1-t, curr, t, 0)
        return result, 'linear'
    elif complexity < threshold_high:
        # Tier 2: MEDIUM - Flow-based warp
        h, w = prev.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        warped = cv2.remap(prev, x + flow[..., 0] * t, y + flow[..., 1] * t, cv2.INTER_LINEAR)
        result = cv2.addWeighted(warped, 1-t, curr, t, 0)
        return result, 'flow'
    else:
        # Tier 3: HARD - Use RIFE for complex motion
        result = rife_interpolate(prev, curr, t, rife_scale)
        return result, 'rife'


def estimate_uncertainty(prev, curr, flow_fwd, flow_bwd):
    """Estimate per-pixel interpolation uncertainty via flow consistency."""
    h, w = prev.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Warp backward flow using forward flow
    flow_bwd_warped = cv2.remap(flow_bwd, x + flow_fwd[..., 0], y + flow_fwd[..., 1], cv2.INTER_LINEAR)

    # Inconsistency = forward + backward should sum to ~0
    consistency_error = np.sqrt((flow_fwd[..., 0] + flow_bwd_warped[..., 0])**2 +
                                 (flow_fwd[..., 1] + flow_bwd_warped[..., 1])**2)

    # Normalize to 0-1 uncertainty
    uncertainty = np.clip(consistency_error / 10.0, 0, 1)

    # Also high uncertainty for large motion
    flow_mag = np.sqrt(flow_fwd[..., 0]**2 + flow_fwd[..., 1]**2)
    motion_uncertainty = np.clip(flow_mag / 50.0, 0, 1)

    # Combined uncertainty
    combined = np.maximum(uncertainty, motion_uncertainty * 0.5)
    return combined


def ughi_interpolate(prev, curr, t):
    """Uncertainty-Guided Hybrid Interpolation (UGHI).

    Innovation: Uses bidirectional flow consistency to estimate uncertainty,
    applying smoother blending for uncertain regions while keeping sharp
    flow-based results for certain regions.

    Returns: (interpolated_frame, mean_uncertainty)
    """
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # Compute bidirectional flow
    flow_fwd = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_bwd = cv2.calcOpticalFlowFarneback(curr_gray, prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Estimate uncertainty
    uncertainty = estimate_uncertainty(prev, curr, flow_fwd, flow_bwd)

    # Base interpolation (bidirectional flow-based)
    h, w = prev.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    warped_fwd = cv2.remap(prev, x + flow_fwd[..., 0] * t, y + flow_fwd[..., 1] * t, cv2.INTER_LINEAR)
    warped_bwd = cv2.remap(curr, x - flow_bwd[..., 0] * (1-t), y - flow_bwd[..., 1] * (1-t), cv2.INTER_LINEAR)

    # Blend based on uncertainty
    base_interp = cv2.addWeighted(warped_fwd, 1-t, warped_bwd, t, 0)
    simple_blend = cv2.addWeighted(prev, 1-t, curr, t, 0)

    # Use base_interp for certain regions, simple blend for uncertain
    uncertainty_3ch = np.stack([uncertainty] * 3, axis=2)
    result = base_interp * (1 - uncertainty_3ch) + simple_blend * uncertainty_3ch

    return result.astype(np.uint8), uncertainty.mean()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_random_start(raw, duration):
    """Get random start time for a clip."""
    cap = cv2.VideoCapture(raw)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / fps
    cap.release()
    max_start = max(0, total_duration - duration - 2)
    return random.uniform(2, max_start) if max_start > 2 else 0


def generate_reference(raw, start_frame, out_w=None, out_h=None, extract_midpoints=False):
    """Generate reference frames for quality comparison.

    Uses source native resolution (via OUT_W, OUT_H which are set dynamically).
    This ensures we compare at the source's native quality for scientific accuracy.

    Args:
        raw: Path to source video
        start_frame: Frame number to start from
        out_w, out_h: Output dimensions (defaults to global OUT_W, OUT_H)
        extract_midpoints: If True, also extract ground truth midpoint frames
                          (the "skipped" frames) for VFI quality evaluation.

    Returns:
        If extract_midpoints=False: List of keyframes (current behavior)
        If extract_midpoints=True: Tuple of (keyframes, midpoint_frames)
            - keyframes: Even-indexed frames (0, 2, 4, ...) used as VFI input
            - midpoints: Odd-indexed frames (1, 3, 5, ...) - ground truth for t=0.5 interpolation
    """
    # Default to current output resolution (set to source native at runtime)
    if out_w is None:
        out_w = OUT_W
    if out_h is None:
        out_h = OUT_H

    cap = cv2.VideoCapture(raw)
    rfps = cap.get(cv2.CAP_PROP_FPS)
    skip = max(1, round(rfps / 30))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    keyframes = []
    midpoints = [] if extract_midpoints else None
    proc, needed = 0, int(DURATION * rfps)

    while proc < needed:
        ret, fr = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(crop16_9(fr), (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        if proc % skip == 0:
            # Even indices (0, 2, 4, ...): Keyframes used as VFI input
            keyframes.append(frame_resized)
        elif extract_midpoints and proc % skip == 1:
            # Odd indices (1, 3, 5, ...): Ground truth midpoint frames
            # These are the ACTUAL frames at t=0.5 between keyframes
            midpoints.append(frame_resized)

        proc += 1

    cap.release()

    if extract_midpoints:
        return keyframes, midpoints
    return keyframes


def run_experiment(raw, experiment, start_frame, ref_frames, log_file, output_dir=None, start_time=0.0, duration=5.0, gt_midpoints=None):
    """Run a single experiment with given configuration.

    STREAMING MODE: Frames are written directly to ffmpeg pipe (no RAM buffering).
    This prevents the 38GB+ memory explosion from buffering 1200 4K frames.

    Experiments can specify quality settings (fps, resolution) or use global defaults.

    Args:
        raw: Path to source video
        experiment: Experiment configuration dict
        start_frame: Frame number to start from
        ref_frames: Reference keyframes for quality comparison
        log_file: File handle for logging
        output_dir: Directory for output videos
        start_time: Start time in seconds (for audio extraction)
        duration: Clip duration in seconds (for audio extraction)
        gt_midpoints: Ground truth midpoint frames for VFI quality evaluation.
                      If provided, VFI-generated t=0.5 frames are compared against these.
    """
    import subprocess

    cap = cv2.VideoCapture(raw)
    if not cap.isOpened():
        return None

    rfps = cap.get(cv2.CAP_PROP_FPS)
    skip = max(1, round(rfps / 30))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    name = experiment['name']
    vfi_method = experiment.get('vfi', 'linear_blend')
    sr_method = experiment.get('sr', 'lanczos')
    edge_strength = experiment.get('edge', 0)
    motion_thresh = experiment.get('motion_thresh', 3.0)
    rife_scale = experiment.get('rife_scale', 0.5)

    # Quality settings: use experiment-specific if present, else global defaults
    exp_fps = experiment.get('quality_fps', FPS)
    exp_resolution = experiment.get('quality_resolution', (OUT_W, OUT_H))
    exp_out_w, exp_out_h = exp_resolution
    quality_level = experiment.get('quality_level', 'default')

    is_control = name == 'control' or name.startswith('control_')
    use_rife = vfi_method == 'rife'
    use_adaptive = vfi_method == 'adaptive'
    use_flow = vfi_method == 'optical_flow'
    use_frame_dup = vfi_method == 'frame_dup'
    # Innovative VFI methods
    use_ui_aware = vfi_method == 'ui_aware'
    use_ughi = vfi_method == 'ughi'
    use_mcar = vfi_method == 'mcar'
    mcar_low = experiment.get('mcar_low', 0.3)  # Complexity threshold for linear blend
    mcar_high = experiment.get('mcar_high', 0.7)  # Complexity threshold for RIFE
    interp = cv2.INTER_CUBIC if sr_method == 'bicubic' else cv2.INTER_LANCZOS4

    cnt, prev, prev_gray, proc = 0, None, None, 0
    needed = int(DURATION * rfps)
    psnr_sum, ssim_sum, metric_cnt = 0.0, 0.0, 0
    rife_frame_count = 0
    total_interp_frames = 0
    ref_idx = 0  # Separate counter for reference frame index (fixes frame mismatch bug)

    # VFI QUALITY METRICS: Compare interpolated t=0.5 frames against ground truth
    vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt = 0.0, 0.0, 0
    vfi_lpips_sum, vfi_lpips_cnt = 0.0, 0  # LPIPS sampled every N frames
    LPIPS_SAMPLE_RATE = 10  # Calculate LPIPS every 10th frame (GPU intensive)
    midpoint_idx = 0  # Index into gt_midpoints array

    def evaluate_vfi_frame(vfi_frame, gt_midpoints, midpoint_idx, vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt,
                           vfi_lpips_sum, vfi_lpips_cnt):
        """Compare VFI-generated frame against ground truth midpoint."""
        if gt_midpoints and midpoint_idx < len(gt_midpoints):
            gt_frame = gt_midpoints[midpoint_idx]
            # Ensure sizes match
            if gt_frame.shape[:2] != vfi_frame.shape[:2]:
                gt_frame = cv2.resize(gt_frame, (vfi_frame.shape[1], vfi_frame.shape[0]),
                                      interpolation=cv2.INTER_LANCZOS4)
            vfi_psnr_sum += calc_psnr(vfi_frame, gt_frame)
            vfi_ssim_sum += calc_ssim(cv2.cvtColor(vfi_frame, cv2.COLOR_BGR2GRAY),
                                       cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY))
            vfi_metric_cnt += 1

            # LPIPS (sampled for efficiency - GPU intensive)
            if vfi_metric_cnt % LPIPS_SAMPLE_RATE == 0:
                lpips_val = calc_lpips(vfi_frame, gt_frame)
                if lpips_val is not None:
                    vfi_lpips_sum += lpips_val
                    vfi_lpips_cnt += 1
        return vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt

    # STREAMING: Write frames directly to ffmpeg instead of buffering in RAM
    # This is the key fix - avoids 38GB+ memory accumulation
    output_path = None
    ff_proc = None
    audio_file = None
    if output_dir:
        output_path = output_dir / f"{name}_experiment.mp4"

        # Extract source audio for this interval (for synced playback)
        audio_file = output_dir / f"{name}_audio.aac"
        try:
            audio_result = subprocess.run([
                'ffmpeg', '-y', '-ss', str(start_time), '-t', str(duration),
                '-i', str(raw), '-vn', '-acodec', 'aac', '-b:a', '192k',
                str(audio_file)
            ], capture_output=True, timeout=60)
            if audio_result.returncode != 0:
                log_file.write(f"[{name}] WARNING: Audio extraction failed, continuing without audio\n")
                audio_file = None
        except Exception as e:
            log_file.write(f"[{name}] WARNING: Audio extraction error: {e}\n")
            audio_file = None

        # Use 120fps for 4x VFI (30fps source → 120fps output)
        # This ensures smooth playback at correct speed
        output_fps = 120

        if audio_file and audio_file.exists():
            # Encode with audio
            ff_proc = subprocess.Popen([
                'ffmpeg', '-y',
                '-f', 'rawvideo', '-s', f'{exp_out_w}x{exp_out_h}',
                '-pix_fmt', 'bgr24', '-r', str(output_fps), '-i', '-',
                '-i', str(audio_file),
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-shortest',
                str(output_path)
            ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        else:
            # Fallback: encode without audio
            ff_proc = subprocess.Popen([
                'ffmpeg', '-y', '-f', 'rawvideo', '-s', f'{exp_out_w}x{exp_out_h}',
                '-pix_fmt', 'bgr24', '-r', str(output_fps), '-i', '-',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                str(output_path)
            ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    t0 = time.time()
    log_file.write(f"[{name}] Starting experiment (streaming mode)...\n")
    log_file.flush()

    while proc < needed:
        ret, fr = cap.read()
        if not ret:
            break
        if proc % skip != 0:
            proc += 1
            continue

        cr = crop16_9(fr)

        if is_control:
            out = cv2.resize(cr, (exp_out_w, exp_out_h), interpolation=cv2.INTER_LANCZOS4)
            out_for_metrics = out  # Control is not enhanced
        else:
            degraded = cv2.resize(cr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
            out = cv2.resize(degraded, (exp_out_w, exp_out_h), interpolation=interp)
            out_for_metrics = out.copy()  # Save non-enhanced version for fair metric comparison
            if edge_strength > 0:
                out = edge_enhance(out, edge_strength)  # Only enhance for video output

        curr_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if (use_flow or use_adaptive or use_mcar) else None

        # VFI: generate 3 intermediate frames - STREAM directly to ffmpeg
        # Also evaluate t=0.5 frame (i=2) against ground truth midpoint
        if prev is not None:
            if use_frame_dup:
                # Frame duplication: midpoint is just prev (worst case baseline)
                for i in range(1, 4):
                    if ff_proc:
                        ff_proc.stdin.write(prev.tobytes())
                    cnt += 1
                    # Evaluate t=0.5 frame (i=2)
                    if i == 2:
                        vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt = evaluate_vfi_frame(
                            prev, gt_midpoints, midpoint_idx, vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt)
                midpoint_idx += 1

            elif use_adaptive and prev_gray is not None:
                motion = calc_motion(
                    cv2.resize(prev_gray, (480, 270)),
                    cv2.resize(curr_gray, (480, 270))
                )
                total_interp_frames += 3
                if motion > motion_thresh:
                    rife_frame_count += 3
                    for i in range(1, 4):
                        mid = rife_interpolate(prev, out, i/4, rife_scale)
                        if ff_proc:
                            ff_proc.stdin.write(mid.tobytes())
                        cnt += 1
                        if i == 2:
                            vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt = evaluate_vfi_frame(
                                mid, gt_midpoints, midpoint_idx, vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt)
                else:
                    for i in range(1, 4):
                        t = i / 4
                        blended = cv2.addWeighted(prev, 1-t, out, t, 0)
                        if ff_proc:
                            ff_proc.stdin.write(blended.tobytes())
                        cnt += 1
                        if i == 2:
                            vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt = evaluate_vfi_frame(
                                blended, gt_midpoints, midpoint_idx, vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt)
                midpoint_idx += 1

            elif use_rife:
                for i in range(1, 4):
                    mid = rife_interpolate(prev, out, i/4, rife_scale)
                    if ff_proc:
                        ff_proc.stdin.write(mid.tobytes())
                    cnt += 1
                    if i == 2:
                        vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt = evaluate_vfi_frame(
                            mid, gt_midpoints, midpoint_idx, vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt)
                midpoint_idx += 1

            elif use_flow:
                # Compute optical flow at 1/2 resolution for better quality
                # (was 1/4 which destroyed flow field accuracy)
                g0_sm = cv2.resize(prev_gray, (exp_out_w//2, exp_out_h//2))
                g1_sm = cv2.resize(curr_gray, (exp_out_w//2, exp_out_h//2))
                fl = cv2.resize(cv2.calcOpticalFlowFarneback(g0_sm, g1_sm, None, 0.5, 3, 15, 3, 5, 1.2, 0),
                               (exp_out_w, exp_out_h)) * 2
                h, w = prev.shape[:2]
                y, x = np.mgrid[0:h, 0:w].astype(np.float32)
                for i in range(1, 4):
                    t = i / 4
                    wr = cv2.remap(prev, x + fl[..., 0]*t, y + fl[..., 1]*t, cv2.INTER_LINEAR)
                    blended = cv2.addWeighted(wr, 1-t, out, t, 0)
                    if ff_proc:
                        ff_proc.stdin.write(blended.tobytes())
                    cnt += 1
                    if i == 2:
                        vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt = evaluate_vfi_frame(
                            blended, gt_midpoints, midpoint_idx, vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt)
                midpoint_idx += 1

            # ============ INNOVATIVE VFI METHODS ============
            elif use_ui_aware:
                # UI-Aware Frame Interpolation: preserve HUD elements
                for i in range(1, 4):
                    t = i / 4
                    mid = ui_aware_interpolate(prev, out, t)
                    if ff_proc:
                        ff_proc.stdin.write(mid.tobytes())
                    cnt += 1
                    if i == 2:
                        vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt = evaluate_vfi_frame(
                            mid, gt_midpoints, midpoint_idx, vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt)
                midpoint_idx += 1

            elif use_ughi:
                # Uncertainty-Guided Hybrid Interpolation
                for i in range(1, 4):
                    t = i / 4
                    mid, _ = ughi_interpolate(prev, out, t)
                    if ff_proc:
                        ff_proc.stdin.write(mid.tobytes())
                    cnt += 1
                    if i == 2:
                        vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt = evaluate_vfi_frame(
                            mid, gt_midpoints, midpoint_idx, vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt)
                midpoint_idx += 1

            elif use_mcar:
                # Motion-Complexity Adaptive Routing
                # PERF FIX: Compute complexity ONCE per frame pair, reuse for all 3 interpolations
                prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                complexity, flow = estimate_motion_complexity(prev_gray, curr_gray)

                total_interp_frames += 3
                for i in range(1, 4):
                    t = i / 4
                    mid, tier = mcar_interpolate_cached(prev, out, t, complexity, flow, rife_scale, mcar_low, mcar_high)
                    if tier == 'rife':
                        rife_frame_count += 1
                    if ff_proc:
                        ff_proc.stdin.write(mid.tobytes())
                    cnt += 1
                    if i == 2:
                        vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt = evaluate_vfi_frame(
                            mid, gt_midpoints, midpoint_idx, vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt)
                midpoint_idx += 1
            # ================================================
            elif is_control:
                # Control experiment: NO VFI - use frame duplication for video output only
                # Do NOT evaluate VFI metrics (control doesn't perform interpolation)
                for i in range(1, 4):
                    if ff_proc:
                        ff_proc.stdin.write(prev.tobytes())  # Duplicate previous frame
                    cnt += 1
                # Skip VFI evaluation for control - it doesn't do VFI
                midpoint_idx += 1

            else:  # linear_blend
                for i in range(1, 4):
                    t = i / 4
                    blended = cv2.addWeighted(prev, 1-t, out, t, 0)
                    if ff_proc:
                        ff_proc.stdin.write(blended.tobytes())
                    cnt += 1
                    if i == 2:
                        vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt = evaluate_vfi_frame(
                            blended, gt_midpoints, midpoint_idx, vfi_psnr_sum, vfi_ssim_sum, vfi_metric_cnt, vfi_lpips_sum, vfi_lpips_cnt)
                midpoint_idx += 1

        # Write the keyframe
        if ff_proc:
            ff_proc.stdin.write(out.tobytes())
        cnt += 1

        # Metrics vs reference - compare EVERY keyframe at full output resolution
        # BUG FIX: Use ref_idx (not proc) to index into ref_frames
        # BUG FIX: Compare at output resolution (not tiny 480x270) for accurate metrics
        if ref_frames and ref_idx < len(ref_frames):
            ref_frame = ref_frames[ref_idx]
            # Ensure sizes match (ref might be different resolution)
            if ref_frame.shape[:2] != out_for_metrics.shape[:2]:
                ref_frame = cv2.resize(ref_frame, (out_for_metrics.shape[1], out_for_metrics.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            psnr_sum += calc_psnr(out_for_metrics, ref_frame)
            ssim_sum += calc_ssim(cv2.cvtColor(out_for_metrics, cv2.COLOR_BGR2GRAY),
                                   cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY))
            metric_cnt += 1
        ref_idx += 1  # Always increment after processing a keyframe

        # Memory check every 100 frames (only for heavy experiments)
        if (use_rife or use_adaptive) and cnt % 100 == 0:
            check_memory_during_processing()

        prev, prev_gray = out.copy(), curr_gray
        proc += 1

    # Close ffmpeg and WAIT for it to finish (strict sequential)
    if ff_proc:
        ff_proc.stdin.close()
        ff_proc.wait(timeout=120)  # Wait up to 2 minutes for encoding to finish
        if ff_proc.returncode != 0:
            log_file.write(f"[{name}] WARNING: ffmpeg returned non-zero\n")

    # Clean up temp audio file
    if audio_file and Path(audio_file).exists():
        try:
            Path(audio_file).unlink()
        except Exception:
            pass  # Ignore cleanup errors

    elapsed = time.time() - t0
    cap.release()

    # Keyframe metrics (degradation + upscaling quality)
    avg_keyframe_psnr = psnr_sum / metric_cnt if metric_cnt > 0 else 0
    avg_keyframe_ssim = ssim_sum / metric_cnt if metric_cnt > 0 else 0

    # VFI metrics (interpolation quality - the key differentiator!)
    avg_vfi_psnr = vfi_psnr_sum / vfi_metric_cnt if vfi_metric_cnt > 0 else 0
    avg_vfi_ssim = vfi_ssim_sum / vfi_metric_cnt if vfi_metric_cnt > 0 else 0
    avg_vfi_lpips = vfi_lpips_sum / vfi_lpips_cnt if vfi_lpips_cnt > 0 else None  # Lower is better

    # Overall weighted metrics (keyframes are 25% of output, VFI is 75%)
    # Weight reflects actual frame distribution: 1 keyframe + 3 interpolated per segment
    if metric_cnt > 0 and vfi_metric_cnt > 0:
        overall_psnr = 0.25 * avg_keyframe_psnr + 0.75 * avg_vfi_psnr
        overall_ssim = 0.25 * avg_keyframe_ssim + 0.75 * avg_vfi_ssim
    elif metric_cnt > 0:
        overall_psnr, overall_ssim = avg_keyframe_psnr, avg_keyframe_ssim
    else:
        overall_psnr, overall_ssim = 0, 0

    rife_pct = (rife_frame_count / total_interp_frames * 100) if total_interp_frames > 0 else 0

    result = {
        'name': name,
        'config': experiment,
        'frames': cnt,
        'time_s': round(elapsed, 2),
        'fps_achieved': round(cnt / elapsed, 1) if elapsed > 0 else 0,

        # SEPARATED METRICS - Now properly measuring both keyframe and VFI quality
        # Using safe_round to handle infinity (identical images)
        'keyframe_psnr_db': safe_round(avg_keyframe_psnr, 2),
        'keyframe_ssim': round(avg_keyframe_ssim, 4),
        'vfi_psnr_db': safe_round(avg_vfi_psnr, 2),
        'vfi_ssim': round(avg_vfi_ssim, 4),
        'vfi_lpips': round(avg_vfi_lpips, 4) if avg_vfi_lpips is not None else None,  # Lower is better (0-1)
        'overall_psnr_db': safe_round(overall_psnr, 2),
        'overall_ssim': round(overall_ssim, 4),

        # Backwards compatibility - now uses overall (weighted) metrics
        'psnr_db': safe_round(overall_psnr, 2),
        'ssim': round(overall_ssim, 4),

        # Evaluation counts
        'keyframes_evaluated': metric_cnt,
        'vfi_frames_evaluated': vfi_metric_cnt,
        'lpips_frames_sampled': vfi_lpips_cnt,

        'rife_frames_pct': round(rife_pct, 1) if use_adaptive or use_mcar else (100.0 if use_rife else 0),
        'realtime_x': round((cnt/exp_fps) / elapsed, 2) if elapsed > 0 else 0,
        'used_rife': use_rife or use_adaptive or use_mcar,
        'output_video': str(output_path) if output_path else None,
        # Quality tracking for science!
        'quality_level': quality_level,
        'target_fps': exp_fps,
        'target_resolution': f"{exp_out_w}x{exp_out_h}",
        'vfi_method': vfi_method,
        'sr_method': sr_method,
        'edge_strength': edge_strength,
    }

    lpips_str = f" | LPIPS: {avg_vfi_lpips:.4f}" if avg_vfi_lpips is not None else ""
    log_file.write(f"[{name}] {cnt} frames in {elapsed:.1f}s | KF_PSNR: {avg_keyframe_psnr:.2f}dB | VFI_PSNR: {avg_vfi_psnr:.2f}dB{lpips_str} | Overall: {overall_psnr:.2f}dB\n")
    log_file.flush()

    return result


def clear_gpu_memory():
    """Clear GPU memory between heavy experiments."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        time.sleep(2)  # Give system time to reclaim memory
    except:
        pass


def cleanup_after_experiment(used_rife=False):
    """Full cleanup between experiments - unload RIFE, clear GPU, GC."""
    # Unload RIFE model if it was loaded
    if used_rife:
        try:
            # Unload from this module's singleton
            if RIFEModel._instance is not None:
                del RIFEModel._instance
                RIFEModel._instance = None
                print("  [CLEANUP] RIFE model unloaded")
        except:
            pass

    # Clear GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass

    # Force garbage collection
    gc.collect()

    # Let system stabilize
    time.sleep(3)


def check_memory_during_processing():
    """Check memory during processing - return False if dangerously low."""
    try:
        import psutil
        ram = psutil.virtual_memory()
        if ram.available < 2_000_000_000:  # 2GB threshold
            print("\n  [!] Memory low - triggering GC...")
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            time.sleep(2)
            return False
    except ImportError:
        pass  # psutil not installed, skip check
    return True


def main():
    global shutdown_requested

    parser = argparse.ArgumentParser(description='Run VFI+SR experiments')
    parser.add_argument('--num-intervals', type=int, default=1, help='Number of random intervals per experiment')
    parser.add_argument('--light-only', action='store_true', help='DEPRECATED: All experiments run by default')
    parser.add_argument('--heavy-only', action='store_true', help='DEPRECATED: All experiments run by default')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint (skip completed experiments)')
    parser.add_argument('--experiment', type=str, help='Run single experiment by name')
    parser.add_argument('--low-mem', action='store_true', help='Low memory mode (1080p output instead of 4K)')
    parser.add_argument('--raw', type=str, default='data/raw/clip1.mp4', help='Source video')
    # Quality testing - run experiments at different FPS/resolution combos
    parser.add_argument('--quality-test', action='store_true',
                        help='Test core methods at all quality levels (high/medium/low)')
    parser.add_argument('--quality', type=str, choices=['high', 'medium', 'low'],
                        help='Run experiments at specific quality level only')
    # Force all experiments to use the same interval (for fair A/B comparison)
    parser.add_argument('--force-interval', type=float, default=None,
                        help='Force all experiments to use this start time (e.g., 9.3)')
    # Pre-computed interval support
    parser.add_argument('--clip', type=str, default=None,
                        help='Use pre-computed intervals from this clip ID (e.g., arc_raiders_001)')
    parser.add_argument('--interval', type=str, default=None,
                        help='Use specific pre-computed interval (e.g., interval_0001)')
    parser.add_argument('--intervals', type=str, default=None,
                        help='Run on multiple intervals: "all" or comma-separated list')
    args = parser.parse_args()

    # Apply low-mem settings globally (overridden by --quality if specified)
    global OUT_W, OUT_H, FPS
    if args.low_mem:
        OUT_W, OUT_H = 1920, 1080
        FPS = 60
        print("[LOW-MEM MODE] Output: 1080p@60fps")

    project_root = Path(__file__).parent.parent
    raw = str(project_root / args.raw)

    # SCIENTIFIC ACCURACY: Get source video properties and use them for output
    # This ensures we compare at the source's native resolution
    src_w, src_h, src_fps = get_source_properties(raw)
    # Crop to 16:9 aspect ratio (same as crop16_9 function)
    if src_w / src_h > 16 / 9:
        crop_w = int(src_h * 16 / 9)
        crop_h = src_h
    else:
        crop_w = src_w
        crop_h = int(src_w * 9 / 16)
    print(f"[SOURCE] {src_w}x{src_h} @ {src_fps:.1f}fps -> Cropped to {crop_w}x{crop_h}")
    OUT_W, OUT_H = crop_w, crop_h
    FPS = int(round(src_fps))  # Use source FPS (rounded)

    output_dir = project_root / 'outputs'
    results_file = output_dir / 'experiment_results.json'
    log_path = output_dir / 'experiment_log.txt'

    # Load existing results or start fresh
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"[CHECKPOINT] Loaded {len(all_results.get('experiments', []))} existing results")
    else:
        all_results = {'experiments': [], 'started_at': datetime.now().isoformat()}

    # Get completed experiments for resume mode
    completed = get_completed_experiments(all_results) if args.resume else set()
    if args.resume and completed:
        print(f"[RESUME] Will skip {len(completed)} completed experiments")

    # Select experiments based on flags
    if args.quality_test:
        # Quality test mode: run core methods at ALL quality levels
        # This creates: 6 core methods × 3 quality levels = 18 experiments
        light_quality = generate_quality_experiments(CORE_LIGHT)
        heavy_quality = generate_quality_experiments(CORE_HEAVY)
        if args.light_only:
            experiments_to_run = light_quality
            mode = "QUALITY TEST - LIGHT ONLY (4 methods × 3 qualities = 12 experiments)"
        elif args.heavy_only:
            experiments_to_run = heavy_quality
            mode = "QUALITY TEST - HEAVY ONLY (2 methods × 3 qualities = 6 experiments)"
        else:
            experiments_to_run = light_quality + heavy_quality
            mode = "QUALITY TEST - ALL (6 methods × 3 qualities = 18 experiments)"
    elif args.quality:
        # Single quality level: run all methods at one quality level
        q = QUALITY_LEVELS[args.quality]
        OUT_W, OUT_H = q['resolution']
        FPS = q['fps']
        # Always run ALL experiments (light/heavy flags deprecated)
        experiments_to_run = LIGHT_EXPERIMENTS + HEAVY_EXPERIMENTS
        mode = f"FULL @ {q['label']}"
    elif args.experiment:
        # Single experiment mode
        all_exp = LIGHT_EXPERIMENTS + HEAVY_EXPERIMENTS
        matching = [e for e in all_exp if e['name'] == args.experiment]
        if not matching:
            print(f"[ERROR] Unknown experiment: {args.experiment}")
            print(f"Available: {[e['name'] for e in all_exp]}")
            sys.exit(1)
        experiments_to_run = matching
        mode = f"SINGLE: {args.experiment}"
    else:
        # Always run ALL experiments (deprecated light/heavy flags ignored)
        experiments_to_run = LIGHT_EXPERIMENTS + HEAVY_EXPERIMENTS
        mode = "FULL (all methods)"

    # Create output directory for experiment videos
    videos_dir = output_dir / 'experiment_videos'
    videos_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VFI+SR EXPERIMENT RUNNER (STREAMING MODE)")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Experiments: {len(experiments_to_run)}")
    print(f"Intervals per experiment: {args.num_intervals}")
    print(f"Output resolution: {OUT_W}x{OUT_H}")
    print(f"Results: {results_file}")
    print(f"Videos: {videos_dir}")
    print(f"Log: {log_path}")
    print(f"[Ctrl+C to stop gracefully]")
    print("=" * 70 + "\n")

    with open(log_path, 'a') as log_file:
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"EXPERIMENT RUN STARTED: {datetime.now().isoformat()}\n")
        log_file.write(f"Mode: {mode} | Resolution: {OUT_W}x{OUT_H}\n")
        log_file.write(f"{'='*60}\n\n")

        # Determine intervals to run
        if args.clip:
            # Using pre-computed intervals from clip
            available = list_available_intervals(args.clip)
            if not available:
                print(f"[ERROR] No intervals found for clip '{args.clip}'. Run extract_intervals.py first.")
                sys.exit(1)

            if args.interval:
                # Single specific interval
                intervals_to_run = [i for i in available if i['interval_id'] == args.interval]
                if not intervals_to_run:
                    print(f"[ERROR] Interval '{args.interval}' not found in clip '{args.clip}'")
                    sys.exit(1)
            elif args.intervals == 'all':
                # All available intervals
                intervals_to_run = available
            elif args.intervals:
                # Comma-separated list
                requested = [i.strip() for i in args.intervals.split(',')]
                intervals_to_run = [i for i in available if i['interval_id'] in requested]
            else:
                # Default: use first interval
                intervals_to_run = available[:1]

            print(f"[CACHED INTERVALS] Using {len(intervals_to_run)} pre-computed interval(s) from '{args.clip}'")
            use_cached = True
        else:
            # Legacy mode: on-the-fly extraction
            intervals_to_run = list(range(args.num_intervals))
            use_cached = False

        for interval_idx, interval_spec in enumerate(intervals_to_run):
            if use_cached:
                # Load pre-computed interval
                interval_data = load_interval(args.clip, interval_spec['interval_id'])
                if not interval_data:
                    print(f"[ERROR] Failed to load interval {interval_spec['interval_id']}")
                    continue

                ref_frames = interval_data['keyframes']
                gt_midpoints = interval_data['midpoints']
                start_time = interval_spec['start_s']
                start_frame = interval_spec['start_frame']
                interval_duration = interval_spec['end_s'] - interval_spec['start_s']
                interval_name = interval_spec['interval_id']

                log_file.write(f"\n--- CACHED INTERVAL {interval_idx + 1}/{len(intervals_to_run)}: {interval_name} ({start_time:.1f}s - {interval_spec['end_s']:.1f}s) ---\n\n")
                print(f"\n--- Cached Interval {interval_idx + 1}: {interval_name} ({start_time:.1f}s - {interval_spec['end_s']:.1f}s, {interval_spec.get('difficulty', 'N/A')}) ---")
                print(f"Loaded: {len(ref_frames)} keyframes, {len(gt_midpoints)} ground truth midpoints\n")

            else:
                # Legacy: on-the-fly extraction
                if args.force_interval is not None:
                    start_time = args.force_interval
                    print(f"[FORCED INTERVAL] Using start time {start_time}s for all experiments")
                else:
                    start_time = get_random_start(raw, DURATION)
                cap = cv2.VideoCapture(raw)
                rfps = cap.get(cv2.CAP_PROP_FPS)
                start_frame = int(start_time * rfps)
                cap.release()
                interval_duration = DURATION
                interval_name = f"{start_time:.1f}s"

                log_file.write(f"\n--- INTERVAL {interval_idx + 1}/{args.num_intervals}: {start_time:.1f}s - {start_time + DURATION:.1f}s ---\n\n")
                print(f"\n--- Interval {interval_idx + 1}: {start_time:.1f}s - {start_time + DURATION:.1f}s ---")

                # Generate reference frames for this interval
                # Extract both keyframes AND ground truth midpoints for VFI evaluation
                print("Generating reference frames...")
                ref_frames, gt_midpoints = generate_reference(raw, start_frame, extract_midpoints=True)
                print(f"Reference: {len(ref_frames)} keyframes, {len(gt_midpoints)} ground truth midpoints\n")

            for exp_idx, experiment in enumerate(experiments_to_run):
                # Check for graceful shutdown
                if shutdown_requested:
                    print("\n[!] Shutdown requested - saving and exiting...")
                    log_file.write(f"\n[SHUTDOWN] User requested stop at {datetime.now().isoformat()}\n")
                    break

                is_heavy = experiment.get('vfi') in ['rife', 'adaptive']
                exp_type = "HEAVY" if is_heavy else "LIGHT"

                # Skip if already completed (resume mode)
                if (experiment['name'], interval_idx) in completed:
                    print(f"[{exp_idx + 1}/{len(experiments_to_run)}] [SKIP] {experiment['name']} - already completed")
                    continue

                print(f"[{exp_idx + 1}/{len(experiments_to_run)}] [{exp_type}] {experiment['name']} - {experiment.get('description', '')}")

                # Clear GPU memory before heavy experiments
                if is_heavy:
                    print("  Clearing GPU memory before heavy experiment...")
                    clear_gpu_memory()
                    # Check if we have enough memory
                    mem_ok, free_mb = check_gpu_memory(1500)
                    if not mem_ok:
                        print(f"  [WARNING] Low GPU memory ({free_mb:.0f}MB free) - consider using --low-mem")

                try:
                    # Pass videos_dir for streaming output, start_time and duration for audio sync
                    # Pass gt_midpoints for VFI quality evaluation
                    result = run_experiment(raw, experiment, start_frame, ref_frames, log_file,
                                           output_dir=videos_dir, start_time=start_time, duration=DURATION,
                                           gt_midpoints=gt_midpoints)
                except Exception as e:
                    print(f"  [ERROR] Experiment failed: {e}")
                    log_file.write(f"[ERROR] {experiment['name']} failed: {e}\n")
                    result = None

                if result:
                    result['interval'] = {'start': round(start_time, 1), 'end': round(start_time + interval_duration, 1)}
                    result['timestamp'] = datetime.now().isoformat()
                    result['interval_idx'] = interval_idx
                    result['experiment_type'] = exp_type
                    result['output_resolution'] = f"{OUT_W}x{OUT_H}"
                    # Add clip/interval IDs for cached intervals
                    if use_cached:
                        result['clip_id'] = args.clip
                        result['interval_id'] = interval_name
                        result['difficulty'] = interval_spec.get('difficulty', 'UNKNOWN')
                    all_results['experiments'].append(result)

                    # Save after each experiment (crash-safe)
                    with open(results_file, 'w') as f:
                        json.dump(all_results, f, indent=2)

                    # Show new separated metrics if available
                    if 'vfi_psnr_db' in result and result['vfi_psnr_db'] is not None:
                        # Handle "inf" strings from safe_round
                        kf_psnr = result['keyframe_psnr_db']
                        vfi_psnr = result['vfi_psnr_db']
                        overall_psnr = result['overall_psnr_db']
                        vfi_lpips = result.get('vfi_lpips')
                        kf_str = f"{kf_psnr:.2f}" if isinstance(kf_psnr, (int, float)) else str(kf_psnr)
                        vfi_str = f"{vfi_psnr:.2f}" if isinstance(vfi_psnr, (int, float)) else str(vfi_psnr)
                        overall_str = f"{overall_psnr:.2f}" if isinstance(overall_psnr, (int, float)) else str(overall_psnr)
                        lpips_str = f" | LPIPS: {vfi_lpips:.4f}" if vfi_lpips is not None else ""
                        print(f"  -> VFI_PSNR: {vfi_str}dB{lpips_str} | Overall: {overall_str}dB | Time: {result['time_s']:.1f}s")
                    else:
                        print(f"  -> PSNR: {result['psnr_db']:.2f}dB | Time: {result['time_s']:.1f}s | RIFE%: {result['rife_frames_pct']:.0f}%")
                    print(f"  [SAVED] {len(all_results['experiments'])} total experiments in checkpoint")

                # FULL CLEANUP after heavy experiments (unload RIFE, clear GPU, GC)
                if is_heavy:
                    used_rife = result.get('used_rife', False) if result else True
                    cleanup_after_experiment(used_rife=used_rife)

            # Check for shutdown after interval
            if shutdown_requested:
                break

        # Final summary
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"EXPERIMENT RUN COMPLETE: {datetime.now().isoformat()}\n")
        log_file.write(f"Total experiments: {len(all_results['experiments'])}\n")
        log_file.write(f"{'='*60}\n")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT RUN COMPLETE")
    print("=" * 70)
    print(f"Total results: {len(all_results['experiments'])}")
    print(f"Results saved to: {results_file}")

    # Quick analysis
    if all_results['experiments']:
        by_name = {}
        for exp in all_results['experiments']:
            name = exp['name']
            if name not in by_name:
                by_name[name] = []
            by_name[name].append(exp)

        print("\nSummary by method:")
        print("-" * 80)
        # Check if we have the new VFI metrics
        has_vfi_metrics = any('vfi_psnr_db' in r and r['vfi_psnr_db'] is not None
                              for r in all_results['experiments'])
        if has_vfi_metrics:
            print(f"{'Method':<25} {'KF_PSNR':>8} {'VFI_PSNR':>9} {'Overall':>8} {'Time':>7}  n")
            print("-" * 80)
            for name, results in sorted(by_name.items()):
                avg_kf = sum(r.get('keyframe_psnr_db', r['psnr_db']) for r in results) / len(results)
                vfi_results = [r for r in results if r.get('vfi_psnr_db') is not None]
                avg_vfi = sum(r['vfi_psnr_db'] for r in vfi_results) / len(vfi_results) if vfi_results else 0
                avg_overall = sum(r.get('overall_psnr_db', r['psnr_db']) for r in results) / len(results)
                avg_time = sum(r['time_s'] for r in results) / len(results)
                print(f"{name:<25} {avg_kf:>7.2f}dB {avg_vfi:>8.2f}dB {avg_overall:>7.2f}dB {avg_time:>6.1f}s  ({len(results)})")
        else:
            for name, results in sorted(by_name.items()):
                avg_psnr = sum(r['psnr_db'] for r in results) / len(results)
                avg_time = sum(r['time_s'] for r in results) / len(results)
                print(f"{name:<25} PSNR: {avg_psnr:.2f}dB  Time: {avg_time:.1f}s  (n={len(results)})")


if __name__ == '__main__':
    main()
