"""
evaluation/metrics.py

Comprehensive quality metrics for VFI+SR evaluation.

## Metric Categories:

### 1. Reconstruction Accuracy (baseline metrics)
- PSNR, SSIM, MS-SSIM — pixel-level accuracy

### 2. Perceptual Quality (better human correlation)
- LPIPS, DISTS — learned perceptual metrics
- UI Legibility — OCR-based text preservation (gaming-specific)

### 3. Temporal Consistency (THE GAP in VFI research)
- tOF (temporal Optical Flow) — flow smoothness across interpolated sequences
- FloLPIPS — motion-weighted perceptual quality
- Flicker Score — direct measurement of high-frequency temporal artifacts
- FVMD — Fréchet Video Motion Distance

### 4. Gaming-Specific
- Motion-difficulty stratification (static/easy/medium/hard/extreme)
- Failure mode detection (UI ghosting, particle trails, edge wobble)
"""

import numpy as np
import cv2
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class MotionDifficulty(Enum):
    """Motion difficulty levels for stratified reporting"""
    STATIC = "static"      # < 1 pixel mean flow
    EASY = "easy"          # 1-5 pixels mean flow
    MEDIUM = "medium"      # 5-15 pixels mean flow
    HARD = "hard"          # 15-30 pixels mean flow
    EXTREME = "extreme"    # > 30 pixels mean flow


@dataclass
class TemporalMetrics:
    """Temporal consistency metrics for VFI evaluation"""
    tof_smoothness: float          # Temporal Optical Flow smoothness (lower = smoother)
    flicker_score: float           # Direct flicker measurement (lower = better)
    flow_consistency: float        # Frame-to-frame flow consistency

    # Per-segment analysis
    motion_variance: float         # Variance of motion magnitude

    def to_dict(self) -> dict:
        return {
            'tof_smoothness': self.tof_smoothness,
            'flicker_score': self.flicker_score,
            'flow_consistency': self.flow_consistency,
            'motion_variance': self.motion_variance,
        }


@dataclass
class StratifiedMetrics:
    """Metrics stratified by motion difficulty"""
    by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)
    frame_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'by_difficulty': self.by_difficulty,
            'frame_counts': self.frame_counts,
        }


@dataclass
class GamingMetrics:
    """Gaming-specific VFI metrics"""
    ui_ghosting_score: float       # UI ghosting detection (lower = better)
    edge_wobble_score: float       # Edge stability (lower = better)

    def to_dict(self) -> dict:
        return {
            'ui_ghosting_score': self.ui_ghosting_score,
            'edge_wobble_score': self.edge_wobble_score,
        }


@dataclass
class QualityResults:
    """Results from quality evaluation"""
    psnr: float
    ssim: float
    lpips: float
    ms_ssim: Optional[float] = None
    flolpips: Optional[float] = None
    
    # Per-frame values for detailed analysis
    psnr_per_frame: Optional[List[float]] = None
    lpips_per_frame: Optional[List[float]] = None
    
    def to_dict(self) -> dict:
        return {
            'psnr': self.psnr,
            'ssim': self.ssim,
            'lpips': self.lpips,
            'ms_ssim': self.ms_ssim,
            'flolpips': self.flolpips,
        }


class QualityEvaluator:
    """
    Evaluate quality of generated frames against ground truth.
    
    Uses pyiqa library for GPU-accelerated metric computation.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self._initialized = False
        
        # Metrics will be initialized lazily
        self._psnr = None
        self._ssim = None
        self._lpips = None
        self._ms_ssim = None
        self._flolpips = None
    
    def _ensure_initialized(self):
        """Lazily initialize metrics"""
        if self._initialized:
            return
        
        print("Initializing quality metrics...")
        
        try:
            import pyiqa
            
            self._psnr = pyiqa.create_metric('psnr', device=self.device)
            self._ssim = pyiqa.create_metric('ssim', device=self.device)
            self._lpips = pyiqa.create_metric('lpips', device=self.device)
            
            try:
                self._ms_ssim = pyiqa.create_metric('ms_ssim', device=self.device)
            except Exception:
                print("  Warning: MS-SSIM not available")

            try:
                self._flolpips = pyiqa.create_metric('flolpips', device=self.device)
            except Exception:
                print("  Warning: FloLPIPS not available (optional)")
            
            self._initialized = True
            print("Quality metrics ready")
            
        except ImportError:
            raise ImportError(
                "pyiqa not installed. Run: pip install pyiqa"
            )
    
    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """Convert numpy frame to tensor for metric computation"""
        # frame is (H, W, C) uint8 RGB
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)
    
    def evaluate_pair(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate a single predicted frame against ground truth.
        
        Args:
            pred: Predicted frame (H, W, C) uint8 RGB
            gt: Ground truth frame (H, W, C) uint8 RGB
            
        Returns:
            Dictionary of metric scores
        """
        self._ensure_initialized()
        
        # Ensure same resolution
        if pred.shape != gt.shape:
            import cv2
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        
        # Convert to tensors
        pred_t = self._to_tensor(pred)
        gt_t = self._to_tensor(gt)
        
        results = {}
        
        with torch.no_grad():
            results['psnr'] = self._psnr(pred_t, gt_t).item()
            results['ssim'] = self._ssim(pred_t, gt_t).item()
            results['lpips'] = self._lpips(pred_t, gt_t).item()
            
            if self._ms_ssim is not None:
                results['ms_ssim'] = self._ms_ssim(pred_t, gt_t).item()
            
            if self._flolpips is not None:
                results['flolpips'] = self._flolpips(pred_t, gt_t).item()
        
        return results
    
    def evaluate(
        self,
        pred_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
    ) -> QualityResults:
        """
        Evaluate a list of predicted frames against ground truth.
        
        Args:
            pred_frames: List of predicted frames (H, W, C) uint8 RGB
            gt_frames: List of ground truth frames (H, W, C) uint8 RGB
            
        Returns:
            QualityResults with averaged metrics
        """
        self._ensure_initialized()
        
        if len(pred_frames) != len(gt_frames):
            raise ValueError(
                f"Number of predicted frames ({len(pred_frames)}) "
                f"doesn't match ground truth ({len(gt_frames)})"
            )
        
        # Collect per-frame metrics
        psnr_values = []
        ssim_values = []
        lpips_values = []
        ms_ssim_values = []
        flolpips_values = []
        
        for pred, gt in zip(pred_frames, gt_frames):
            metrics = self.evaluate_pair(pred, gt)
            
            psnr_values.append(metrics['psnr'])
            ssim_values.append(metrics['ssim'])
            lpips_values.append(metrics['lpips'])
            
            if 'ms_ssim' in metrics:
                ms_ssim_values.append(metrics['ms_ssim'])
            if 'flolpips' in metrics:
                flolpips_values.append(metrics['flolpips'])
        
        return QualityResults(
            psnr=float(np.mean(psnr_values)),
            ssim=float(np.mean(ssim_values)),
            lpips=float(np.mean(lpips_values)),
            ms_ssim=float(np.mean(ms_ssim_values)) if ms_ssim_values else None,
            flolpips=float(np.mean(flolpips_values)) if flolpips_values else None,
            psnr_per_frame=psnr_values,
            lpips_per_frame=lpips_values,
        )
    
    def evaluate_temporal_consistency(
        self,
        frames: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Evaluate temporal consistency of a frame sequence.
        
        Good VFI should produce smooth motion with consistent frame-to-frame changes.
        High variance in LPIPS between consecutive frames = jittery/inconsistent.
        
        Args:
            frames: List of frames in temporal order
            
        Returns:
            Dictionary with temporal consistency metrics
        """
        self._ensure_initialized()
        
        if len(frames) < 2:
            return {'temporal_lpips_mean': 0, 'temporal_lpips_std': 0}
        
        lpips_diffs = []
        
        for i in range(len(frames) - 1):
            f1 = self._to_tensor(frames[i])
            f2 = self._to_tensor(frames[i + 1])
            
            with torch.no_grad():
                diff = self._lpips(f1, f2).item()
                lpips_diffs.append(diff)
        
        return {
            'temporal_lpips_mean': float(np.mean(lpips_diffs)),
            'temporal_lpips_std': float(np.std(lpips_diffs)),  # Lower = smoother
            'temporal_lpips_max': float(np.max(lpips_diffs)),
            'temporal_lpips_min': float(np.min(lpips_diffs)),
        }


def compute_psnr_simple(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Simple PSNR computation without pyiqa (for debugging/testing).
    
    PSNR = 10 * log10(MAX^2 / MSE)
    """
    if pred.shape != gt.shape:
        import cv2
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    
    mse = np.mean((pred.astype(float) - gt.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_val = 255.0
    psnr = 10 * np.log10(max_val ** 2 / mse)
    return float(psnr)


def compute_ssim_simple(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Simple SSIM computation using skimage.
    """
    from skimage.metrics import structural_similarity as ssim

    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    # Convert to grayscale for SSIM
    if len(pred.shape) == 3:
        pred_gray = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
        gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
    else:
        pred_gray = pred
        gt_gray = gt

    return float(ssim(pred_gray, gt_gray))


# =============================================================================
# TEMPORAL CONSISTENCY METRICS (THE GAP IN VFI RESEARCH)
# =============================================================================

def classify_motion_difficulty(flow: np.ndarray) -> MotionDifficulty:
    """
    Classify a frame pair's motion difficulty based on optical flow.

    Args:
        flow: Optical flow array (H, W, 2)

    Returns:
        MotionDifficulty enum value
    """
    flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    mean_motion = np.mean(flow_mag)

    if mean_motion < 1:
        return MotionDifficulty.STATIC
    elif mean_motion < 5:
        return MotionDifficulty.EASY
    elif mean_motion < 15:
        return MotionDifficulty.MEDIUM
    elif mean_motion < 30:
        return MotionDifficulty.HARD
    else:
        return MotionDifficulty.EXTREME


def compute_optical_flow(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """Compute optical flow between two frames using Farneback method."""
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = frame1

    if len(frame2.shape) == 3:
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = frame2

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow


def compute_tof_smoothness(flows: List[np.ndarray]) -> float:
    """
    Compute temporal Optical Flow smoothness (tOF).

    Measures how smoothly the optical flow changes between consecutive frames.
    Lower values = smoother motion = better VFI.

    This is THE metric that catches jittery/inconsistent interpolation.

    Args:
        flows: List of optical flow arrays between consecutive frames

    Returns:
        tOF smoothness score (lower is better)
    """
    if len(flows) < 2:
        return 0.0

    flow_diffs = []
    for i in range(len(flows) - 1):
        # Compute flow acceleration (second derivative of motion)
        flow_diff = flows[i+1] - flows[i]
        diff_mag = np.sqrt(flow_diff[..., 0]**2 + flow_diff[..., 1]**2)
        flow_diffs.append(np.mean(diff_mag))

    return float(np.mean(flow_diffs))


def compute_flicker_score(frames: List[np.ndarray]) -> float:
    """
    Compute flicker score for a sequence of frames.

    Detects high-frequency temporal artifacts (the "AI-generated" look).

    Method: Compute second-order temporal differences in luminance.
    High values indicate flickering/inconsistent brightness.

    Args:
        frames: List of frames in temporal order

    Returns:
        Flicker score (lower is better)
    """
    if len(frames) < 3:
        return 0.0

    # Convert to grayscale luminance
    luminances = []
    for frame in frames:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        else:
            gray = frame.astype(float)
        luminances.append(gray)

    # Compute second-order temporal derivative (acceleration)
    flicker_scores = []
    for i in range(1, len(luminances) - 1):
        # Second derivative: f(t+1) - 2*f(t) + f(t-1)
        second_deriv = luminances[i+1] - 2*luminances[i] + luminances[i-1]
        flicker_scores.append(np.mean(np.abs(second_deriv)))

    return float(np.mean(flicker_scores))


def compute_flow_consistency(flows_fwd: List[np.ndarray], flows_bwd: List[np.ndarray]) -> float:
    """
    Compute bidirectional flow consistency.

    For good VFI: forward flow + backward flow should sum to ~zero.

    Args:
        flows_fwd: Forward optical flows (frame i -> frame i+1)
        flows_bwd: Backward optical flows (frame i+1 -> frame i)

    Returns:
        Flow consistency score (lower is better)
    """
    if len(flows_fwd) == 0 or len(flows_bwd) == 0:
        return 0.0

    consistency_errors = []
    for fwd, bwd in zip(flows_fwd, flows_bwd):
        # Perfect consistency: fwd + bwd = 0
        h, w = fwd.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)

        # Warp backward flow using forward flow
        bwd_warped = cv2.remap(
            bwd, x + fwd[..., 0], y + fwd[..., 1],
            cv2.INTER_LINEAR
        )

        # Consistency error
        error = np.sqrt((fwd[..., 0] + bwd_warped[..., 0])**2 +
                       (fwd[..., 1] + bwd_warped[..., 1])**2)
        consistency_errors.append(np.mean(error))

    return float(np.mean(consistency_errors))


def compute_temporal_metrics(frames: List[np.ndarray]) -> TemporalMetrics:
    """
    Compute comprehensive temporal consistency metrics for a frame sequence.

    Args:
        frames: List of frames in temporal order (RGB or BGR uint8)

    Returns:
        TemporalMetrics dataclass with all temporal scores
    """
    if len(frames) < 3:
        return TemporalMetrics(
            tof_smoothness=0.0,
            flicker_score=0.0,
            flow_consistency=0.0,
            motion_variance=0.0,
        )

    # Compute forward and backward flows
    flows_fwd = []
    flows_bwd = []
    flow_magnitudes = []

    for i in range(len(frames) - 1):
        flow_fwd = compute_optical_flow(frames[i], frames[i+1])
        flow_bwd = compute_optical_flow(frames[i+1], frames[i])
        flows_fwd.append(flow_fwd)
        flows_bwd.append(flow_bwd)

        flow_mag = np.sqrt(flow_fwd[..., 0]**2 + flow_fwd[..., 1]**2)
        flow_magnitudes.append(np.mean(flow_mag))

    return TemporalMetrics(
        tof_smoothness=compute_tof_smoothness(flows_fwd),
        flicker_score=compute_flicker_score(frames),
        flow_consistency=compute_flow_consistency(flows_fwd, flows_bwd),
        motion_variance=float(np.var(flow_magnitudes)) if flow_magnitudes else 0.0,
    )


# =============================================================================
# GAMING-SPECIFIC METRICS
# =============================================================================

def detect_ui_regions(frame: np.ndarray) -> np.ndarray:
    """
    Detect likely UI/HUD regions in a gaming frame.

    UI regions typically have:
    - High contrast edges
    - Flat color regions
    - Located near screen edges

    Returns:
        Binary mask of UI regions
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    h, w = gray.shape

    # Detect edges (UI has sharp edges)
    edges = cv2.Canny(gray, 100, 200)

    # Detect flat regions (UI often has solid colors)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    flat_mask = np.abs(laplacian) < 5

    # Weight by screen position (HUD typically at edges)
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    edge_weight = np.minimum(
        np.minimum(y_coords, h - y_coords),
        np.minimum(x_coords, w - x_coords)
    ) / min(h, w) * 4
    edge_weight = np.clip(1 - edge_weight, 0, 1)

    # Combine: high edges + flat neighbors + screen edge position
    ui_score = (edges > 0).astype(float) * 0.4 + flat_mask.astype(float) * 0.3 + edge_weight * 0.3
    ui_mask = (ui_score > 0.5).astype(np.uint8)

    # Clean up with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    ui_mask = cv2.morphologyEx(ui_mask, cv2.MORPH_CLOSE, kernel)

    return ui_mask


def compute_ui_ghosting_score(
    pred_frames: List[np.ndarray],
    gt_frames: List[np.ndarray]
) -> float:
    """
    Compute UI ghosting score for interpolated frames.

    UI ghosting occurs when UI elements are incorrectly interpolated,
    creating double/blurry text and icons.

    Args:
        pred_frames: Predicted/interpolated frames
        gt_frames: Ground truth frames

    Returns:
        UI ghosting score (lower is better)
    """
    if len(pred_frames) != len(gt_frames):
        return 0.0

    ghosting_scores = []

    for pred, gt in zip(pred_frames, gt_frames):
        # Detect UI regions in GT
        ui_mask = detect_ui_regions(gt)

        if np.sum(ui_mask) < 100:  # Skip if no significant UI detected
            continue

        # Compute error only in UI regions
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        error = np.abs(pred.astype(float) - gt.astype(float))
        if len(error.shape) == 3:
            error = np.mean(error, axis=2)

        # Weight by UI mask
        ui_error = error * ui_mask
        ghosting_scores.append(np.mean(ui_error[ui_mask > 0]))

    return float(np.mean(ghosting_scores)) if ghosting_scores else 0.0


def compute_edge_wobble_score(frames: List[np.ndarray]) -> float:
    """
    Compute edge wobble/stability score.

    Detects unstable edges that "wobble" between frames,
    a common artifact in poor VFI.

    Args:
        frames: List of frames in temporal order

    Returns:
        Edge wobble score (lower is better)
    """
    if len(frames) < 3:
        return 0.0

    # Detect edges in each frame
    edge_maps = []
    for frame in frames:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        edges = cv2.Canny(gray, 50, 150).astype(float) / 255.0
        edge_maps.append(edges)

    # Compute edge position variance over time
    wobble_scores = []
    for i in range(1, len(edge_maps) - 1):
        # For each edge pixel, check if it's stable across 3 frames
        prev_edges = edge_maps[i-1]
        curr_edges = edge_maps[i]
        next_edges = edge_maps[i+1]

        # Dilate edges to allow small position tolerance
        kernel = np.ones((3, 3), np.uint8)
        prev_dilated = cv2.dilate(prev_edges, kernel)
        next_dilated = cv2.dilate(next_edges, kernel)

        # Edge is "wobbly" if present in current but not consistently present in neighbors
        wobbly = curr_edges * (1 - prev_dilated * next_dilated)
        wobble_scores.append(np.mean(wobbly))

    return float(np.mean(wobble_scores)) if wobble_scores else 0.0


def compute_gaming_metrics(
    pred_frames: List[np.ndarray],
    gt_frames: List[np.ndarray]
) -> GamingMetrics:
    """
    Compute gaming-specific VFI metrics.

    Args:
        pred_frames: Predicted/interpolated frames
        gt_frames: Ground truth frames

    Returns:
        GamingMetrics dataclass
    """
    return GamingMetrics(
        ui_ghosting_score=compute_ui_ghosting_score(pred_frames, gt_frames),
        edge_wobble_score=compute_edge_wobble_score(pred_frames),
    )


# =============================================================================
# STRATIFIED METRICS
# =============================================================================

def compute_stratified_metrics(
    pred_frames: List[np.ndarray],
    gt_frames: List[np.ndarray]
) -> StratifiedMetrics:
    """
    Compute metrics stratified by motion difficulty.

    This is crucial for gaming VFI evaluation because:
    - Method A might be +0.3dB overall
    - But Method B is +3.4dB on EXTREME motion (which matters more)

    Args:
        pred_frames: Predicted frames
        gt_frames: Ground truth frames

    Returns:
        StratifiedMetrics with per-difficulty breakdown
    """
    if len(pred_frames) != len(gt_frames) or len(pred_frames) < 2:
        return StratifiedMetrics()

    # Group frames by difficulty
    difficulty_frames = {d.value: {'psnr': [], 'ssim': []} for d in MotionDifficulty}
    frame_counts = {d.value: 0 for d in MotionDifficulty}

    for i in range(len(gt_frames) - 1):
        # Classify based on motion between this and next GT frame
        flow = compute_optical_flow(gt_frames[i], gt_frames[i+1])
        difficulty = classify_motion_difficulty(flow)

        # Compute metrics for this frame pair
        pred = pred_frames[i]
        gt = gt_frames[i]

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        psnr = compute_psnr_simple(pred, gt)
        ssim = compute_ssim_simple(pred, gt)

        difficulty_frames[difficulty.value]['psnr'].append(psnr)
        difficulty_frames[difficulty.value]['ssim'].append(ssim)
        frame_counts[difficulty.value] += 1

    # Aggregate per difficulty
    by_difficulty = {}
    for d in MotionDifficulty:
        if frame_counts[d.value] > 0:
            by_difficulty[d.value] = {
                'psnr': float(np.mean(difficulty_frames[d.value]['psnr'])),
                'ssim': float(np.mean(difficulty_frames[d.value]['ssim'])),
            }

    return StratifiedMetrics(
        by_difficulty=by_difficulty,
        frame_counts={k: v for k, v in frame_counts.items() if v > 0},
    )


# =============================================================================
# COMPREHENSIVE BENCHMARK RUNNER
# =============================================================================

@dataclass
class ComprehensiveBenchmarkResults:
    """Complete benchmark results across all metric dimensions"""
    # Reconstruction accuracy
    psnr: float
    ssim: float

    # Temporal consistency
    temporal: TemporalMetrics

    # Gaming-specific
    gaming: GamingMetrics

    # Stratified by difficulty
    stratified: StratifiedMetrics

    # Timing
    total_frames: int

    def to_dict(self) -> dict:
        return {
            'reconstruction': {
                'psnr': self.psnr,
                'ssim': self.ssim,
            },
            'temporal': self.temporal.to_dict(),
            'gaming': self.gaming.to_dict(),
            'stratified': self.stratified.to_dict(),
            'total_frames': self.total_frames,
        }

    def summary_table(self) -> str:
        """Generate summary table for paper/report"""
        lines = [
            "=" * 60,
            "COMPREHENSIVE VFI BENCHMARK RESULTS",
            "=" * 60,
            "",
            "## Reconstruction Accuracy",
            f"  PSNR: {self.psnr:.2f} dB",
            f"  SSIM: {self.ssim:.4f}",
            "",
            "## Temporal Consistency (lower is better)",
            f"  tOF Smoothness: {self.temporal.tof_smoothness:.4f}",
            f"  Flicker Score: {self.temporal.flicker_score:.4f}",
            f"  Flow Consistency: {self.temporal.flow_consistency:.4f}",
            "",
            "## Gaming-Specific (lower is better)",
            f"  UI Ghosting: {self.gaming.ui_ghosting_score:.2f}",
            f"  Edge Wobble: {self.gaming.edge_wobble_score:.4f}",
            "",
            "## By Motion Difficulty",
        ]

        for diff, metrics in self.stratified.by_difficulty.items():
            count = self.stratified.frame_counts.get(diff, 0)
            lines.append(f"  {diff.upper()}: PSNR={metrics['psnr']:.2f}dB, "
                        f"SSIM={metrics['ssim']:.4f} (n={count})")

        lines.append("")
        lines.append(f"Total Frames: {self.total_frames}")
        lines.append("=" * 60)

        return "\n".join(lines)


def run_comprehensive_benchmark(
    pred_frames: List[np.ndarray],
    gt_frames: List[np.ndarray],
    verbose: bool = True
) -> ComprehensiveBenchmarkResults:
    """
    Run comprehensive VFI benchmark across all metric dimensions.

    This is THE function to call for complete evaluation.

    Args:
        pred_frames: Predicted/interpolated frames (list of HxWxC uint8)
        gt_frames: Ground truth frames (list of HxWxC uint8)
        verbose: Print progress

    Returns:
        ComprehensiveBenchmarkResults with all metrics
    """
    if verbose:
        print("Running comprehensive VFI benchmark...")

    # Ensure same number of frames
    n_frames = min(len(pred_frames), len(gt_frames))
    pred_frames = pred_frames[:n_frames]
    gt_frames = gt_frames[:n_frames]

    # 1. Reconstruction accuracy
    if verbose:
        print("  Computing reconstruction metrics...")
    psnr_values = []
    ssim_values = []
    for pred, gt in zip(pred_frames, gt_frames):
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        psnr_values.append(compute_psnr_simple(pred, gt))
        ssim_values.append(compute_ssim_simple(pred, gt))

    # 2. Temporal consistency
    if verbose:
        print("  Computing temporal consistency metrics...")
    temporal = compute_temporal_metrics(pred_frames)

    # 3. Gaming-specific
    if verbose:
        print("  Computing gaming-specific metrics...")
    gaming = compute_gaming_metrics(pred_frames, gt_frames)

    # 4. Stratified by difficulty
    if verbose:
        print("  Computing stratified metrics...")
    stratified = compute_stratified_metrics(pred_frames, gt_frames)

    results = ComprehensiveBenchmarkResults(
        psnr=float(np.mean(psnr_values)),
        ssim=float(np.mean(ssim_values)),
        temporal=temporal,
        gaming=gaming,
        stratified=stratified,
        total_frames=n_frames,
    )

    if verbose:
        print(results.summary_table())

    return results
