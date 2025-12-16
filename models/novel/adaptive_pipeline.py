"""
models/novel/adaptive_pipeline.py

NOVEL: Adaptive Content-Aware VFI+SR Pipeline

This is the key innovation of the project.

Idea: Different content types need different models:
- Easy frames (low motion, simple content): Use fast RIFE
- Hard frames (high motion, particles, explosions): Use quality-focused VFIMamba
- HUD/UI regions: Don't interpolate at all (copy from nearest frame)
- Scene changes: Don't interpolate (would create artifacts)

This adaptive routing gives us the best of both worlds:
- Fast processing for easy content (most frames)
- High quality for hard content (when it matters)
- Special handling for gaming-specific challenges
"""

import cv2
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque

from ..base import BaseModel, ModelInfo, InferenceResult


@dataclass
class ContentAnalysis:
    """Analysis of frame pair content"""
    motion_mean: float
    motion_max: float
    motion_std: float
    has_particles: bool
    is_scene_change: bool
    hud_coverage: float
    recommended_model: str
    confidence: float


@dataclass
class RoutingStats:
    """Statistics on routing decisions"""
    total: int = 0
    rife_count: int = 0
    vfimamba_count: int = 0
    scene_change_count: int = 0
    
    def add(self, model: str):
        self.total += 1
        if model == 'rife':
            self.rife_count += 1
        elif model == 'vfimamba':
            self.vfimamba_count += 1
        elif model == 'scene_change':
            self.scene_change_count += 1
    
    def to_dict(self) -> dict:
        if self.total == 0:
            return {'total': 0}
        return {
            'total': self.total,
            'rife': self.rife_count,
            'rife_pct': self.rife_count / self.total * 100,
            'vfimamba': self.vfimamba_count,
            'vfimamba_pct': self.vfimamba_count / self.total * 100,
            'scene_change': self.scene_change_count,
            'scene_change_pct': self.scene_change_count / self.total * 100,
        }


class AdaptiveRouter:
    """
    Analyzes frame content and decides which model to use.
    
    This is the "brain" of the adaptive pipeline.
    """
    
    def __init__(
        self,
        # Motion thresholds (in pixels of average optical flow magnitude)
        motion_threshold_low: float = 5.0,   # Below this: easy content, use RIFE
        motion_threshold_high: float = 25.0,  # Above this: hard content, use VFIMamba
        
        # Scene change detection
        scene_change_threshold: float = 0.65,  # SSIM below this = scene cut
        
        # Particle detection
        particle_threshold: float = 0.4,  # Combined score above this = particles
        
        # HUD detection
        hud_variance_threshold: float = 10.0,  # Pixel variance below this = static (HUD)
        hud_history_frames: int = 10,  # How many frames to use for HUD detection
    ):
        self.motion_threshold_low = motion_threshold_low
        self.motion_threshold_high = motion_threshold_high
        self.scene_change_threshold = scene_change_threshold
        self.particle_threshold = particle_threshold
        self.hud_variance_threshold = hud_variance_threshold
        self.hud_history_frames = hud_history_frames
        
        # Frame history for HUD detection
        self.frame_history: deque = deque(maxlen=hud_history_frames)
        self.hud_mask: Optional[np.ndarray] = None
    
    def compute_motion(self, frame0: np.ndarray, frame1: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
        """
        Compute optical flow and motion statistics.
        
        Returns: (mean_magnitude, max_magnitude, std_magnitude, flow_magnitude_map)
        """
        # Convert to grayscale
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        
        # Compute optical flow using Farneback
        flow = cv2.calcOpticalFlowFarneback(
            gray0, gray1, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Compute magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        return (
            float(np.mean(magnitude)),
            float(np.max(magnitude)),
            float(np.std(magnitude)),
            magnitude
        )
    
    def detect_scene_change(self, frame0: np.ndarray, frame1: np.ndarray) -> Tuple[bool, float]:
        """
        Detect scene cuts using SSIM.
        
        Returns: (is_scene_change, ssim_score)
        """
        from skimage.metrics import structural_similarity as ssim
        
        # Convert to grayscale
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        
        # Downsample for speed (scene detection doesn't need full resolution)
        scale = 0.25
        small0 = cv2.resize(gray0, None, fx=scale, fy=scale)
        small1 = cv2.resize(gray1, None, fx=scale, fy=scale)
        
        # Compute SSIM
        score = ssim(small0, small1)
        
        return score < self.scene_change_threshold, score
    
    def detect_particles(
        self,
        frame0: np.ndarray,
        flow_magnitude: np.ndarray,
        flow_std: float
    ) -> Tuple[bool, float]:
        """
        Detect particle effects (explosions, fire, smoke).
        
        Particle effects are characterized by:
        1. High local variance in optical flow (chaotic motion)
        2. High-frequency content (small bright spots)
        
        Returns: (has_particles, particle_score)
        """
        # Normalize flow std by expected range
        flow_score = min(flow_std / 20.0, 1.0)
        
        # High-frequency content detection using Laplacian
        gray = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # Normalize by expected range
        freq_score = min(laplacian_var / 500.0, 1.0)
        
        # Combined score (geometric mean)
        particle_score = np.sqrt(flow_score * freq_score)
        
        return particle_score > self.particle_threshold, particle_score
    
    def detect_hud(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect HUD/UI regions by finding areas that don't change across frames.
        
        Gaming HUDs are typically static overlays that should NOT be interpolated.
        
        Returns: (hud_mask, hud_coverage_percentage)
        """
        # Add frame to history
        # Store as smaller grayscale for efficiency
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (320, 180))  # Low-res for analysis
        self.frame_history.append(small)
        
        if len(self.frame_history) < 5:
            # Not enough history yet
            return np.zeros(frame.shape[:2], dtype=bool), 0.0
        
        # Stack recent frames
        frames_array = np.array(list(self.frame_history)[-5:])
        
        # Compute variance across frames
        variance = np.var(frames_array, axis=0)
        
        # Low variance = likely HUD (static content)
        hud_mask_small = variance < self.hud_variance_threshold
        
        # Upscale mask to original resolution
        hud_mask = cv2.resize(
            hud_mask_small.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        hud_mask_clean = cv2.morphologyEx(hud_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        hud_mask_clean = cv2.morphologyEx(hud_mask_clean, cv2.MORPH_OPEN, kernel)
        
        self.hud_mask = hud_mask_clean.astype(bool)
        hud_coverage = float(self.hud_mask.mean())
        
        return self.hud_mask, hud_coverage
    
    def analyze(self, frame0: np.ndarray, frame1: np.ndarray) -> ContentAnalysis:
        """
        Analyze frame pair and return routing decision.
        """
        # 1. Scene change detection (fast, do first)
        is_scene_change, ssim_score = self.detect_scene_change(frame0, frame1)
        
        if is_scene_change:
            return ContentAnalysis(
                motion_mean=0,
                motion_max=0,
                motion_std=0,
                has_particles=False,
                is_scene_change=True,
                hud_coverage=0,
                recommended_model='scene_change',
                confidence=1.0 - ssim_score,  # Lower SSIM = higher confidence of scene change
            )
        
        # 2. Motion analysis
        motion_mean, motion_max, motion_std, flow_mag = self.compute_motion(frame0, frame1)
        
        # 3. Particle detection
        has_particles, particle_score = self.detect_particles(frame0, flow_mag, motion_std)
        
        # 4. HUD detection
        hud_mask, hud_coverage = self.detect_hud(frame0)
        
        # 5. Routing decision
        if has_particles or motion_max > self.motion_threshold_high:
            recommended_model = 'vfimamba'
            confidence = min(particle_score + motion_max / 50.0, 1.0)
        elif motion_mean < self.motion_threshold_low:
            recommended_model = 'rife'
            confidence = 1.0 - motion_mean / self.motion_threshold_low
        else:
            # Medium motion - RIFE can still handle it well
            recommended_model = 'rife'
            confidence = 0.7
        
        return ContentAnalysis(
            motion_mean=motion_mean,
            motion_max=motion_max,
            motion_std=motion_std,
            has_particles=has_particles,
            is_scene_change=False,
            hud_coverage=hud_coverage,
            recommended_model=recommended_model,
            confidence=confidence,
        )


class AdaptivePipeline(BaseModel):
    """
    NOVEL: Adaptive VFI+SR Pipeline
    
    Routes frames to RIFE (fast) or VFIMamba (quality) based on content analysis.
    Handles HUD regions specially (no interpolation).
    Detects scene changes (no interpolation).
    
    This is the main contribution of the project.
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        # Router parameters
        motion_threshold_low: float = 5.0,
        motion_threshold_high: float = 25.0,
        # Whether to actually use VFIMamba (set False to only use RIFE for speed testing)
        enable_vfimamba: bool = True,
        # SR model to use
        sr_model_name: str = 'lanczos',  # 'lanczos', 'span', etc.
    ):
        super().__init__(device)
        
        self.enable_vfimamba = enable_vfimamba
        self.sr_model_name = sr_model_name
        
        # Initialize router
        self.router = AdaptiveRouter(
            motion_threshold_low=motion_threshold_low,
            motion_threshold_high=motion_threshold_high,
        )
        
        # Statistics
        self.stats = RoutingStats()
        
        # Models will be loaded lazily
        self._rife = None
        self._vfimamba = None
        self._sr = None
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name='AdaptivePipeline',
            type='novel',
            supports_vfi=True,
            supports_sr=True,
            supports_joint=False,
            parameters=27_700_000,  # RIFE + VFIMamba + SR
            requires_gpu=True,
            description="Novel adaptive routing: fast RIFE for easy content, quality VFIMamba for hard content",
        )
    
    def load(self) -> None:
        """Load all sub-models"""
        print("Loading Adaptive Pipeline...")
        
        # Load RIFE
        from .rife_wrapper import RIFEModel
        self._rife = RIFEModel(device=str(self.device))
        self._rife.load()
        
        # Load VFIMamba if enabled
        if self.enable_vfimamba:
            try:
                from .vfimamba_wrapper import VFIMambaModel
                self._vfimamba = VFIMambaModel(device=str(self.device))
                self._vfimamba.load()
            except Exception as e:
                print(f"Warning: Could not load VFIMamba: {e}")
                print("Will use RIFE for all frames")
                self.enable_vfimamba = False
        
        # For SR, we'll use Lanczos by default (fast, good quality)
        # Can be extended to use SPAN or other SR models
        
        self._loaded = True
        print("Adaptive Pipeline ready")
    
    def interpolate(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_frames: int = 3,
        timestamps: Optional[List[float]] = None,
    ) -> List[np.ndarray]:
        """
        Adaptive interpolation based on content analysis.
        """
        self.ensure_loaded()
        
        # Analyze content
        analysis = self.router.analyze(frame0, frame1)
        
        # Handle scene change
        if analysis.is_scene_change:
            self.stats.add('scene_change')
            # Return copies of first frame (or could do crossfade)
            return [frame0.copy() for _ in range(num_frames)]
        
        # Select model based on analysis
        if analysis.recommended_model == 'vfimamba' and self.enable_vfimamba:
            self.stats.add('vfimamba')
            interpolated = self._vfimamba.interpolate(frame0, frame1, num_frames, timestamps)
        else:
            self.stats.add('rife')
            interpolated = self._rife.interpolate(frame0, frame1, num_frames, timestamps)
        
        # Handle HUD regions (copy from nearest input frame)
        if analysis.hud_coverage > 0.01:  # More than 1% HUD
            hud_mask = self.router.hud_mask
            if hud_mask is not None:
                if timestamps is None:
                    timestamps = self._rife.get_default_timestamps(num_frames)
                
                for i, frame in enumerate(interpolated):
                    t = timestamps[i]
                    # Copy HUD from nearest input frame
                    source = frame0 if t < 0.5 else frame1
                    frame[hud_mask] = source[hud_mask]
        
        return interpolated
    
    def upscale(self, frame: np.ndarray, scale: float = 1.333) -> np.ndarray:
        """
        Super-resolution.
        
        Currently uses Lanczos for simplicity.
        Can be extended to use SPAN or other SR models.
        """
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    def process_pair(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_intermediate: int = 3,
        target_scale: float = 1.333,
    ) -> InferenceResult:
        """
        Full adaptive pipeline with timing.
        """
        self.ensure_loaded()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        import time
        start = time.perf_counter()
        
        # Get content analysis for extra_info
        analysis = self.router.analyze(frame0, frame1)
        
        # Interpolate (adaptive)
        interpolated = self.interpolate(frame0, frame1, num_intermediate)
        
        # Upscale all frames
        all_frames = [frame0] + interpolated + [frame1]
        upscaled = [self.upscale(f, target_scale) for f in all_frames]
        
        end = time.perf_counter()
        
        if torch.cuda.is_available():
            vram_peak = torch.cuda.max_memory_allocated() / 1e6
        else:
            vram_peak = 0
        
        return InferenceResult(
            frames=upscaled,
            inference_time_ms=(end - start) * 1000,
            vram_peak_mb=vram_peak,
            model_used=self.info.name,
            extra_info={
                'analysis': {
                    'motion_mean': analysis.motion_mean,
                    'motion_max': analysis.motion_max,
                    'has_particles': analysis.has_particles,
                    'is_scene_change': analysis.is_scene_change,
                    'hud_coverage': analysis.hud_coverage,
                    'recommended_model': analysis.recommended_model,
                },
                'routing_stats': self.stats.to_dict(),
            },
        )
    
    def get_stats(self) -> dict:
        """Get routing statistics"""
        return self.stats.to_dict()
    
    def reset_stats(self) -> None:
        """Reset routing statistics"""
        self.stats = RoutingStats()
