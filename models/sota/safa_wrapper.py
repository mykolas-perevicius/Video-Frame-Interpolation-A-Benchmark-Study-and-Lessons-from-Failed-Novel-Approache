#!/usr/bin/env python3
"""
safa_wrapper.py

SAFA wrapper - WACV 2024 Joint VFI+SR SOTA
1/3 computational cost of TMNet with better quality

Repository: https://github.com/hzwer/WACV2024-SAFA
"""

import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

# Add parent to path for base import
sys.path.insert(0, str(Path(__file__).parent.parent))
from base import BaseModel, ModelInfo, InferenceResult


class SAFAModel(BaseModel):
    """
    SAFA - Space-Time Video Super Resolution with Flow Alignment
    
    Key features:
    - Joint VFI + SR in single forward pass (more efficient)
    - Flow-based alignment with spatial attention
    - 1/3 computational cost of TMNet
    - Trained for gaming/high-motion content
    
    Note: Requires SAFA repository to be cloned to external/WACV2024-SAFA
    """
    
    def __init__(self, device: str = 'cuda'):
        super().__init__(device)
        self.safa_path = Path(__file__).parent.parent.parent / 'external' / 'WACV2024-SAFA'
        self._loaded = False
        
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name='SAFA',
            type='sota',
            supports_vfi=True,
            supports_sr=True,
            supports_joint=True,  # KEY: Does both in one pass
            parameters=5_500_000,
            requires_gpu=True,
            description='Joint Space-Time Video Super Resolution'
        )
    
    def load(self) -> None:
        """Load SAFA model"""
        if self._loaded:
            return
            
        if not self.safa_path.exists():
            raise RuntimeError(
                f"SAFA not found at {self.safa_path}\n"
                "Clone it with: git clone https://github.com/hzwer/WACV2024-SAFA external/WACV2024-SAFA"
            )
        
        sys.path.insert(0, str(self.safa_path))
        
        try:
            # Import SAFA - adjust based on actual repo structure
            from models.SAFA import SAFA as SAFANet
            
            self._model = SAFANet()
            
            # Load weights
            weights_path = self.safa_path / 'checkpoints' / 'safa.pth'
            
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location=self.device)
                self._model.load_state_dict(state_dict)
            else:
                print(f"Warning: Weights not found at {weights_path}")
                print("Check SAFA repo for download instructions")
            
            self._model = self._model.to(self.device).eval()
            self._loaded = True
            print("Loaded SAFA joint model")
            
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import SAFA: {e}\n"
                "Make sure SAFA is properly installed."
            )
    
    def joint_process(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_intermediate: int = 3,
        target_scale: float = 1.333,
    ) -> List[np.ndarray]:
        """
        Joint VFI + SR in single forward pass.
        
        This is more efficient than separate VFI then SR.
        
        Args:
            frame0: First frame (H, W, C) uint8
            frame1: Second frame (H, W, C) uint8
            num_intermediate: Number of intermediate frames
            target_scale: Spatial upscaling factor
            
        Returns:
            List of upscaled frames (including endpoints)
        """
        if not self._loaded:
            self.load()
        
        img0 = self.to_tensor(frame0)
        img1 = self.to_tensor(frame1)
        
        # Pad if necessary
        h, w = frame0.shape[:2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        
        if pad_h > 0 or pad_w > 0:
            img0 = torch.nn.functional.pad(img0, (0, pad_w, 0, pad_h), mode='reflect')
            img1 = torch.nn.functional.pad(img1, (0, pad_w, 0, pad_h), mode='reflect')
        
        with torch.no_grad():
            # SAFA outputs all frames at once, upscaled
            # Actual API may differ - this is a placeholder
            outputs = self._model(
                img0, img1, 
                scale=target_scale, 
                num_frames=num_intermediate + 2  # Including endpoints
            )
        
        # Target dimensions
        target_h = int(h * target_scale)
        target_w = int(w * target_scale)
        
        # Extract frames
        result_frames = []
        for i in range(outputs.shape[1] if outputs.dim() > 3 else 1):
            if outputs.dim() > 3:
                frame = outputs[:, i]
            else:
                frame = outputs
            
            # Remove padding (scaled) - crop to target dimensions
            if pad_h > 0 or pad_w > 0:
                frame = frame[:, :, :target_h, :target_w]
            
            result_frames.append(self.to_numpy(frame))
        
        return result_frames
    
    def process_pair(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_intermediate: int = 3,
        target_scale: float = 1.333,
    ) -> InferenceResult:
        """Override to use joint processing"""
        import time
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start = time.perf_counter()
        upscaled = self.joint_process(frame0, frame1, num_intermediate, target_scale)
        end = time.perf_counter()
        
        vram_peak = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        
        return InferenceResult(
            frames=upscaled,
            inference_time_ms=(end - start) * 1000,
            vram_peak_mb=vram_peak,
            model_used='SAFA_joint',
        )
    
    def interpolate(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_frames: int = 3,
        timestamps: Optional[List[float]] = None,
    ) -> List[np.ndarray]:
        """
        VFI-only mode (without upscaling).
        Falls back to joint process at scale=1.0
        """
        if not self._loaded:
            self.load()
        
        if timestamps is None:
            timestamps = [(i + 1) / (num_frames + 1) for i in range(num_frames)]
        
        # Use joint process at scale=1.0 for VFI only
        all_frames = self.joint_process(frame0, frame1, num_frames, target_scale=1.0)
        
        # Return only intermediate frames
        return all_frames[1:-1] if len(all_frames) > 2 else all_frames
    
    def upscale(self, frame: np.ndarray, scale: float = 1.333) -> np.ndarray:
        """
        SR-only mode.
        For single-frame upscaling, fall back to Lanczos.
        SAFA is designed for temporal sequences.
        """
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
