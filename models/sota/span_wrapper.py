#!/usr/bin/env python3
"""
span_wrapper.py

SPAN wrapper - NTIRE 2024 winner for real-time SR
Swift Parameter-free Attention Network

Repository: https://github.com/hongyuanyu/SPAN
"""

import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

# Add parent to path for base import
sys.path.insert(0, str(Path(__file__).parent.parent))
from base import BaseModel, ModelInfo


class SPANModel(BaseModel):
    """
    SPAN - Swift Parameter-free Attention Network
    
    NTIRE 2024 Efficient SR Challenge winner
    Key features:
    - Parameter-free attention (no learned attention weights)
    - Very fast inference (~21 fps @ 480p)
    - Only ~400K parameters
    - 2x and 4x upscaling variants
    
    Note: Requires SPAN repository to be cloned to external/SPAN
    """
    
    def __init__(self, scale: int = 2, device: str = 'cuda'):
        """
        Args:
            scale: Upscaling factor (2 or 4)
            device: 'cuda' or 'cpu'
        """
        super().__init__(device)
        self.scale = scale
        self.span_path = Path(__file__).parent.parent.parent / 'external' / 'SPAN'
        self._loaded = False
        
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f'SPAN_x{self.scale}',
            type='sota',
            supports_vfi=False,
            supports_sr=True,
            supports_joint=False,
            parameters=400_000,
            requires_gpu=True,
            description='Swift Parameter-free Attention Network for SR'
        )
    
    def load(self) -> None:
        """Load SPAN model"""
        if self._loaded:
            return
            
        if not self.span_path.exists():
            raise RuntimeError(
                f"SPAN not found at {self.span_path}\n"
                "Clone it with: git clone https://github.com/hongyuanyu/SPAN external/SPAN"
            )
        
        sys.path.insert(0, str(self.span_path))
        
        try:
            # Import SPAN - adjust based on actual repo structure
            from models.span import SPAN as SPANNet
            
            self._model = SPANNet(
                num_in_ch=3,
                num_out_ch=3,
                upscale=self.scale
            )
            
            # Load weights
            weights_path = self.span_path / 'weights' / f'SPAN_x{self.scale}.pth'
            
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location=self.device)
                self._model.load_state_dict(state_dict)
            else:
                print(f"Warning: Weights not found at {weights_path}")
                print("Download from: https://github.com/hongyuanyu/SPAN/releases")
            
            self._model = self._model.to(self.device).eval()
            self._loaded = True
            print(f"Loaded SPAN x{self.scale}")
            
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import SPAN: {e}\n"
                "Make sure SPAN is properly installed."
            )
    
    def interpolate(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_frames: int = 3,
        timestamps: Optional[List[float]] = None,
    ) -> List[np.ndarray]:
        """
        SPAN doesn't do frame interpolation.
        Falls back to simple blending.
        """
        if timestamps is None:
            timestamps = [(i + 1) / (num_frames + 1) for i in range(num_frames)]
        
        # Simple linear blend (not real interpolation)
        return [
            cv2.addWeighted(frame0, 1 - t, frame1, t, 0)
            for t in timestamps
        ]
    
    def upscale(self, frame: np.ndarray, scale: float = 1.333) -> np.ndarray:
        """
        Super-resolve a frame using SPAN.
        
        Args:
            frame: Input frame (H, W, C) uint8
            scale: Target scale factor
            
        Returns:
            Upscaled frame
        """
        if not self._loaded:
            self.load()
        
        # SPAN is trained for fixed scales (2x or 4x)
        # For non-matching scales, we upscale then resize
        tensor = self.to_tensor(frame)
        
        with torch.no_grad():
            upscaled = self._model(tensor)
        
        result = self.to_numpy(upscaled)
        
        # Resize to exact target if needed
        h, w = frame.shape[:2]
        target_h, target_w = int(h * scale), int(w * scale)
        
        if result.shape[:2] != (target_h, target_w):
            result = cv2.resize(result, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        return result


class TwoStageModel(BaseModel):
    """
    Combine separate VFI and SR models into a two-stage pipeline.
    
    This allows mixing and matching different VFI and SR models,
    e.g., RIFE + SPAN, VFIMamba + SPAN, etc.
    """
    
    def __init__(self, vfi_model: BaseModel, sr_model: BaseModel):
        """
        Args:
            vfi_model: Model for frame interpolation
            sr_model: Model for super resolution
        """
        device = vfi_model.device if hasattr(vfi_model, 'device') else 'cuda'
        super().__init__(device)
        
        self.vfi = vfi_model
        self.sr = sr_model
        self._loaded = False
    
    @property
    def info(self) -> ModelInfo:
        vfi_params = self.vfi.info.parameters or 0
        sr_params = self.sr.info.parameters or 0
        
        return ModelInfo(
            name=f'{self.vfi.info.name}+{self.sr.info.name}',
            type='sota',
            supports_vfi=True,
            supports_sr=True,
            supports_joint=False,  # Two-stage, not joint
            parameters=vfi_params + sr_params,
            requires_gpu=self.vfi.info.requires_gpu or self.sr.info.requires_gpu,
            description=f'Two-stage: {self.vfi.info.name} for VFI, {self.sr.info.name} for SR'
        )
    
    def load(self) -> None:
        """Load both models"""
        if self._loaded:
            return
        
        self.vfi.load()
        self.sr.load()
        self._loaded = True
        print(f"Loaded two-stage: {self.vfi.info.name} + {self.sr.info.name}")
    
    def interpolate(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_frames: int = 3,
        timestamps: Optional[List[float]] = None,
    ) -> List[np.ndarray]:
        """Use VFI model for interpolation"""
        if not self._loaded:
            self.load()
        return self.vfi.interpolate(frame0, frame1, num_frames, timestamps)
    
    def upscale(self, frame: np.ndarray, scale: float = 1.333) -> np.ndarray:
        """Use SR model for upscaling"""
        if not self._loaded:
            self.load()
        return self.sr.upscale(frame, scale)
    
    def process_pair(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_intermediate: int = 3,
        target_scale: float = 1.333,
    ):
        """
        Full two-stage pipeline: VFI then SR.
        """
        import time
        from base import InferenceResult
        
        if not self._loaded:
            self.load()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start = time.perf_counter()
        
        # Stage 1: Interpolate
        interpolated = self.vfi.interpolate(frame0, frame1, num_intermediate)
        
        # Stage 2: Upscale all frames (including endpoints)
        all_frames = [frame0] + interpolated + [frame1]
        upscaled = [self.sr.upscale(f, target_scale) for f in all_frames]
        
        end = time.perf_counter()
        
        vram_peak = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        
        return InferenceResult(
            frames=upscaled,
            inference_time_ms=(end - start) * 1000,
            vram_peak_mb=vram_peak,
            model_used=self.info.name,
        )


# Convenience factory functions
def create_rife_span(device: str = 'cuda') -> TwoStageModel:
    """Create RIFE + SPAN two-stage model"""
    from rife_wrapper import RIFEModel
    return TwoStageModel(
        vfi_model=RIFEModel(device=device),
        sr_model=SPANModel(device=device),
    )


def create_vfimamba_span(device: str = 'cuda') -> TwoStageModel:
    """Create VFIMamba + SPAN two-stage model"""
    from vfimamba_wrapper import VFIMambaModel
    return TwoStageModel(
        vfi_model=VFIMambaModel(device=device),
        sr_model=SPANModel(device=device),
    )
