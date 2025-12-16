#!/usr/bin/env python3
"""
vfimamba_wrapper.py

VFIMamba wrapper - NeurIPS 2024 SOTA with linear complexity
Quality champion: 36.5+ dB PSNR on Vimeo90K

Repository: https://github.com/MCG-NJU/VFIMamba
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


class VFIMambaModel(BaseModel):
    """
    VFIMamba - State Space Model for Video Frame Interpolation
    
    Key features:
    - Linear complexity O(n) vs quadratic for transformers
    - Bidirectional Mamba blocks
    - Multi-scale feature extraction
    - SOTA quality on standard benchmarks
    
    Note: Requires VFIMamba repository to be cloned to external/VFIMamba
    and weights downloaded from their releases.
    """
    
    def __init__(self, variant: str = 'full', device: str = 'cuda'):
        """
        Args:
            variant: 'full' for best quality, 'S' for faster/smaller
            device: 'cuda' or 'cpu'
        """
        super().__init__(device)
        self.variant = variant
        self.mamba_path = Path(__file__).parent.parent.parent / 'external' / 'VFIMamba'
        self._loaded = False
        
    @property
    def info(self) -> ModelInfo:
        params = 17_000_000 if self.variant == 'full' else 8_000_000
        return ModelInfo(
            name=f'VFIMamba_{self.variant}',
            type='sota',
            supports_vfi=True,
            supports_sr=False,
            supports_joint=False,
            parameters=params,
            requires_gpu=True,
            description='State Space Model VFI with linear complexity'
        )
    
    def load(self) -> None:
        """Load VFIMamba model weights"""
        if self._loaded:
            return
            
        if not self.mamba_path.exists():
            raise RuntimeError(
                f"VFIMamba not found at {self.mamba_path}\n"
                "Clone it with: git clone https://github.com/MCG-NJU/VFIMamba external/VFIMamba"
            )
        
        # Add to path and import
        sys.path.insert(0, str(self.mamba_path))
        
        try:
            # Import depends on actual repo structure
            # This is a placeholder - adjust based on actual VFIMamba API
            from models.VFIMamba import VFIMamba as VFIMambaNet
            
            self._model = VFIMambaNet()
            
            # Load weights
            if self.variant == 'full':
                weights_path = self.mamba_path / 'checkpoints' / 'VFIMamba.pth'
            else:
                weights_path = self.mamba_path / 'checkpoints' / 'VFIMamba_S.pth'
            
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location=self.device)
                self._model.load_state_dict(state_dict)
            else:
                print(f"Warning: Weights not found at {weights_path}")
                print("Download from: https://github.com/MCG-NJU/VFIMamba/releases")
            
            self._model = self._model.to(self.device).eval()
            self._loaded = True
            print(f"Loaded VFIMamba {self.variant}")
            
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import VFIMamba: {e}\n"
                "Make sure VFIMamba is properly installed with its dependencies."
            )
    
    def interpolate(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_frames: int = 3,
        timestamps: Optional[List[float]] = None,
    ) -> List[np.ndarray]:
        """
        Interpolate between two frames using VFIMamba.
        
        Args:
            frame0: First frame (H, W, C) uint8 RGB
            frame1: Second frame (H, W, C) uint8 RGB
            num_frames: Number of intermediate frames
            timestamps: Optional specific timestamps [0-1]
            
        Returns:
            List of interpolated frames
        """
        if not self._loaded:
            self.load()
        
        if timestamps is None:
            timestamps = [(i + 1) / (num_frames + 1) for i in range(num_frames)]
        
        # Convert to tensors
        img0 = self.to_tensor(frame0)
        img1 = self.to_tensor(frame1)
        
        # Pad to multiple of 32 (common requirement)
        h, w = frame0.shape[:2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        
        if pad_h > 0 or pad_w > 0:
            img0 = torch.nn.functional.pad(img0, (0, pad_w, 0, pad_h), mode='reflect')
            img1 = torch.nn.functional.pad(img1, (0, pad_w, 0, pad_h), mode='reflect')
        
        interpolated = []
        
        with torch.no_grad():
            for t in timestamps:
                # VFIMamba inference
                # Actual API may differ - adjust based on repo
                mid = self._model(img0, img1, timestep=t)
                
                # Remove padding
                if pad_h > 0 or pad_w > 0:
                    mid = mid[:, :, :h, :w]
                
                interpolated.append(self.to_numpy(mid))
        
        return interpolated
    
    def upscale(self, frame: np.ndarray, scale: float = 1.333) -> np.ndarray:
        """
        VFIMamba doesn't do super resolution.
        Falls back to Lanczos for upscaling.
        """
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


class VFIMambaLite(VFIMambaModel):
    """Smaller/faster VFIMamba variant"""
    
    def __init__(self, device: str = 'cuda'):
        super().__init__(variant='S', device=device)
