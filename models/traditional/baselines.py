"""
models/traditional/baselines.py

Traditional (non-deep-learning) methods for baseline comparison.

These methods establish the lower bound of quality that AI methods should beat.
"""

import cv2
import numpy as np
from typing import List, Optional

from ..base import BaseModel, ModelInfo


class BicubicBaseline(BaseModel):
    """
    Bicubic interpolation baseline.
    
    For VFI: Simple linear blending (crossfade) between frames.
    For SR: Bicubic upsampling.
    
    This is the simplest possible baseline - any AI method should beat this.
    """
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name='Bicubic',
            type='traditional',
            supports_vfi=False,  # Only does blending, not true VFI
            supports_sr=True,
            supports_joint=False,
            parameters=0,
            requires_gpu=False,
            description="Bicubic interpolation - simplest baseline",
        )
    
    def load(self) -> None:
        """No model to load"""
        self._loaded = True
    
    def interpolate(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_frames: int = 3,
        timestamps: Optional[List[float]] = None,
    ) -> List[np.ndarray]:
        """
        Simple linear blending (crossfade) between frames.
        
        NOT true motion interpolation - just temporal blending.
        Will produce ghosting on moving objects.
        """
        if timestamps is None:
            timestamps = self.get_default_timestamps(num_frames)
        
        interpolated = []
        for t in timestamps:
            # Linear blend
            blended = cv2.addWeighted(
                frame0.astype(np.float32), 1 - t,
                frame1.astype(np.float32), t,
                0
            ).astype(np.uint8)
            interpolated.append(blended)
        
        return interpolated
    
    def upscale(self, frame: np.ndarray, scale: float = 1.333) -> np.ndarray:
        """Bicubic upscaling"""
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


class LanczosBaseline(BaseModel):
    """
    Lanczos interpolation baseline.
    
    Higher quality than bicubic for SR, but same limitations for VFI.
    """
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name='Lanczos',
            type='traditional',
            supports_vfi=False,
            supports_sr=True,
            supports_joint=False,
            parameters=0,
            requires_gpu=False,
            description="Lanczos interpolation - higher quality traditional SR",
        )
    
    def load(self) -> None:
        self._loaded = True
    
    def interpolate(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_frames: int = 3,
        timestamps: Optional[List[float]] = None,
    ) -> List[np.ndarray]:
        """Same as bicubic - just linear blending"""
        if timestamps is None:
            timestamps = self.get_default_timestamps(num_frames)
        
        return [
            cv2.addWeighted(
                frame0.astype(np.float32), 1 - t,
                frame1.astype(np.float32), t,
                0
            ).astype(np.uint8)
            for t in timestamps
        ]
    
    def upscale(self, frame: np.ndarray, scale: float = 1.333) -> np.ndarray:
        """Lanczos4 upscaling"""
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


class OpticalFlowVFI(BaseModel):
    """
    OpenCV optical flow-based frame interpolation.
    
    Traditional CV approach without deep learning.
    Uses Farneback optical flow to estimate motion, then warps frames.
    
    This represents the best traditional VFI before deep learning.
    """
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name='OpticalFlow_Farneback',
            type='traditional',
            supports_vfi=True,
            supports_sr=True,
            supports_joint=False,
            parameters=0,
            requires_gpu=False,
            description="OpenCV Farneback optical flow - traditional VFI",
        )
    
    def load(self) -> None:
        self._loaded = True
    
    def interpolate(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_frames: int = 3,
        timestamps: Optional[List[float]] = None,
    ) -> List[np.ndarray]:
        """
        Optical flow-based interpolation.
        
        1. Compute bidirectional optical flow
        2. For each timestamp, warp both frames toward the middle
        3. Blend the warped frames
        """
        if timestamps is None:
            timestamps = self.get_default_timestamps(num_frames)
        
        # Convert to grayscale for flow estimation
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        
        # Compute bidirectional optical flow
        # Forward: where do pixels in frame0 go in frame1?
        flow_forward = cv2.calcOpticalFlowFarneback(
            gray0, gray1, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Backward: where do pixels in frame1 come from in frame0?
        flow_backward = cv2.calcOpticalFlowFarneback(
            gray1, gray0, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        h, w = frame0.shape[:2]
        
        # Create coordinate grids
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        interpolated = []
        
        for t in timestamps:
            # Scale flows by timestamp
            # frame0 needs to move forward by t
            # frame1 needs to move backward by (1-t)
            flow_t_0 = flow_forward * t
            flow_t_1 = flow_backward * (1 - t)
            
            # Warp frame0 forward toward time t
            map_x_0 = grid_x + flow_t_0[..., 0]
            map_y_0 = grid_y + flow_t_0[..., 1]
            warped_0 = cv2.remap(
                frame0, map_x_0, map_y_0,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            
            # Warp frame1 backward toward time t
            map_x_1 = grid_x + flow_t_1[..., 0]
            map_y_1 = grid_y + flow_t_1[..., 1]
            warped_1 = cv2.remap(
                frame1, map_x_1, map_y_1,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            
            # Blend warped frames
            # Weight by distance from source frame
            blended = cv2.addWeighted(
                warped_0.astype(np.float32), 1 - t,
                warped_1.astype(np.float32), t,
                0
            ).astype(np.uint8)
            
            interpolated.append(blended)
        
        return interpolated
    
    def upscale(self, frame: np.ndarray, scale: float = 1.333) -> np.ndarray:
        """Use Lanczos for SR (best traditional quality)"""
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


# Convenience function to get all traditional baselines
def get_traditional_models() -> dict:
    """Return dictionary of all traditional baseline models"""
    return {
        'bicubic': BicubicBaseline,
        'lanczos': LanczosBaseline,
        'optical_flow': OpticalFlowVFI,
    }
