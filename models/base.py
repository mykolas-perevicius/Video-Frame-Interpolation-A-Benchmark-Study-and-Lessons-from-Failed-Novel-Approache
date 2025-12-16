"""
models/base.py

Abstract base class for all VFI+SR models.
Ensures consistent interface across traditional, SOTA, and novel methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time

import numpy as np
import torch


@dataclass
class ModelInfo:
    """Model metadata"""
    name: str
    type: str  # 'traditional', 'sota', 'novel'
    supports_vfi: bool
    supports_sr: bool
    supports_joint: bool  # Does both VFI+SR in one pass
    parameters: Optional[int] = None
    requires_gpu: bool = True
    description: str = ""


@dataclass 
class InferenceResult:
    """Result from model inference"""
    frames: List[np.ndarray]  # Output frames (H, W, C) uint8 RGB
    inference_time_ms: float
    vram_peak_mb: float
    model_used: str = ""
    extra_info: dict = None
    
    def __post_init__(self):
        if self.extra_info is None:
            self.extra_info = {}


class BaseModel(ABC):
    """
    Abstract base class for all upscaling models.
    
    All models must implement:
    - info: Return model metadata
    - load: Load model weights
    - interpolate: Generate intermediate frames between two input frames
    - upscale: Super-resolve a single frame
    
    The process_pair method provides the full pipeline.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._model = None
        self._loaded = False
    
    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        """Return model information"""
        pass
    
    @abstractmethod
    def load(self) -> None:
        """Load model weights. Called once before inference."""
        pass
    
    @abstractmethod
    def interpolate(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_frames: int = 3,
        timestamps: Optional[List[float]] = None,
    ) -> List[np.ndarray]:
        """
        Interpolate between two frames.
        
        Args:
            frame0: First frame (H, W, C) uint8 RGB
            frame1: Second frame (H, W, C) uint8 RGB
            num_frames: Number of intermediate frames to generate
            timestamps: Optional specific timestamps [0-1] for each frame
                       If None, uses evenly spaced timestamps
            
        Returns:
            List of interpolated frames (num_frames items)
        """
        pass
    
    @abstractmethod
    def upscale(
        self,
        frame: np.ndarray,
        scale: float = 1.333,  # 1080p -> 1440p
    ) -> np.ndarray:
        """
        Upscale a single frame.
        
        Args:
            frame: Input frame (H, W, C) uint8 RGB
            scale: Scale factor (e.g., 1.333 for 1080p->1440p)
            
        Returns:
            Upscaled frame
        """
        pass
    
    def process_pair(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_intermediate: int = 3,
        target_scale: float = 1.333,
    ) -> InferenceResult:
        """
        Full pipeline: interpolate then upscale.
        
        This is the main entry point for benchmarking.
        Returns all frames: [upscaled_frame0, upscaled_interp_1, ..., upscaled_interp_n, upscaled_frame1]
        
        Args:
            frame0: First input frame (H, W, C) uint8 RGB
            frame1: Second input frame (H, W, C) uint8 RGB
            num_intermediate: Number of intermediate frames to generate
            target_scale: Spatial upscaling factor
            
        Returns:
            InferenceResult with all processed frames and timing info
        """
        if not self._loaded:
            raise RuntimeError(f"Model {self.info.name} not loaded. Call load() first.")
        
        # Reset VRAM stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start = time.perf_counter()
        
        # Interpolate
        interpolated = self.interpolate(frame0, frame1, num_intermediate)
        
        # Upscale all frames (endpoints + intermediates)
        all_frames = [frame0] + interpolated + [frame1]
        upscaled = [self.upscale(f, target_scale) for f in all_frames]
        
        end = time.perf_counter()
        
        # Get VRAM usage
        if torch.cuda.is_available():
            vram_peak = torch.cuda.max_memory_allocated() / 1e6
        else:
            vram_peak = 0
        
        return InferenceResult(
            frames=upscaled,
            inference_time_ms=(end - start) * 1000,
            vram_peak_mb=vram_peak,
            model_used=self.info.name,
        )
    
    def ensure_loaded(self) -> None:
        """Ensure model is loaded"""
        if not self._loaded:
            self.load()
            self._loaded = True
    
    # ==================== Utility Methods ====================
    
    def to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert numpy frame to model input tensor.
        
        Args:
            frame: (H, W, C) uint8 RGB numpy array
            
        Returns:
            (1, C, H, W) float32 tensor in [0, 1], on self.device
        """
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert model output tensor to numpy frame.
        
        Args:
            tensor: (1, C, H, W) or (C, H, W) float tensor in [0, 1]
            
        Returns:
            (H, W, C) uint8 RGB numpy array
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        frame = tensor.permute(1, 2, 0).cpu().numpy()
        return (frame * 255).clip(0, 255).astype(np.uint8)
    
    def pad_to_multiple(self, tensor: torch.Tensor, multiple: int = 32) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """
        Pad tensor to be divisible by multiple.
        
        Many models require input dimensions to be multiples of 32 or 64.
        
        Args:
            tensor: (N, C, H, W) tensor
            multiple: Pad to this multiple (default 32)
            
        Returns:
            Padded tensor and padding tuple (left, right, top, bottom)
        """
        _, _, h, w = tensor.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        
        if pad_h == 0 and pad_w == 0:
            return tensor, (0, 0, 0, 0)
        
        padding = (0, pad_w, 0, pad_h)  # left, right, top, bottom
        padded = torch.nn.functional.pad(tensor, padding, mode='reflect')
        return padded, padding
    
    def unpad(self, tensor: torch.Tensor, original_h: int, original_w: int) -> torch.Tensor:
        """Remove padding to restore original dimensions"""
        return tensor[:, :, :original_h, :original_w]
    
    def get_default_timestamps(self, num_frames: int) -> List[float]:
        """Generate evenly spaced timestamps between 0 and 1 (exclusive)"""
        return [(i + 1) / (num_frames + 1) for i in range(num_frames)]


class JointModel(BaseModel):
    """
    Base class for joint VFI+SR models (like SAFA).
    
    These models perform both temporal interpolation and spatial upscaling
    in a single forward pass, which is more efficient than cascaded approaches.
    """
    
    @abstractmethod
    def joint_process(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_intermediate: int = 3,
        target_scale: float = 1.333,
    ) -> List[np.ndarray]:
        """
        Joint interpolation and upscaling in one pass.
        
        Returns all frames at target resolution:
        [frame0_upscaled, interp_1_upscaled, ..., frame1_upscaled]
        """
        pass
    
    def process_pair(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_intermediate: int = 3,
        target_scale: float = 1.333,
    ) -> InferenceResult:
        """Override to use joint processing"""
        if not self._loaded:
            raise RuntimeError(f"Model {self.info.name} not loaded. Call load() first.")
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start = time.perf_counter()
        upscaled = self.joint_process(frame0, frame1, num_intermediate, target_scale)
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
        )


class TwoStageModel(BaseModel):
    """
    Combines separate VFI and SR models into a two-stage pipeline.
    
    Useful for testing different VFI+SR combinations.
    """
    
    def __init__(self, vfi_model: BaseModel, sr_model: BaseModel):
        # Don't call super().__init__() as we manage our own state
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
            supports_joint=False,
            parameters=vfi_params + sr_params,
            requires_gpu=self.vfi.info.requires_gpu or self.sr.info.requires_gpu,
            description=f"Two-stage: {self.vfi.info.name} for VFI, {self.sr.info.name} for SR",
        )
    
    def load(self) -> None:
        """Load both sub-models"""
        print(f"Loading two-stage model: {self.info.name}")
        self.vfi.load()
        self.sr.load()
        self._loaded = True
    
    def interpolate(self, frame0, frame1, num_frames=3, timestamps=None):
        """Delegate to VFI model"""
        return self.vfi.interpolate(frame0, frame1, num_frames, timestamps)
    
    def upscale(self, frame, scale=1.333):
        """Delegate to SR model"""
        return self.sr.upscale(frame, scale)
    
    def process_pair(self, frame0, frame1, num_intermediate=3, target_scale=1.333):
        """Time the complete two-stage pipeline"""
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start = time.perf_counter()
        
        # VFI stage
        interpolated = self.vfi.interpolate(frame0, frame1, num_intermediate)
        
        # SR stage
        all_frames = [frame0] + interpolated + [frame1]
        upscaled = [self.sr.upscale(f, target_scale) for f in all_frames]
        
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
            extra_info={'vfi_model': self.vfi.info.name, 'sr_model': self.sr.info.name},
        )
