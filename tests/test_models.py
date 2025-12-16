"""
Tests for VFI+SR models.

Run with: pytest tests/test_models.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.traditional.baselines import BicubicBaseline, LanczosBaseline, OpticalFlowVFI
from models.base import ModelInfo


class TestBicubicBaseline:
    """Tests for BicubicBaseline model."""

    @pytest.fixture
    def model(self):
        return BicubicBaseline()

    @pytest.fixture
    def sample_frame(self):
        """Create a sample RGB frame."""
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def test_model_info(self, model):
        """Test model info returns correct metadata."""
        info = model.info  # info is a property, not a method
        assert isinstance(info, ModelInfo)
        assert info.name == "Bicubic"
        assert info.type == "traditional"
        assert info.supports_sr is True
        assert info.requires_gpu is False

    def test_upscale_dimensions(self, model, sample_frame):
        """Test upscaling produces correct dimensions."""
        scale = 1.5
        result = model.upscale(sample_frame, scale=scale)

        expected_h = int(480 * scale)
        expected_w = int(640 * scale)

        assert result.shape == (expected_h, expected_w, 3)
        assert result.dtype == np.uint8

    def test_upscale_different_scales(self, model, sample_frame):
        """Test various upscaling factors."""
        for scale in [1.0, 1.333, 1.5, 2.0]:
            result = model.upscale(sample_frame, scale=scale)
            expected_h = int(480 * scale)
            expected_w = int(640 * scale)
            assert result.shape == (expected_h, expected_w, 3)

    def test_interpolate_produces_frames(self, model, sample_frame):
        """Test that interpolate produces frames (uses linear blending)."""
        result = model.interpolate(sample_frame, sample_frame, num_frames=2)
        assert len(result) == 2
        for frame in result:
            assert frame.shape == sample_frame.shape


class TestLanczosBaseline:
    """Tests for LanczosBaseline model."""

    @pytest.fixture
    def model(self):
        return LanczosBaseline()

    @pytest.fixture
    def sample_frame(self):
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def test_model_info(self, model):
        """Test model info returns correct metadata."""
        info = model.info  # info is a property, not a method
        assert info.name == "Lanczos"
        assert info.type == "traditional"
        assert info.supports_sr is True

    def test_upscale_quality_different_from_bicubic(self, model, sample_frame):
        """Test that Lanczos produces different results than Bicubic."""
        bicubic = BicubicBaseline()

        result_lanczos = model.upscale(sample_frame, scale=2.0)
        result_bicubic = bicubic.upscale(sample_frame, scale=2.0)

        # Results should be different (different interpolation algorithms)
        assert not np.array_equal(result_lanczos, result_bicubic)


class TestOpticalFlowVFI:
    """Tests for OpticalFlowVFI model."""

    @pytest.fixture
    def model(self):
        return OpticalFlowVFI()

    @pytest.fixture
    def sample_frames(self):
        """Create two consecutive sample frames with slight difference."""
        frame0 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame0[100:200, 100:200] = 255  # White square

        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1[110:210, 110:210] = 255  # Shifted white square

        return frame0, frame1

    def test_model_info(self, model):
        """Test model info returns correct metadata."""
        info = model.info  # info is a property, not a method
        assert info.name == "OpticalFlow_Farneback"
        assert info.supports_vfi is True
        assert info.supports_sr is True

    def test_interpolate_produces_frames(self, model, sample_frames):
        """Test that interpolation produces the expected number of frames."""
        frame0, frame1 = sample_frames

        result = model.interpolate(frame0, frame1, num_frames=3)

        assert len(result) == 3
        for frame in result:
            assert frame.shape == frame0.shape
            assert frame.dtype == np.uint8

    def test_interpolate_single_frame(self, model, sample_frames):
        """Test interpolation with single intermediate frame."""
        frame0, frame1 = sample_frames

        result = model.interpolate(frame0, frame1, num_frames=1)

        assert len(result) == 1
        assert result[0].shape == frame0.shape

    def test_upscale_and_interpolate(self, model, sample_frames):
        """Test combined upscaling and interpolation."""
        frame0, frame1 = sample_frames
        scale = 1.5

        # Upscale
        up0 = model.upscale(frame0, scale=scale)
        up1 = model.upscale(frame1, scale=scale)

        expected_h = int(480 * scale)
        expected_w = int(640 * scale)

        assert up0.shape == (expected_h, expected_w, 3)
        assert up1.shape == (expected_h, expected_w, 3)


class TestModelConsistency:
    """Tests for consistency across models."""

    @pytest.fixture
    def sample_frame(self):
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def test_all_models_return_uint8(self, sample_frame):
        """All models should return uint8 arrays."""
        models = [BicubicBaseline(), LanczosBaseline(), OpticalFlowVFI()]

        for model in models:
            result = model.upscale(sample_frame, scale=1.5)
            assert result.dtype == np.uint8, f"{model.info.name} returned wrong dtype"

    def test_all_models_preserve_channels(self, sample_frame):
        """All models should preserve RGB channels."""
        models = [BicubicBaseline(), LanczosBaseline(), OpticalFlowVFI()]

        for model in models:
            result = model.upscale(sample_frame, scale=1.5)
            assert result.shape[2] == 3, f"{model.info.name} changed channel count"

    def test_scale_one_preserves_size(self, sample_frame):
        """Scale of 1.0 should preserve dimensions."""
        models = [BicubicBaseline(), LanczosBaseline(), OpticalFlowVFI()]

        for model in models:
            result = model.upscale(sample_frame, scale=1.0)
            assert result.shape == sample_frame.shape, f"{model.info.name} changed size at scale=1.0"
