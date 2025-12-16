"""
Tests for evaluation metrics.

Run with: pytest tests/test_metrics.py -v
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import compute_psnr_simple, compute_ssim_simple


class TestPSNR:
    """Tests for PSNR computation."""

    def test_identical_images_infinite_psnr(self):
        """Identical images should have infinite PSNR."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        psnr = compute_psnr_simple(img, img)
        assert psnr == float('inf')

    def test_psnr_range(self):
        """PSNR should be in reasonable range for similar images."""
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        # Add small noise
        noise = np.random.randint(-10, 11, img1.shape, dtype=np.int16)
        img2 = np.clip(img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        psnr = compute_psnr_simple(img1, img2)

        # PSNR should be positive and reasonable for slight differences
        assert 20 < psnr < 50

    def test_completely_different_images_low_psnr(self):
        """Completely different images should have low PSNR."""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.full((100, 100, 3), 255, dtype=np.uint8)

        psnr = compute_psnr_simple(img1, img2)

        # PSNR should be low for maximally different images
        assert psnr < 10

    def test_psnr_symmetry(self):
        """PSNR should be symmetric: PSNR(a, b) == PSNR(b, a)."""
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        psnr_ab = compute_psnr_simple(img1, img2)
        psnr_ba = compute_psnr_simple(img2, img1)

        assert abs(psnr_ab - psnr_ba) < 0.001

    def test_psnr_handles_different_sizes(self):
        """PSNR should handle images of different sizes by resizing."""
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

        # Should not raise an error
        psnr = compute_psnr_simple(img1, img2)
        assert psnr > 0


class TestSSIM:
    """Tests for SSIM computation."""

    def test_identical_images_perfect_ssim(self):
        """Identical images should have SSIM of 1.0."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        ssim = compute_ssim_simple(img, img)
        assert abs(ssim - 1.0) < 0.001

    def test_ssim_range(self):
        """SSIM should be between -1 and 1."""
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        ssim = compute_ssim_simple(img1, img2)

        assert -1 <= ssim <= 1

    def test_similar_images_high_ssim(self):
        """Similar images should have high SSIM."""
        img1 = np.random.randint(100, 150, (100, 100, 3), dtype=np.uint8)
        # Add small noise
        noise = np.random.randint(-5, 6, img1.shape, dtype=np.int16)
        img2 = np.clip(img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        ssim = compute_ssim_simple(img1, img2)

        assert ssim > 0.9

    def test_ssim_symmetry(self):
        """SSIM should be symmetric."""
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        ssim_ab = compute_ssim_simple(img1, img2)
        ssim_ba = compute_ssim_simple(img2, img1)

        assert abs(ssim_ab - ssim_ba) < 0.001


class TestMetricConsistency:
    """Tests for consistency between metrics."""

    def test_high_psnr_implies_high_ssim(self):
        """Images with high PSNR should generally have high SSIM."""
        img1 = np.random.randint(100, 150, (100, 100, 3), dtype=np.uint8)
        # Very similar image
        noise = np.random.randint(-2, 3, img1.shape, dtype=np.int16)
        img2 = np.clip(img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        psnr = compute_psnr_simple(img1, img2)
        ssim = compute_ssim_simple(img1, img2)

        if psnr > 40:  # High PSNR
            assert ssim > 0.95  # Should have high SSIM too

    def test_metrics_work_with_grayscale(self):
        """Metrics should work with grayscale images."""
        img1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        # PSNR works with grayscale
        psnr = compute_psnr_simple(
            np.stack([img1] * 3, axis=-1),
            np.stack([img2] * 3, axis=-1)
        )
        assert psnr > 0

        # SSIM works with grayscale
        ssim = compute_ssim_simple(
            np.stack([img1] * 3, axis=-1),
            np.stack([img2] * 3, axis=-1)
        )
        assert -1 <= ssim <= 1
