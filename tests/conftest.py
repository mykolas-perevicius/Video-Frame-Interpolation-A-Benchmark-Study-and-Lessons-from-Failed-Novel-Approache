"""
Pytest configuration and shared fixtures.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_rgb_frame():
    """Create a sample RGB frame for testing."""
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_pair():
    """Create a pair of similar frames for VFI testing."""
    frame0 = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
    # Create frame1 as a slightly shifted version
    frame1 = np.roll(frame0, shift=5, axis=1)
    return frame0, frame1


@pytest.fixture
def sample_gradient_frame():
    """Create a gradient frame (useful for testing interpolation)."""
    h, w = 480, 640
    x = np.linspace(0, 255, w, dtype=np.uint8)
    gradient = np.tile(x, (h, 1))
    return np.stack([gradient, gradient, gradient], axis=-1)


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root):
    """Return the data directory."""
    return project_root / "data"


@pytest.fixture
def processed_data_dir(project_root):
    """Return the processed data directory."""
    return project_root / "data" / "processed"
