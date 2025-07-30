"""
Pytest configuration and fixtures for photonic neuromorphics tests.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_spike_train():
    """Generate a sample spike train for testing."""
    return np.random.poisson(0.1, size=(100, 784))


@pytest.fixture
def sample_weights():
    """Generate sample synaptic weights."""
    return torch.randn(784, 256)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_optical_parameters():
    """Standard optical parameters for testing."""
    return {
        "wavelength": 1550e-9,
        "propagation_loss": 0.1,  # dB/cm
        "coupling_efficiency": 0.9,
        "detector_efficiency": 0.8,
    }


# Markers for different test categories
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
]