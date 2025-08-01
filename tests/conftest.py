"""
Pytest configuration and fixtures for photonic neuromorphics tests.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
import os
from unittest.mock import Mock, MagicMock

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return TEST_DATA_DIR


@pytest.fixture
def sample_spike_train():
    """Generate a sample spike train for testing."""
    np.random.seed(42)  # Reproducible results
    return np.random.poisson(0.1, size=(100, 784))


@pytest.fixture
def sample_dense_spike_train():
    """Generate a dense spike train for stress testing."""
    np.random.seed(42)
    return np.random.poisson(0.5, size=(1000, 784))


@pytest.fixture
def sample_weights():
    """Generate sample synaptic weights."""
    torch.manual_seed(42)
    return torch.randn(784, 256)


@pytest.fixture
def sample_network_topology():
    """Standard network topology for testing."""
    return {
        "input_size": 784,
        "hidden_layers": [256, 128, 64],
        "output_size": 10,
        "neuron_type": "lif",
        "synapse_type": "static"
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def temp_layout_dir(tmp_path):
    """Create a temporary directory for layout files."""
    layout_dir = tmp_path / "layouts"
    layout_dir.mkdir()
    return layout_dir


@pytest.fixture
def temp_simulation_dir(tmp_path):
    """Create a temporary directory for simulation files."""
    sim_dir = tmp_path / "simulation"
    sim_dir.mkdir()
    return sim_dir


@pytest.fixture
def mock_optical_parameters():
    """Standard optical parameters for testing."""
    return {
        "wavelength": 1550e-9,
        "propagation_loss": 0.1,  # dB/cm
        "coupling_efficiency": 0.9,
        "detector_efficiency": 0.8,
        "waveguide_width": 450e-9,
        "waveguide_height": 220e-9,
        "coupling_gap": 200e-9,
        "bend_radius": 5e-6,
    }


@pytest.fixture
def mock_device_parameters():
    """Standard device parameters for testing."""
    return {
        "mzi_length": 100e-6,
        "phase_shifter_efficiency": 10e-6,  # rad/V
        "modulation_depth": 0.9,
        "extinction_ratio": 20,  # dB
        "insertion_loss": 1.0,  # dB
        "crosstalk": -30,  # dB
    }


@pytest.fixture
def mock_simulation_config():
    """Standard simulation configuration."""
    return {
        "timestep": 1e-12,  # 1 ps
        "duration": 1e-6,   # 1 μs
        "temperature": 300,  # K
        "solver": "runge_kutta",
        "tolerance": 1e-9,
        "max_iterations": 10000,
    }


@pytest.fixture
def mock_pdk_config():
    """Mock PDK configuration for testing."""
    return {
        "name": "TestPDK",
        "version": "1.0.0",
        "process": "test_220nm",
        "min_feature_size": 45e-9,
        "min_spacing": 100e-9,
        "metal_layers": 5,
        "via_size": 150e-9,
        "design_rules": {
            "min_width": 450e-9,
            "min_bend_radius": 5e-6,
            "min_coupling_gap": 150e-9,
        }
    }


@pytest.fixture
def mock_spice_simulator():
    """Mock SPICE simulator for testing."""
    simulator = Mock()
    simulator.run.return_value = {
        "time": np.linspace(0, 1e-6, 1000),
        "voltage": np.sin(2 * np.pi * 1e6 * np.linspace(0, 1e-6, 1000)),
        "current": np.cos(2 * np.pi * 1e6 * np.linspace(0, 1e-6, 1000)) * 1e-3,
    }
    return simulator


@pytest.fixture
def mock_optical_solver():
    """Mock optical solver for testing."""
    solver = Mock()
    solver.solve.return_value = {
        "transmission": np.random.random(1000) * 0.8 + 0.1,
        "phase": np.random.random(1000) * 2 * np.pi,
        "loss": np.random.random(1000) * 0.1,
    }
    return solver


@pytest.fixture
def sample_mnist_data():
    """Sample MNIST-like data for testing."""
    # Generate synthetic MNIST-like data
    images = np.random.rand(100, 28, 28)
    labels = np.random.randint(0, 10, 100)
    return images, labels


@pytest.fixture
def sample_gds_layout(temp_layout_dir):
    """Create a simple test GDS layout file."""
    gds_file = temp_layout_dir / "test_layout.gds"
    # Create a minimal GDS file (placeholder)
    with open(gds_file, 'wb') as f:
        f.write(b'GDS_HEADER_PLACEHOLDER')
    return gds_file


@pytest.fixture
def sample_spice_netlist(temp_simulation_dir):
    """Create a sample SPICE netlist for testing."""
    netlist_file = temp_simulation_dir / "test_circuit.sp"
    netlist_content = """
* Test SPICE Netlist for Photonic Neuromorphics
.title Test Photonic Circuit

* Voltage sources
Vdd vdd 0 DC 3.3V
Vss vss 0 DC 0V

* Photodetector model
Rpd pd_out 0 1k
Cpd pd_out 0 10p

* Simple RC circuit
R1 in pd_out 10k  
C1 pd_out 0 1p

.tran 1p 1n
.end
"""
    with open(netlist_file, 'w') as f:
        f.write(netlist_content)
    return netlist_file


@pytest.fixture
def sample_verilog_module(temp_simulation_dir):
    """Create a sample Verilog module for testing."""
    verilog_file = temp_simulation_dir / "test_module.v"
    verilog_content = """
// Test Verilog module for photonic neuromorphics
module photonic_neuron #(
    parameter THRESHOLD = 16'h1000
)(
    input wire clk,
    input wire rst_n,
    input wire [15:0] optical_input,
    output reg spike_out
);

    reg [15:0] membrane_potential;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            membrane_potential <= 16'h0000;
            spike_out <= 1'b0;
        end else begin
            membrane_potential <= membrane_potential + optical_input;
            if (membrane_potential >= THRESHOLD) begin
                spike_out <= 1'b1;
                membrane_potential <= 16'h0000;
            end else begin
                spike_out <= 1'b0;
            end
        end
    end

endmodule
"""
    with open(verilog_file, 'w') as f:
        f.write(verilog_content)
    return verilog_file


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup global test environment."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create test data directory if it doesn't exist
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables for testing
    os.environ["TESTING"] = "true"
    os.environ["PHOTONIC_WAVELENGTH"] = "1550e-9"
    os.environ["LOG_LEVEL"] = "WARNING"  # Reduce log noise during tests
    
    yield
    
    # Cleanup after all tests
    # Remove any global test artifacts if needed


@pytest.fixture
def performance_benchmark():
    """Fixture for performance benchmarking."""
    import time
    
    class BenchmarkContext:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def __enter__(self):
            self.start_time = time.perf_counter()
            return self
            
        def __exit__(self, *args):
            self.end_time = time.perf_counter()
            
        @property
        def elapsed_time(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return BenchmarkContext


# Custom markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "regression: mark test as regression test")
    config.addinivalue_line("markers", "contract: mark test as API contract test")
    config.addinivalue_line("markers", "security: mark test as security test")
    config.addinivalue_line("markers", "photonic: mark test as photonic-specific")
    config.addinivalue_line("markers", "simulation: mark test as simulation test")
    config.addinivalue_line("markers", "layout: mark test as layout test")


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "contract" in str(item.fspath):
            item.add_marker(pytest.mark.contract)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


# Global filters for warnings
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning:torch.*"),
    pytest.mark.filterwarnings("ignore::FutureWarning:numpy.*"),
]