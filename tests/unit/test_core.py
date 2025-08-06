"""
Comprehensive unit tests for core photonic neuromorphics functionality.
"""

import pytest
import numpy as np
import torch
import logging
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from photonic_neuromorphics.core import (
    PhotonicSNN, WaveguideNeuron, OpticalParameters, 
    create_mnist_photonic_snn, encode_to_spikes
)
from photonic_neuromorphics.exceptions import (
    ValidationError, NetworkTopologyError, OpticalModelError
)
from photonic_neuromorphics.monitoring import MetricsCollector


class TestOpticalParameters:
    """Test cases for OpticalParameters dataclass."""
    
    def test_default_initialization(self):
        """Test default parameter initialization."""
        params = OpticalParameters()
        assert params.wavelength == 1550e-9
        assert params.power == 1e-3
        assert params.loss == 0.1
        assert params.coupling_efficiency == 0.9
        assert params.detector_efficiency == 0.8
    
    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        params = OpticalParameters(
            wavelength=1310e-9,
            power=5e-3,
            loss=0.2
        )
        assert params.wavelength == 1310e-9
        assert params.power == 5e-3
        assert params.loss == 0.2


class TestWaveguideNeuron:
    """Test cases for WaveguideNeuron class."""
    
    def test_default_initialization(self):
        """Test neuron initialization with defaults."""
        neuron = WaveguideNeuron()
        assert neuron.arm_length == 100e-6
        assert neuron.phase_shifter_type == "thermal"
        assert neuron.modulation_depth == 0.9
        assert neuron.threshold_power == 1e-6
        assert neuron.wavelength == 1550e-9
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test valid parameters
        neuron = WaveguideNeuron(
            arm_length=50e-6,
            modulation_depth=0.8,
            threshold_power=500e-9,
            wavelength=1310e-9
        )
        assert neuron.arm_length == 50e-6
        
        # Test invalid arm length
        with pytest.raises(ValueError, match="Arm length"):
            WaveguideNeuron(arm_length=-10e-6)
        
        # Test invalid modulation depth
        with pytest.raises(ValueError, match="Modulation depth"):
            WaveguideNeuron(modulation_depth=1.5)
        
        # Test invalid threshold power
        with pytest.raises(ValueError, match="Threshold power"):
            WaveguideNeuron(threshold_power=-1e-6)
        
        # Test invalid wavelength
        with pytest.raises(ValueError, match="Wavelength"):
            WaveguideNeuron(wavelength=2000e-9)
    
    def test_forward_pass_basic(self):
        """Test basic neuron forward pass."""
        neuron = WaveguideNeuron(threshold_power=1e-6)
        
        # Test below threshold
        result = neuron.forward(0.5e-6, 0.0)  # 0.5 μW
        assert result is False
        
        # Test above threshold
        result = neuron.forward(2e-6, 1e-9)  # 2 μW
        assert result is True
    
    def test_membrane_potential_dynamics(self):
        """Test membrane potential integration and leak."""
        neuron = WaveguideNeuron(threshold_power=10e-6)  # High threshold
        
        # Test integration
        neuron.forward(1e-6, 0.0)
        initial_potential = neuron._membrane_potential
        assert initial_potential > 0
        
        # Test leak over time
        neuron.forward(0.1e-6, 1e-9)
        leaked_potential = neuron._membrane_potential
        assert leaked_potential < initial_potential * 1.01  # Should have leaked
    
    def test_refractory_period(self):
        """Test neuron refractory period behavior."""
        neuron = WaveguideNeuron(threshold_power=1e-6)
        
        # Generate first spike
        result1 = neuron.forward(2e-6, 0.0)
        assert result1 is True
        
        # Should not generate spike immediately after (refractory)
        result2 = neuron.forward(2e-6, 1e-12)  # 1 ps later
        assert result2 is False
    
    def test_metrics_integration(self):
        """Test metrics collector integration."""
        neuron = WaveguideNeuron()
        metrics_collector = Mock()
        
        neuron.set_metrics_collector(metrics_collector)
        assert neuron._metrics_collector is not None
        
        # Generate spike and verify metrics
        neuron.forward(2e-6, 0.0)
        metrics_collector.increment_counter.assert_called()
    
    def test_error_handling(self):
        """Test neuron error handling."""
        neuron = WaveguideNeuron()
        
        # Test negative optical input
        with pytest.raises(OpticalModelError):
            neuron.forward(-1e-6, 0.0)
    
    def test_high_power_warning(self):
        """Test high power warning system."""
        neuron = WaveguideNeuron()
        metrics_collector = Mock()
        neuron.set_metrics_collector(metrics_collector)
        
        with patch.object(neuron._logger, 'warning') as mock_warning:
            neuron.forward(2.0, 0.0)  # 2W - very high power
            mock_warning.assert_called()
            metrics_collector.increment_counter.assert_called_with("high_power_warnings")
    
    def test_transfer_function(self):
        """Test transfer function generation."""
        neuron = WaveguideNeuron(
            arm_length=200e-6,
            wavelength=1310e-9
        )
        
        transfer_func = neuron.get_transfer_function()
        assert transfer_func["type"] == "mach_zehnder"
        assert transfer_func["arm_length"] == 200e-6
        assert transfer_func["wavelength"] == 1310e-9
    
    def test_spice_generation(self):
        """Test SPICE model generation."""
        neuron = WaveguideNeuron(threshold_power=2e-6)
        
        spice_model = neuron.to_spice()
        assert "photonic_neuron" in spice_model
        assert "Mach-Zehnder" in spice_model
        assert str(neuron.threshold_power) in spice_model


class TestPhotonicSNN:
    """Test cases for PhotonicSNN class."""
    
    def test_basic_initialization(self):
        """Test basic SNN initialization."""
        topology = [10, 5, 2]
        snn = PhotonicSNN(topology)
        
        assert snn.topology == topology
        assert len(snn.neurons) == 3  # Including input layer
        assert len(snn.layers) == 2   # Weight matrices
        assert snn.wavelength == 1550e-9
    
    def test_custom_parameters(self):
        """Test SNN with custom parameters."""
        topology = [784, 256, 10]
        optical_params = OpticalParameters(wavelength=1310e-9, power=2e-3)
        
        snn = PhotonicSNN(
            topology=topology,
            neuron_type=WaveguideNeuron,
            synapse_type="microring",
            wavelength=1310e-9,
            optical_params=optical_params
        )
        
        assert snn.topology == topology
        assert snn.synapse_type == "microring"
        assert snn.wavelength == 1310e-9
        assert snn.optical_params.power == 2e-3
    
    def test_topology_validation(self):
        """Test network topology validation."""
        # Valid topology
        PhotonicSNN([10, 5, 2])
        
        # Invalid topologies
        with pytest.raises(NetworkTopologyError):
            PhotonicSNN([10])  # Too few layers
        
        with pytest.raises(NetworkTopologyError):
            PhotonicSNN([10, -5, 2])  # Negative size
        
        with pytest.raises(NetworkTopologyError):
            PhotonicSNN([10, 0, 2])  # Zero size
    
    def test_synapse_type_validation(self):
        """Test synapse type validation."""
        # Valid synapse types
        PhotonicSNN([10, 5, 2], synapse_type="phase_change")
        PhotonicSNN([10, 5, 2], synapse_type="microring")
        
        # Invalid synapse type
        with pytest.raises(ValidationError):
            PhotonicSNN([10, 5, 2], synapse_type="invalid_type")
    
    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        snn = PhotonicSNN([10, 5, 2])
        spike_train = torch.randn(100, 10)  # 100 time steps, 10 features
        
        output = snn.forward(spike_train)
        
        assert output.shape == (100, 2)  # 100 time steps, 2 outputs
        assert output.dtype == torch.float32
    
    def test_forward_pass_validation(self):
        """Test forward pass input validation."""
        snn = PhotonicSNN([10, 5, 2])
        
        # Wrong number of dimensions
        with pytest.raises(ValidationError):
            snn.forward(torch.randn(10))  # 1D instead of 2D
        
        # Wrong input size
        with pytest.raises(ValidationError):
            snn.forward(torch.randn(100, 20))  # 20 features instead of 10
        
        # Negative duration
        with pytest.raises(ValidationError):
            snn.forward(torch.randn(100, 10), duration=-1.0)
    
    def test_invalid_spikes_handling(self):
        """Test handling of invalid spike inputs."""
        snn = PhotonicSNN([5, 3, 2])
        metrics_collector = Mock()
        snn.set_metrics_collector(metrics_collector)
        
        # Create spike train with NaN values
        spike_train = torch.randn(10, 5)
        spike_train[5, 2] = float('nan')
        
        with patch.object(snn._logger, 'warning') as mock_warning:
            output = snn.forward(spike_train)
            mock_warning.assert_called()
            metrics_collector.increment_counter.assert_called()
        
        assert output.shape == (10, 2)
    
    def test_metrics_integration(self):
        """Test metrics collector integration."""
        snn = PhotonicSNN([5, 3, 2])
        metrics_collector = Mock()
        
        snn.set_metrics_collector(metrics_collector)
        
        # Verify metrics collector is set on neurons
        for layer_neurons in snn.neurons:
            for neuron in layer_neurons:
                assert neuron._metrics_collector is metrics_collector
    
    def test_network_info(self):
        """Test network information retrieval."""
        topology = [784, 256, 128, 10]
        snn = PhotonicSNN(topology, wavelength=1310e-9)
        
        info = snn.get_network_info()
        
        assert info["topology"] == topology
        assert info["neuron_type"] == "WaveguideNeuron"
        assert info["wavelength"] == 1310e-9
        assert info["total_neurons"] == sum(topology)
        assert info["total_synapses"] == 784*256 + 256*128 + 128*10
    
    def test_energy_estimation(self):
        """Test energy consumption estimation."""
        snn = PhotonicSNN([10, 5, 2])
        spike_train = torch.randn(50, 10)
        
        energy_metrics = snn.estimate_energy_consumption(spike_train)
        
        assert "total_spikes" in energy_metrics
        assert "energy_per_spike" in energy_metrics
        assert "total_energy" in energy_metrics
        assert "power_consumption" in energy_metrics
        
        assert energy_metrics["total_energy"] > 0
        assert energy_metrics["power_consumption"] > 0
    
    def test_optical_power_clamping(self):
        """Test optical power safety clamping."""
        snn = PhotonicSNN([3, 2, 1])
        metrics_collector = Mock()
        snn.set_metrics_collector(metrics_collector)
        
        # Create spike train that could cause high optical power
        spike_train = torch.ones(10, 3) * 100  # Very high values
        
        with patch.object(snn._logger, 'error') as mock_error:
            output = snn.forward(spike_train)
            # Should have clamped power and logged errors
            assert output.shape == (10, 1)
    
    def test_layer_processing_error_recovery(self):
        """Test error recovery in layer processing."""
        snn = PhotonicSNN([5, 3, 2])
        metrics_collector = Mock()
        snn.set_metrics_collector(metrics_collector)
        
        # Mock a neuron to raise an exception
        with patch.object(snn.neurons[1][0], 'forward', side_effect=Exception("Test error")):
            spike_train = torch.randn(5, 5)
            
            with patch.object(snn._logger, 'error') as mock_error:
                output = snn.forward(spike_train)
                
                # Should continue processing despite error
                assert output.shape == (5, 2)
                mock_error.assert_called()
                metrics_collector.increment_counter.assert_called_with("layer_processing_errors")


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_encode_to_spikes_basic(self):
        """Test basic spike encoding."""
        data = np.array([0.5, 1.0, 0.0, 0.8])
        spike_train = encode_to_spikes(data, duration=100e-9, dt=1e-9)
        
        assert spike_train.shape == (100, 4)  # 100 time steps, 4 features
        assert spike_train.dtype == torch.float32
        
        # Higher values should have higher spike rates
        spike_rates = torch.mean(spike_train, dim=0)
        assert spike_rates[1] > spike_rates[0]  # 1.0 > 0.5
        assert spike_rates[1] > spike_rates[2]  # 1.0 > 0.0
    
    def test_encode_to_spikes_edge_cases(self):
        """Test spike encoding edge cases."""
        # All zeros
        data = np.zeros(5)
        spike_train = encode_to_spikes(data)
        assert torch.sum(spike_train) == 0  # Should produce no spikes
        
        # All ones
        data = np.ones(5)
        spike_train = encode_to_spikes(data, duration=10e-9, dt=1e-9)
        assert torch.sum(spike_train) > 0  # Should produce some spikes
    
    def test_create_mnist_photonic_snn(self):
        """Test MNIST-specific SNN creation."""
        snn = create_mnist_photonic_snn()
        
        assert snn.topology == [784, 256, 128, 10]
        assert snn.neuron_type == WaveguideNeuron
        assert snn.synapse_type == "phase_change"
        assert snn.wavelength == 1550e-9
    
    def test_benchmark_photonic_vs_electronic(self):
        """Test benchmarking functionality."""
        photonic_model = PhotonicSNN([10, 5, 2])
        
        # Mock electronic model
        electronic_model = Mock()
        electronic_model.return_value = torch.randn(1, 2)
        
        test_data = torch.randn(1, 10)
        
        with patch('time.time', side_effect=[0.0, 0.1, 0.1, 0.2]):  # Mock timing
            benchmark_results = photonic_model.benchmark_photonic_vs_electronic(
                photonic_model, electronic_model, test_data
            )
        
        assert "photonic" in benchmark_results
        assert "electronic" in benchmark_results
        assert "inference_time" in benchmark_results["photonic"]
        assert "total_energy" in benchmark_results["photonic"]


@pytest.fixture
def sample_spike_train():
    """Create sample spike train for testing."""
    return torch.randn(100, 784)  # 100 time steps, 784 features (MNIST)


@pytest.fixture
def mock_optical_parameters():
    """Create mock optical parameters for testing."""
    return {
        "wavelength": 1550e-9,
        "power": 1e-3,
        "loss": 0.1,
        "coupling_efficiency": 0.9
    }


@pytest.fixture
def basic_photonic_snn():
    """Create basic PhotonicSNN for testing."""
    return PhotonicSNN([10, 5, 2])


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_simulation(self, basic_photonic_snn):
        """Test complete end-to-end simulation pipeline."""
        snn = basic_photonic_snn
        
        # Create input data
        input_data = np.random.rand(10)
        spike_train = encode_to_spikes(input_data, duration=50e-9)
        
        # Run simulation
        output_spikes = snn.forward(spike_train)
        
        # Verify output
        assert output_spikes.shape[0] == spike_train.shape[0]
        assert output_spikes.shape[1] == snn.topology[-1]
        assert torch.all(torch.isfinite(output_spikes))
    
    def test_metrics_collection_integration(self):
        """Test full metrics collection integration."""
        snn = PhotonicSNN([5, 3, 2])
        metrics_collector = MetricsCollector(enable_system_metrics=False)
        
        snn.set_metrics_collector(metrics_collector)
        
        # Run simulation
        spike_train = torch.randn(20, 5)
        output = snn.forward(spike_train)
        
        # Verify metrics were collected
        current_metrics = metrics_collector.get_current_metrics()
        assert len(current_metrics) > 0
    
    def test_large_network_stability(self):
        """Test stability with larger networks."""
        # Create larger network
        topology = [100, 50, 25, 10]
        snn = PhotonicSNN(topology)
        
        # Create larger input
        spike_train = torch.randn(10, 100)
        
        # Should complete without errors
        output = snn.forward(spike_train)
        assert output.shape == (10, 10)
        assert torch.all(torch.isfinite(output))
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs."""
        snn = PhotonicSNN([5, 3, 2])
        
        # Test with very small values
        small_spikes = torch.randn(10, 5) * 1e-10
        output_small = snn.forward(small_spikes)
        assert torch.all(torch.isfinite(output_small))
        
        # Test with large values (should be clamped)
        large_spikes = torch.randn(10, 5) * 1e10
        output_large = snn.forward(large_spikes)
        assert torch.all(torch.isfinite(output_large))


# Performance benchmarks
@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmark tests."""
    
    def test_forward_pass_performance(self, basic_photonic_snn):
        """Benchmark forward pass performance."""
        snn = basic_photonic_snn
        spike_train = torch.randn(1000, 10)  # Large input
        
        import time
        start_time = time.time()
        output = snn.forward(spike_train)
        end_time = time.time()
        
        execution_time = end_time - start_time
        throughput = spike_train.shape[0] / execution_time
        
        # Log performance metrics
        print(f"Forward pass: {execution_time:.3f}s, {throughput:.0f} time steps/s")
        
        # Performance should be reasonable
        assert execution_time < 5.0  # Should complete in under 5 seconds
        assert throughput > 100     # Should process at least 100 time steps/second
    
    def test_memory_usage(self):
        """Test memory usage patterns."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large network
        snn = PhotonicSNN([1000, 500, 100])
        spike_train = torch.randn(100, 1000)
        
        output = snn.forward(spike_train)
        
        peak_memory = process.memory_info().rss
        memory_increase = (peak_memory - initial_memory) / 1024**2  # MB
        
        print(f"Memory increase: {memory_increase:.1f} MB")
        
        # Memory usage should be reasonable
        assert memory_increase < 1000  # Should not use more than 1GB additional memory