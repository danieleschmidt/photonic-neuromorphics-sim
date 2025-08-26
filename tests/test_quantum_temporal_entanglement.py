"""
Comprehensive test suite for Quantum Temporal Entanglement module.

Tests cover quantum state management, temporal coherence processing,
Bell state creation, and performance validation with statistical verification.
"""

import pytest
import numpy as np
import torch
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple

from src.photonic_neuromorphics.quantum_temporal_entanglement import (
    QuantumTemporalEntanglementProcessor,
    QuantumTemporalState,
    QuantumTemporalParameters,
    create_quantum_temporal_entanglement_demo,
    run_quantum_temporal_entanglement_benchmark,
    validate_quantum_temporal_advantage
)


class TestQuantumTemporalState:
    """Test suite for QuantumTemporalState class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dimensions = 4
        self.state = QuantumTemporalState(self.dimensions)
    
    def test_initialization(self):
        """Test quantum state initialization."""
        assert self.state.dimensions == self.dimensions
        assert len(self.state.amplitude) == self.dimensions
        assert self.state.amplitude[0] == 1.0  # Ground state
        assert all(self.state.amplitude[i] == 0.0 for i in range(1, self.dimensions))
    
    def test_superposition_initialization(self):
        """Test superposition state initialization."""
        state = QuantumTemporalState(4, initial_state="superposition")
        expected_amplitude = 1.0 / np.sqrt(4)
        
        for amplitude in state.amplitude:
            assert abs(amplitude - expected_amplitude) < 1e-6
    
    def test_bell_state_initialization(self):
        """Test Bell state initialization."""
        state = QuantumTemporalState(4, initial_state="bell")
        
        # Check Bell state structure
        assert abs(state.amplitude[0] - 1.0/np.sqrt(2)) < 1e-6
        assert abs(state.amplitude[1] - 1.0/np.sqrt(2)) < 1e-6
        assert state.amplitude[2] == 0.0
        assert state.amplitude[3] == 0.0
    
    def test_quantum_gate_application(self):
        """Test quantum gate application."""
        # Pauli-X gate
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        
        # Apply to 2-qubit system
        state_2qubit = QuantumTemporalState(2)  # |00⟩ state
        initial_amplitude = state_2qubit.amplitude.copy()
        
        # This is a simplified test - full implementation would be more complex
        state_2qubit.apply_quantum_gate(pauli_x, [0])
        
        # State should change after gate application
        assert not np.allclose(state_2qubit.amplitude, initial_amplitude)
    
    def test_measurement(self):
        """Test quantum measurement with state collapse."""
        state = QuantumTemporalState(2, initial_state="superposition")
        
        # Measure the state multiple times
        measurements = []
        for _ in range(100):
            test_state = QuantumTemporalState(2, initial_state="superposition")
            result = test_state.measure()
            measurements.append(result)
        
        # Check measurement distribution (should be roughly equal for superposition)
        count_00 = sum(1 for m in measurements if m == [0, 0])
        count_11 = sum(1 for m in measurements if m == [1, 1])
        count_01 = sum(1 for m in measurements if m == [0, 1])
        count_10 = sum(1 for m in measurements if m == [1, 0])
        
        # For equal superposition, all outcomes should be roughly equal
        total = len(measurements)
        assert 0.15 < count_00/total < 0.35  # Allow some statistical variation
        assert 0.15 < count_11/total < 0.35
        assert 0.15 < count_01/total < 0.35
        assert 0.15 < count_10/total < 0.35
    
    def test_entanglement_entropy(self):
        """Test von Neumann entropy calculation."""
        # Ground state should have zero entropy
        ground_state = QuantumTemporalState(4)
        assert ground_state.get_entanglement_entropy() == 0.0
        
        # Superposition state should have maximum entropy
        superposition_state = QuantumTemporalState(4, initial_state="superposition")
        entropy = superposition_state.get_entanglement_entropy()
        expected_entropy = np.log2(4)  # Maximum entropy for 4 states
        assert abs(entropy - expected_entropy) < 1e-6
    
    def test_normalization(self):
        """Test quantum state normalization."""
        state = QuantumTemporalState(4)
        
        # Manually modify amplitudes
        state.amplitude = np.array([2.0, 3.0, 1.0, 0.5], dtype=complex)
        state._normalize()
        
        # Check normalization
        norm_squared = np.sum(np.abs(state.amplitude) ** 2)
        assert abs(norm_squared - 1.0) < 1e-10


class TestQuantumTemporalParameters:
    """Test suite for QuantumTemporalParameters class."""
    
    def test_default_initialization(self):
        """Test default parameter initialization."""
        params = QuantumTemporalParameters()
        
        assert params.coherence_time == 100e-9
        assert params.entanglement_fidelity == 0.99
        assert params.synchronization_accuracy == 1e-15
        assert params.temporal_entanglement_depth == 4
    
    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        params = QuantumTemporalParameters(
            coherence_time=200e-9,
            entanglement_fidelity=0.95,
            synchronization_accuracy=0.5e-15
        )
        
        assert params.coherence_time == 200e-9
        assert params.entanglement_fidelity == 0.95
        assert params.synchronization_accuracy == 0.5e-15
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters should not raise exception
        QuantumTemporalParameters(entanglement_fidelity=0.8)
        
        # Invalid fidelity should raise exception
        with pytest.raises(ValueError):
            QuantumTemporalParameters(entanglement_fidelity=0.3)
        
        # Invalid coherence time should raise exception
        with pytest.raises(ValueError):
            QuantumTemporalParameters(coherence_time=-1.0)


class TestQuantumTemporalEntanglementProcessor:
    """Test suite for QuantumTemporalEntanglementProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = QuantumTemporalParameters(
            coherence_time=50e-9,
            entanglement_fidelity=0.95,
            temporal_entanglement_depth=4
        )
        self.processor = QuantumTemporalEntanglementProcessor(
            num_neurons=10,
            temporal_modes=4,
            quantum_params=self.params
        )
    
    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.num_neurons == 10
        assert self.processor.temporal_modes == 4
        assert len(self.processor.quantum_states) == 10
        assert self.processor.entanglement_network.shape == (10, 10)
        
        # Check quantum states are initialized
        for i in range(10):
            assert i in self.processor.quantum_states
            state = self.processor.quantum_states[i]
            assert isinstance(state, QuantumTemporalState)
            assert state.dimensions == 4
    
    def test_entanglement_network_structure(self):
        """Test entanglement network connectivity."""
        network = self.processor.entanglement_network
        
        # Should be symmetric for undirected graph
        assert np.allclose(network, network.T.conj())
        
        # Diagonal should be zero (no self-entanglement)
        assert np.allclose(np.diag(network), 0)
        
        # Check network is not all zeros
        assert np.sum(np.abs(network)) > 0
    
    @pytest.mark.parametrize("time_steps,num_neurons", [
        (10, 5),
        (20, 8),
        (50, 10)
    ])
    def test_spike_train_processing(self, time_steps, num_neurons):
        """Test quantum spike train processing."""
        # Create test spike train
        spike_train = torch.rand(time_steps, num_neurons)
        spike_train = (spike_train > 0.7).float()  # Create sparse spikes
        
        # Process with quantum entanglement
        processed_spikes, quantum_metrics = self.processor.process_spike_train_quantum(
            spike_train, temporal_window=10e-9
        )
        
        # Check output shape
        assert processed_spikes.shape == spike_train.shape
        
        # Check metrics
        assert "entanglement_fidelity" in quantum_metrics
        assert "coherence_time" in quantum_metrics
        assert "synchronization_accuracy" in quantum_metrics
        assert "quantum_advantage" in quantum_metrics
        
        # Check metric values are reasonable
        assert len(quantum_metrics["entanglement_fidelity"]) > 0
        assert all(0.5 <= f <= 1.0 for f in quantum_metrics["entanglement_fidelity"])
        assert quantum_metrics["quantum_advantage"] > 0
    
    def test_bell_state_creation(self):
        """Test Bell state entanglement between neurons."""
        neuron1, neuron2 = 0, 1
        
        success, entanglement = self.processor.create_bell_state_pair(neuron1, neuron2)
        
        # Check Bell state creation
        assert isinstance(success, bool)
        assert isinstance(entanglement, float)
        assert 0 <= entanglement <= 2  # von Neumann entropy bounds
        
        # For successful entanglement, entropy should be high
        if success:
            assert entanglement > 0.5
    
    def test_bell_state_invalid_neurons(self):
        """Test Bell state creation with invalid neuron indices."""
        with pytest.raises(ValueError):
            self.processor.create_bell_state_pair(0, 100)  # Out of range
        
        with pytest.raises(ValueError):
            self.processor.create_bell_state_pair(-1, 5)  # Negative index
    
    def test_quantum_network_state(self):
        """Test quantum network state reporting."""
        state_info = self.processor.get_quantum_network_state()
        
        # Check required fields
        required_fields = [
            "num_quantum_neurons",
            "temporal_modes", 
            "average_entanglement",
            "active_entanglement_pairs",
            "coherence_time",
            "entanglement_fidelity",
            "processing_stats"
        ]
        
        for field in required_fields:
            assert field in state_info
        
        # Check values are reasonable
        assert state_info["num_quantum_neurons"] == 10
        assert state_info["temporal_modes"] == 4
        assert 0 <= state_info["average_entanglement"] <= 2
        assert state_info["active_entanglement_pairs"] >= 0
        assert state_info["coherence_time"] > 0
        assert 0.5 <= state_info["entanglement_fidelity"] <= 1.0
    
    def test_processing_statistics_update(self):
        """Test processing statistics tracking."""
        initial_stats = self.processor._processing_stats.copy()
        
        # Process some spike trains
        spike_train = torch.rand(20, 5)
        self.processor.process_spike_train_quantum(spike_train)
        
        # Check statistics were updated
        final_stats = self.processor._processing_stats
        assert final_stats["quantum_operations"] >= initial_stats["quantum_operations"]
        assert final_stats["entanglement_operations"] >= initial_stats["entanglement_operations"]
    
    @patch('src.photonic_neuromorphics.quantum_temporal_entanglement.PhotonicLogger')
    def test_logging_integration(self, mock_logger):
        """Test logging integration."""
        # Create processor with mocked logger
        processor = QuantumTemporalEntanglementProcessor(num_neurons=5)
        
        # Process spike train
        spike_train = torch.rand(10, 5)
        processor.process_spike_train_quantum(spike_train)
        
        # Verify logging calls were made
        assert mock_logger.called


class TestIntegrationFunctions:
    """Test suite for module integration functions."""
    
    def test_create_quantum_temporal_entanglement_demo(self):
        """Test demo creation function."""
        processor, spike_train, params = create_quantum_temporal_entanglement_demo(
            num_neurons=20,
            simulation_time=50e-9
        )
        
        # Check processor
        assert isinstance(processor, QuantumTemporalEntanglementProcessor)
        assert processor.num_neurons == 20
        assert processor.temporal_modes == 8
        
        # Check spike train
        assert isinstance(spike_train, torch.Tensor)
        assert spike_train.shape[1] == 20  # Number of neurons
        assert spike_train.shape[0] > 0    # Time steps
        
        # Check parameters
        assert isinstance(params, dict)
        assert "coherence_time" in params
        assert "entanglement_fidelity" in params
    
    @pytest.mark.asyncio
    async def test_run_quantum_temporal_entanglement_benchmark(self):
        """Test benchmark execution."""
        processor, spike_train, _ = create_quantum_temporal_entanglement_demo(
            num_neurons=10,
            simulation_time=20e-9
        )
        
        # Run benchmark with reduced trials for testing
        results = run_quantum_temporal_entanglement_benchmark(
            processor, spike_train, num_trials=2
        )
        
        # Check benchmark results structure
        required_fields = [
            "processing_times",
            "quantum_advantages", 
            "entanglement_fidelities",
            "synchronization_accuracies",
            "coherence_times",
            "error_rates"
        ]
        
        for field in required_fields:
            assert field in results
            if isinstance(results[field], dict):
                # Statistical summary
                assert "mean" in results[field]
                assert "std" in results[field]
                assert "min" in results[field]
                assert "max" in results[field]
    
    def test_validate_quantum_temporal_advantage(self):
        """Test quantum advantage validation."""
        validation_results = validate_quantum_temporal_advantage()
        
        # Check validation results
        required_metrics = [
            "quantum_speedup",
            "synchronization_improvement",
            "entanglement_stability", 
            "coherence_preservation"
        ]
        
        for metric in required_metrics:
            assert metric in validation_results
            assert isinstance(validation_results[metric], (int, float))
            assert validation_results[metric] >= 0


class TestPerformanceAndScalability:
    """Test suite for performance and scalability."""
    
    @pytest.mark.parametrize("num_neurons", [5, 10, 20, 50])
    def test_scalability_with_neuron_count(self, num_neurons):
        """Test scalability with increasing neuron count."""
        processor = QuantumTemporalEntanglementProcessor(num_neurons=num_neurons)
        
        # Create proportional spike train
        time_steps = 20
        spike_train = torch.rand(time_steps, num_neurons)
        
        start_time = time.time()
        processed_spikes, metrics = processor.process_spike_train_quantum(spike_train)
        processing_time = time.time() - start_time
        
        # Check processing completes successfully
        assert processed_spikes.shape == spike_train.shape
        assert processing_time < 10.0  # Should complete within 10 seconds
        
        # Check quantum advantage scales reasonably
        assert metrics["quantum_advantage"] > 0
    
    @pytest.mark.parametrize("temporal_modes", [2, 4, 8, 16])
    def test_scalability_with_temporal_modes(self, temporal_modes):
        """Test scalability with increasing temporal modes."""
        processor = QuantumTemporalEntanglementProcessor(
            num_neurons=10,
            temporal_modes=temporal_modes
        )
        
        spike_train = torch.rand(15, 10)
        
        start_time = time.time()
        processed_spikes, metrics = processor.process_spike_train_quantum(spike_train)
        processing_time = time.time() - start_time
        
        # Check successful processing
        assert processed_spikes.shape == spike_train.shape
        assert processing_time < 15.0
        
        # Check quantum states have correct dimensions
        for state in processor.quantum_states.values():
            assert state.dimensions == temporal_modes
    
    def test_memory_usage(self):
        """Test memory usage remains reasonable."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create large processor
        processor = QuantumTemporalEntanglementProcessor(
            num_neurons=100,
            temporal_modes=8
        )
        
        # Process multiple spike trains
        for _ in range(5):
            spike_train = torch.rand(30, 100)
            processor.process_spike_train_quantum(spike_train)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (less than 500MB)
        assert memory_increase < 500


class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases."""
    
    def test_empty_spike_train(self):
        """Test handling of empty spike train."""
        processor = QuantumTemporalEntanglementProcessor(num_neurons=5)
        empty_spike_train = torch.zeros(0, 5)
        
        # Should handle gracefully
        processed_spikes, metrics = processor.process_spike_train_quantum(empty_spike_train)
        
        assert processed_spikes.shape == empty_spike_train.shape
        assert len(metrics["entanglement_fidelity"]) == 0
    
    def test_single_timestep_spike_train(self):
        """Test handling of single timestep."""
        processor = QuantumTemporalEntanglementProcessor(num_neurons=5)
        single_step_spike_train = torch.rand(1, 5)
        
        processed_spikes, metrics = processor.process_spike_train_quantum(single_step_spike_train)
        
        assert processed_spikes.shape == single_step_spike_train.shape
        assert "quantum_advantage" in metrics
    
    def test_zero_spike_train(self):
        """Test handling of all-zero spike train."""
        processor = QuantumTemporalEntanglementProcessor(num_neurons=5)
        zero_spike_train = torch.zeros(20, 5)
        
        processed_spikes, metrics = processor.process_spike_train_quantum(zero_spike_train)
        
        assert processed_spikes.shape == zero_spike_train.shape
        # Should still produce valid metrics
        assert "entanglement_fidelity" in metrics
    
    def test_invalid_temporal_window(self):
        """Test handling of invalid temporal window."""
        processor = QuantumTemporalEntanglementProcessor(num_neurons=5)
        spike_train = torch.rand(10, 5)
        
        # Very small temporal window
        processed_spikes, metrics = processor.process_spike_train_quantum(
            spike_train, temporal_window=1e-15
        )
        
        # Should still process successfully
        assert processed_spikes.shape == spike_train.shape
    
    def test_processor_with_zero_neurons(self):
        """Test processor creation with zero neurons."""
        with pytest.raises((ValueError, IndexError)):
            QuantumTemporalEntanglementProcessor(num_neurons=0)
    
    def test_processor_with_excessive_neurons(self):
        """Test processor creation with excessive neurons."""
        # Should handle large numbers gracefully or raise appropriate error
        try:
            processor = QuantumTemporalEntanglementProcessor(num_neurons=10000)
            # If creation succeeds, basic functionality should work
            spike_train = torch.rand(5, 10000)
            processor.process_spike_train_quantum(spike_train)
        except (MemoryError, ValueError, RuntimeError):
            # Expected for very large systems
            pass


class TestQuantumPhysicsCorrectness:
    """Test suite for quantum physics correctness."""
    
    def test_unitary_evolution(self):
        """Test that quantum evolution preserves unitarity."""
        state = QuantumTemporalState(4, initial_state="superposition")
        initial_norm = np.sum(np.abs(state.amplitude) ** 2)
        
        # Apply some quantum gates
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        state.apply_quantum_gate(pauli_x, [0])
        
        final_norm = np.sum(np.abs(state.amplitude) ** 2)
        
        # Norm should be preserved (unitarity)
        assert abs(initial_norm - final_norm) < 1e-10
    
    def test_measurement_probabilities(self):
        """Test that measurement probabilities sum to 1."""
        state = QuantumTemporalState(4, initial_state="superposition")
        probabilities = state.get_probability_distribution()
        
        # Probabilities should sum to 1
        assert abs(np.sum(probabilities) - 1.0) < 1e-10
        
        # All probabilities should be non-negative
        assert all(p >= 0 for p in probabilities)
    
    def test_entanglement_properties(self):
        """Test entanglement properties are physically meaningful."""
        processor = QuantumTemporalEntanglementProcessor(num_neurons=4)
        
        # Create Bell states between pairs
        entanglements = []
        for i in range(3):
            success, entanglement = processor.create_bell_state_pair(i, i + 1)
            if success:
                entanglements.append(entanglement)
        
        if entanglements:
            # Entanglement entropy should be bounded
            for ent in entanglements:
                assert 0 <= ent <= 2  # von Neumann entropy bounds for 2-qubit system
    
    def test_coherence_time_effects(self):
        """Test quantum coherence time effects."""
        # Short coherence time
        short_params = QuantumTemporalParameters(coherence_time=1e-9)
        short_processor = QuantumTemporalEntanglementProcessor(
            num_neurons=5, quantum_params=short_params
        )
        
        # Long coherence time
        long_params = QuantumTemporalParameters(coherence_time=100e-9)
        long_processor = QuantumTemporalEntanglementProcessor(
            num_neurons=5, quantum_params=long_params
        )
        
        spike_train = torch.rand(20, 5)
        
        # Process with both systems
        _, short_metrics = short_processor.process_spike_train_quantum(spike_train)
        _, long_metrics = long_processor.process_spike_train_quantum(spike_train)
        
        # Longer coherence time should generally maintain higher fidelity
        short_fidelity = np.mean(short_metrics["entanglement_fidelity"])
        long_fidelity = np.mean(long_metrics["entanglement_fidelity"])
        
        # This is a statistical test - may occasionally fail
        if len(short_metrics["entanglement_fidelity"]) > 0 and len(long_metrics["entanglement_fidelity"]) > 0:
            # Allow some tolerance for statistical variation
            assert long_fidelity >= short_fidelity - 0.1


@pytest.fixture
def sample_processor():
    """Pytest fixture providing a sample processor for tests."""
    return QuantumTemporalEntanglementProcessor(
        num_neurons=8,
        temporal_modes=4,
        quantum_params=QuantumTemporalParameters(entanglement_fidelity=0.95)
    )


@pytest.fixture
def sample_spike_train():
    """Pytest fixture providing a sample spike train."""
    return torch.rand(25, 8) > 0.8  # Sparse spike train


class TestWithFixtures:
    """Test suite using pytest fixtures."""
    
    def test_processor_consistency(self, sample_processor):
        """Test processor produces consistent results."""
        spike_train = torch.rand(15, 8)
        
        # Process same input multiple times
        results = []
        for _ in range(3):
            processed, metrics = sample_processor.process_spike_train_quantum(spike_train)
            results.append((processed, metrics))
        
        # Results should be similar (allowing for quantum randomness)
        for i in range(1, len(results)):
            # Check output shapes are consistent
            assert results[i][0].shape == results[0][0].shape
            
            # Check metrics are within reasonable bounds
            assert abs(results[i][1]["quantum_advantage"] - results[0][1]["quantum_advantage"]) < 10
    
    def test_spike_train_processing(self, sample_processor, sample_spike_train):
        """Test spike train processing with fixtures."""
        processed_spikes, metrics = sample_processor.process_spike_train_quantum(sample_spike_train)
        
        assert processed_spikes.shape == sample_spike_train.shape
        assert "entanglement_fidelity" in metrics
        assert len(metrics["entanglement_fidelity"]) >= 0