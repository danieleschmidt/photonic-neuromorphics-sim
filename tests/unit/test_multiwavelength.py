"""
Unit tests for multi-wavelength neuromorphic computing functionality.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from photonic_neuromorphics.multiwavelength import (
    WDMMultiplexer, MultiWavelengthNeuron, WDMCrossbar, AttentionMechanism,
    create_multiwavelength_mnist_network, simulate_multiwavelength_network
)
from photonic_neuromorphics.core import WaveguideNeuron, MultiWavelengthParameters


class TestWDMMultiplexer:
    """Test WDM multiplexer functionality."""
    
    def test_multiplexer_initialization(self):
        """Test WDM multiplexer initialization."""
        mux = WDMMultiplexer(
            channel_count=4,
            insertion_loss=0.5,
            crosstalk=-30.0
        )
        
        assert mux.channel_count == 4
        assert mux.insertion_loss == 0.5
        assert mux.crosstalk == -30.0
        assert mux._transfer_matrix.shape == (4, 4)
    
    def test_multiplex_operation(self):
        """Test multiplexing operation."""
        mux = WDMMultiplexer(channel_count=4)
        channel_powers = [1.0, 0.5, 0.8, 0.3]
        
        output = mux.multiplex(channel_powers)
        
        assert len(output) == 4
        assert all(isinstance(p, (int, float, np.number)) for p in output)
        assert np.all(output >= 0)  # No negative powers
    
    def test_demultiplex_operation(self):
        """Test demultiplexing operation."""
        mux = WDMMultiplexer(channel_count=4)
        combined_signal = 2.0
        
        extracted = mux.demultiplex(combined_signal, target_channel=1)
        
        assert isinstance(extracted, (int, float, np.number))
        assert extracted >= 0
    
    def test_invalid_channel_count(self):
        """Test error handling for invalid channel count."""
        mux = WDMMultiplexer(channel_count=4)
        
        with pytest.raises(Exception):
            mux.multiplex([1.0, 2.0])  # Wrong number of channels


class TestMultiWavelengthNeuron:
    """Test multi-wavelength neuron functionality."""
    
    def test_neuron_initialization(self):
        """Test multi-wavelength neuron initialization."""
        base_neuron = WaveguideNeuron()
        wdm_params = MultiWavelengthParameters(channel_count=4)
        
        mw_neuron = MultiWavelengthNeuron(
            base_neuron=base_neuron,
            wdm_params=wdm_params,
            wavelength_weights=[1.0, 1.0, 1.0, 1.0]
        )
        
        assert mw_neuron.wdm_params.channel_count == 4
        assert len(mw_neuron.wavelength_weights) == 4
        assert mw_neuron.adaptive_weighting == True
    
    def test_forward_multiwavelength(self):
        """Test multi-wavelength forward processing."""
        base_neuron = WaveguideNeuron(threshold_power=1e-6)
        wdm_params = MultiWavelengthParameters(channel_count=4)
        
        mw_neuron = MultiWavelengthNeuron(
            base_neuron=base_neuron,
            wdm_params=wdm_params
        )
        
        wavelength_inputs = [1e-6, 2e-6, 1.5e-6, 0.5e-6]  # Powers above threshold
        time = 0.0
        
        spike_generated, analysis = mw_neuron.forward_multiwavelength(wavelength_inputs, time)
        
        assert isinstance(spike_generated, bool)
        assert isinstance(analysis, dict)
        assert 'channel_powers' in analysis
        assert 'weighted_powers' in analysis
        assert 'total_power' in analysis
        assert 'dominant_channel' in analysis
        assert len(analysis['channel_powers']) == 4
    
    def test_adaptive_weight_update(self):
        """Test adaptive weight updating."""
        base_neuron = WaveguideNeuron(threshold_power=1e-6)
        wdm_params = MultiWavelengthParameters(channel_count=4)
        
        mw_neuron = MultiWavelengthNeuron(
            base_neuron=base_neuron,
            wdm_params=wdm_params,
            adaptive_weighting=True
        )
        
        initial_weights = mw_neuron.wavelength_weights.copy()
        wavelength_inputs = [2e-6, 1e-6, 1e-6, 1e-6]  # High power inputs
        
        # Process input that should generate spike
        spike_generated, _ = mw_neuron.forward_multiwavelength(wavelength_inputs, 0.0)
        
        if spike_generated:
            # Weights should have changed
            assert mw_neuron.wavelength_weights != initial_weights
    
    def test_invalid_wavelength_inputs(self):
        """Test error handling for invalid wavelength inputs."""
        base_neuron = WaveguideNeuron()
        wdm_params = MultiWavelengthParameters(channel_count=4)
        
        mw_neuron = MultiWavelengthNeuron(
            base_neuron=base_neuron,
            wdm_params=wdm_params
        )
        
        with pytest.raises(Exception):
            mw_neuron.forward_multiwavelength([1.0, 2.0], 0.0)  # Wrong number of inputs


class TestWDMCrossbar:
    """Test WDM crossbar functionality."""
    
    def test_crossbar_initialization(self):
        """Test WDM crossbar initialization."""
        wdm_params = MultiWavelengthParameters(channel_count=4)
        
        crossbar = WDMCrossbar(
            rows=8,
            cols=8,
            wavelength_channels=4,
            wdm_params=wdm_params
        )
        
        assert crossbar.rows == 8
        assert crossbar.cols == 8
        assert crossbar.wavelength_channels == 4
        assert len(crossbar.neurons) == 8
        assert len(crossbar.neurons[0]) == 8
        assert crossbar.weights.shape == (8, 8, 4)
    
    def test_crossbar_forward(self):
        """Test crossbar forward processing."""
        wdm_params = MultiWavelengthParameters(channel_count=4)
        
        crossbar = WDMCrossbar(
            rows=4,
            cols=4,
            wavelength_channels=4,
            wdm_params=wdm_params
        )
        
        input_matrix = torch.ones(4, 4) * 1e-6  # Uniform input
        time = 0.0
        
        output_spikes, analysis = crossbar.forward(input_matrix, time)
        
        assert output_spikes.shape == (4, 4)
        assert output_spikes.dtype == torch.bool
        assert isinstance(analysis, dict)
        assert 'network' in analysis
        assert 'total_spikes' in analysis['network']
        assert 'wavelength_efficiency' in analysis['network']
    
    def test_wavelength_efficiency_calculation(self):
        """Test wavelength efficiency calculation."""
        wdm_params = MultiWavelengthParameters(channel_count=4)
        
        crossbar = WDMCrossbar(
            rows=2,
            cols=2,
            wavelength_channels=4,
            wdm_params=wdm_params
        )
        
        # Create mock analysis data
        analysis = {
            'neuron_0_0': {
                'channel_powers': [1.0, 0.5, 0.8, 0.3],
                'weighted_powers': [0.8, 0.4, 0.6, 0.2]
            },
            'neuron_0_1': {
                'channel_powers': [0.5, 1.0, 0.3, 0.8],
                'weighted_powers': [0.4, 0.8, 0.2, 0.6]
            }
        }
        
        efficiency = crossbar._calculate_wavelength_efficiency(analysis)
        
        assert isinstance(efficiency, float)
        assert 0 <= efficiency <= 1


class TestAttentionMechanism:
    """Test optical attention mechanism."""
    
    def test_attention_initialization(self):
        """Test attention mechanism initialization."""
        attention = AttentionMechanism(
            attention_channels=4,
            focus_bandwidth=0.1e-9,
            attention_strength=2.0
        )
        
        assert attention.attention_channels == 4
        assert attention.focus_bandwidth == 0.1e-9
        assert attention.attention_strength == 2.0
        assert len(attention.attention_weights) == 4
    
    def test_apply_attention(self):
        """Test attention application."""
        attention = AttentionMechanism(attention_channels=4)
        wavelength_inputs = [1.0, 0.5, 0.8, 0.3]
        query_wavelength = 1550e-9
        
        attended_outputs, analysis = attention.apply_attention(wavelength_inputs, query_wavelength)
        
        assert len(attended_outputs) == 4
        assert isinstance(analysis, dict)
        assert 'attention_profile' in analysis
        assert 'attention_entropy' in analysis
        assert 'total_attention' in analysis
        assert len(analysis['attention_profile']) == 4
    
    def test_attention_update(self):
        """Test attention weight updates."""
        attention = AttentionMechanism(attention_channels=4)
        initial_weights = attention.attention_weights.copy()
        
        # Apply positive reward
        attention.update_attention(1.0)
        
        # Weights should change but still be normalized
        assert not np.array_equal(attention.attention_weights, initial_weights)
        assert np.isclose(np.sum(attention.attention_weights), 1.0)
    
    def test_attention_entropy_calculation(self):
        """Test attention entropy calculation."""
        attention = AttentionMechanism(attention_channels=4)
        
        # Uniform attention (high entropy)
        uniform_profile = np.ones(4)
        entropy_uniform = attention._calculate_attention_entropy(uniform_profile)
        
        # Focused attention (low entropy)
        focused_profile = np.array([10.0, 0.1, 0.1, 0.1])
        entropy_focused = attention._calculate_attention_entropy(focused_profile)
        
        assert entropy_uniform > entropy_focused
        assert entropy_uniform > 0
        assert entropy_focused >= 0


class TestNetworkCreation:
    """Test network creation functions."""
    
    def test_create_multiwavelength_mnist_network(self):
        """Test multi-wavelength MNIST network creation."""
        network = create_multiwavelength_mnist_network(
            input_size=784,
            hidden_size=256,
            output_size=10,
            wavelength_channels=4
        )
        
        assert isinstance(network, dict)
        assert 'input_crossbar' in network
        assert 'output_crossbar' in network
        assert 'attention_mechanism' in network
        assert 'wdm_parameters' in network
        assert 'network_config' in network
        
        config = network['network_config']
        assert config['input_size'] == 784
        assert config['hidden_size'] == 256
        assert config['output_size'] == 10
        assert config['wavelength_channels'] == 4
    
    @patch('photonic_neuromorphics.multiwavelength.torch.tensor')
    def test_simulate_multiwavelength_network(self, mock_tensor):
        """Test multi-wavelength network simulation."""
        # Create a minimal network for testing
        network = create_multiwavelength_mnist_network(
            input_size=16,  # Smaller for testing
            hidden_size=8,
            output_size=4,
            wavelength_channels=2
        )
        
        # Mock input data
        input_data = torch.randn(2, 16)  # 2 samples, 16 features
        
        # Mock tensor creation to avoid complex tensor operations
        mock_tensor.return_value = torch.ones(4, 2)
        
        results = simulate_multiwavelength_network(
            network, input_data, simulation_time=100e-9
        )
        
        assert isinstance(results, dict)
        assert 'batch_outputs' in results
        assert 'attention_analysis' in results
        assert 'wavelength_efficiency' in results
        assert 'total_spikes' in results
        assert 'average_wavelength_efficiency' in results
        assert len(results['batch_outputs']) == 2  # 2 input samples


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_power_inputs(self):
        """Test handling of zero power inputs."""
        base_neuron = WaveguideNeuron()
        wdm_params = MultiWavelengthParameters(channel_count=4)
        
        mw_neuron = MultiWavelengthNeuron(
            base_neuron=base_neuron,
            wdm_params=wdm_params
        )
        
        zero_inputs = [0.0, 0.0, 0.0, 0.0]
        spike_generated, analysis = mw_neuron.forward_multiwavelength(zero_inputs, 0.0)
        
        assert isinstance(spike_generated, bool)
        assert analysis['total_power'] == 0.0
    
    def test_large_power_inputs(self):
        """Test handling of large power inputs."""
        base_neuron = WaveguideNeuron(threshold_power=1e-6)
        wdm_params = MultiWavelengthParameters(channel_count=4)
        
        mw_neuron = MultiWavelengthNeuron(
            base_neuron=base_neuron,
            wdm_params=wdm_params
        )
        
        large_inputs = [1.0, 1.0, 1.0, 1.0]  # Very high powers
        spike_generated, analysis = mw_neuron.forward_multiwavelength(large_inputs, 0.0)
        
        assert isinstance(spike_generated, bool)
        assert analysis['total_power'] > 0
    
    def test_single_channel_activation(self):
        """Test single channel activation scenario."""
        base_neuron = WaveguideNeuron(threshold_power=1e-6)
        wdm_params = MultiWavelengthParameters(channel_count=4)
        
        mw_neuron = MultiWavelengthNeuron(
            base_neuron=base_neuron,
            wdm_params=wdm_params
        )
        
        single_channel_inputs = [2e-6, 0.0, 0.0, 0.0]  # Only first channel active
        spike_generated, analysis = mw_neuron.forward_multiwavelength(single_channel_inputs, 0.0)
        
        assert analysis['dominant_channel'] == 0
        assert analysis['channel_powers'][0] > 0
        assert all(p == 0 for p in analysis['channel_powers'][1:])


if __name__ == '__main__':
    pytest.main([__file__])