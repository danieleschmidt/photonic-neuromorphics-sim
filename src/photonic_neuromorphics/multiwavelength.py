"""
Multi-wavelength neuromorphic computing implementation.

This module provides advanced wavelength division multiplexing (WDM) capabilities
for photonic neural networks, enabling massive parallelization through wavelength channels.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import logging
from concurrent.futures import ThreadPoolExecutor
from scipy import signal
from scipy.optimize import minimize

from .core import OpticalParameters, MultiWavelengthParameters, WaveguideNeuron
from .exceptions import OpticalModelError, ValidationError
from .monitoring import MetricsCollector


class WDMMultiplexer(BaseModel):
    """Wavelength Division Multiplexer for combining multiple optical channels."""
    
    channel_count: int = Field(default=4, description="Number of WDM channels")
    insertion_loss: float = Field(default=0.5, description="Insertion loss in dB")
    crosstalk: float = Field(default=-30.0, description="Crosstalk between channels in dB")
    passband_width: float = Field(default=0.2e-9, description="Passband width in meters")
    
    def __init__(self, **data):
        super().__init__(**data)
        self._transfer_matrix = self._calculate_transfer_matrix()
        self._logger = logging.getLogger(__name__)
    
    def _calculate_transfer_matrix(self) -> np.ndarray:
        """Calculate transfer matrix for WDM device."""
        matrix = np.eye(self.channel_count) * (10 ** (-self.insertion_loss / 10))
        
        # Add crosstalk between adjacent channels
        crosstalk_linear = 10 ** (self.crosstalk / 10)
        for i in range(self.channel_count - 1):
            matrix[i, i + 1] = crosstalk_linear
            matrix[i + 1, i] = crosstalk_linear
            
        return matrix
    
    def multiplex(self, channel_powers: List[float]) -> np.ndarray:
        """Multiplex multiple wavelength channels."""
        if len(channel_powers) != self.channel_count:
            raise ValidationError(f"Expected {self.channel_count} channels, got {len(channel_powers)}")
        
        input_vector = np.array(channel_powers)
        output_vector = self._transfer_matrix @ input_vector
        
        return output_vector
    
    def demultiplex(self, combined_signal: float, target_channel: int) -> float:
        """Demultiplex signal to extract specific wavelength channel."""
        # Simplified demultiplexing with wavelength selectivity
        selectivity = np.exp(-(target_channel - self.channel_count//2)**2 / 2)
        extracted_power = combined_signal * selectivity * (10 ** (-self.insertion_loss / 10))
        
        return extracted_power


class MultiWavelengthNeuron(BaseModel):
    """Multi-wavelength photonic neuron with WDM capability."""
    
    base_neuron: WaveguideNeuron
    wdm_params: MultiWavelengthParameters
    wavelength_weights: List[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    adaptive_weighting: bool = Field(default=True, description="Enable adaptive wavelength weighting")
    
    def __init__(self, **data):
        super().__init__(**data)
        self._multiplexer = WDMMultiplexer(channel_count=self.wdm_params.channel_count)
        self._wavelength_states = np.zeros(self.wdm_params.channel_count)
        self._adaptation_rates = np.ones(self.wdm_params.channel_count) * 0.01
        self._logger = logging.getLogger(__name__)
    
    def forward_multiwavelength(
        self, 
        wavelength_inputs: List[float], 
        time: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Process multi-wavelength input and generate spike.
        
        Args:
            wavelength_inputs: List of optical powers for each wavelength channel
            time: Current simulation time
            
        Returns:
            Tuple of (spike_generated, wavelength_analysis)
        """
        if len(wavelength_inputs) != self.wdm_params.channel_count:
            raise ValidationError(
                f"Expected {self.wdm_params.channel_count} wavelength inputs, "
                f"got {len(wavelength_inputs)}"
            )
        
        # Multiplex wavelength channels
        multiplexed_power = self._multiplexer.multiplex(wavelength_inputs)
        
        # Apply wavelength-specific weights
        weighted_powers = np.array(wavelength_inputs) * np.array(self.wavelength_weights)
        total_weighted_power = np.sum(weighted_powers)
        
        # Generate spike based on combined power
        spike_generated = self.base_neuron.forward(total_weighted_power, time)
        
        # Update adaptive weights if enabled
        if self.adaptive_weighting and spike_generated:
            self._update_wavelength_weights(wavelength_inputs, spike_generated)
        
        # Analyze wavelength contributions
        wavelength_analysis = {
            'channel_powers': wavelength_inputs,
            'weighted_powers': weighted_powers.tolist(),
            'total_power': total_weighted_power,
            'multiplexed_power': multiplexed_power.tolist(),
            'wavelength_weights': self.wavelength_weights.copy(),
            'dominant_channel': int(np.argmax(weighted_powers))
        }
        
        return spike_generated, wavelength_analysis
    
    def _update_wavelength_weights(self, inputs: List[float], spike_generated: bool):
        """Update wavelength weights based on spike-timing dependent plasticity."""
        for i, power in enumerate(inputs):
            if power > 0:  # Only update active channels
                if spike_generated:
                    # Strengthen weights for channels that contributed to spike
                    self.wavelength_weights[i] *= (1 + self._adaptation_rates[i])
                else:
                    # Weaken weights for channels that didn't lead to spike
                    self.wavelength_weights[i] *= (1 - self._adaptation_rates[i] * 0.5)
                
                # Normalize weights to prevent runaway growth
                self.wavelength_weights[i] = np.clip(self.wavelength_weights[i], 0.1, 10.0)


class WDMCrossbar(BaseModel):
    """WDM-enabled photonic crossbar array."""
    
    rows: int = Field(default=8, description="Number of rows")
    cols: int = Field(default=8, description="Number of columns")
    wavelength_channels: int = Field(default=4, description="Number of WDM channels per connection")
    wdm_params: MultiWavelengthParameters
    
    def __init__(self, **data):
        super().__init__(**data)
        self.neurons = self._create_multiwavelength_neurons()
        self.weights = self._initialize_weights()
        self._logger = logging.getLogger(__name__)
    
    def _create_multiwavelength_neurons(self) -> List[List[MultiWavelengthNeuron]]:
        """Create grid of multi-wavelength neurons."""
        neurons = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                base_neuron = WaveguideNeuron(
                    threshold_power=1e-6 * (1 + 0.1 * np.random.randn())  # Add variation
                )
                mw_neuron = MultiWavelengthNeuron(
                    base_neuron=base_neuron,
                    wdm_params=self.wdm_params,
                    wavelength_weights=[1.0] * self.wavelength_channels
                )
                row.append(mw_neuron)
            neurons.append(row)
        return neurons
    
    def _initialize_weights(self) -> torch.Tensor:
        """Initialize synaptic weights for crossbar."""
        # Initialize with small random weights
        weights = torch.randn(self.rows, self.cols, self.wavelength_channels) * 0.1
        return weights
    
    def forward(
        self, 
        input_matrix: torch.Tensor, 
        time: float
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process input through WDM crossbar.
        
        Args:
            input_matrix: Input tensor of shape (rows, wavelength_channels)
            time: Current simulation time
            
        Returns:
            Tuple of (output_spikes, detailed_analysis)
        """
        if input_matrix.shape != (self.rows, self.wavelength_channels):
            raise ValidationError(
                f"Expected input shape ({self.rows}, {self.wavelength_channels}), "
                f"got {input_matrix.shape}"
            )
        
        output_spikes = torch.zeros(self.rows, self.cols, dtype=torch.bool)
        wavelength_analysis = {}
        
        for i in range(self.rows):
            for j in range(self.cols):
                # Apply synaptic weights to wavelength channels
                weighted_input = input_matrix[i] * self.weights[i, j]
                
                # Process through multi-wavelength neuron
                spike, analysis = self.neurons[i][j].forward_multiwavelength(
                    weighted_input.tolist(), time
                )
                
                output_spikes[i, j] = spike
                wavelength_analysis[f'neuron_{i}_{j}'] = analysis
        
        # Calculate network-level statistics
        network_analysis = {
            'total_spikes': output_spikes.sum().item(),
            'spike_rate': output_spikes.float().mean().item(),
            'active_wavelengths': self._analyze_active_wavelengths(wavelength_analysis),
            'wavelength_efficiency': self._calculate_wavelength_efficiency(wavelength_analysis)
        }
        
        return output_spikes, {**wavelength_analysis, 'network': network_analysis}
    
    def _analyze_active_wavelengths(self, analysis: Dict[str, Any]) -> Dict[int, int]:
        """Analyze which wavelength channels are most active."""
        channel_activity = {i: 0 for i in range(self.wavelength_channels)}
        
        for neuron_key, neuron_analysis in analysis.items():
            if neuron_key.startswith('neuron_'):
                dominant_channel = neuron_analysis.get('dominant_channel', 0)
                channel_activity[dominant_channel] += 1
        
        return channel_activity
    
    def _calculate_wavelength_efficiency(self, analysis: Dict[str, Any]) -> float:
        """Calculate how efficiently wavelength channels are being utilized."""
        total_power = 0
        utilized_power = 0
        
        for neuron_key, neuron_analysis in analysis.items():
            if neuron_key.startswith('neuron_'):
                channel_powers = neuron_analysis.get('channel_powers', [])
                weighted_powers = neuron_analysis.get('weighted_powers', [])
                
                total_power += sum(channel_powers)
                utilized_power += sum(weighted_powers)
        
        return utilized_power / total_power if total_power > 0 else 0.0


class AttentionMechanism(BaseModel):
    """Optical attention mechanism using wavelength selectivity."""
    
    attention_channels: int = Field(default=4, description="Number of attention channels")
    focus_bandwidth: float = Field(default=0.1e-9, description="Focus bandwidth in meters")
    attention_strength: float = Field(default=2.0, description="Attention amplification factor")
    
    def __init__(self, **data):
        super().__init__(**data)
        self.attention_weights = np.ones(self.attention_channels)
        self._adaptation_rate = 0.1
        self._logger = logging.getLogger(__name__)
    
    def apply_attention(
        self, 
        wavelength_inputs: List[float], 
        query_wavelength: float = 1550e-9
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Apply wavelength-based attention to input signals.
        
        Args:
            wavelength_inputs: Input powers for each wavelength channel
            query_wavelength: Target wavelength for attention focus
            
        Returns:
            Tuple of (attended_outputs, attention_analysis)
        """
        wavelength_grid = np.linspace(1540e-9, 1560e-9, self.attention_channels)
        
        # Calculate attention weights based on wavelength proximity
        attention_profile = np.exp(
            -((wavelength_grid - query_wavelength) / self.focus_bandwidth) ** 2
        )
        attention_profile *= self.attention_strength
        
        # Apply attention to inputs
        attended_outputs = [
            power * weight * attention 
            for power, weight, attention in zip(
                wavelength_inputs, self.attention_weights, attention_profile
            )
        ]
        
        attention_analysis = {
            'attention_profile': attention_profile.tolist(),
            'attention_weights': self.attention_weights.tolist(),
            'focus_wavelength': query_wavelength,
            'total_attention': sum(attended_outputs),
            'attention_entropy': self._calculate_attention_entropy(attention_profile)
        }
        
        return attended_outputs, attention_analysis
    
    def _calculate_attention_entropy(self, attention_profile: np.ndarray) -> float:
        """Calculate entropy of attention distribution."""
        normalized_attention = attention_profile / np.sum(attention_profile)
        entropy = -np.sum(normalized_attention * np.log(normalized_attention + 1e-10))
        return float(entropy)
    
    def update_attention(self, reward_signal: float):
        """Update attention weights based on reward signal."""
        # Simple reinforcement learning update
        self.attention_weights *= (1 + self._adaptation_rate * reward_signal)
        self.attention_weights /= np.sum(self.attention_weights)  # Normalize


def create_multiwavelength_mnist_network(
    input_size: int = 784,
    hidden_size: int = 256,
    output_size: int = 10,
    wavelength_channels: int = 4
) -> Dict[str, Any]:
    """
    Create a multi-wavelength photonic network for MNIST classification.
    
    Args:
        input_size: Input layer size
        hidden_size: Hidden layer size  
        output_size: Output layer size
        wavelength_channels: Number of WDM channels
        
    Returns:
        Dictionary containing network components and configuration
    """
    wdm_params = MultiWavelengthParameters(
        channel_count=wavelength_channels,
        center_wavelength=1550e-9,
        channel_spacing=0.8e-9,
        power_per_channel=250e-6
    )
    
    # Create multi-wavelength crossbars for each layer
    input_crossbar = WDMCrossbar(
        rows=input_size // 4,  # Reduce size for practical implementation
        cols=hidden_size // 4,
        wavelength_channels=wavelength_channels,
        wdm_params=wdm_params
    )
    
    output_crossbar = WDMCrossbar(
        rows=hidden_size // 4,
        cols=output_size,
        wavelength_channels=wavelength_channels,
        wdm_params=wdm_params
    )
    
    # Add attention mechanism
    attention = AttentionMechanism(
        attention_channels=wavelength_channels,
        focus_bandwidth=0.1e-9,
        attention_strength=2.0
    )
    
    return {
        'input_crossbar': input_crossbar,
        'output_crossbar': output_crossbar,
        'attention_mechanism': attention,
        'wdm_parameters': wdm_params,
        'network_config': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'wavelength_channels': wavelength_channels,
            'total_wavelengths': wavelength_channels * 2  # Input + output layers
        }
    }


def simulate_multiwavelength_network(
    network: Dict[str, Any],
    input_data: torch.Tensor,
    simulation_time: float = 100e-9
) -> Dict[str, Any]:
    """
    Simulate multi-wavelength photonic network.
    
    Args:
        network: Network created by create_multiwavelength_mnist_network
        input_data: Input tensor of shape (batch_size, input_size)
        simulation_time: Simulation duration in seconds
        
    Returns:
        Comprehensive simulation results
    """
    input_crossbar = network['input_crossbar']
    output_crossbar = network['output_crossbar']
    attention = network['attention_mechanism']
    
    batch_size = input_data.shape[0]
    wavelength_channels = network['network_config']['wavelength_channels']
    
    results = {
        'batch_outputs': [],
        'attention_analysis': [],
        'wavelength_efficiency': [],
        'energy_consumption': 0.0,
        'total_spikes': 0
    }
    
    for batch_idx in range(batch_size):
        # Convert input to wavelength-encoded format
        input_sample = input_data[batch_idx].reshape(-1, wavelength_channels)
        
        # Apply attention mechanism
        attended_input = []
        for i in range(input_sample.shape[0]):
            attended_channels, attention_analysis = attention.apply_attention(
                input_sample[i].tolist()
            )
            attended_input.append(attended_channels)
        
        attended_input_tensor = torch.tensor(attended_input)
        
        # Process through input crossbar
        hidden_spikes, hidden_analysis = input_crossbar.forward(
            attended_input_tensor, simulation_time
        )
        
        # Convert hidden spikes to wavelength format for output layer
        hidden_wavelength = hidden_spikes.float().unsqueeze(-1).repeat(1, 1, wavelength_channels)
        hidden_reshaped = hidden_wavelength.reshape(-1, wavelength_channels)
        
        # Process through output crossbar
        output_spikes, output_analysis = output_crossbar.forward(
            hidden_reshaped, simulation_time
        )
        
        # Collect results
        results['batch_outputs'].append({
            'hidden_spikes': hidden_spikes,
            'output_spikes': output_spikes,
            'hidden_analysis': hidden_analysis,
            'output_analysis': output_analysis
        })
        
        results['attention_analysis'].append(attention_analysis)
        results['wavelength_efficiency'].append(
            output_analysis['network']['wavelength_efficiency']
        )
        results['total_spikes'] += output_analysis['network']['total_spikes']
    
    # Calculate aggregate metrics
    results['average_wavelength_efficiency'] = np.mean(results['wavelength_efficiency'])
    results['spike_rate'] = results['total_spikes'] / (batch_size * simulation_time * 1e9)  # spikes/ns
    results['energy_per_spike'] = 0.1e-12  # 0.1 pJ per spike (estimated)
    results['total_energy'] = results['total_spikes'] * results['energy_per_spike']
    
    return results