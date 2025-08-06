"""
Core photonic neuromorphics simulation classes.

This module provides the fundamental building blocks for photonic spiking neural networks,
including photonic neurons and network architectures.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass
import logging
import warnings

from .exceptions import (
    OpticalModelError, ValidationError, NetworkTopologyError,
    validate_optical_parameters, validate_network_topology,
    ExceptionContext
)
from .monitoring import MetricsCollector


@dataclass
class OpticalParameters:
    """Optical parameters for photonic components."""
    wavelength: float = 1550e-9  # 1550 nm
    power: float = 1e-3  # 1 mW
    loss: float = 0.1  # dB/cm
    coupling_efficiency: float = 0.9
    detector_efficiency: float = 0.8
    propagation_loss: float = 0.1  # dB/cm


class WaveguideNeuron(BaseModel):
    """
    Waveguide-based photonic neuron implementation.
    
    Models a Mach-Zehnder interferometer-based neuron with configurable
    optical parameters and spiking behavior.
    """
    
    arm_length: float = Field(default=100e-6, description="Arm length in meters")
    phase_shifter_type: str = Field(default="thermal", description="Type of phase shifter")
    modulation_depth: float = Field(default=0.9, description="Modulation depth")
    threshold_power: float = Field(default=1e-6, description="Threshold power in watts")
    wavelength: float = Field(default=1550e-9, description="Operating wavelength in meters")
    
    @validator('arm_length')
    def validate_arm_length(cls, v):
        if v <= 0 or v > 1e-2:  # Max 1 cm
            raise ValueError(f"Arm length {v*1e6:.1f} μm out of valid range (0-10000 μm)")
        return v
    
    @validator('modulation_depth')
    def validate_modulation_depth(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"Modulation depth {v} must be between 0 and 1")
        return v
    
    @validator('threshold_power')
    def validate_threshold_power(cls, v):
        if v <= 0 or v > 1e-3:  # Max 1 mW threshold
            raise ValueError(f"Threshold power {v*1e6:.1f} μW out of valid range (0-1000 μW)")
        return v
    
    @validator('wavelength')
    def validate_wavelength(cls, v):
        if not (1260e-9 <= v <= 1675e-9):
            raise ValueError(f"Wavelength {v*1e9:.0f} nm outside optical communication bands")
        return v
    
    def __init__(self, **data):
        super().__init__(**data)
        self._membrane_potential = 0.0
        self._refractory_time = 0.0
        self._last_spike_time = -float('inf')
        self._logger = logging.getLogger(__name__)
        self._metrics_collector = None
    
    def set_metrics_collector(self, collector: Optional[MetricsCollector]):
        """Set metrics collector for monitoring."""
        self._metrics_collector = collector
    
    def forward(self, optical_input: float, time: float) -> bool:
        """
        Process optical input and generate spike if threshold is exceeded.
        
        Args:
            optical_input: Input optical power in watts
            time: Current simulation time in seconds
            
        Returns:
            bool: True if spike is generated, False otherwise
            
        Raises:
            OpticalModelError: If optical parameters are invalid
        """
        try:
            # Validate inputs
            if optical_input < 0:
                raise OpticalModelError("waveguide_neuron", "optical_input", optical_input,
                                      "Optical input power cannot be negative")
            
            if optical_input > 1.0:  # 1W safety limit
                self._logger.warning(f"High optical input power: {optical_input*1e3:.1f} mW")
                if self._metrics_collector:
                    self._metrics_collector.increment_counter("high_power_warnings")
            
            # Simple leaky integrate-and-fire behavior with error checking
            if time - self._last_spike_time > self._refractory_time:
                self._membrane_potential += optical_input * 1e6  # Scale for numerical stability
                self._membrane_potential *= 0.99  # Leak
                
                # Check for numerical overflow
                if abs(self._membrane_potential) > 1e12:
                    self._logger.error(f"Membrane potential overflow: {self._membrane_potential:.2e}")
                    self._membrane_potential = 0.0
                    if self._metrics_collector:
                        self._metrics_collector.increment_counter("numerical_overflow_errors")
                
                if self._membrane_potential > self.threshold_power * 1e6:
                    self._membrane_potential = 0.0
                    self._last_spike_time = time
                    
                    # Record metrics
                    if self._metrics_collector:
                        self._metrics_collector.increment_counter("spikes_generated")
                        self._metrics_collector.record_metric("spike_amplitude", optical_input)
                    
                    return True
            
            return False
            
        except Exception as e:
            if self._metrics_collector:
                self._metrics_collector.increment_counter("neuron_processing_errors")
            raise OpticalModelError("waveguide_neuron", "forward", optical_input, str(e))
    
    def get_transfer_function(self) -> Dict[str, Any]:
        """Get the optical transfer function of the neuron."""
        return {
            "type": "mach_zehnder",
            "arm_length": self.arm_length,
            "wavelength": self.wavelength,
            "modulation_depth": self.modulation_depth
        }
    
    def to_spice(self) -> str:
        """Generate SPICE model for the photonic neuron."""
        return f"""
* Photonic Neuron SPICE Model
.subckt photonic_neuron optical_in spike_out
* Mach-Zehnder Interferometer
R_thermal 0 thermal_control 1k
C_thermal thermal_control 0 10p
* Phase shifter (thermal)
V_phase thermal_control 0 DC 0
* Photodetector
I_photo optical_in 0 DC {self.threshold_power}
R_load spike_out 0 50
.ends photonic_neuron
"""


class PhotonicSNN(nn.Module):
    """
    Photonic Spiking Neural Network implementation.
    
    A complete photonic SNN with configurable topology, neuron types,
    and synaptic connections for neuromorphic computing applications.
    """
    
    def __init__(
        self,
        topology: List[int],
        neuron_type: type = WaveguideNeuron,
        synapse_type: str = "phase_change",
        wavelength: float = 1550e-9,
        optical_params: Optional[OpticalParameters] = None
    ):
        """
        Initialize photonic spiking neural network.
        
        Args:
            topology: List of layer sizes [input, hidden1, hidden2, ..., output]
            neuron_type: Type of photonic neuron to use
            synapse_type: Type of photonic synapse ("phase_change", "microring")
            wavelength: Operating wavelength in meters
            optical_params: Optical parameters for the system
            
        Raises:
            NetworkTopologyError: If topology is invalid
            ValidationError: If parameters are invalid
        """
        super().__init__()
        
        # Validate inputs
        validate_network_topology(topology)
        
        if synapse_type not in ["phase_change", "microring", "thermal"]:
            raise ValidationError("synapse_type", synapse_type, "string",
                                "Must be 'phase_change', 'microring', or 'thermal'")
        
        self.topology = topology
        self.neuron_type = neuron_type
        self.synapse_type = synapse_type
        self.wavelength = wavelength
        self.optical_params = optical_params or OpticalParameters(wavelength=wavelength)
        
        # Validate optical parameters
        validate_optical_parameters(
            self.optical_params.wavelength,
            self.optical_params.power,
            self.optical_params.loss,
            self.optical_params.coupling_efficiency
        )
        
        # Initialize logging and monitoring
        self._logger = logging.getLogger(__name__)
        self._metrics_collector = None
        
        # Create layers of photonic neurons
        self.layers = nn.ModuleList()
        self.neurons = []
        
        try:
            with ExceptionContext("neuron_creation", topology=topology):
                for i, layer_size in enumerate(topology):
                    layer_neurons = []
                    for j in range(layer_size):
                        neuron = self._create_neuron()
                        layer_neurons.append(neuron)
                    self.neurons.append(layer_neurons)
                    
                    # Create weight matrices for synaptic connections
                    if i > 0:  # Skip input layer
                        weight_matrix = nn.Parameter(
                            torch.randn(topology[i-1], layer_size) * 0.1
                        )
                        self.layers.append(weight_matrix)
                        
                        # Validate weight initialization
                        if torch.any(torch.isnan(weight_matrix)) or torch.any(torch.isinf(weight_matrix)):
                            raise ValidationError("weight_matrix", "NaN/Inf values", "finite numbers")
        
        except Exception as e:
            self._logger.error(f"Failed to create network: {e}")
            raise NetworkTopologyError(topology, f"Network creation failed: {str(e)}")
        
        self.current_time = 0.0
        self.dt = 1e-9  # 1 ns time step
        
        self._logger.info(f"Created PhotonicSNN: {topology} topology, {self.wavelength*1e9:.0f}nm wavelength")
    
    def set_metrics_collector(self, collector: MetricsCollector):
        """Set metrics collector for monitoring."""
        self._metrics_collector = collector
        
        # Propagate to all neurons
        for layer_neurons in self.neurons:
            for neuron in layer_neurons:
                neuron.set_metrics_collector(collector)
    
    def _create_neuron(self) -> WaveguideNeuron:
        """Create a new photonic neuron with current parameters."""
        try:
            neuron = self.neuron_type(
                wavelength=self.wavelength,
                threshold_power=self.optical_params.power * 0.001  # 0.1% of input power
            )
            if self._metrics_collector:
                neuron.set_metrics_collector(self._metrics_collector)
            return neuron
        except Exception as e:
            raise OpticalModelError("photonic_snn", "neuron_creation", self.wavelength, str(e))
    
    def forward(self, spike_train: torch.Tensor, duration: float = 100e-9) -> torch.Tensor:
        """
        Process spike train through the photonic neural network.
        
        Args:
            spike_train: Input spike train [time_steps, input_size]
            duration: Simulation duration in seconds
            
        Returns:
            torch.Tensor: Output spike train [time_steps, output_size]
            
        Raises:
            ValidationError: If input dimensions are invalid
            OpticalModelError: If processing fails
        """
        # Input validation
        if spike_train.dim() != 2:
            raise ValidationError("spike_train", f"{spike_train.dim()}D", "2D tensor")
        
        if spike_train.shape[1] != self.topology[0]:
            raise ValidationError(
                "spike_train_width", spike_train.shape[1], f"int={self.topology[0]}",
                f"Expected {self.topology[0]} input features"
            )
        
        if duration <= 0:
            raise ValidationError("duration", duration, "positive float")
        
        time_steps = spike_train.shape[0]
        output_size = self.topology[-1]
        
        try:
            with ExceptionContext("forward_pass", 
                                time_steps=time_steps, 
                                input_size=spike_train.shape[1],
                                duration=duration):
                
                output_spikes = torch.zeros(time_steps, output_size)
                
                # Track processing statistics
                total_spikes_processed = 0
                processing_errors = 0
                
                for t in range(time_steps):
                    self.current_time = t * self.dt
                    layer_activities = [spike_train[t].float()]
                    
                    # Validate input spikes
                    if torch.any(torch.isnan(spike_train[t])) or torch.any(torch.isinf(spike_train[t])):
                        self._logger.warning(f"Invalid spikes at time step {t}")
                        if self._metrics_collector:
                            self._metrics_collector.increment_counter("invalid_input_spikes")
                        continue
                    
                    # Process through each layer
                    for layer_idx, weight_matrix in enumerate(self.layers):
                        try:
                            prev_activity = layer_activities[-1]
                            current_layer_spikes = torch.zeros(weight_matrix.shape[1])
                            
                            # Process each neuron in current layer
                            for neuron_idx in range(weight_matrix.shape[1]):
                                # Calculate weighted optical input
                                optical_input = torch.sum(
                                    prev_activity * weight_matrix[:, neuron_idx]
                                ).item()
                                optical_input = max(0, optical_input * self.optical_params.power)
                                
                                # Safety check for extreme values
                                if optical_input > 10.0:  # 10W safety limit
                                    self._logger.error(f"Extreme optical input: {optical_input:.2f} W")
                                    optical_input = min(optical_input, 1.0)
                                    if self._metrics_collector:
                                        self._metrics_collector.increment_counter("optical_power_clamping")
                                
                                # Process through photonic neuron
                                neuron = self.neurons[layer_idx + 1][neuron_idx]
                                spike = neuron.forward(optical_input, self.current_time)
                                current_layer_spikes[neuron_idx] = float(spike)
                                
                                total_spikes_processed += 1
                            
                            layer_activities.append(current_layer_spikes)
                            
                        except Exception as e:
                            processing_errors += 1
                            self._logger.error(f"Layer {layer_idx} processing error at time {t}: {e}")
                            if self._metrics_collector:
                                self._metrics_collector.increment_counter("layer_processing_errors")
                            # Continue with zeros for this layer
                            layer_activities.append(torch.zeros(weight_matrix.shape[1]))
                    
                    # Store output layer activity
                    if len(layer_activities) > 1:
                        output_spikes[t] = layer_activities[-1]
                
                # Log processing statistics
                if self._metrics_collector:
                    self._metrics_collector.record_metric("total_spikes_processed", total_spikes_processed)
                    self._metrics_collector.record_metric("processing_error_rate", 
                                                        processing_errors / max(total_spikes_processed, 1))
                
                self._logger.debug(f"Processed {time_steps} time steps, "
                                 f"{total_spikes_processed} operations, "
                                 f"{processing_errors} errors")
                
                return output_spikes
                
        except Exception as e:
            if self._metrics_collector:
                self._metrics_collector.increment_counter("forward_pass_failures")
            self._logger.error(f"Forward pass failed: {e}")
            raise OpticalModelError("photonic_snn", "forward", spike_train.shape, str(e))
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get comprehensive network information."""
        return {
            "topology": self.topology,
            "neuron_type": self.neuron_type.__name__,
            "synapse_type": self.synapse_type,
            "wavelength": self.wavelength,
            "total_neurons": sum(self.topology),
            "total_synapses": sum(
                self.topology[i] * self.topology[i+1] 
                for i in range(len(self.topology)-1)
            ),
            "optical_parameters": {
                "wavelength": self.optical_params.wavelength,
                "power": self.optical_params.power,
                "loss": self.optical_params.loss,
                "coupling_efficiency": self.optical_params.coupling_efficiency
            }
        }
    
    def estimate_energy_consumption(self, spike_train: torch.Tensor) -> Dict[str, float]:
        """Estimate energy consumption for given spike train."""
        total_spikes = torch.sum(spike_train).item()
        energy_per_spike = 0.1e-12  # 0.1 pJ per spike (photonic advantage)
        
        return {
            "total_spikes": total_spikes,
            "energy_per_spike": energy_per_spike,
            "total_energy": total_spikes * energy_per_spike,
            "power_consumption": total_spikes * energy_per_spike / (spike_train.shape[0] * self.dt)
        }


def encode_to_spikes(data: np.ndarray, duration: float = 100e-9, dt: float = 1e-9) -> torch.Tensor:
    """
    Encode input data to spike trains using rate coding.
    
    Args:
        data: Input data array
        duration: Encoding duration in seconds
        dt: Time step in seconds
        
    Returns:
        torch.Tensor: Spike train [time_steps, features]
    """
    time_steps = int(duration / dt)
    spike_train = torch.zeros(time_steps, len(data.flatten()))
    
    # Rate coding: higher values -> higher spike rates
    normalized_data = (data.flatten() - data.min()) / (data.max() - data.min() + 1e-8)
    
    for t in range(time_steps):
        # Poisson spike generation
        rand_vals = torch.rand(len(normalized_data))
        spikes = rand_vals < (normalized_data * dt * 1000)  # Max 1kHz spike rate
        spike_train[t] = spikes.float()
    
    return spike_train


def create_mnist_photonic_snn() -> PhotonicSNN:
    """Create a photonic SNN optimized for MNIST classification."""
    return PhotonicSNN(
        topology=[784, 256, 128, 10],
        neuron_type=WaveguideNeuron,
        synapse_type="phase_change",
        wavelength=1550e-9
    )


def benchmark_photonic_vs_electronic(
    photonic_model: PhotonicSNN,
    electronic_model: nn.Module,
    test_data: torch.Tensor
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark photonic vs electronic neural networks.
    
    Args:
        photonic_model: Photonic neural network
        electronic_model: Electronic neural network
        test_data: Test dataset
        
    Returns:
        Dict containing benchmark results
    """
    # Convert test data to spikes for photonic model
    spike_data = encode_to_spikes(test_data.numpy())
    
    # Time photonic inference
    import time
    start_time = time.time()
    photonic_output = photonic_model(spike_data)
    photonic_time = time.time() - start_time
    
    # Time electronic inference
    start_time = time.time()
    electronic_output = electronic_model(test_data)
    electronic_time = time.time() - start_time
    
    # Calculate energy estimates
    photonic_energy = photonic_model.estimate_energy_consumption(spike_data)
    electronic_energy = {
        "total_energy": electronic_time * 10e-3,  # Estimate 10W power consumption
        "energy_per_inference": electronic_time * 10e-3 / len(test_data)
    }
    
    return {
        "photonic": {
            "inference_time": photonic_time,
            "total_energy": photonic_energy["total_energy"],
            "energy_per_inference": photonic_energy["total_energy"] / len(test_data),
            "power_consumption": photonic_energy["power_consumption"]
        },
        "electronic": {
            "inference_time": electronic_time,
            "total_energy": electronic_energy["total_energy"],
            "energy_per_inference": electronic_energy["energy_per_inference"],
            "power_consumption": electronic_energy["total_energy"] / electronic_time
        }
    }