"""
Advanced Photonic Neural Network Architectures.

This module implements sophisticated architectures including photonic crossbars,
reservoir computing systems, and neuromorphic processors optimized for
silicon-photonic implementation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod

from .core import PhotonicSNN, WaveguideNeuron, OpticalParameters
from .components import (
    MachZehnderNeuron, MicroringResonator, PhaseChangeMaterial, 
    WaveguideCrossing, PhotonicComponent
)
from .exceptions import NetworkTopologyError, OpticalModelError
from .monitoring import MetricsCollector


@dataclass
class ArchitectureConfig:
    """Configuration for photonic neural network architectures."""
    wavelength: float = 1550e-9
    temperature: float = 300.0
    power_budget: float = 10e-3  # 10 mW total
    area_budget: float = 1e-3    # 1 mm² total
    target_frequency: float = 1e9 # 1 GHz operation
    noise_tolerance: float = 0.1  # 10% noise tolerance
    routing_algorithm: str = "minimize_crossings"
    weight_resolution: int = 8    # 8-bit weights
    enable_plasticity: bool = True


class PhotonicArchitecture(ABC):
    """Abstract base class for photonic neural architectures."""
    
    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_collector: Optional[MetricsCollector] = None
    
    @abstractmethod
    def build_network(self) -> PhotonicSNN:
        """Build the photonic neural network."""
        pass
    
    @abstractmethod
    def estimate_resources(self) -> Dict[str, float]:
        """Estimate resource requirements."""
        pass
    
    @abstractmethod
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze expected performance metrics."""
        pass
    
    def set_metrics_collector(self, collector: MetricsCollector) -> None:
        """Set metrics collector for monitoring."""
        self.metrics_collector = collector


class PhotonicCrossbar(PhotonicArchitecture):
    """
    Advanced photonic crossbar array for matrix-vector multiplication.
    
    Implements a sophisticated crossbar with optimized routing, weight programming,
    and advanced optical components for high-density neural computation.
    """
    
    def __init__(
        self,
        rows: int,
        cols: int,
        weight_bits: int = 8,
        modulator_type: str = "microring",
        routing_algorithm: str = "minimize_crossings",
        config: Optional[ArchitectureConfig] = None
    ):
        super().__init__(config or ArchitectureConfig())
        self.rows = rows
        self.cols = cols
        self.weight_bits = weight_bits
        self.modulator_type = modulator_type
        self.routing_algorithm = routing_algorithm
        
        # Architecture components
        self.weight_matrix = torch.randn(rows, cols) * 0.1
        self.modulators: List[List[PhotonicComponent]] = []
        self.crossings: List[WaveguideCrossing] = []
        self.routing_losses: Dict[Tuple[int, int], float] = {}
        
        # Performance metrics
        self.total_crossings = 0
        self.max_path_length = 0.0
        self.insertion_loss_map: Dict[Tuple[int, int], float] = {}
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize crossbar components and routing."""
        # Create modulator array
        for i in range(self.rows):
            modulator_row = []
            for j in range(self.cols):
                if self.modulator_type == "microring":
                    modulator = MicroringResonator()
                elif self.modulator_type == "mz":
                    modulator = MachZehnderNeuron()
                else:
                    raise ValueError(f"Unknown modulator type: {self.modulator_type}")
                
                modulator_row.append(modulator)
            self.modulators.append(modulator_row)
        
        # Calculate routing and crossings
        self._design_routing()
        
        self.logger.info(f"Initialized {self.rows}x{self.cols} crossbar with "
                        f"{self.total_crossings} crossings")
    
    def _design_routing(self) -> None:
        """Design optimal waveguide routing."""
        if self.routing_algorithm == "minimize_crossings":
            self._minimize_crossings_routing()
        elif self.routing_algorithm == "minimize_loss":
            self._minimize_loss_routing()
        else:
            self._default_routing()
    
    def _minimize_crossings_routing(self) -> None:
        """Optimize routing to minimize waveguide crossings."""
        # Snake routing pattern to minimize crossings
        crossing_count = 0
        
        for i in range(self.rows):
            for j in range(self.cols):
                # Calculate number of crossings to reach (i,j)
                h_crossings = j  # Horizontal crossings
                v_crossings = i  # Vertical crossings
                total_crossings = h_crossings + v_crossings
                
                # Path length calculation
                path_length = (i + j) * 50e-6  # Assume 50 μm per grid unit
                
                # Store routing information
                self.routing_losses[(i, j)] = 0.1 * total_crossings  # 0.1 dB per crossing
                
                crossing_count += total_crossings
        
        self.total_crossings = crossing_count
        
        # Create crossing components
        for _ in range(min(crossing_count, 1000)):  # Limit to reasonable number
            crossing = WaveguideCrossing(
                crossing_loss=0.1,  # dB
                crosstalk=-30       # dB
            )
            self.crossings.append(crossing)
    
    def _minimize_loss_routing(self) -> None:
        """Optimize routing to minimize optical loss."""
        # Use shortest path routing
        for i in range(self.rows):
            for j in range(self.cols):
                # Manhattan distance for path length
                path_length = (abs(i) + abs(j)) * 25e-6  # 25 μm per grid unit
                
                # Loss calculation
                propagation_loss = path_length * 0.1  # 0.1 dB/cm
                crossing_loss = min(i, j) * 0.05  # Optimized crossings
                
                self.routing_losses[(i, j)] = propagation_loss + crossing_loss
                self.insertion_loss_map[(i, j)] = self.routing_losses[(i, j)]
        
        self.total_crossings = sum(min(i, j) for i in range(self.rows) for j in range(self.cols))
    
    def _default_routing(self) -> None:
        """Default grid routing."""
        for i in range(self.rows):
            for j in range(self.cols):
                self.routing_losses[(i, j)] = 0.2  # Fixed loss
    
    def program_weights(self, weights: torch.Tensor) -> None:
        """Program synaptic weights into the crossbar."""
        if weights.shape != (self.rows, self.cols):
            raise ValueError(f"Weight matrix shape {weights.shape} doesn't match "
                           f"crossbar dimensions ({self.rows}, {self.cols})")
        
        self.weight_matrix = weights.clone()
        
        # Program individual modulators
        for i in range(self.rows):
            for j in range(self.cols):
                weight_value = weights[i, j].item()
                
                # Normalize weight to 0-1 range for programming
                normalized_weight = (weight_value + 1) / 2  # Assume weights in [-1, 1]
                normalized_weight = torch.clamp(torch.tensor(normalized_weight), 0, 1).item()
                
                # Program modulator based on type
                modulator = self.modulators[i][j]
                if hasattr(modulator, 'set_weight_value'):
                    modulator.set_weight_value(normalized_weight)
                elif hasattr(modulator, 'resonance_shift'):
                    # For microring modulators
                    modulator.resonance_shift = normalized_weight * 2e-9 - 1e-9  # ±1 nm range
        
        if self.metrics_collector:
            self.metrics_collector.increment_counter("weight_programming_operations")
            self.metrics_collector.record_metric("programmed_weights", weights.numel())
        
        self.logger.debug(f"Programmed {weights.numel()} weights into crossbar")
    
    def forward(
        self, 
        optical_inputs: torch.Tensor, 
        wavelength: float = None
    ) -> torch.Tensor:
        """Perform matrix-vector multiplication through optical crossbar."""
        if optical_inputs.size(0) != self.rows:
            raise ValueError(f"Input size {optical_inputs.size(0)} doesn't match "
                           f"crossbar rows {self.rows}")
        
        if wavelength is None:
            wavelength = self.config.wavelength
        
        outputs = torch.zeros(self.cols)
        
        # Process each output column
        for j in range(self.cols):
            column_sum = 0.0
            
            # Accumulate weighted inputs for this column
            for i in range(self.rows):
                input_power = optical_inputs[i].item()
                weight = self.weight_matrix[i, j].item()
                modulator = self.modulators[i][j]
                
                # Apply modulator transfer function
                transmission, phase = modulator.transfer_function(wavelength, input_power)
                
                # Apply routing loss
                routing_loss_db = self.routing_losses.get((i, j), 0.2)
                routing_loss_linear = 10**(-routing_loss_db / 10)
                
                # Weighted optical signal
                weighted_signal = input_power * abs(weight) * transmission * routing_loss_linear
                
                # Phase modulation for negative weights (simplified)
                if weight < 0:
                    weighted_signal *= -1  # Phase shift representation
                
                column_sum += weighted_signal
            
            outputs[j] = column_sum
        
        # Add optical noise
        if self.config.noise_tolerance > 0:
            noise = torch.randn_like(outputs) * self.config.noise_tolerance * outputs.abs()
            outputs += noise
        
        if self.metrics_collector:
            self.metrics_collector.record_metric("crossbar_operations", 1)
            self.metrics_collector.record_metric("total_optical_power", outputs.sum().item())
        
        return outputs
    
    def analyze_losses(self) -> Dict[str, float]:
        """Analyze optical losses in the crossbar."""
        loss_analysis = {
            "min_loss": float('inf'),
            "max_loss": 0.0,
            "mean_loss": 0.0,
            "total_crossings": self.total_crossings,
            "crosstalk_db": -30,  # Assumed crosstalk level
        }
        
        all_losses = list(self.routing_losses.values())
        
        if all_losses:
            loss_analysis["min_loss"] = min(all_losses)
            loss_analysis["max_loss"] = max(all_losses)
            loss_analysis["mean_loss"] = np.mean(all_losses)
            loss_analysis["std_loss"] = np.std(all_losses)
        
        # Calculate power efficiency
        total_loss_db = loss_analysis["mean_loss"] * self.rows * self.cols
        power_efficiency = 10**(-total_loss_db / 10)
        loss_analysis["power_efficiency"] = power_efficiency
        
        return loss_analysis
    
    def generate_layout(self) -> Dict[str, Any]:
        """Generate photonic layout information."""
        # Calculate layout dimensions
        pitch = 25e-6  # 25 μm pitch
        total_width = self.cols * pitch
        total_height = self.rows * pitch
        
        layout_info = {
            "dimensions": {
                "width": total_width,
                "height": total_height,
                "area": total_width * total_height
            },
            "component_count": {
                "modulators": self.rows * self.cols,
                "crossings": len(self.crossings),
                "waveguides": self.rows + self.cols
            },
            "routing": {
                "algorithm": self.routing_algorithm,
                "total_length": sum(self.routing_losses.keys()) * pitch,
                "average_path_length": np.mean([i+j for i, j in self.routing_losses.keys()]) * pitch
            }
        }
        
        return layout_info
    
    def build_network(self) -> PhotonicSNN:
        """Build PhotonicSNN using the crossbar architecture."""
        # Create a simple topology that uses the crossbar
        topology = [self.rows, self.cols]
        
        network = PhotonicSNN(
            topology=topology,
            wavelength=self.config.wavelength
        )
        
        # Override the forward pass to use crossbar
        def crossbar_forward(spike_train, duration=100e-9):
            time_steps = spike_train.shape[0]
            outputs = torch.zeros(time_steps, self.cols)
            
            for t in range(time_steps):
                outputs[t] = self.forward(spike_train[t])
            
            return outputs
        
        network.forward = crossbar_forward
        return network
    
    def estimate_resources(self) -> Dict[str, float]:
        """Estimate resource requirements for the crossbar."""
        # Area estimates
        modulator_area = 10e-12  # 10 μm² per modulator
        crossing_area = 5e-12    # 5 μm² per crossing
        routing_area = (self.rows + self.cols) * 25e-6 * 1e-6  # Waveguide routing
        
        total_area = (
            self.rows * self.cols * modulator_area +
            len(self.crossings) * crossing_area +
            routing_area
        )
        
        # Power estimates
        modulator_power = 10e-6   # 10 μW per modulator
        crossing_power = 1e-6     # 1 μW per crossing
        
        total_power = (
            self.rows * self.cols * modulator_power +
            len(self.crossings) * crossing_power
        )
        
        return {
            "total_area_m2": total_area,
            "total_power_w": total_power,
            "modulators": self.rows * self.cols,
            "crossings": len(self.crossings),
            "area_efficiency": (self.rows * self.cols) / total_area,  # Operations per m²
            "power_efficiency": (self.rows * self.cols) / total_power  # Operations per W
        }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze crossbar performance characteristics."""
        loss_analysis = self.analyze_losses()
        resource_estimates = self.estimate_resources()
        
        # Bandwidth analysis
        optical_bandwidth = 3e8 / self.config.wavelength  # Optical frequency
        electronic_bandwidth = self.config.target_frequency
        bandwidth_bottleneck = min(optical_bandwidth, electronic_bandwidth)
        
        # Throughput analysis
        operations_per_inference = self.rows * self.cols
        theoretical_throughput = bandwidth_bottleneck / operations_per_inference
        
        # Realistic throughput with losses
        realistic_throughput = theoretical_throughput * loss_analysis["power_efficiency"]
        
        return {
            "loss_analysis": loss_analysis,
            "resource_estimates": resource_estimates,
            "bandwidth": {
                "optical_hz": optical_bandwidth,
                "electronic_hz": electronic_bandwidth,
                "bottleneck_hz": bandwidth_bottleneck
            },
            "throughput": {
                "theoretical_ops_per_sec": theoretical_throughput,
                "realistic_ops_per_sec": realistic_throughput,
                "efficiency_factor": realistic_throughput / theoretical_throughput
            },
            "latency": {
                "optical_delay_s": loss_analysis["mean_loss"] * 1e-12,  # Rough estimate
                "electronic_delay_s": 1 / self.config.target_frequency
            }
        }


class PhotonicReservoir(PhotonicArchitecture):
    """
    Photonic reservoir computing system for temporal processing.
    
    Implements a recurrent photonic network with fixed random connectivity
    and trainable readout layer for processing temporal data.
    """
    
    def __init__(
        self,
        nodes: int,
        connectivity: float = 0.1,
        delay_distribution: str = "exponential",
        nonlinearity: str = "semiconductor_optical_amplifier",
        spectral_radius: float = 1.2,
        config: Optional[ArchitectureConfig] = None
    ):
        super().__init__(config or ArchitectureConfig())
        self.nodes = nodes
        self.connectivity = connectivity
        self.delay_distribution = delay_distribution
        self.nonlinearity = nonlinearity
        self.spectral_radius = spectral_radius
        
        # Reservoir components
        self.reservoir_matrix = self._create_reservoir_matrix()
        self.delay_matrix = self._create_delay_matrix()
        self.node_components: List[PhotonicComponent] = []
        
        # Readout layer (trainable)
        self.readout_weights = torch.randn(nodes, 1) * 0.1
        self.reservoir_states = torch.zeros(nodes)
        
        self._initialize_reservoir()
    
    def _create_reservoir_matrix(self) -> torch.Tensor:
        """Create random reservoir connectivity matrix."""
        # Random sparse matrix
        matrix = torch.randn(self.nodes, self.nodes)
        
        # Apply sparsity
        mask = torch.rand(self.nodes, self.nodes) < self.connectivity
        matrix = matrix * mask.float()
        
        # Normalize to desired spectral radius
        eigenvalues = torch.linalg.eigvals(matrix)
        current_spectral_radius = torch.max(torch.abs(eigenvalues)).item()
        
        if current_spectral_radius > 0:
            matrix = matrix * (self.spectral_radius / current_spectral_radius)
        
        return matrix
    
    def _create_delay_matrix(self) -> torch.Tensor:
        """Create matrix of optical delays between nodes."""
        delays = torch.zeros(self.nodes, self.nodes)
        
        if self.delay_distribution == "exponential":
            # Exponential distribution of delays
            base_delay = 1e-12  # 1 ps base delay
            for i in range(self.nodes):
                for j in range(self.nodes):
                    if self.reservoir_matrix[i, j] != 0:
                        delays[i, j] = base_delay * np.random.exponential(5.0)
        
        elif self.delay_distribution == "uniform":
            # Uniform distribution
            min_delay, max_delay = 1e-12, 10e-12  # 1-10 ps
            for i in range(self.nodes):
                for j in range(self.nodes):
                    if self.reservoir_matrix[i, j] != 0:
                        delays[i, j] = np.random.uniform(min_delay, max_delay)
        
        return delays
    
    def _initialize_reservoir(self) -> None:
        """Initialize reservoir components."""
        # Create photonic components for each node
        for i in range(self.nodes):
            if self.nonlinearity == "semiconductor_optical_amplifier":
                # SOA-based nonlinear node (simplified)
                component = MachZehnderNeuron(
                    threshold_power=1e-7,  # 100 nW
                    modulation_depth=0.8
                )
            elif self.nonlinearity == "microring_resonator":
                component = MicroringResonator(
                    quality_factor=5000,  # Medium Q for nonlinearity
                    coupling_gap=300e-9
                )
            else:
                component = MachZehnderNeuron()  # Default
            
            self.node_components.append(component)
        
        self.logger.info(f"Initialized photonic reservoir with {self.nodes} nodes, "
                        f"{self.connectivity:.1%} connectivity")
    
    def forward(
        self, 
        input_sequence: torch.Tensor, 
        collect_states: bool = True
    ) -> torch.Tensor:
        """Process input sequence through photonic reservoir."""
        sequence_length, input_dim = input_sequence.shape
        
        if collect_states:
            state_history = torch.zeros(sequence_length, self.nodes)
        
        outputs = torch.zeros(sequence_length, 1)
        
        # Process each time step
        for t in range(sequence_length):
            # Update reservoir states
            input_contribution = input_sequence[t] @ torch.randn(input_dim, self.nodes) * 0.1
            
            # Reservoir dynamics with optical delays (simplified)
            reservoir_contribution = self.reservoir_states @ self.reservoir_matrix
            
            # Apply nonlinearity through photonic components
            new_states = torch.zeros(self.nodes)
            for i, component in enumerate(self.node_components):
                total_input = (input_contribution[i] + reservoir_contribution[i]).item()
                
                # Convert to optical power (ensure positive)
                optical_power = max(0, total_input * 1e-6)  # Scale to μW
                
                # Apply component nonlinearity
                if hasattr(component, 'forward'):
                    # For neuron-like components
                    spike = component.forward(optical_power, t * 1e-9)
                    new_states[i] = float(spike)
                else:
                    # For other components, use transfer function
                    transmission, phase = component.transfer_function(
                        self.config.wavelength, optical_power
                    )
                    new_states[i] = transmission
            
            self.reservoir_states = new_states
            
            if collect_states:
                state_history[t] = self.reservoir_states
            
            # Compute output through readout layer
            outputs[t] = (self.reservoir_states @ self.readout_weights).squeeze()
        
        if self.metrics_collector:
            self.metrics_collector.record_metric("reservoir_processed_timesteps", sequence_length)
            self.metrics_collector.record_metric("reservoir_active_nodes", 
                                               torch.count_nonzero(self.reservoir_states).item())
        
        return outputs if not collect_states else (outputs, state_history)
    
    def train_readout(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """Train the readout layer using ridge regression."""
        # Collect reservoir states for training data
        _, state_history = self.forward(X_train, collect_states=True)
        
        # Ridge regression
        alpha = 1e-6  # Regularization parameter
        I = torch.eye(self.nodes)
        
        # Normal equations: w = (X^T X + α I)^(-1) X^T y
        XTX = state_history.T @ state_history + alpha * I
        XTy = state_history.T @ y_train
        
        try:
            self.readout_weights = torch.linalg.solve(XTX, XTy).unsqueeze(1)
            
            if self.metrics_collector:
                self.metrics_collector.increment_counter("readout_training_operations")
            
            self.logger.info("Readout layer trained successfully")
            
        except torch.linalg.LinAlgError:
            self.logger.warning("Training failed, using pseudo-inverse")
            self.readout_weights = torch.pinverse(state_history) @ y_train
    
    def predict(self, X_test: torch.Tensor) -> torch.Tensor:
        """Make predictions on test data."""
        with torch.no_grad():
            predictions = self.forward(X_test, collect_states=False)
        return predictions
    
    def build_network(self) -> PhotonicSNN:
        """Build PhotonicSNN representation of the reservoir."""
        # Create a network topology that represents the reservoir
        input_size = 10  # Assume 10-dimensional input
        topology = [input_size, self.nodes, 1]
        
        network = PhotonicSNN(
            topology=topology,
            wavelength=self.config.wavelength
        )
        
        # Override forward pass to use reservoir dynamics
        network.forward = lambda x, duration=None: self.forward(x)
        
        return network
    
    def estimate_resources(self) -> Dict[str, float]:
        """Estimate reservoir resource requirements."""
        # Each node requires a nonlinear element and connections
        node_area = 50e-12  # 50 μm² per node
        connection_area = 5e-12  # 5 μm² per connection
        
        total_connections = torch.count_nonzero(self.reservoir_matrix).item()
        total_area = self.nodes * node_area + total_connections * connection_area
        
        # Power estimates
        node_power = 50e-6  # 50 μW per node
        connection_power = 1e-6  # 1 μW per connection
        
        total_power = self.nodes * node_power + total_connections * connection_power
        
        return {
            "total_area_m2": total_area,
            "total_power_w": total_power,
            "nodes": self.nodes,
            "connections": total_connections,
            "memory_capacity": self._estimate_memory_capacity()
        }
    
    def _estimate_memory_capacity(self) -> float:
        """Estimate memory capacity of the reservoir."""
        # Memory capacity based on spectral radius and connectivity
        effective_connectivity = torch.count_nonzero(self.reservoir_matrix).item() / (self.nodes**2)
        memory_capacity = self.nodes * effective_connectivity * (self.spectral_radius / 2.0)
        return memory_capacity
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze reservoir performance characteristics."""
        resource_estimates = self.estimate_resources()
        
        # Analyze reservoir properties
        eigenvalues = torch.linalg.eigvals(self.reservoir_matrix)
        spectral_radius = torch.max(torch.abs(eigenvalues)).item()
        
        # Effective connectivity
        effective_connectivity = torch.count_nonzero(self.reservoir_matrix).item() / (self.nodes**2)
        
        # Delay characteristics
        active_delays = self.delay_matrix[self.reservoir_matrix != 0]
        
        return {
            "resource_estimates": resource_estimates,
            "reservoir_properties": {
                "spectral_radius": spectral_radius,
                "effective_connectivity": effective_connectivity,
                "nodes": self.nodes,
                "nonlinearity_type": self.nonlinearity
            },
            "delay_characteristics": {
                "min_delay_s": torch.min(active_delays).item() if len(active_delays) > 0 else 0,
                "max_delay_s": torch.max(active_delays).item() if len(active_delays) > 0 else 0,
                "mean_delay_s": torch.mean(active_delays).item() if len(active_delays) > 0 else 0
            },
            "performance_metrics": {
                "memory_capacity": resource_estimates["memory_capacity"],
                "processing_bandwidth_hz": 1 / torch.mean(active_delays).item() if len(active_delays) > 0 else 1e9,
                "computational_efficiency": self.nodes / resource_estimates["total_power_w"]
            }
        }


class ConvolutionalPhotonicNetwork(PhotonicArchitecture):
    """
    Photonic convolutional neural network for image processing.
    
    Implements optical convolution operations using photonic components
    for high-speed parallel image processing.
    """
    
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        config: Optional[ArchitectureConfig] = None
    ):
        super().__init__(config or ArchitectureConfig())
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Convolution weights (trainable)
        self.conv_weights = torch.randn(
            output_channels, input_channels, kernel_size, kernel_size
        ) * 0.1
        
        # Photonic implementation components
        self.optical_correlators = self._create_optical_correlators()
        
    def _create_optical_correlators(self) -> List[PhotonicCrossbar]:
        """Create optical correlators for parallel convolution."""
        correlators = []
        
        for out_ch in range(self.output_channels):
            # Each output channel has its own correlator
            correlator = PhotonicCrossbar(
                rows=self.input_channels * self.kernel_size * self.kernel_size,
                cols=1,  # Single output per correlator
                weight_bits=self.config.weight_resolution,
                config=self.config
            )
            
            # Program kernel weights
            kernel_weights = self.conv_weights[out_ch].flatten().unsqueeze(1)
            correlator.program_weights(kernel_weights)
            
            correlators.append(correlator)
        
        return correlators
    
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        """Perform optical convolution on input image."""
        batch_size, channels, height, width = input_image.shape
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        output = torch.zeros(batch_size, self.output_channels, out_height, out_width)
        
        # Process each sample in the batch
        for b in range(batch_size):
            # Extract patches for convolution
            patches = self._extract_patches(input_image[b])
            
            # Apply optical correlators in parallel
            for out_ch, correlator in enumerate(self.optical_correlators):
                # Process all patches for this output channel
                channel_output = torch.zeros(out_height, out_width)
                
                patch_idx = 0
                for i in range(0, out_height, self.stride):
                    for j in range(0, out_width, self.stride):
                        if patch_idx < patches.shape[0]:
                            patch = patches[patch_idx].flatten()
                            result = correlator.forward(patch)
                            channel_output[i, j] = result[0]  # Single output
                            patch_idx += 1
                
                output[b, out_ch] = channel_output
        
        return output
    
    def _extract_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Extract convolution patches from image."""
        channels, height, width = image.shape
        
        # Apply padding
        padded = torch.nn.functional.pad(image, 
                                       (self.padding, self.padding, self.padding, self.padding))
        
        patches = []
        for i in range(0, height, self.stride):
            for j in range(0, width, self.stride):
                if i + self.kernel_size <= height + 2 * self.padding and \
                   j + self.kernel_size <= width + 2 * self.padding:
                    patch = padded[:, i:i+self.kernel_size, j:j+self.kernel_size]
                    patches.append(patch)
        
        return torch.stack(patches)
    
    def build_network(self) -> PhotonicSNN:
        """Build PhotonicSNN representation (simplified)."""
        # Flatten convolution for SNN representation
        input_size = self.input_channels * 32 * 32  # Assume 32x32 input
        output_size = self.output_channels * 32 * 32  # Same size output
        
        topology = [input_size, output_size]
        
        network = PhotonicSNN(
            topology=topology,
            wavelength=self.config.wavelength
        )
        
        return network
    
    def estimate_resources(self) -> Dict[str, float]:
        """Estimate convolution network resources."""
        total_area = 0.0
        total_power = 0.0
        
        for correlator in self.optical_correlators:
            resources = correlator.estimate_resources()
            total_area += resources["total_area_m2"]
            total_power += resources["total_power_w"]
        
        return {
            "total_area_m2": total_area,
            "total_power_w": total_power,
            "correlators": len(self.optical_correlators),
            "total_weights": self.conv_weights.numel(),
            "parallelism_factor": self.output_channels
        }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze convolution performance."""
        resource_estimates = self.estimate_resources()
        
        # Calculate theoretical throughput
        operations_per_conv = (
            self.input_channels * self.kernel_size * self.kernel_size * 
            self.output_channels
        )
        
        optical_bandwidth = 3e8 / self.config.wavelength
        throughput = optical_bandwidth / operations_per_conv
        
        return {
            "resource_estimates": resource_estimates,
            "convolution_properties": {
                "kernel_size": self.kernel_size,
                "input_channels": self.input_channels,
                "output_channels": self.output_channels,
                "operations_per_convolution": operations_per_conv
            },
            "performance_metrics": {
                "theoretical_throughput_fps": throughput,
                "parallelism_advantage": self.output_channels,
                "power_efficiency_ops_per_watt": operations_per_conv / resource_estimates["total_power_w"]
            }
        }


# Factory functions for common architectures
def create_mnist_photonic_crossbar() -> PhotonicCrossbar:
    """Create photonic crossbar optimized for MNIST."""
    config = ArchitectureConfig(
        wavelength=1550e-9,
        power_budget=5e-3,  # 5 mW
        area_budget=500e-6,  # 0.5 mm²
        weight_resolution=6   # 6-bit weights sufficient for MNIST
    )
    
    return PhotonicCrossbar(
        rows=784,  # 28x28 pixel input
        cols=256,  # Hidden layer size
        weight_bits=6,
        modulator_type="microring",
        routing_algorithm="minimize_loss",
        config=config
    )


def create_temporal_photonic_reservoir() -> PhotonicReservoir:
    """Create photonic reservoir for temporal signal processing."""
    config = ArchitectureConfig(
        wavelength=1550e-9,
        power_budget=20e-3,  # 20 mW for reservoir
        target_frequency=10e9,  # 10 GHz for high-speed processing
        enable_plasticity=False  # Fixed reservoir
    )
    
    return PhotonicReservoir(
        nodes=200,
        connectivity=0.15,
        delay_distribution="exponential",
        nonlinearity="semiconductor_optical_amplifier",
        spectral_radius=1.1,
        config=config
    )


def create_vision_photonic_cnn() -> ConvolutionalPhotonicNetwork:
    """Create photonic CNN for computer vision."""
    config = ArchitectureConfig(
        wavelength=1550e-9,
        power_budget=50e-3,  # 50 mW for CNN
        area_budget=2e-3,    # 2 mm²
        weight_resolution=8,  # 8-bit for vision
        routing_algorithm="minimize_crossings"
    )
    
    return ConvolutionalPhotonicNetwork(
        input_channels=3,   # RGB
        output_channels=32, # Feature maps
        kernel_size=3,
        stride=1,
        padding=1,
        config=config
    )
