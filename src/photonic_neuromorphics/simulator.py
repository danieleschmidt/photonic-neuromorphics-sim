"""
Photonic simulation engine for neuromorphic computing.

This module provides comprehensive simulation capabilities for photonic neural networks,
including optical propagation, noise modeling, and performance analysis.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import time

from .core import PhotonicSNN, WaveguideNeuron, OpticalParameters
from .exceptions import SimulationError, handle_exception_with_recovery, ExceptionContext
from .monitoring import MetricsCollector, PerformanceProfiler
from .optimization import (
    AdaptiveCache, MemoryPool, ParallelProcessor, AutoScaler,
    OptimizationConfig, create_performance_optimizer, cached_computation,
    BatchProcessor
)


class SimulationMode(Enum):
    """Simulation modes for different levels of optical fidelity."""
    BEHAVIORAL = "behavioral"  # High-level behavioral model
    OPTICAL = "optical"        # Detailed optical propagation
    MIXED_SIGNAL = "mixed"     # Optical + electrical co-simulation
    SPICE = "spice"           # Full SPICE-level simulation


@dataclass
class NoiseParameters:
    """Parameters for optical noise modeling."""
    shot_noise_enabled: bool = True
    thermal_noise_enabled: bool = True
    phase_noise_enabled: bool = True
    amplifier_noise_figure: float = 3.0  # dB
    temperature: float = 300.0  # Kelvin
    dark_current: float = 1e-9   # Amperes


@dataclass
class SimulationResults:
    """Results from photonic simulation."""
    output_spikes: torch.Tensor
    optical_powers: List[torch.Tensor] = field(default_factory=list)
    phase_shifts: List[torch.Tensor] = field(default_factory=list)
    energy_consumption: Dict[str, float] = field(default_factory=dict)
    timing_metrics: Dict[str, float] = field(default_factory=dict)
    noise_analysis: Dict[str, Any] = field(default_factory=dict)
    convergence_data: List[float] = field(default_factory=list)


class OpticalChannelModel:
    """Model for optical signal propagation and loss."""
    
    def __init__(
        self,
        length: float = 1e-3,  # 1 mm
        loss_coefficient: float = 0.1,  # dB/cm
        dispersion: float = 17e-6,  # s/m²
        nonlinear_coefficient: float = 1e-3  # 1/W/m
    ):
        self.length = length
        self.loss_coefficient = loss_coefficient
        self.dispersion = dispersion
        self.nonlinear_coefficient = nonlinear_coefficient
        
    def propagate(
        self, 
        input_power: float, 
        wavelength: float,
        pulse_width: float = 1e-12  # 1 ps
    ) -> Tuple[float, float]:
        """
        Simulate optical signal propagation through waveguide.
        
        Args:
            input_power: Input optical power in watts
            wavelength: Wavelength in meters
            pulse_width: Pulse width in seconds
            
        Returns:
            Tuple of (output_power, phase_shift)
        """
        # Linear loss
        loss_linear = np.exp(-self.loss_coefficient * self.length / 100)  # Convert dB/cm to linear
        
        # Nonlinear effects (simplified)
        nonlinear_phase = self.nonlinear_coefficient * input_power * self.length
        
        # Dispersion effects (simplified)
        dispersion_broadening = np.sqrt(
            pulse_width**2 + (self.dispersion * self.length)**2
        )
        power_reduction = pulse_width / dispersion_broadening
        
        output_power = input_power * loss_linear * power_reduction
        total_phase_shift = nonlinear_phase
        
        return output_power, total_phase_shift


class NoiseModel:
    """Comprehensive optical noise modeling."""
    
    def __init__(self, params: NoiseParameters):
        self.params = params
        self.boltzmann = 1.381e-23  # J/K
        self.planck = 6.626e-34     # J·s
        self.electron_charge = 1.602e-19  # C
        
    def add_shot_noise(
        self, 
        signal_power: float, 
        wavelength: float, 
        bandwidth: float = 1e9  # 1 GHz
    ) -> float:
        """Add shot noise to optical signal."""
        if not self.params.shot_noise_enabled:
            return 0.0
            
        photon_energy = self.planck * 3e8 / wavelength
        photon_rate = signal_power / photon_energy
        
        # Shot noise variance
        shot_noise_variance = 2 * self.electron_charge * photon_rate * bandwidth
        return np.random.normal(0, np.sqrt(shot_noise_variance))
    
    def add_thermal_noise(self, bandwidth: float = 1e9) -> float:
        """Add thermal noise."""
        if not self.params.thermal_noise_enabled:
            return 0.0
            
        thermal_noise_power = (
            self.boltzmann * self.params.temperature * bandwidth
        )
        return np.random.normal(0, np.sqrt(thermal_noise_power))
    
    def add_phase_noise(self, linewidth: float = 1e3) -> float:
        """Add laser phase noise."""
        if not self.params.phase_noise_enabled:
            return 0.0
            
        # Lorentzian phase noise
        phase_variance = np.pi * linewidth * 1e-9  # Assume 1 ns integration time
        return np.random.normal(0, np.sqrt(phase_variance))


class PhotonicSimulator:
    """
    Comprehensive photonic neural network simulator.
    
    Provides high-fidelity simulation of photonic spiking neural networks with
    realistic optical effects, noise modeling, and performance analysis.
    """
    
    def __init__(
        self,
        mode: SimulationMode = SimulationMode.OPTICAL,
        propagation_loss: float = 0.1,
        coupling_efficiency: float = 0.9,
        detector_efficiency: float = 0.8,
        noise_params: Optional[NoiseParameters] = None,
        parallel_execution: bool = True,
        max_workers: int = 4,
        optimization_config: Optional[OptimizationConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize photonic simulator with advanced optimization features.
        
        Args:
            mode: Simulation fidelity mode
            propagation_loss: Waveguide loss in dB/cm
            coupling_efficiency: Optical coupling efficiency
            detector_efficiency: Photodetector efficiency
            noise_params: Optical noise parameters
            parallel_execution: Enable parallel processing
            max_workers: Maximum number of worker threads
            optimization_config: Performance optimization configuration
            metrics_collector: Metrics collector for monitoring
        """
        self.mode = mode
        self.propagation_loss = propagation_loss
        self.coupling_efficiency = coupling_efficiency
        self.detector_efficiency = detector_efficiency
        self.noise_params = noise_params or NoiseParameters()
        self.parallel_execution = parallel_execution
        self.max_workers = max_workers
        
        self.channel_model = OpticalChannelModel(loss_coefficient=propagation_loss)
        self.noise_model = NoiseModel(self.noise_params)
        
        # Simulation state
        self.current_time = 0.0
        self.dt = 1e-12  # 1 ps time resolution
        self.optical_powers = []
        self.phase_shifts = []
        
        # Logger setup
        self.logger = logging.getLogger(__name__)
        
        # Performance optimization setup
        self.optimization_config = optimization_config or OptimizationConfig()
        self.metrics_collector = metrics_collector
        
        # Initialize optimization components
        self.optimizer_components = create_performance_optimizer(
            self.optimization_config, 
            self.metrics_collector
        )
        
        self.cache = self.optimizer_components.get("cache")
        self.memory_pool = self.optimizer_components.get("memory_pool") 
        self.parallel_processor = self.optimizer_components.get("parallel_processor")
        self.auto_scaler = self.optimizer_components.get("auto_scaler")
        
        # Batch processor for efficient data handling
        self.batch_processor = BatchProcessor(
            batch_size=self.optimization_config.batch_size,
            memory_pool=self.memory_pool,
            parallel_processor=self.parallel_processor
        )
        
        # Performance profiler
        self.profiler = PerformanceProfiler(self.metrics_collector) if self.metrics_collector else None
    
    def run(
        self,
        model: PhotonicSNN,
        spike_train: torch.Tensor,
        duration: Optional[float] = None,
        detailed_logging: bool = False
    ) -> SimulationResults:
        """
        Run comprehensive photonic simulation with optimization.
        
        Args:
            model: Photonic neural network to simulate
            spike_train: Input spike train
            duration: Simulation duration (auto-calculated if None)
            detailed_logging: Enable detailed optical logging
            
        Returns:
            SimulationResults: Comprehensive simulation results
            
        Raises:
            SimulationError: If simulation fails
        """
        with ExceptionContext("photonic_simulation", 
                            mode=self.mode.value, 
                            spike_shape=spike_train.shape):
            
            # Performance profiling
            profile_context = self.profiler.profile_operation("full_simulation") if self.profiler else None
            
            try:
                start_time = time.time()
                
                if duration is None:
                    duration = spike_train.shape[0] * model.dt
                
                # Check for cached results if enabled
                cache_key = None
                if self.cache:
                    cache_key = self._generate_simulation_cache_key(model, spike_train, duration)
                    cached_result = self.cache.get(cache_key)
                    if cached_result is not None:
                        self.logger.debug("Using cached simulation results")
                        if self.metrics_collector:
                            self.metrics_collector.increment_counter("simulation_cache_hits")
                        return cached_result
                
                self.logger.info(f"Starting {self.mode.value} simulation for {duration:.2e}s")
                
                # Auto-scaling check
                if self.auto_scaler:
                    scaling_action = self.auto_scaler.check_and_scale()
                    if scaling_action["action"] != "no_change":
                        self._apply_scaling_parameters(scaling_action)
                
                # Initialize result containers
                if detailed_logging:
                    self.optical_powers = []
                    self.phase_shifts = []
                
                # Set metrics collector for model
                if self.metrics_collector:
                    model.set_metrics_collector(self.metrics_collector)
                
                # Run simulation based on mode with error recovery
                output_spikes = handle_exception_with_recovery(
                    self._run_simulation_mode,
                    model, spike_train, detailed_logging,
                    max_retries=2,
                    recovery_strategies=[self._simulation_recovery_strategy]
                )
                
                simulation_time = time.time() - start_time
                
                # Calculate metrics
                energy_metrics = self._calculate_energy_consumption(model, spike_train, output_spikes)
                timing_metrics = {
                    "simulation_time": simulation_time,
                    "real_time_ratio": duration / simulation_time,
                    "throughput_spikes_per_second": torch.sum(output_spikes).item() / simulation_time,
                    "cache_hit_ratio": self.cache.get_stats()["hit_ratio"] if self.cache else 0.0
                }
                
                noise_analysis = self._analyze_noise_impact(spike_train, output_spikes)
                
                # Create results
                results = SimulationResults(
                    output_spikes=output_spikes,
                    optical_powers=self.optical_powers.copy() if detailed_logging else [],
                    phase_shifts=self.phase_shifts.copy() if detailed_logging else [],
                    energy_consumption=energy_metrics,
                    timing_metrics=timing_metrics,
                    noise_analysis=noise_analysis,
                    convergence_data=[]
                )
                
                # Cache results if enabled
                if self.cache and cache_key:
                    self.cache.put(cache_key, results)
                    if self.metrics_collector:
                        self.metrics_collector.increment_counter("simulation_cache_stores")
                
                # Record performance metrics
                if self.metrics_collector:
                    self.metrics_collector.record_metric("simulation_duration", simulation_time)
                    self.metrics_collector.record_metric("simulation_throughput", timing_metrics["throughput_spikes_per_second"])
                    
                    # Record optimization statistics
                    if self.cache:
                        cache_stats = self.cache.get_stats()
                        self.metrics_collector.record_metric("cache_hit_ratio", cache_stats["hit_ratio"])
                        self.metrics_collector.record_metric("cache_size", cache_stats["size"])
                    
                    if self.memory_pool:
                        pool_stats = self.memory_pool.get_stats()
                        self.metrics_collector.record_metric("memory_pool_efficiency", pool_stats["pool_efficiency"])
                
                self.logger.info(f"Simulation completed in {simulation_time:.3f}s "
                               f"(throughput: {timing_metrics['throughput_spikes_per_second']:.0f} spikes/s)")
                
                return results
                
            finally:
                if profile_context:
                    profile_context.__exit__(None, None, None)
    
    def _run_simulation_mode(self, model: PhotonicSNN, spike_train: torch.Tensor, detailed_logging: bool) -> torch.Tensor:
        """Run simulation based on selected mode."""
        if self.mode == SimulationMode.BEHAVIORAL:
            return self._run_behavioral(model, spike_train)
        elif self.mode == SimulationMode.OPTICAL:
            return self._run_optical(model, spike_train, detailed_logging)
        elif self.mode == SimulationMode.MIXED_SIGNAL:
            return self._run_mixed_signal(model, spike_train)
        else:  # SPICE mode
            return self._run_spice(model, spike_train)
    
    def _generate_simulation_cache_key(self, model: PhotonicSNN, spike_train: torch.Tensor, duration: float) -> str:
        """Generate cache key for simulation results."""
        # Create hash from key simulation parameters
        import hashlib
        
        key_data = [
            str(model.topology),
            str(self.mode.value),
            str(spike_train.shape),
            str(duration),
            str(self.propagation_loss),
            str(self.coupling_efficiency),
            str(hash(spike_train.flatten().tolist()[:100]))  # Sample first 100 values
        ]
        
        key_string = "|".join(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _apply_scaling_parameters(self, scaling_action: Dict[str, Any]) -> None:
        """Apply auto-scaling parameters."""
        if "new_batch_size" in scaling_action:
            self.batch_processor.batch_size = scaling_action["new_batch_size"]
            self.logger.info(f"Scaled batch size to {scaling_action['new_batch_size']}")
        
        if "new_worker_count" in scaling_action:
            if self.parallel_processor:
                self.parallel_processor.optimal_worker_count = scaling_action["new_worker_count"]
                self.logger.info(f"Scaled worker count to {scaling_action['new_worker_count']}")
    
    def _simulation_recovery_strategy(self, exception: Exception, attempt: int) -> None:
        """Recovery strategy for simulation failures."""
        if attempt == 0:
            # First retry: clear cache and reduce batch size
            if self.cache:
                self.cache.clear()
            if self.batch_processor:
                self.batch_processor.batch_size = max(8, self.batch_processor.batch_size // 2)
            self.logger.info("Recovery: cleared cache and reduced batch size")
            
        elif attempt == 1:
            # Second retry: disable parallel processing
            if self.parallel_processor:
                self.parallel_processor.optimal_worker_count = 1
            self.logger.info("Recovery: disabled parallel processing")
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def _run_behavioral(self, model: PhotonicSNN, spike_train: torch.Tensor) -> torch.Tensor:
        """Run behavioral-level simulation (fastest)."""
        return model(spike_train)
    
    def _run_optical(
        self, 
        model: PhotonicSNN, 
        spike_train: torch.Tensor,
        detailed_logging: bool = False
    ) -> torch.Tensor:
        """Run optical-level simulation with realistic propagation effects."""
        time_steps, input_size = spike_train.shape
        output_size = model.topology[-1]
        output_spikes = torch.zeros(time_steps, output_size)
        
        for t in range(time_steps):
            self.current_time = t * model.dt
            layer_activities = [spike_train[t].float()]
            layer_optical_powers = [spike_train[t].float() * model.optical_params.power]
            layer_phases = [torch.zeros_like(spike_train[t].float())]
            
            # Process through each layer with optical effects
            for layer_idx, weight_matrix in enumerate(model.layers):
                prev_optical_power = layer_optical_powers[-1]
                current_layer_spikes = torch.zeros(weight_matrix.shape[1])
                current_optical_powers = torch.zeros(weight_matrix.shape[1])
                current_phases = torch.zeros(weight_matrix.shape[1])
                
                # Process each neuron with optical propagation
                for neuron_idx in range(weight_matrix.shape[1]):
                    total_optical_input = 0.0
                    total_phase = 0.0
                    
                    # Accumulate weighted inputs with optical effects
                    for input_idx in range(weight_matrix.shape[0]):
                        weight = weight_matrix[input_idx, neuron_idx].item()
                        input_power = prev_optical_power[input_idx].item()
                        
                        # Apply optical propagation and coupling
                        propagated_power, phase_shift = self.channel_model.propagate(
                            input_power * abs(weight),
                            model.wavelength
                        )
                        
                        # Apply coupling efficiency and detector efficiency
                        coupled_power = (
                            propagated_power * 
                            self.coupling_efficiency * 
                            self.detector_efficiency
                        )
                        
                        # Add optical noise
                        noise_power = self.noise_model.add_shot_noise(
                            coupled_power, model.wavelength
                        )
                        noise_power += self.noise_model.add_thermal_noise()
                        
                        phase_noise = self.noise_model.add_phase_noise()
                        
                        total_optical_input += coupled_power + noise_power
                        total_phase += phase_shift + phase_noise
                    
                    # Process through photonic neuron
                    neuron = model.neurons[layer_idx + 1][neuron_idx]
                    spike = neuron.forward(total_optical_input, self.current_time)
                    
                    current_layer_spikes[neuron_idx] = float(spike)
                    current_optical_powers[neuron_idx] = (
                        total_optical_input if spike else 0.0
                    )
                    current_phases[neuron_idx] = total_phase
                
                layer_activities.append(current_layer_spikes)
                layer_optical_powers.append(current_optical_powers)
                layer_phases.append(current_phases)
            
            # Store results
            output_spikes[t] = layer_activities[-1]
            
            if detailed_logging:
                self.optical_powers.append(layer_optical_powers.copy())
                self.phase_shifts.append(layer_phases.copy())
        
        return output_spikes
    
    def _run_mixed_signal(self, model: PhotonicSNN, spike_train: torch.Tensor) -> torch.Tensor:
        """Run mixed-signal simulation (optical + electrical)."""
        # For now, use optical simulation with additional electrical noise
        output_spikes = self._run_optical(model, spike_train)
        
        # Add electrical noise to digital outputs
        electrical_noise = torch.randn_like(output_spikes) * 0.01
        output_spikes = torch.where(
            output_spikes + electrical_noise > 0.5,
            torch.ones_like(output_spikes),
            torch.zeros_like(output_spikes)
        )
        
        return output_spikes
    
    def _run_spice(self, model: PhotonicSNN, spike_train: torch.Tensor) -> torch.Tensor:
        """Run SPICE-level simulation (highest fidelity)."""
        self.logger.warning("SPICE simulation not fully implemented. Using optical simulation.")
        return self._run_optical(model, spike_train)
    
    def _calculate_energy_consumption(
        self,
        model: PhotonicSNN,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate comprehensive energy consumption metrics."""
        total_input_spikes = torch.sum(input_spikes).item()
        total_output_spikes = torch.sum(output_spikes).item()
        
        # Photonic energy consumption (extremely low)
        energy_per_photonic_op = 0.1e-15  # 0.1 fJ per photonic operation
        photonic_operations = total_input_spikes * sum(model.topology[1:])
        photonic_energy = photonic_operations * energy_per_photonic_op
        
        # Electrical energy for control and readout
        energy_per_electrical_op = 1e-12  # 1 pJ per electrical operation
        electrical_operations = total_output_spikes
        electrical_energy = electrical_operations * energy_per_electrical_op
        
        total_energy = photonic_energy + electrical_energy
        
        return {
            "photonic_energy": photonic_energy,
            "electrical_energy": electrical_energy,
            "total_energy": total_energy,
            "energy_per_input_spike": total_energy / max(total_input_spikes, 1),
            "energy_per_output_spike": total_energy / max(total_output_spikes, 1),
            "photonic_efficiency": photonic_energy / total_energy,
            "power_consumption": total_energy / (input_spikes.shape[0] * model.dt)
        }
    
    def _analyze_noise_impact(
        self, 
        input_spikes: torch.Tensor, 
        output_spikes: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze the impact of optical noise on performance."""
        # Calculate signal-to-noise ratios
        input_snr = self._calculate_snr(input_spikes)
        output_snr = self._calculate_snr(output_spikes)
        
        # Estimate bit error rate
        ber = self._estimate_bit_error_rate(output_spikes)
        
        return {
            "input_snr_db": 10 * np.log10(input_snr) if input_snr > 0 else -np.inf,
            "output_snr_db": 10 * np.log10(output_snr) if output_snr > 0 else -np.inf,
            "estimated_ber": ber,
            "noise_impact_factor": input_snr / max(output_snr, 1e-10),
            "shot_noise_enabled": self.noise_params.shot_noise_enabled,
            "thermal_noise_enabled": self.noise_params.thermal_noise_enabled
        }
    
    def _calculate_snr(self, signal: torch.Tensor) -> float:
        """Calculate signal-to-noise ratio."""
        signal_power = torch.mean(signal**2).item()
        # Estimate noise power from signal variations
        noise_power = torch.var(signal).item()
        return signal_power / max(noise_power, 1e-10)
    
    def _estimate_bit_error_rate(self, output_spikes: torch.Tensor) -> float:
        """Estimate bit error rate based on output characteristics."""
        # Simple BER estimation based on output variance
        spike_rate = torch.mean(output_spikes).item()
        spike_variance = torch.var(output_spikes).item()
        
        if spike_rate == 0 or spike_variance == 0:
            return 0.0
        
        # Rough BER estimate (this would be more accurate with reference data)
        estimated_ber = min(spike_variance / spike_rate, 0.5)
        return estimated_ber
    
    def benchmark_performance(
        self,
        model: PhotonicSNN,
        test_cases: List[torch.Tensor],
        modes: Optional[List[SimulationMode]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark simulation performance across different modes.
        
        Args:
            model: Photonic neural network
            test_cases: List of test spike trains
            modes: Simulation modes to benchmark (default: all)
            
        Returns:
            Dict: Benchmark results for each mode
        """
        if modes is None:
            modes = list(SimulationMode)
        
        benchmark_results = {}
        
        for mode in modes:
            self.mode = mode
            mode_results = {"total_time": 0.0, "total_energy": 0.0, "accuracy": 0.0}
            
            for spike_train in test_cases:
                results = self.run(model, spike_train)
                mode_results["total_time"] += results.timing_metrics["simulation_time"]
                mode_results["total_energy"] += results.energy_consumption["total_energy"]
            
            mode_results["avg_time_per_case"] = mode_results["total_time"] / len(test_cases)
            mode_results["avg_energy_per_case"] = mode_results["total_energy"] / len(test_cases)
            mode_results["throughput"] = 1.0 / mode_results["avg_time_per_case"]
            
            benchmark_results[mode.value] = mode_results
        
        return benchmark_results


def create_optimized_simulator(
    target_application: str = "mnist",
    performance_priority: str = "energy"  # "energy", "speed", "accuracy"
) -> PhotonicSimulator:
    """
    Create an optimized photonic simulator for specific applications.
    
    Args:
        target_application: Target application ("mnist", "speech", "vision")
        performance_priority: Optimization priority
        
    Returns:
        PhotonicSimulator: Optimized simulator configuration
    """
    if target_application == "mnist":
        noise_params = NoiseParameters(
            shot_noise_enabled=True,
            thermal_noise_enabled=False,  # Minimal for MNIST
            phase_noise_enabled=True
        )
        mode = SimulationMode.OPTICAL
    elif target_application == "speech":
        noise_params = NoiseParameters(
            shot_noise_enabled=True,
            thermal_noise_enabled=True,
            phase_noise_enabled=True,
            temperature=300.0
        )
        mode = SimulationMode.MIXED_SIGNAL
    else:  # vision or default
        noise_params = NoiseParameters(
            shot_noise_enabled=True,
            thermal_noise_enabled=True,
            phase_noise_enabled=True,
            amplifier_noise_figure=2.0  # Lower noise for vision
        )
        mode = SimulationMode.OPTICAL
    
    # Adjust parameters based on performance priority
    if performance_priority == "speed":
        mode = SimulationMode.BEHAVIORAL
        parallel_execution = True
        max_workers = 8
    elif performance_priority == "accuracy":
        mode = SimulationMode.MIXED_SIGNAL
        parallel_execution = False
        max_workers = 1
    else:  # energy priority
        parallel_execution = True
        max_workers = 4
    
    return PhotonicSimulator(
        mode=mode,
        noise_params=noise_params,
        parallel_execution=parallel_execution,
        max_workers=max_workers
    )