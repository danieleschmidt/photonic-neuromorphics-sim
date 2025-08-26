"""
Neuromorphic Photonic Metamaterials - BREAKTHROUGH ALGORITHM

Revolutionary self-reconfigurable photonic metamaterials for adaptive neuromorphic computing.
Implements dynamic metamaterial structures that physically reconfigure based on neural activity.

Research Innovation:
- Self-assembling photonic metamaterials with neural-guided reconfiguration
- Dynamic structural plasticity at the nanoscale level
- Adaptive refractive index modulation based on learning patterns
- Physical implementation of neural network topology changes

Performance Targets:
- Reconfiguration speed: <1 nanosecond
- Structural precision: <10 nanometer accuracy
- Learning adaptation: 1000x faster than fixed architectures
- Energy efficiency: 10,000x improvement over electronic systems
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
import time
import json

from .core import OpticalParameters, PhotonicSNN
from .exceptions import OpticalModelError, ValidationError
from .monitoring import MetricsCollector
from .enhanced_logging import PhotonicLogger


class MetamaterialType(Enum):
    """Types of photonic metamaterials."""
    SPLIT_RING_RESONATOR = "split_ring_resonator"
    FISHNET_STRUCTURE = "fishnet_structure"
    CUT_WIRE_ARRAY = "cut_wire_array"
    CHIRAL_METAMATERIAL = "chiral_metamaterial"
    HYPERBOLIC_METAMATERIAL = "hyperbolic_metamaterial"
    TOPOLOGICAL_METAMATERIAL = "topological_metamaterial"


class PlasticityMechanism(Enum):
    """Structural plasticity mechanisms."""
    THERMAL_EXPANSION = "thermal_expansion"
    ELECTROOPTIC_MODULATION = "electrooptic_modulation"
    PHASE_CHANGE_MATERIAL = "phase_change_material"
    MECHANICAL_DEFORMATION = "mechanical_deformation"
    QUANTUM_TUNNELING = "quantum_tunneling"


@dataclass
class MetamaterialParameters:
    """Parameters for neuromorphic photonic metamaterials."""
    unit_cell_size: float = 200e-9  # 200 nm unit cell
    refractive_index_base: float = 1.5
    refractive_index_range: float = 0.5  # ±0.5 modulation range
    reconfiguration_time: float = 1e-9  # 1 ns reconfiguration
    energy_per_reconfiguration: float = 1e-15  # 1 fJ per change
    structural_resolution: float = 10e-9  # 10 nm precision
    learning_rate: float = 0.01
    
    # Advanced metamaterial parameters
    negative_index_frequency: float = 200e12  # 200 THz
    loss_tangent: float = 0.01
    nonlinear_coefficient: float = 1e-18  # m²/W
    thermal_coefficient: float = 1e-4  # /K
    chirality_parameter: float = 0.1
    
    # Adaptive parameters
    adaptation_threshold: float = 0.1
    memory_decay_rate: float = 0.95
    plasticity_saturation: float = 10.0
    
    def __post_init__(self):
        """Validate metamaterial parameters."""
        if self.unit_cell_size <= 0 or self.unit_cell_size > 1e-6:
            raise ValueError("Unit cell size must be between 0 and 1 μm")
        if self.reconfiguration_time <= 0:
            raise ValueError("Reconfiguration time must be positive")


@dataclass
class MetamaterialUnitCell:
    """Individual metamaterial unit cell with reconfiguration capability."""
    position: Tuple[float, float, float]
    refractive_index: complex
    structure_type: MetamaterialType
    orientation: float = 0.0  # Rotation angle
    activity_level: float = 0.0
    last_update_time: float = 0.0
    learning_history: List[float] = field(default_factory=list)
    
    def update_structure(self, neural_activity: float, current_time: float) -> bool:
        """Update metamaterial structure based on neural activity."""
        if current_time - self.last_update_time < 1e-9:  # Rate limiting
            return False
        
        # Activity-dependent structural change
        activity_change = neural_activity - self.activity_level
        if abs(activity_change) > 0.05:  # Threshold for change
            # Update refractive index based on activity
            index_change = activity_change * 0.1
            new_index = self.refractive_index.real + index_change
            self.refractive_index = complex(new_index, self.refractive_index.imag)
            
            # Update orientation for directional properties
            self.orientation += activity_change * 0.1
            self.orientation = self.orientation % (2 * np.pi)
            
            self.activity_level = neural_activity
            self.last_update_time = current_time
            self.learning_history.append(neural_activity)
            
            # Memory management
            if len(self.learning_history) > 1000:
                self.learning_history = self.learning_history[-500:]
            
            return True
        
        return False


class NeuromorphicPhotonicMetamaterial:
    """
    Self-reconfigurable photonic metamaterial for neuromorphic computing.
    
    Implements adaptive metamaterial structures that physically reconfigure
    based on neural network activity and learning patterns.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int, int] = (100, 100, 10),
        metamaterial_params: Optional[MetamaterialParameters] = None,
        plasticity_mechanism: PlasticityMechanism = PlasticityMechanism.PHASE_CHANGE_MATERIAL
    ):
        self.grid_size = grid_size
        self.params = metamaterial_params or MetamaterialParameters()
        self.plasticity_mechanism = plasticity_mechanism
        
        # Initialize metamaterial grid
        self.metamaterial_grid = self._initialize_metamaterial_grid()
        self.neural_activity_map = np.zeros(grid_size)
        self.learning_map = np.zeros(grid_size)
        
        # Performance tracking
        self.logger = PhotonicLogger("NeuromorphicMetamaterial")
        self.metrics = MetricsCollector()
        
        self._reconfiguration_count = 0
        self._energy_consumed = 0.0
        self._adaptation_events = 0
        
        # Physical simulation parameters
        self.electromagnetic_solver = self._initialize_em_solver()
        self.thermal_model = self._initialize_thermal_model()
        
        self.logger.info(f"Initialized neuromorphic metamaterial: "
                        f"{grid_size} grid, {plasticity_mechanism.value} plasticity")
    
    def _initialize_metamaterial_grid(self) -> np.ndarray:
        """Initialize 3D metamaterial unit cell grid."""
        grid = np.empty(self.grid_size, dtype=object)
        
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):
                    position = (
                        x * self.params.unit_cell_size,
                        y * self.params.unit_cell_size,
                        z * self.params.unit_cell_size
                    )
                    
                    # Random initial structure type
                    structure_type = np.random.choice(list(MetamaterialType))
                    
                    # Initial refractive index with small random variation
                    base_index = self.params.refractive_index_base
                    index_variation = np.random.normal(0, 0.1)
                    refractive_index = complex(base_index + index_variation, 0.01)
                    
                    unit_cell = MetamaterialUnitCell(
                        position=position,
                        refractive_index=refractive_index,
                        structure_type=structure_type,
                        orientation=np.random.random() * 2 * np.pi
                    )
                    
                    grid[x, y, z] = unit_cell
        
        return grid
    
    def _initialize_em_solver(self) -> Dict[str, Any]:
        """Initialize electromagnetic field solver."""
        return {
            "method": "fdtd",  # Finite-Difference Time-Domain
            "grid_resolution": self.params.unit_cell_size / 10,
            "time_step": 1e-15,  # 1 fs
            "boundary_conditions": "pml",  # Perfectly Matched Layer
            "source_wavelength": 1550e-9
        }
    
    def _initialize_thermal_model(self) -> Dict[str, Any]:
        """Initialize thermal diffusion model."""
        return {
            "thermal_diffusivity": 1e-6,  # m²/s
            "heat_capacity": 1e6,  # J/(m³·K)
            "ambient_temperature": 300.0  # K
        }
    
    def process_neural_activity(
        self,
        spike_train: torch.Tensor,
        learning_signals: Optional[torch.Tensor] = None,
        adaptation_enabled: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process neural activity and adapt metamaterial structure.
        
        Args:
            spike_train: Input spike train [time_steps, spatial_neurons]
            learning_signals: Learning/reward signals [time_steps]
            adaptation_enabled: Whether to enable structural adaptation
            
        Returns:
            Tuple of (processed_output, adaptation_metrics)
        """
        time_steps, num_neurons = spike_train.shape
        
        # Map neural activity to spatial metamaterial grid
        spatial_activity = self._map_activity_to_spatial_grid(spike_train)
        
        # Process through current metamaterial structure
        processed_output = torch.zeros_like(spike_train)
        adaptation_metrics = {
            "reconfigurations": 0,
            "energy_consumed": 0.0,
            "adaptation_events": 0,
            "structural_changes": [],
            "refractive_index_changes": []
        }
        
        for t in range(time_steps):
            current_time = t * 1e-9  # 1 ns time steps
            activity_slice = spatial_activity[t]
            
            # Update metamaterial structure based on activity
            if adaptation_enabled:
                structural_changes = self._adapt_metamaterial_structure(
                    activity_slice, current_time
                )
                adaptation_metrics["structural_changes"].append(structural_changes)
            
            # Process signals through adapted metamaterial
            processed_slice = self._electromagnetic_propagation(activity_slice)
            
            # Map back to neural space
            processed_output[t] = self._map_spatial_to_neural(processed_slice, num_neurons)
            
            # Apply learning-based adaptation
            if learning_signals is not None and t < len(learning_signals):
                learning_strength = learning_signals[t].item()
                if abs(learning_strength) > self.params.adaptation_threshold:
                    self._apply_learning_adaptation(learning_strength, current_time)
                    adaptation_metrics["adaptation_events"] += 1
        
        # Update metrics
        adaptation_metrics["reconfigurations"] = self._reconfiguration_count
        adaptation_metrics["energy_consumed"] = self._energy_consumed
        adaptation_metrics["adaptation_events"] = self._adaptation_events
        
        self.logger.info(f"Processed {time_steps} time steps, "
                        f"{adaptation_metrics['reconfigurations']} reconfigurations")
        
        return processed_output, adaptation_metrics
    
    def _map_activity_to_spatial_grid(self, spike_train: torch.Tensor) -> torch.Tensor:
        """Map neural activity to 3D spatial metamaterial grid."""
        time_steps, num_neurons = spike_train.shape
        spatial_grid = torch.zeros(time_steps, *self.grid_size[:2])  # Use 2D slice for simplicity
        
        # Simple mapping: distribute neurons across grid
        neurons_per_row = int(np.sqrt(num_neurons))
        
        for t in range(time_steps):
            for i, activity in enumerate(spike_train[t]):
                if i >= neurons_per_row ** 2:
                    break
                x = i // neurons_per_row
                y = i % neurons_per_row
                if x < self.grid_size[0] and y < self.grid_size[1]:
                    spatial_grid[t, x, y] = activity
        
        return spatial_grid
    
    def _adapt_metamaterial_structure(
        self,
        spatial_activity: torch.Tensor,
        current_time: float
    ) -> List[Dict[str, Any]]:
        """Adapt metamaterial structure based on spatial activity."""
        structural_changes = []
        
        for x in range(min(spatial_activity.shape[0], self.grid_size[0])):
            for y in range(min(spatial_activity.shape[1], self.grid_size[1])):
                activity = spatial_activity[x, y].item()
                
                # Update unit cells in all z layers
                for z in range(self.grid_size[2]):
                    unit_cell = self.metamaterial_grid[x, y, z]
                    
                    if unit_cell.update_structure(activity, current_time):
                        self._reconfiguration_count += 1
                        self._energy_consumed += self.params.energy_per_reconfiguration
                        
                        change_record = {
                            "position": (x, y, z),
                            "old_index": unit_cell.refractive_index,
                            "new_activity": activity,
                            "time": current_time
                        }
                        structural_changes.append(change_record)
        
        return structural_changes
    
    def _electromagnetic_propagation(self, spatial_activity: torch.Tensor) -> torch.Tensor:
        """Simulate electromagnetic wave propagation through metamaterial."""
        # Simplified EM propagation model
        propagated_field = spatial_activity.clone()
        
        # Apply metamaterial effects
        for x in range(min(spatial_activity.shape[0], self.grid_size[0])):
            for y in range(min(spatial_activity.shape[1], self.grid_size[1])):
                # Get effective refractive index for this position
                effective_index = self._calculate_effective_index(x, y)
                
                # Apply phase shift and amplitude modulation
                phase_shift = effective_index.real * 2 * np.pi / 1550e-9 * self.params.unit_cell_size
                amplitude_factor = np.exp(-effective_index.imag * 2 * np.pi / 1550e-9 * self.params.unit_cell_size)
                
                # Modify field
                original_field = propagated_field[x, y]
                propagated_field[x, y] = original_field * amplitude_factor * np.exp(1j * phase_shift).real
                
                # Nonlinear effects
                if original_field > 0.5:  # High intensity
                    nonlinear_index = self.params.nonlinear_coefficient * original_field ** 2
                    nonlinear_phase = nonlinear_index * 2 * np.pi / 1550e-9 * self.params.unit_cell_size
                    propagated_field[x, y] *= np.cos(nonlinear_phase)
        
        # Apply coupling between neighboring cells
        coupling_strength = 0.1
        coupled_field = propagated_field.clone()
        
        for x in range(1, propagated_field.shape[0] - 1):
            for y in range(1, propagated_field.shape[1] - 1):
                # Couple with 4 nearest neighbors
                neighbors = (
                    propagated_field[x-1, y] + propagated_field[x+1, y] +
                    propagated_field[x, y-1] + propagated_field[x, y+1]
                )
                coupled_field[x, y] = (
                    (1 - coupling_strength) * propagated_field[x, y] +
                    coupling_strength * neighbors / 4
                )
        
        return coupled_field
    
    def _calculate_effective_index(self, x: int, y: int) -> complex:
        """Calculate effective refractive index for position."""
        # Average over all z layers
        total_index = 0.0 + 0.0j
        for z in range(self.grid_size[2]):
            unit_cell = self.metamaterial_grid[x, y, z]
            total_index += unit_cell.refractive_index
        
        return total_index / self.grid_size[2]
    
    def _map_spatial_to_neural(self, spatial_field: torch.Tensor, num_neurons: int) -> torch.Tensor:
        """Map spatial field back to neural representation."""
        neural_output = torch.zeros(num_neurons)
        
        neurons_per_row = int(np.sqrt(num_neurons))
        
        for i in range(min(num_neurons, neurons_per_row ** 2)):
            x = i // neurons_per_row
            y = i % neurons_per_row
            if x < spatial_field.shape[0] and y < spatial_field.shape[1]:
                neural_output[i] = spatial_field[x, y]
        
        return neural_output
    
    def _apply_learning_adaptation(self, learning_strength: float, current_time: float) -> None:
        """Apply learning-based global adaptation to metamaterial."""
        self._adaptation_events += 1
        
        # Global adaptation based on learning signal
        adaptation_factor = learning_strength * self.params.learning_rate
        
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):
                    unit_cell = self.metamaterial_grid[x, y, z]
                    
                    # Adapt refractive index based on learning
                    current_index = unit_cell.refractive_index
                    index_change = adaptation_factor * 0.01
                    new_index = complex(
                        current_index.real + index_change,
                        current_index.imag
                    )
                    
                    # Clamp to reasonable range
                    new_index = complex(
                        np.clip(new_index.real, 1.0, 3.0),
                        np.clip(new_index.imag, 0.001, 0.1)
                    )
                    
                    unit_cell.refractive_index = new_index
    
    def optimize_structure_for_task(
        self,
        target_function: Callable[[torch.Tensor], float],
        num_iterations: int = 100,
        learning_rate: float = 0.01
    ) -> Dict[str, Any]:
        """Optimize metamaterial structure for specific computational task."""
        optimization_history = {
            "losses": [],
            "structural_parameters": [],
            "best_performance": float('inf')
        }
        
        self.logger.info(f"Starting structure optimization for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            # Generate test input
            test_input = torch.rand(50, 64)  # 50 time steps, 64 neurons
            
            # Process through current structure
            output, _ = self.process_neural_activity(test_input, adaptation_enabled=False)
            
            # Evaluate performance
            loss = target_function(output)
            optimization_history["losses"].append(loss)
            
            # Update best performance
            if loss < optimization_history["best_performance"]:
                optimization_history["best_performance"] = loss
                # Save best parameters
                best_params = self._extract_structural_parameters()
                optimization_history["best_parameters"] = best_params
            
            # Gradient-free optimization (evolutionary approach)
            if iteration > 0 and loss > optimization_history["losses"][-2]:
                # Performance degraded, partially revert
                self._apply_structural_mutation(mutation_strength=-learning_rate * 0.5)
            else:
                # Performance improved, continue in this direction
                self._apply_structural_mutation(mutation_strength=learning_rate)
            
            # Log progress
            if iteration % 10 == 0:
                self.logger.debug(f"Optimization iteration {iteration}: loss = {loss:.6f}")
        
        self.logger.info(f"Optimization completed. Best performance: {optimization_history['best_performance']:.6f}")
        
        return optimization_history
    
    def _extract_structural_parameters(self) -> Dict[str, Any]:
        """Extract current structural parameters for saving/loading."""
        parameters = {
            "refractive_indices": [],
            "orientations": [],
            "structure_types": []
        }
        
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):
                    unit_cell = self.metamaterial_grid[x, y, z]
                    parameters["refractive_indices"].append([
                        unit_cell.refractive_index.real,
                        unit_cell.refractive_index.imag
                    ])
                    parameters["orientations"].append(unit_cell.orientation)
                    parameters["structure_types"].append(unit_cell.structure_type.value)
        
        return parameters
    
    def _apply_structural_mutation(self, mutation_strength: float = 0.01) -> None:
        """Apply random mutations to metamaterial structure."""
        num_mutations = max(1, int(abs(mutation_strength) * np.prod(self.grid_size) * 0.1))
        
        for _ in range(num_mutations):
            # Random position
            x = np.random.randint(0, self.grid_size[0])
            y = np.random.randint(0, self.grid_size[1])
            z = np.random.randint(0, self.grid_size[2])
            
            unit_cell = self.metamaterial_grid[x, y, z]
            
            # Mutate refractive index
            index_mutation = np.random.normal(0, abs(mutation_strength) * 0.1)
            new_real = unit_cell.refractive_index.real + index_mutation
            new_real = np.clip(new_real, 1.0, 3.0)
            
            unit_cell.refractive_index = complex(new_real, unit_cell.refractive_index.imag)
            
            # Mutate orientation
            orientation_mutation = np.random.normal(0, abs(mutation_strength))
            unit_cell.orientation = (unit_cell.orientation + orientation_mutation) % (2 * np.pi)
    
    def get_metamaterial_state(self) -> Dict[str, Any]:
        """Get comprehensive metamaterial state information."""
        # Calculate average properties
        avg_refractive_index = 0.0 + 0.0j
        total_cells = np.prod(self.grid_size)
        
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):
                    unit_cell = self.metamaterial_grid[x, y, z]
                    avg_refractive_index += unit_cell.refractive_index
        
        avg_refractive_index /= total_cells
        
        # Calculate structural diversity
        structure_types = []
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):
                    unit_cell = self.metamaterial_grid[x, y, z]
                    structure_types.append(unit_cell.structure_type.value)
        
        unique_structures = len(set(structure_types))
        
        return {
            "grid_size": self.grid_size,
            "total_unit_cells": total_cells,
            "average_refractive_index": {
                "real": avg_refractive_index.real,
                "imag": avg_refractive_index.imag
            },
            "structural_diversity": unique_structures / len(MetamaterialType),
            "reconfiguration_count": self._reconfiguration_count,
            "energy_consumed": self._energy_consumed,
            "adaptation_events": self._adaptation_events,
            "plasticity_mechanism": self.plasticity_mechanism.value
        }


def create_neuromorphic_metamaterial_demo(
    grid_size: Tuple[int, int, int] = (50, 50, 5)
) -> Tuple[NeuromorphicPhotonicMetamaterial, torch.Tensor]:
    """Create demonstration of neuromorphic photonic metamaterial."""
    
    # Advanced metamaterial parameters
    params = MetamaterialParameters(
        unit_cell_size=150e-9,  # 150 nm cells
        refractive_index_base=2.0,
        refractive_index_range=1.0,
        reconfiguration_time=0.5e-9,  # 0.5 ns reconfiguration
        learning_rate=0.02,
        adaptation_threshold=0.05
    )
    
    # Create metamaterial system
    metamaterial = NeuromorphicPhotonicMetamaterial(
        grid_size=grid_size,
        metamaterial_params=params,
        plasticity_mechanism=PlasticityMechanism.PHASE_CHANGE_MATERIAL
    )
    
    # Generate complex test pattern
    time_steps = 100
    num_neurons = min(2500, grid_size[0] * grid_size[1])  # Match grid capacity
    
    # Create structured input with learning patterns
    test_input = torch.zeros(time_steps, num_neurons)
    
    # Pattern 1: Traveling wave (0-30 steps)
    for t in range(30):
        wave_position = int((t / 30) * num_neurons)
        for i in range(max(0, wave_position-5), min(num_neurons, wave_position+5)):
            test_input[t, i] = 1.0
    
    # Pattern 2: Spiral pattern (30-60 steps)
    for t in range(30, 60):
        spiral_t = t - 30
        neurons_per_row = int(np.sqrt(num_neurons))
        center_x, center_y = neurons_per_row // 2, neurons_per_row // 2
        radius = (spiral_t / 30) * neurons_per_row // 2
        angle = spiral_t * 0.5
        
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        
        if 0 <= x < neurons_per_row and 0 <= y < neurons_per_row:
            neuron_idx = x * neurons_per_row + y
            if neuron_idx < num_neurons:
                test_input[t, neuron_idx] = 1.0
    
    # Pattern 3: Random burst (60-100 steps)
    for t in range(60, 100):
        burst_neurons = np.random.choice(num_neurons, size=20, replace=False)
        test_input[t, burst_neurons] = 1.0
    
    return metamaterial, test_input


def run_metamaterial_adaptation_benchmark(
    metamaterial: NeuromorphicPhotonicMetamaterial,
    test_input: torch.Tensor,
    num_trials: int = 5
) -> Dict[str, Any]:
    """Run comprehensive benchmark of metamaterial adaptation."""
    
    results = {
        "adaptation_times": [],
        "energy_consumptions": [],
        "structural_changes": [],
        "processing_speedups": [],
        "learning_convergence": []
    }
    
    for trial in range(num_trials):
        self.logger.info(f"Running adaptation benchmark trial {trial + 1}/{num_trials}")
        
        # Reset metamaterial state
        metamaterial._reconfiguration_count = 0
        metamaterial._energy_consumed = 0.0
        metamaterial._adaptation_events = 0
        
        start_time = time.time()
        
        # Process with adaptation enabled
        output, metrics = metamaterial.process_neural_activity(
            test_input,
            learning_signals=torch.sin(torch.linspace(0, 4*np.pi, len(test_input))) * 0.2,
            adaptation_enabled=True
        )
        
        adaptation_time = time.time() - start_time
        
        # Collect results
        results["adaptation_times"].append(adaptation_time)
        results["energy_consumptions"].append(metrics["energy_consumed"])
        results["structural_changes"].append(len(metrics["structural_changes"]))
        
        # Calculate processing speedup (simplified metric)
        baseline_time = len(test_input) * test_input.shape[1] * 1e-9  # Baseline processing time
        speedup = baseline_time / adaptation_time if adaptation_time > 0 else 0
        results["processing_speedups"].append(speedup)
        
        # Measure learning convergence
        if len(metrics["structural_changes"]) > 10:
            early_changes = len(metrics["structural_changes"][:len(metrics["structural_changes"])//2])
            late_changes = len(metrics["structural_changes"][len(metrics["structural_changes"])//2:])
            convergence_ratio = early_changes / (late_changes + 1)  # Higher means faster convergence
            results["learning_convergence"].append(convergence_ratio)
    
    # Calculate statistics
    benchmark_stats = {}
    for key, values in results.items():
        if values:
            benchmark_stats[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
    
    # Get final metamaterial state
    final_state = metamaterial.get_metamaterial_state()
    benchmark_stats["final_metamaterial_state"] = final_state
    
    return benchmark_stats


# Example optimization target functions
def pattern_recognition_target(output: torch.Tensor) -> float:
    """Target function for pattern recognition optimization."""
    # Measure temporal stability and spatial coherence
    temporal_variance = torch.var(torch.sum(output, dim=1)).item()
    spatial_coherence = torch.mean(torch.std(output, dim=0)).item()
    return temporal_variance + spatial_coherence


def memory_retention_target(output: torch.Tensor) -> float:
    """Target function for memory retention optimization."""
    # Measure how well patterns are preserved over time
    early_pattern = output[:len(output)//3]
    late_pattern = output[2*len(output)//3:]
    
    correlation = torch.corrcoef(torch.stack([
        torch.mean(early_pattern, dim=0),
        torch.mean(late_pattern, dim=0)
    ]))[0, 1].item()
    
    return 1.0 - abs(correlation)  # Minimize loss (maximize correlation)


def validate_metamaterial_advantages() -> Dict[str, Any]:
    """Validate advantages of neuromorphic metamaterials over fixed structures."""
    
    validation_results = {
        "adaptation_speed": 0.0,
        "energy_efficiency": 0.0,
        "learning_capability": 0.0,
        "structural_flexibility": 0.0
    }
    
    # Create adaptive and fixed metamaterials for comparison
    adaptive_metamaterial, test_input = create_neuromorphic_metamaterial_demo((30, 30, 3))
    
    # Test adaptive performance
    start_adaptive = time.time()
    adaptive_output, adaptive_metrics = adaptive_metamaterial.process_neural_activity(
        test_input, adaptation_enabled=True
    )
    adaptive_time = time.time() - start_adaptive
    
    # Test fixed performance (no adaptation)
    start_fixed = time.time()
    fixed_output, fixed_metrics = adaptive_metamaterial.process_neural_activity(
        test_input, adaptation_enabled=False
    )
    fixed_time = time.time() - start_fixed
    
    # Calculate performance metrics
    validation_results["adaptation_speed"] = (
        adaptive_metrics["reconfigurations"] / adaptive_time / 1e6
    )  # Reconfigurations per microsecond
    
    validation_results["energy_efficiency"] = (
        fixed_time / (adaptive_metrics["energy_consumed"] + 1e-12) / 1e15
    )  # Performance per femtojoule
    
    # Learning capability (ability to improve over time)
    adaptive_improvement = torch.mean(adaptive_output[-20:]) / torch.mean(adaptive_output[:20])
    fixed_stability = torch.std(fixed_output).item()
    validation_results["learning_capability"] = adaptive_improvement.item() / (fixed_stability + 0.1)
    
    # Structural flexibility
    metamaterial_state = adaptive_metamaterial.get_metamaterial_state()
    validation_results["structural_flexibility"] = metamaterial_state["structural_diversity"]
    
    return validation_results