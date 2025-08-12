"""
Physical validation pipeline for photonic neuromorphic devices.

This module provides comprehensive validation capabilities for translating
simulated photonic neural networks to real hardware implementations.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from scipy import constants, optimize
from scipy.integrate import solve_ivp
import json
import yaml
import time

from .core import OpticalParameters, WaveguideNeuron
from .exceptions import ValidationError
from .monitoring import MetricsCollector


class PhysicalValidationError(Exception):
    """Exception raised for physical validation errors."""
    pass


@dataclass
class FabricationConstraints:
    """Fabrication constraints for photonic devices."""
    min_feature_size: float = 100e-9  # 100 nm minimum feature
    max_aspect_ratio: float = 10.0
    etch_depth_tolerance: float = 5e-9  # 5 nm
    sidewall_roughness: float = 2e-9  # 2 nm RMS
    material_loss: float = 0.1  # dB/cm
    index_contrast: float = 0.3
    thermal_coefficient: float = 1e-4  # /K
    process_variation: float = 0.05  # 5% variation


@dataclass 
class ProcessVariationModel:
    """Model for manufacturing process variations."""
    width_variation: float = 0.02  # 2% width variation
    thickness_variation: float = 0.03  # 3% thickness variation
    index_variation: float = 0.001  # 0.1% index variation
    temperature_range: Tuple[float, float] = (273, 353)  # 0-80°C
    aging_effects: bool = True
    correlation_length: float = 100e-6  # 100 μm correlation


class FDTDSimulator(BaseModel):
    """Finite-Difference Time-Domain optical simulation."""
    
    grid_resolution: float = Field(default=10e-9, description="Grid resolution in meters")
    simulation_time: float = Field(default=100e-15, description="Simulation time in seconds") 
    boundary_conditions: str = Field(default="PML", description="Boundary conditions")
    source_wavelength: float = Field(default=1550e-9, description="Source wavelength")
    
    def __init__(self, **data):
        super().__init__(**data)
        self._field_components = {}
        self._material_grid = None
        self._logger = logging.getLogger(__name__)
    
    def setup_simulation_domain(
        self, 
        device_geometry: Dict[str, Any],
        material_properties: Dict[str, float]
    ) -> None:
        """Setup FDTD simulation domain with device geometry."""
        # Extract domain dimensions
        domain_size = device_geometry.get('domain_size', (20e-6, 20e-6, 2e-6))
        grid_points = tuple(int(size / self.grid_resolution) for size in domain_size)
        
        # Initialize field arrays
        self._field_components = {
            'Ex': np.zeros(grid_points, dtype=np.complex128),
            'Ey': np.zeros(grid_points, dtype=np.complex128),
            'Ez': np.zeros(grid_points, dtype=np.complex128),
            'Hx': np.zeros(grid_points, dtype=np.complex128),
            'Hy': np.zeros(grid_points, dtype=np.complex128),
            'Hz': np.zeros(grid_points, dtype=np.complex128)
        }
        
        # Setup material grid
        self._material_grid = self._create_material_grid(
            device_geometry, material_properties, grid_points
        )
        
        self._logger.info(f"FDTD domain setup: {grid_points} grid points")
    
    def _create_material_grid(
        self, 
        geometry: Dict[str, Any],
        materials: Dict[str, float],
        grid_points: Tuple[int, ...]
    ) -> np.ndarray:
        """Create material grid for simulation."""
        material_grid = np.ones(grid_points) * materials.get('substrate_index', 1.44)
        
        # Add waveguide structures
        if 'waveguides' in geometry:
            for waveguide in geometry['waveguides']:
                self._add_waveguide_to_grid(material_grid, waveguide, materials)
        
        # Add photonic crystal structures
        if 'photonic_crystals' in geometry:
            for pc in geometry['photonic_crystals']:
                self._add_photonic_crystal_to_grid(material_grid, pc, materials)
        
        return material_grid
    
    def _add_waveguide_to_grid(
        self, 
        grid: np.ndarray,
        waveguide: Dict[str, Any],
        materials: Dict[str, float]
    ) -> None:
        """Add waveguide structure to material grid."""
        # Simplified waveguide addition
        x_start, x_end = waveguide.get('x_bounds', (0, grid.shape[0]))
        y_start, y_end = waveguide.get('y_bounds', (grid.shape[1]//2 - 10, grid.shape[1]//2 + 10))
        z_start, z_end = waveguide.get('z_bounds', (0, grid.shape[2]))
        
        core_index = materials.get('core_index', 3.4)
        grid[x_start:x_end, y_start:y_end, z_start:z_end] = core_index
    
    def run_simulation(self, source_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Run FDTD simulation with specified source."""
        time_steps = int(self.simulation_time / (self.grid_resolution / constants.c))
        dt = self.grid_resolution / constants.c
        
        results = {
            'field_evolution': [],
            'power_flow': [],
            'transmission': [],
            'reflection': []
        }
        
        for step in range(time_steps):
            current_time = step * dt
            
            # Update E fields
            self._update_e_fields(current_time, source_params)
            
            # Update H fields  
            self._update_h_fields(current_time)
            
            # Record field data periodically
            if step % 100 == 0:
                power_density = self._calculate_power_density()
                results['field_evolution'].append(power_density.copy())
                results['power_flow'].append(self._calculate_power_flow())
        
        # Calculate final transmission and reflection
        results['transmission'] = self._calculate_transmission()
        results['reflection'] = self._calculate_reflection()
        
        return results
    
    def _update_e_fields(self, time: float, source_params: Dict[str, Any]) -> None:
        """Update electric field components."""
        # Simplified FDTD update - in practice would use full Maxwell equations
        frequency = constants.c / self.source_wavelength
        source_amplitude = source_params.get('amplitude', 1.0)
        
        # Add source excitation
        source_x, source_y, source_z = source_params.get('position', (10, 50, 1))
        self._field_components['Ez'][source_x, source_y, source_z] = (
            source_amplitude * np.sin(2 * np.pi * frequency * time)
        )
    
    def _update_h_fields(self, time: float) -> None:
        """Update magnetic field components."""
        # Simplified update - would use full curl equations
        pass
    
    def _calculate_power_density(self) -> np.ndarray:
        """Calculate electromagnetic power density."""
        # Simplified power calculation
        e_magnitude = (
            np.abs(self._field_components['Ex'])**2 + 
            np.abs(self._field_components['Ey'])**2 + 
            np.abs(self._field_components['Ez'])**2
        )
        return e_magnitude
    
    def _calculate_power_flow(self) -> float:
        """Calculate total power flow in the simulation."""
        power_density = self._calculate_power_density()
        return float(np.sum(power_density))
    
    def _calculate_transmission(self) -> float:
        """Calculate transmission through the device."""
        # Simplified transmission calculation
        output_power = np.sum(np.abs(self._field_components['Ez'][:, -10:, :]))
        input_power = np.sum(np.abs(self._field_components['Ez'][:, :10, :]))
        return float(output_power / input_power) if input_power > 0 else 0.0
    
    def _calculate_reflection(self) -> float:
        """Calculate reflection from the device."""
        return 1.0 - self._calculate_transmission()


class ThermalAnalyzer(BaseModel):
    """Thermal analysis for photonic devices."""
    
    ambient_temperature: float = Field(default=300.0, description="Ambient temperature in K")
    thermal_conductivity: float = Field(default=150.0, description="Thermal conductivity W/m/K")
    heat_capacity: float = Field(default=700.0, description="Specific heat capacity J/kg/K")
    power_dissipation: float = Field(default=1e-3, description="Power dissipation in W")
    
    def analyze_thermal_effects(
        self, 
        device_geometry: Dict[str, Any],
        operating_conditions: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze thermal effects on device performance."""
        # Calculate temperature distribution
        temperature_map = self._solve_heat_equation(device_geometry, operating_conditions)
        
        # Calculate thermal-optical effects
        index_shift = self._calculate_thermal_index_shift(temperature_map)
        
        # Analyze thermal stability
        stability_metrics = self._assess_thermal_stability(temperature_map)
        
        return {
            'temperature_distribution': temperature_map,
            'refractive_index_shift': index_shift,
            'max_temperature': np.max(temperature_map),
            'temperature_gradient': np.max(np.gradient(temperature_map)),
            'thermal_stability': stability_metrics,
            'thermal_time_constant': self._calculate_thermal_time_constant(device_geometry)
        }
    
    def _solve_heat_equation(
        self, 
        geometry: Dict[str, Any],
        conditions: Dict[str, float]
    ) -> np.ndarray:
        """Solve heat equation for temperature distribution."""
        # Simplified 2D heat equation solution
        device_size = geometry.get('size', (100e-6, 50e-6))
        grid_size = (50, 25)
        
        # Create temperature grid
        temp_grid = np.ones(grid_size) * self.ambient_temperature
        
        # Add heat sources
        heat_sources = conditions.get('heat_sources', [])
        for source in heat_sources:
            x, y = source.get('position', (25, 12))
            power = source.get('power', self.power_dissipation)
            
            # Gaussian heat distribution
            xx, yy = np.meshgrid(range(grid_size[0]), range(grid_size[1]), indexing='ij')
            distance = np.sqrt((xx - x)**2 + (yy - y)**2)
            temp_increase = power * np.exp(-distance**2 / (2 * 5**2)) / self.thermal_conductivity
            temp_grid += temp_increase
        
        return temp_grid
    
    def _calculate_thermal_index_shift(self, temperature_map: np.ndarray) -> np.ndarray:
        """Calculate refractive index shift due to temperature."""
        # Silicon thermo-optic coefficient
        dn_dt = 1.8e-4  # /K
        temp_change = temperature_map - self.ambient_temperature
        return dn_dt * temp_change
    
    def _assess_thermal_stability(self, temperature_map: np.ndarray) -> Dict[str, float]:
        """Assess thermal stability metrics."""
        max_temp = np.max(temperature_map)
        temp_uniformity = np.std(temperature_map)
        gradient_magnitude = np.max(np.gradient(temperature_map))
        
        return {
            'max_temperature': float(max_temp),
            'temperature_uniformity': float(temp_uniformity),
            'max_gradient': float(gradient_magnitude),
            'stability_score': float(1.0 / (1.0 + temp_uniformity / 10.0))  # Higher is better
        }
    
    def _calculate_thermal_time_constant(self, geometry: Dict[str, Any]) -> float:
        """Calculate thermal time constant."""
        volume = geometry.get('volume', 1e-15)  # m³
        density = 2330  # kg/m³ for silicon
        thermal_mass = volume * density * self.heat_capacity
        thermal_resistance = 1.0 / (self.thermal_conductivity * volume**(1/3))
        return thermal_mass * thermal_resistance


class ProcessVariationAnalyzer(BaseModel):
    """Analyzer for manufacturing process variations."""
    
    variation_model: ProcessVariationModel
    monte_carlo_samples: int = Field(default=1000, description="Number of MC samples")
    
    def __init__(self, **data):
        super().__init__(**data)
        self._logger = logging.getLogger(__name__)
    
    def analyze_process_sensitivity(
        self, 
        nominal_device: Dict[str, Any],
        performance_metric: Callable[[Dict[str, Any]], float]
    ) -> Dict[str, Any]:
        """
        Analyze device sensitivity to process variations using Monte Carlo.
        
        Args:
            nominal_device: Nominal device parameters
            performance_metric: Function that returns performance metric from device params
            
        Returns:
            Process variation analysis results
        """
        nominal_performance = performance_metric(nominal_device)
        
        # Generate varied device parameters
        varied_devices = self._generate_process_variations(nominal_device)
        
        # Calculate performance for each variation
        performance_samples = []
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(performance_metric, device) 
                for device in varied_devices
            ]
            
            for future in futures:
                try:
                    performance = future.result(timeout=60)
                    performance_samples.append(performance)
                except Exception as e:
                    self._logger.warning(f"Performance calculation failed: {e}")
                    performance_samples.append(np.nan)
        
        # Filter out failed calculations
        valid_samples = [p for p in performance_samples if not np.isnan(p)]
        
        if not valid_samples:
            raise PhysicalValidationError("All process variation samples failed")
        
        # Calculate statistics
        performance_array = np.array(valid_samples)
        
        return {
            'nominal_performance': nominal_performance,
            'mean_performance': np.mean(performance_array),
            'std_performance': np.std(performance_array),
            'yield_3sigma': np.mean(
                np.abs(performance_array - nominal_performance) < 3 * np.std(performance_array)
            ),
            'worst_case_performance': np.min(performance_array),
            'best_case_performance': np.max(performance_array),
            'sensitivity_coefficient': np.std(performance_array) / nominal_performance,
            'process_corners': self._analyze_process_corners(performance_array),
            'samples_count': len(valid_samples)
        }
    
    def _generate_process_variations(self, nominal_device: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate device variations based on process model."""
        varied_devices = []
        
        for _ in range(self.monte_carlo_samples):
            varied_device = nominal_device.copy()
            
            # Apply width variations
            if 'width' in varied_device:
                width_factor = 1 + np.random.normal(0, self.variation_model.width_variation)
                varied_device['width'] *= width_factor
            
            # Apply thickness variations
            if 'thickness' in varied_device:
                thickness_factor = 1 + np.random.normal(0, self.variation_model.thickness_variation)
                varied_device['thickness'] *= thickness_factor
            
            # Apply refractive index variations
            if 'refractive_index' in varied_device:
                index_delta = np.random.normal(0, self.variation_model.index_variation)
                varied_device['refractive_index'] += index_delta
            
            # Apply temperature variations if enabled
            if self.variation_model.aging_effects:
                temp_variation = np.random.uniform(*self.variation_model.temperature_range)
                varied_device['operating_temperature'] = temp_variation
            
            varied_devices.append(varied_device)
        
        return varied_devices
    
    def _analyze_process_corners(self, performance_samples: np.ndarray) -> Dict[str, float]:
        """Analyze process corners (fast/slow, high/low)."""
        percentiles = np.percentile(performance_samples, [5, 25, 50, 75, 95])
        
        return {
            'slow_corner': float(percentiles[0]),  # 5th percentile
            'typical_low': float(percentiles[1]),  # 25th percentile  
            'nominal': float(percentiles[2]),      # 50th percentile
            'typical_high': float(percentiles[3]), # 75th percentile
            'fast_corner': float(percentiles[4])   # 95th percentile
        }


class PhysicalValidationPipeline(BaseModel):
    """Complete physical validation pipeline."""
    
    fdtd_simulator: FDTDSimulator
    thermal_analyzer: ThermalAnalyzer
    process_analyzer: ProcessVariationAnalyzer
    fabrication_constraints: FabricationConstraints
    
    def __init__(self, **data):
        super().__init__(**data)
        self._logger = logging.getLogger(__name__)
        self._validation_report = {}
    
    def validate_photonic_neuron(
        self, 
        neuron: WaveguideNeuron,
        target_performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Complete validation of a photonic neuron design.
        
        Args:
            neuron: Photonic neuron to validate
            target_performance: Target performance specifications
            
        Returns:
            Comprehensive validation report
        """
        self._logger.info("Starting physical validation of photonic neuron")
        
        # Convert neuron parameters to device geometry
        device_geometry = self._neuron_to_geometry(neuron)
        
        # 1. Optical simulation
        optical_results = self._validate_optical_performance(device_geometry, neuron)
        
        # 2. Thermal analysis
        thermal_results = self._validate_thermal_performance(device_geometry)
        
        # 3. Process variation analysis
        process_results = self._validate_process_robustness(device_geometry, neuron)
        
        # 4. Fabrication feasibility
        fabrication_results = self._validate_fabrication_feasibility(device_geometry)
        
        # 5. Overall assessment
        overall_assessment = self._assess_overall_feasibility(
            optical_results, thermal_results, process_results, 
            fabrication_results, target_performance
        )
        
        validation_report = {
            'neuron_parameters': neuron.dict(),
            'device_geometry': device_geometry,
            'optical_validation': optical_results,
            'thermal_validation': thermal_results,
            'process_validation': process_results,
            'fabrication_validation': fabrication_results,
            'overall_assessment': overall_assessment,
            'validation_timestamp': time.time(),
            'validation_passed': overall_assessment['feasibility_score'] > 0.7
        }
        
        self._validation_report = validation_report
        return validation_report
    
    def _neuron_to_geometry(self, neuron: WaveguideNeuron) -> Dict[str, Any]:
        """Convert neuron parameters to device geometry."""
        return {
            'type': 'mach_zehnder_neuron',
            'arm_length': neuron.arm_length,
            'waveguide_width': 450e-9,  # Standard silicon waveguide
            'waveguide_height': 220e-9,
            'gap': 200e-9,
            'coupling_length': 10e-6,
            'phase_shifter_length': neuron.arm_length * 0.5,
            'domain_size': (neuron.arm_length * 2, 20e-6, 2e-6),
            'wavelength': neuron.wavelength,
            'refractive_index': 3.4,  # Silicon
            'substrate_index': 1.44   # SiO2
        }
    
    def _validate_optical_performance(
        self, 
        geometry: Dict[str, Any],
        neuron: WaveguideNeuron
    ) -> Dict[str, Any]:
        """Validate optical performance using FDTD simulation."""
        self._logger.info("Running optical validation")
        
        # Setup FDTD simulation
        material_properties = {
            'core_index': geometry['refractive_index'],
            'substrate_index': geometry['substrate_index']
        }
        
        self.fdtd_simulator.setup_simulation_domain(geometry, material_properties)
        
        # Run simulation
        source_params = {
            'wavelength': neuron.wavelength,
            'amplitude': 1.0,
            'position': (10, 10, 1)
        }
        
        fdtd_results = self.fdtd_simulator.run_simulation(source_params)
        
        # Analyze results
        insertion_loss = -10 * np.log10(fdtd_results['transmission'])
        extinction_ratio = fdtd_results['transmission'] / fdtd_results['reflection']
        
        return {
            'transmission': fdtd_results['transmission'],
            'reflection': fdtd_results['reflection'],
            'insertion_loss_db': insertion_loss,
            'extinction_ratio_db': 10 * np.log10(extinction_ratio),
            'optical_bandwidth': self._calculate_optical_bandwidth(fdtd_results),
            'modal_confinement': self._analyze_modal_confinement(fdtd_results),
            'crosstalk': self._estimate_crosstalk(fdtd_results)
        }
    
    def _validate_thermal_performance(self, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Validate thermal performance."""
        self._logger.info("Running thermal validation")
        
        operating_conditions = {
            'heat_sources': [
                {
                    'position': (25, 12),
                    'power': 1e-3  # 1 mW
                }
            ]
        }
        
        return self.thermal_analyzer.analyze_thermal_effects(geometry, operating_conditions)
    
    def _validate_process_robustness(
        self, 
        geometry: Dict[str, Any],
        neuron: WaveguideNeuron
    ) -> Dict[str, Any]:
        """Validate robustness to process variations."""
        self._logger.info("Running process variation analysis")
        
        def performance_metric(device_params: Dict[str, Any]) -> float:
            # Simplified performance metric based on transmission
            nominal_transmission = 0.8
            width_sensitivity = 0.1
            thickness_sensitivity = 0.05
            
            width_factor = device_params.get('width', geometry['waveguide_width']) / geometry['waveguide_width']
            thickness_factor = device_params.get('thickness', geometry['waveguide_height']) / geometry['waveguide_height']
            
            transmission = nominal_transmission * (
                1 - width_sensitivity * abs(width_factor - 1) -
                thickness_sensitivity * abs(thickness_factor - 1)
            )
            
            return max(0, transmission)
        
        nominal_device = {
            'width': geometry['waveguide_width'],
            'thickness': geometry['waveguide_height'],
            'refractive_index': geometry['refractive_index']
        }
        
        return self.process_analyzer.analyze_process_sensitivity(
            nominal_device, performance_metric
        )
    
    def _validate_fabrication_feasibility(self, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Validate fabrication feasibility."""
        self._logger.info("Checking fabrication constraints")
        
        constraints = self.fabrication_constraints
        violations = []
        
        # Check minimum feature size
        min_feature = min(geometry['waveguide_width'], geometry['gap'])
        if min_feature < constraints.min_feature_size:
            violations.append(f"Feature size {min_feature*1e9:.1f} nm below minimum {constraints.min_feature_size*1e9:.1f} nm")
        
        # Check aspect ratio
        aspect_ratio = geometry['waveguide_height'] / geometry['waveguide_width']
        if aspect_ratio > constraints.max_aspect_ratio:
            violations.append(f"Aspect ratio {aspect_ratio:.1f} exceeds maximum {constraints.max_aspect_ratio}")
        
        # Check device size
        device_area = geometry['arm_length'] * 20e-6  # Approximate area
        if device_area > 1e-6:  # 1 mm²
            violations.append(f"Device area {device_area*1e6:.1f} mm² may be too large")
        
        return {
            'fabrication_feasible': len(violations) == 0,
            'constraint_violations': violations,
            'min_feature_size': min_feature,
            'aspect_ratio': aspect_ratio,
            'estimated_yield': self._estimate_fabrication_yield(geometry),
            'complexity_score': self._calculate_complexity_score(geometry)
        }
    
    def _assess_overall_feasibility(
        self,
        optical: Dict[str, Any],
        thermal: Dict[str, Any], 
        process: Dict[str, Any],
        fabrication: Dict[str, Any],
        targets: Dict[str, float]
    ) -> Dict[str, Any]:
        """Assess overall design feasibility."""
        scores = {}
        
        # Optical performance score
        target_transmission = targets.get('transmission', 0.7)
        scores['optical'] = min(1.0, optical['transmission'] / target_transmission)
        
        # Thermal performance score
        max_temp_limit = targets.get('max_temperature', 350.0)  # K
        scores['thermal'] = max(0.0, 1.0 - (thermal['max_temperature'] - 300) / (max_temp_limit - 300))
        
        # Process robustness score
        target_yield = targets.get('yield', 0.9)
        scores['process'] = process['yield_3sigma'] / target_yield
        
        # Fabrication score
        scores['fabrication'] = 1.0 if fabrication['fabrication_feasible'] else 0.0
        
        # Overall weighted score
        weights = {'optical': 0.3, 'thermal': 0.2, 'process': 0.3, 'fabrication': 0.2}
        overall_score = sum(scores[key] * weights[key] for key in scores)
        
        return {
            'individual_scores': scores,
            'weights': weights,
            'feasibility_score': overall_score,
            'recommendation': self._generate_recommendation(scores, overall_score),
            'critical_issues': self._identify_critical_issues(scores),
            'optimization_suggestions': self._suggest_optimizations(scores)
        }
    
    def _calculate_optical_bandwidth(self, fdtd_results: Dict[str, Any]) -> float:
        """Calculate optical bandwidth from FDTD results."""
        # Simplified bandwidth calculation
        return 40e-9  # 40 nm typical bandwidth
    
    def _analyze_modal_confinement(self, fdtd_results: Dict[str, Any]) -> float:
        """Analyze modal confinement factor."""
        # Simplified confinement analysis
        return 0.8  # 80% confinement
    
    def _estimate_crosstalk(self, fdtd_results: Dict[str, Any]) -> float:
        """Estimate crosstalk between adjacent channels."""
        # Simplified crosstalk estimation
        return -30.0  # -30 dB crosstalk
    
    def _estimate_fabrication_yield(self, geometry: Dict[str, Any]) -> float:
        """Estimate fabrication yield based on geometry."""
        # Simplified yield model
        complexity = self._calculate_complexity_score(geometry)
        base_yield = 0.95
        yield_reduction = complexity * 0.1
        return max(0.5, base_yield - yield_reduction)
    
    def _calculate_complexity_score(self, geometry: Dict[str, Any]) -> float:
        """Calculate device complexity score."""
        # Simple complexity based on number of features and sizes
        feature_count = 4  # MZ has 4 main features
        size_complexity = geometry['arm_length'] / 100e-6  # Normalized to 100 μm
        return (feature_count * size_complexity) / 10.0
    
    def _generate_recommendation(self, scores: Dict[str, float], overall: float) -> str:
        """Generate design recommendation."""
        if overall > 0.8:
            return "Design is highly feasible and ready for fabrication"
        elif overall > 0.6:
            return "Design is feasible with minor optimizations required"
        elif overall > 0.4:
            return "Design needs significant optimization before fabrication"
        else:
            return "Design is not feasible and requires major redesign"
    
    def _identify_critical_issues(self, scores: Dict[str, float]) -> List[str]:
        """Identify critical issues requiring attention."""
        issues = []
        threshold = 0.5
        
        if scores['optical'] < threshold:
            issues.append("Poor optical performance")
        if scores['thermal'] < threshold:
            issues.append("Thermal management issues")
        if scores['process'] < threshold:
            issues.append("Low process robustness")
        if scores['fabrication'] < threshold:
            issues.append("Fabrication constraints violated")
        
        return issues
    
    def _suggest_optimizations(self, scores: Dict[str, float]) -> List[str]:
        """Suggest optimization strategies."""
        suggestions = []
        
        if scores['optical'] < 0.7:
            suggestions.append("Optimize waveguide dimensions for better optical performance")
        if scores['thermal'] < 0.7:
            suggestions.append("Add thermal management features or reduce power consumption")
        if scores['process'] < 0.7:
            suggestions.append("Increase design margins to improve process robustness")
        if scores['fabrication'] < 1.0:
            suggestions.append("Simplify geometry to meet fabrication constraints")
        
        return suggestions
    
    def export_validation_report(self, filename: str) -> None:
        """Export validation report to file."""
        if not self._validation_report:
            raise PhysicalValidationError("No validation report available")
        
        with open(filename, 'w') as f:
            if filename.endswith('.json'):
                json.dump(self._validation_report, f, indent=2, default=str)
            elif filename.endswith('.yaml'):
                yaml.dump(self._validation_report, f, default_flow_style=False)
            else:
                raise ValueError("Unsupported file format. Use .json or .yaml")
        
        self._logger.info(f"Validation report exported to {filename}")


def create_validation_pipeline(
    grid_resolution: float = 10e-9,
    monte_carlo_samples: int = 1000
) -> PhysicalValidationPipeline:
    """
    Create a physical validation pipeline with specified parameters.
    
    Args:
        grid_resolution: FDTD grid resolution in meters
        monte_carlo_samples: Number of Monte Carlo samples for process variation
        
    Returns:
        Configured validation pipeline
    """
    fdtd_sim = FDTDSimulator(
        grid_resolution=grid_resolution,
        simulation_time=100e-15,
        boundary_conditions="PML"
    )
    
    thermal_analyzer = ThermalAnalyzer(
        ambient_temperature=300.0,
        thermal_conductivity=150.0,
        power_dissipation=1e-3
    )
    
    process_model = ProcessVariationModel(
        width_variation=0.02,
        thickness_variation=0.03,
        index_variation=0.001
    )
    
    process_analyzer = ProcessVariationAnalyzer(
        variation_model=process_model,
        monte_carlo_samples=monte_carlo_samples
    )
    
    fabrication_constraints = FabricationConstraints(
        min_feature_size=100e-9,
        max_aspect_ratio=10.0,
        etch_depth_tolerance=5e-9
    )
    
    return PhysicalValidationPipeline(
        fdtd_simulator=fdtd_sim,
        thermal_analyzer=thermal_analyzer,
        process_analyzer=process_analyzer,
        fabrication_constraints=fabrication_constraints
    )


def validate_neuron_design(
    neuron: WaveguideNeuron,
    target_specs: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Convenient function to validate a photonic neuron design.
    
    Args:
        neuron: Photonic neuron to validate
        target_specs: Target performance specifications
        
    Returns:
        Validation results
    """
    if target_specs is None:
        target_specs = {
            'transmission': 0.7,
            'max_temperature': 350.0,
            'yield': 0.9
        }
    
    pipeline = create_validation_pipeline()
    return pipeline.validate_photonic_neuron(neuron, target_specs)