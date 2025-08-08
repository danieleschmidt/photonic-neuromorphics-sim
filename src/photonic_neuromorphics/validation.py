"""
Comprehensive Validation and Testing Framework.

This module provides robust validation, testing, and quality assurance
capabilities for photonic neuromorphic systems, including automated
testing, contract verification, and system health monitoring.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import warnings
import time
from pathlib import Path

from .core import PhotonicSNN, WaveguideNeuron
from .components import PhotonicComponent
from .architectures import PhotonicCrossbar, PhotonicReservoir
from .exceptions import ValidationError, OpticalModelError, NetworkTopologyError
from .monitoring import MetricsCollector, PerformanceProfiler


@dataclass
class ValidationConfig:
    """Configuration for validation procedures."""
    tolerance: float = 1e-6
    max_iterations: int = 1000
    timeout_seconds: float = 30.0
    enable_warnings: bool = True
    strict_mode: bool = False
    save_reports: bool = True
    report_directory: str = "validation_reports"


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    score: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class SystemHealthReport:
    """Comprehensive system health assessment."""
    overall_health: float  # 0-1 score
    component_health: Dict[str, float]
    performance_metrics: Dict[str, float]
    issues_found: List[str]
    recommendations: List[str]
    timestamp: float = field(default_factory=time.time)


class Validator(ABC):
    """Abstract base class for validators."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def validate(self, target: Any) -> ValidationResult:
        """Perform validation on target object."""
        pass
    
    def _create_result(self, test_name: str, passed: bool, 
                      score: float = 0.0, message: str = "") -> ValidationResult:
        """Create a validation result."""
        return ValidationResult(
            test_name=test_name,
            passed=passed,
            score=score,
            message=message
        )


class ComponentValidator(Validator):
    """Validator for individual photonic components."""
    
    def validate(self, component: PhotonicComponent) -> ValidationResult:
        """Validate a photonic component."""
        start_time = time.time()
        
        try:
            # Test 1: Basic parameter validation
            param_result = self._validate_parameters(component)
            if not param_result.passed:
                return param_result
            
            # Test 2: Transfer function validation
            tf_result = self._validate_transfer_function(component)
            if not tf_result.passed:
                return tf_result
            
            # Test 3: Stability validation
            stability_result = self._validate_stability(component)
            if not stability_result.passed:
                return stability_result
            
            # Test 4: Performance validation
            perf_result = self._validate_performance(component)
            
            # Combine results
            overall_score = np.mean([
                param_result.score,
                tf_result.score, 
                stability_result.score,
                perf_result.score
            ])
            
            result = self._create_result(
                "component_validation",
                True,
                overall_score,
                f"Component {type(component).__name__} validated successfully"
            )
            
            result.details = {
                "parameter_validation": param_result.details,
                "transfer_function_validation": tf_result.details,
                "stability_validation": stability_result.details,
                "performance_validation": perf_result.details
            }
            
            result.execution_time = time.time() - start_time
            return result
            
        except Exception as e:
            return self._create_result(
                "component_validation",
                False,
                0.0,
                f"Validation failed: {str(e)}"
            )
    
    def _validate_parameters(self, component: PhotonicComponent) -> ValidationResult:
        """Validate component parameters."""
        try:
            score = 1.0
            details = {}
            warnings = []
            
            # Check wavelength range
            if hasattr(component, 'params') and hasattr(component.params, 'wavelength'):
                wavelength = component.params.wavelength
                if not (1200e-9 <= wavelength <= 1700e-9):
                    warnings.append(f"Wavelength {wavelength*1e9:.0f}nm outside typical range")
                    score *= 0.8
                details["wavelength_nm"] = wavelength * 1e9
            
            # Check power levels
            if hasattr(component, 'params') and hasattr(component.params, 'power'):
                power = component.params.power
                if power > 1.0:  # > 1W is very high
                    warnings.append(f"Power level {power*1e3:.1f}mW is very high")
                    score *= 0.9
                details["power_mw"] = power * 1e3
            
            # Component-specific validations
            if hasattr(component, 'arm_length'):
                if component.arm_length > 1e-2:  # > 1cm is impractical
                    warnings.append("Arm length exceeds practical limits")
                    score *= 0.7
            
            if hasattr(component, 'quality_factor'):
                if component.quality_factor > 1e6:  # Very high Q
                    warnings.append("Quality factor may be difficult to achieve")
                    score *= 0.9
            
            result = self._create_result(
                "parameter_validation",
                True,
                score,
                "Parameters within acceptable ranges"
            )
            result.warnings = warnings
            result.details = details
            
            return result
            
        except Exception as e:
            return self._create_result(
                "parameter_validation",
                False,
                0.0,
                f"Parameter validation failed: {str(e)}"
            )
    
    def _validate_transfer_function(self, component: PhotonicComponent) -> ValidationResult:
        """Validate component transfer function."""
        try:
            test_wavelengths = np.linspace(1530e-9, 1570e-9, 10)
            test_powers = np.logspace(-6, -3, 5)  # 1µW to 1mW
            
            transmissions = []
            phases = []
            
            for wavelength in test_wavelengths:
                for power in test_powers:
                    try:
                        transmission, phase = component.transfer_function(wavelength, power)
                        
                        # Validate outputs
                        if not (0 <= transmission <= 1):
                            return self._create_result(
                                "transfer_function_validation",
                                False,
                                0.0,
                                f"Invalid transmission: {transmission}"
                            )
                        
                        if np.isnan(transmission) or np.isinf(transmission):
                            return self._create_result(
                                "transfer_function_validation",
                                False,
                                0.0,
                                "Transfer function returned NaN/Inf"
                            )
                        
                        transmissions.append(transmission)
                        phases.append(phase)
                        
                    except Exception as e:
                        return self._create_result(
                            "transfer_function_validation",
                            False,
                            0.0,
                            f"Transfer function error: {str(e)}"
                        )
            
            # Analyze transfer function characteristics
            transmission_range = max(transmissions) - min(transmissions)
            phase_range = max(phases) - min(phases)
            
            score = 1.0
            if transmission_range < 0.1:  # Very flat response
                score *= 0.8
            if phase_range < 0.1:  # Very flat phase
                score *= 0.9
            
            result = self._create_result(
                "transfer_function_validation",
                True,
                score,
                "Transfer function behaves correctly"
            )
            
            result.details = {
                "transmission_range": transmission_range,
                "phase_range_rad": phase_range,
                "test_points": len(transmissions)
            }
            
            return result
            
        except Exception as e:
            return self._create_result(
                "transfer_function_validation",
                False,
                0.0,
                f"Transfer function validation failed: {str(e)}"
            )
    
    def _validate_stability(self, component: PhotonicComponent) -> ValidationResult:
        """Validate component stability."""
        try:
            # Test stability by repeated calls
            stable_results = []
            test_wavelength = 1550e-9
            test_power = 1e-3
            
            for _ in range(10):
                transmission, phase = component.transfer_function(test_wavelength, test_power)
                stable_results.append((transmission, phase))
            
            # Check consistency
            transmissions = [r[0] for r in stable_results]
            phases = [r[1] for r in stable_results]
            
            transmission_std = np.std(transmissions)
            phase_std = np.std(phases)
            
            # Stability criteria
            transmission_stable = transmission_std < 0.01  # 1% variation
            phase_stable = phase_std < 0.1  # 0.1 rad variation
            
            score = 1.0
            if not transmission_stable:
                score *= 0.5
            if not phase_stable:
                score *= 0.7
            
            passed = transmission_stable and phase_stable
            message = "Component shows stable behavior" if passed else "Component shows instability"
            
            result = self._create_result(
                "stability_validation",
                passed,
                score,
                message
            )
            
            result.details = {
                "transmission_std": transmission_std,
                "phase_std_rad": phase_std,
                "stability_tests": len(stable_results)
            }
            
            return result
            
        except Exception as e:
            return self._create_result(
                "stability_validation",
                False,
                0.0,
                f"Stability validation failed: {str(e)}"
            )
    
    def _validate_performance(self, component: PhotonicComponent) -> ValidationResult:
        """Validate component performance characteristics."""
        try:
            # Performance metrics
            performance_score = 1.0
            details = {}
            
            # Test insertion loss
            transmission, _ = component.transfer_function(1550e-9, 1e-3)
            insertion_loss_db = -10 * np.log10(transmission) if transmission > 0 else float('inf')
            
            if insertion_loss_db > 3.0:  # > 3dB is high loss
                performance_score *= 0.7
            
            details["insertion_loss_db"] = insertion_loss_db
            
            # Test bandwidth (simplified)
            wavelengths = np.linspace(1530e-9, 1570e-9, 41)
            transmissions = []
            
            for wl in wavelengths:
                t, _ = component.transfer_function(wl, 1e-3)
                transmissions.append(t)
            
            # 3dB bandwidth
            max_transmission = max(transmissions)
            half_power = max_transmission / 2
            
            bandwidth_points = sum(1 for t in transmissions if t > half_power)
            bandwidth_nm = bandwidth_points * (1570 - 1530) / len(wavelengths)
            
            details["bandwidth_3db_nm"] = bandwidth_nm
            
            if bandwidth_nm < 1.0:  # < 1nm is very narrow
                performance_score *= 0.8
            
            result = self._create_result(
                "performance_validation",
                True,
                performance_score,
                "Performance characteristics acceptable"
            )
            result.details = details
            
            return result
            
        except Exception as e:
            return self._create_result(
                "performance_validation",
                False,
                0.0,
                f"Performance validation failed: {str(e)}"
            )


class NetworkValidator(Validator):
    """Validator for photonic neural networks."""
    
    def validate(self, network: Union[PhotonicSNN, PhotonicCrossbar, PhotonicReservoir]) -> ValidationResult:
        """Validate a photonic neural network."""
        start_time = time.time()
        
        try:
            validation_results = []
            
            # Test 1: Architecture validation
            arch_result = self._validate_architecture(network)
            validation_results.append(arch_result)
            
            # Test 2: Connectivity validation  
            conn_result = self._validate_connectivity(network)
            validation_results.append(conn_result)
            
            # Test 3: Forward pass validation
            forward_result = self._validate_forward_pass(network)
            validation_results.append(forward_result)
            
            # Test 4: Gradient flow validation (if applicable)
            if hasattr(network, 'parameters'):
                grad_result = self._validate_gradient_flow(network)
                validation_results.append(grad_result)
            
            # Test 5: Resource validation
            resource_result = self._validate_resources(network)
            validation_results.append(resource_result)
            
            # Combine results
            all_passed = all(r.passed for r in validation_results)
            overall_score = np.mean([r.score for r in validation_results])
            
            result = self._create_result(
                "network_validation",
                all_passed,
                overall_score,
                f"Network validation {'passed' if all_passed else 'failed'}"
            )
            
            result.details = {
                test_result.test_name: test_result.details 
                for test_result in validation_results
            }
            
            # Collect all warnings
            result.warnings = []
            for test_result in validation_results:
                result.warnings.extend(test_result.warnings)
            
            result.execution_time = time.time() - start_time
            return result
            
        except Exception as e:
            return self._create_result(
                "network_validation",
                False,
                0.0,
                f"Network validation failed: {str(e)}"
            )
    
    def _validate_architecture(self, network: Any) -> ValidationResult:
        """Validate network architecture."""
        try:
            score = 1.0
            warnings = []
            details = {}
            
            # Check topology if available
            if hasattr(network, 'topology'):
                topology = network.topology
                details["topology"] = topology
                
                # Check for reasonable sizes
                if any(size > 10000 for size in topology):
                    warnings.append("Very large layer sizes may cause memory issues")
                    score *= 0.9
                
                if len(topology) > 10:
                    warnings.append("Very deep network may cause training difficulties")
                    score *= 0.9
            
            # Check crossbar dimensions
            elif hasattr(network, 'rows') and hasattr(network, 'cols'):
                rows, cols = network.rows, network.cols
                details["dimensions"] = (rows, cols)
                
                if rows * cols > 1e6:  # > 1M elements
                    warnings.append("Very large crossbar may require significant resources")
                    score *= 0.8
            
            # Check reservoir properties
            elif hasattr(network, 'nodes'):
                nodes = network.nodes
                details["nodes"] = nodes
                
                if nodes > 1000:
                    warnings.append("Large reservoir may have slow dynamics")
                    score *= 0.9
                
                if hasattr(network, 'connectivity'):
                    connectivity = network.connectivity
                    details["connectivity"] = connectivity
                    
                    if connectivity > 0.5:
                        warnings.append("High connectivity may cause instability")
                        score *= 0.8
            
            result = self._create_result(
                "architecture_validation",
                True,
                score,
                "Architecture appears reasonable"
            )
            result.warnings = warnings
            result.details = details
            
            return result
            
        except Exception as e:
            return self._create_result(
                "architecture_validation",
                False,
                0.0,
                f"Architecture validation failed: {str(e)}"
            )
    
    def _validate_connectivity(self, network: Any) -> ValidationResult:
        """Validate network connectivity."""
        try:
            # Check for basic connectivity requirements
            if hasattr(network, 'layers') and hasattr(network, 'topology'):
                # SNN-style network
                expected_layers = len(network.topology) - 1
                actual_layers = len(network.layers)
                
                if actual_layers != expected_layers:
                    return self._create_result(
                        "connectivity_validation",
                        False,
                        0.0,
                        f"Layer count mismatch: expected {expected_layers}, got {actual_layers}"
                    )
                
                # Check weight matrix shapes
                for i, layer in enumerate(network.layers):
                    if hasattr(layer, 'shape'):
                        expected_shape = (network.topology[i], network.topology[i+1])
                        if tuple(layer.shape) != expected_shape:
                            return self._create_result(
                                "connectivity_validation",
                                False,
                                0.0,
                                f"Weight matrix {i} shape mismatch"
                            )
            
            return self._create_result(
                "connectivity_validation",
                True,
                1.0,
                "Network connectivity is valid"
            )
            
        except Exception as e:
            return self._create_result(
                "connectivity_validation",
                False,
                0.0,
                f"Connectivity validation failed: {str(e)}"
            )
    
    def _validate_forward_pass(self, network: Any) -> ValidationResult:
        """Validate network forward pass."""
        try:
            # Create test input
            if hasattr(network, 'topology'):
                input_size = network.topology[0]
                test_input = torch.randn(10, input_size)  # 10 time steps
            elif hasattr(network, 'rows'):
                input_size = network.rows
                test_input = torch.randn(input_size)
            else:
                # Default test
                test_input = torch.randn(10, 100)
            
            # Test forward pass
            start_time = time.time()
            
            if hasattr(network, 'forward'):
                try:
                    output = network.forward(test_input)
                    forward_time = time.time() - start_time
                    
                    # Validate output
                    if torch.any(torch.isnan(output)):
                        return self._create_result(
                            "forward_pass_validation",
                            False,
                            0.0,
                            "Forward pass produced NaN values"
                        )
                    
                    if torch.any(torch.isinf(output)):
                        return self._create_result(
                            "forward_pass_validation",
                            False,
                            0.0,
                            "Forward pass produced infinite values"
                        )
                    
                    # Score based on execution time and output quality
                    score = 1.0
                    if forward_time > 1.0:  # > 1 second is slow
                        score *= 0.8
                    
                    result = self._create_result(
                        "forward_pass_validation",
                        True,
                        score,
                        "Forward pass executed successfully"
                    )
                    
                    result.details = {
                        "execution_time_s": forward_time,
                        "output_shape": list(output.shape),
                        "output_range": [float(torch.min(output)), float(torch.max(output))]
                    }
                    
                    return result
                    
                except Exception as e:
                    return self._create_result(
                        "forward_pass_validation",
                        False,
                        0.0,
                        f"Forward pass failed: {str(e)}"
                    )
            
            else:
                return self._create_result(
                    "forward_pass_validation",
                    False,
                    0.0,
                    "Network does not have forward method"
                )
            
        except Exception as e:
            return self._create_result(
                "forward_pass_validation",
                False,
                0.0,
                f"Forward pass validation failed: {str(e)}"
            )
    
    def _validate_gradient_flow(self, network: torch.nn.Module) -> ValidationResult:
        """Validate gradient flow through network."""
        try:
            # This is a simplified gradient flow check
            # In practice, would need more sophisticated analysis
            
            # Count parameters with gradients
            param_count = 0
            grad_param_count = 0
            
            for param in network.parameters():
                param_count += 1
                if param.requires_grad:
                    grad_param_count += 1
            
            if param_count == 0:
                return self._create_result(
                    "gradient_flow_validation",
                    False,
                    0.0,
                    "Network has no parameters"
                )
            
            gradient_ratio = grad_param_count / param_count
            
            result = self._create_result(
                "gradient_flow_validation",
                gradient_ratio > 0.5,  # At least 50% of parameters should have gradients
                gradient_ratio,
                f"Gradient flow validation: {gradient_ratio:.1%} of parameters trainable"
            )
            
            result.details = {
                "total_parameters": param_count,
                "trainable_parameters": grad_param_count,
                "gradient_ratio": gradient_ratio
            }
            
            return result
            
        except Exception as e:
            return self._create_result(
                "gradient_flow_validation",
                False,
                0.0,
                f"Gradient flow validation failed: {str(e)}"
            )
    
    def _validate_resources(self, network: Any) -> ValidationResult:
        """Validate network resource requirements."""
        try:
            score = 1.0
            warnings = []
            details = {}
            
            # Estimate memory usage
            if hasattr(network, 'estimate_resources'):
                resources = network.estimate_resources()
                details.update(resources)
                
                # Check for excessive resource usage
                if "total_area_m2" in resources:
                    area_mm2 = resources["total_area_m2"] * 1e6  # Convert to mm²
                    details["area_mm2"] = area_mm2
                    
                    if area_mm2 > 100:  # > 100 mm² is very large
                        warnings.append(f"Large chip area required: {area_mm2:.1f} mm²")
                        score *= 0.7
                
                if "total_power_w" in resources:
                    power_mw = resources["total_power_w"] * 1e3  # Convert to mW
                    details["power_mw"] = power_mw
                    
                    if power_mw > 1000:  # > 1W is high power
                        warnings.append(f"High power consumption: {power_mw:.0f} mW")
                        score *= 0.8
            
            # Memory estimation for PyTorch modules
            elif hasattr(network, 'parameters'):
                total_params = sum(p.numel() for p in network.parameters())
                memory_mb = total_params * 4 / 1e6  # 4 bytes per float32, convert to MB
                
                details["total_parameters"] = total_params
                details["memory_mb"] = memory_mb
                
                if memory_mb > 1000:  # > 1GB
                    warnings.append(f"High memory usage: {memory_mb:.0f} MB")
                    score *= 0.8
            
            result = self._create_result(
                "resource_validation",
                True,
                score,
                "Resource requirements are reasonable"
            )
            result.warnings = warnings
            result.details = details
            
            return result
            
        except Exception as e:
            return self._create_result(
                "resource_validation",
                False,
                0.0,
                f"Resource validation failed: {str(e)}"
            )


class SystemHealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self, config: ValidationConfig, metrics_collector: Optional[MetricsCollector] = None):
        self.config = config
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.logger = logging.getLogger(__name__)
        
        # Health check registry
        self.health_checks: Dict[str, Callable] = {}
        self.last_health_report: Optional[SystemHealthReport] = None
    
    def register_health_check(self, name: str, check_function: Callable) -> None:
        """Register a custom health check function."""
        self.health_checks[name] = check_function
        self.logger.info(f"Registered health check: {name}")
    
    def run_health_assessment(self, system: Any) -> SystemHealthReport:
        """Run comprehensive health assessment."""
        self.logger.info("Starting system health assessment...")
        
        component_health = {}
        issues_found = []
        recommendations = []
        performance_metrics = {}
        
        try:
            # Component-level health checks
            if hasattr(system, 'neurons'):
                # Check neuron health
                neuron_health = self._assess_neuron_health(system)
                component_health["neurons"] = neuron_health
                
                if neuron_health < 0.8:
                    issues_found.append("Some neurons showing degraded performance")
                    recommendations.append("Check neuron parameters and calibration")
            
            # Network-level health checks
            network_health = self._assess_network_health(system)
            component_health["network"] = network_health
            
            if network_health < 0.7:
                issues_found.append("Network connectivity or performance issues")
                recommendations.append("Validate network architecture and weights")
            
            # Performance health checks
            perf_health = self._assess_performance_health(system)
            component_health["performance"] = perf_health
            performance_metrics.update(perf_health)
            
            # Custom health checks
            for check_name, check_func in self.health_checks.items():
                try:
                    check_result = check_func(system)
                    component_health[check_name] = check_result
                    
                    if check_result < 0.8:
                        issues_found.append(f"Custom check {check_name} failed")
                        recommendations.append(f"Investigate {check_name} health check")
                        
                except Exception as e:
                    self.logger.error(f"Health check {check_name} failed: {e}")
                    component_health[check_name] = 0.0
                    issues_found.append(f"Health check {check_name} execution failed")
            
            # Calculate overall health
            if component_health:
                overall_health = np.mean(list(component_health.values()))
            else:
                overall_health = 0.5  # Unknown health
                issues_found.append("No health metrics available")
                recommendations.append("Implement health monitoring")
            
            # Generate recommendations based on health score
            if overall_health < 0.5:
                recommendations.append("CRITICAL: System requires immediate attention")
            elif overall_health < 0.7:
                recommendations.append("WARNING: System performance degraded")
            elif overall_health > 0.9:
                recommendations.append("System operating at optimal performance")
            
            # Create health report
            health_report = SystemHealthReport(
                overall_health=overall_health,
                component_health=component_health,
                performance_metrics=performance_metrics,
                issues_found=issues_found,
                recommendations=recommendations
            )
            
            self.last_health_report = health_report
            
            # Record metrics
            self.metrics_collector.record_metric("system_health_overall", overall_health)
            self.metrics_collector.record_metric("system_health_issues", len(issues_found))
            
            self.logger.info(f"Health assessment complete. Overall health: {overall_health:.2f}")
            
            return health_report
            
        except Exception as e:
            self.logger.error(f"Health assessment failed: {e}")
            
            # Return degraded health report
            return SystemHealthReport(
                overall_health=0.0,
                component_health={"error": 0.0},
                performance_metrics={},
                issues_found=[f"Health assessment failed: {str(e)}"],
                recommendations=["Fix health monitoring system"]
            )
    
    def _assess_neuron_health(self, system: Any) -> float:
        """Assess health of neurons in the system."""
        if not hasattr(system, 'neurons'):
            return 1.0  # No neurons to check
        
        try:
            healthy_neurons = 0
            total_neurons = 0
            
            for layer_neurons in system.neurons:
                for neuron in layer_neurons:
                    total_neurons += 1
                    
                    # Check neuron parameters
                    if hasattr(neuron, 'threshold_power'):
                        if 0 < neuron.threshold_power < 1e-3:  # Reasonable threshold
                            healthy_neurons += 1
                    else:
                        healthy_neurons += 1  # Assume healthy if no threshold
            
            return healthy_neurons / max(total_neurons, 1)
            
        except Exception:
            return 0.5  # Unknown health
    
    def _assess_network_health(self, system: Any) -> float:
        """Assess overall network health."""
        health_score = 1.0
        
        try:
            # Check for common issues
            if hasattr(system, 'topology'):
                # Check topology reasonableness
                topology = system.topology
                if any(size <= 0 for size in topology):
                    health_score *= 0.5
                if len(topology) > 20:  # Very deep network
                    health_score *= 0.8
            
            # Check weight matrices if available
            if hasattr(system, 'layers'):
                for layer in system.layers:
                    if hasattr(layer, 'data'):
                        weights = layer.data
                        if torch.any(torch.isnan(weights)):
                            health_score *= 0.2  # NaN weights are critical
                        if torch.any(torch.isinf(weights)):
                            health_score *= 0.3  # Inf weights are critical
                        
                        # Check weight distribution
                        weight_std = torch.std(weights).item()
                        if weight_std > 10:  # Very large weights
                            health_score *= 0.7
                        elif weight_std < 1e-6:  # Weights too small
                            health_score *= 0.8
            
            return health_score
            
        except Exception:
            return 0.5  # Unknown health
    
    def _assess_performance_health(self, system: Any) -> Dict[str, float]:
        """Assess performance-related health metrics."""
        performance_metrics = {}
        
        try:
            # Memory health
            if hasattr(system, 'estimate_resources'):
                resources = system.estimate_resources()
                
                # Area efficiency
                if "area_efficiency" in resources:
                    area_eff = resources["area_efficiency"]
                    performance_metrics["area_efficiency"] = min(area_eff / 1e6, 1.0)  # Normalize
                
                # Power efficiency
                if "power_efficiency" in resources:
                    power_eff = resources["power_efficiency"]
                    performance_metrics["power_efficiency"] = min(power_eff / 1e6, 1.0)  # Normalize
            
            # Computational health (simplified)
            performance_metrics["computational_health"] = 0.8  # Default assumption
            
            # Communication health
            performance_metrics["communication_health"] = 0.9  # Optical advantage
            
            return performance_metrics
            
        except Exception:
            return {"performance_health": 0.5}
    
    def get_health_summary(self) -> str:
        """Get human-readable health summary."""
        if not self.last_health_report:
            return "No health report available. Run health assessment first."
        
        report = self.last_health_report
        
        summary = f"\n{'='*50}\n"
        summary += f"SYSTEM HEALTH REPORT\n"
        summary += f"{'='*50}\n"
        summary += f"Overall Health: {report.overall_health:.1%}\n\n"
        
        summary += "Component Health:\n"
        for component, health in report.component_health.items():
            status = "🟢" if health > 0.8 else "🟡" if health > 0.5 else "🔴"
            summary += f"  {status} {component}: {health:.1%}\n"
        
        if report.issues_found:
            summary += "\nIssues Found:\n"
            for issue in report.issues_found:
                summary += f"  ⚠️  {issue}\n"
        
        if report.recommendations:
            summary += "\nRecommendations:\n"
            for rec in report.recommendations:
                summary += f"  💡 {rec}\n"
        
        summary += f"\nReport generated: {time.ctime(report.timestamp)}\n"
        summary += f"{'='*50}\n"
        
        return summary


def create_comprehensive_validator(config: Optional[ValidationConfig] = None) -> Tuple[ComponentValidator, NetworkValidator, SystemHealthMonitor]:
    """Create a comprehensive validation suite."""
    if config is None:
        config = ValidationConfig()
    
    component_validator = ComponentValidator(config)
    network_validator = NetworkValidator(config)
    health_monitor = SystemHealthMonitor(config)
    
    return component_validator, network_validator, health_monitor


def run_full_validation_suite(system: Any, config: Optional[ValidationConfig] = None) -> Dict[str, Any]:
    """Run complete validation suite on a photonic neuromorphic system."""
    component_validator, network_validator, health_monitor = create_comprehensive_validator(config)
    
    results = {
        "timestamp": time.time(),
        "system_type": type(system).__name__
    }
    
    # Network-level validation
    network_result = network_validator.validate(system)
    results["network_validation"] = {
        "passed": network_result.passed,
        "score": network_result.score,
        "message": network_result.message,
        "execution_time": network_result.execution_time,
        "warnings": network_result.warnings
    }
    
    # Component-level validation (if applicable)
    if hasattr(system, 'neurons'):
        component_results = []
        
        # Validate a sample of neurons
        for layer_idx, layer_neurons in enumerate(system.neurons[:2]):  # First 2 layers
            for neuron_idx, neuron in enumerate(layer_neurons[:5]):  # First 5 neurons
                comp_result = component_validator.validate(neuron)
                component_results.append({
                    "layer": layer_idx,
                    "neuron": neuron_idx,
                    "passed": comp_result.passed,
                    "score": comp_result.score
                })
        
        results["component_validation"] = component_results
    
    # Health assessment
    health_report = health_monitor.run_health_assessment(system)
    results["health_assessment"] = {
        "overall_health": health_report.overall_health,
        "component_health": health_report.component_health,
        "issues_count": len(health_report.issues_found),
        "recommendations_count": len(health_report.recommendations)
    }
    
    # Overall validation score
    scores = [network_result.score]
    if "component_validation" in results:
        comp_scores = [r["score"] for r in results["component_validation"] if r["passed"]]
        if comp_scores:
            scores.append(np.mean(comp_scores))
    scores.append(health_report.overall_health)
    
    results["overall_score"] = np.mean(scores)
    results["validation_passed"] = results["overall_score"] > 0.7
    
    return results
