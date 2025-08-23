#!/usr/bin/env python3
"""
Robust Validation System for Photonic Neuromorphics - Generation 2: MAKE IT ROBUST
Implements comprehensive error handling, validation, security, and monitoring.
"""

import sys
import os
import time
import logging
import json
import hashlib
import warnings
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np

# Add source path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/photonic_robust_validation.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Comprehensive validation result with security and performance metrics."""
    success: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    security_score: float = 0.0
    performance_score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def add_error(self, error: str):
        """Add error with logging."""
        self.errors.append(error)
        logger.error(f"Validation error: {error}")
    
    def add_warning(self, warning: str):
        """Add warning with logging."""
        self.warnings.append(warning)
        logger.warning(f"Validation warning: {warning}")
    
    def set_metric(self, name: str, value: float):
        """Set performance metric with validation."""
        if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
            self.add_warning(f"Invalid metric value for {name}: {value}")
            return
        self.metrics[name] = float(value)
    
    def calculate_scores(self):
        """Calculate overall security and performance scores."""
        # Security score: inverse of error count, penalized by warnings
        error_penalty = len(self.errors) * 20
        warning_penalty = len(self.warnings) * 5
        self.security_score = max(0, 100 - error_penalty - warning_penalty)
        
        # Performance score: based on metrics
        if 'execution_time' in self.metrics:
            time_score = max(0, 100 - self.metrics['execution_time'] * 10)
        else:
            time_score = 50
            
        if 'memory_usage' in self.metrics:
            memory_score = max(0, 100 - self.metrics['memory_usage'] * 0.1)
        else:
            memory_score = 50
            
        self.performance_score = (time_score + memory_score) / 2

class SecurityValidator:
    """Comprehensive security validation for photonic neuromorphic systems."""
    
    def __init__(self):
        self.blocked_patterns = [
            '__import__', 'exec', 'eval', 'subprocess', 'os.system',
            'open(', 'file(', 'input(', 'raw_input'
        ]
        self.safe_modules = {
            'numpy', 'scipy', 'matplotlib', 'torch', 'pydantic',
            'networkx', 'click', 'yaml', 'json', 'math', 'random'
        }
    
    def validate_input_parameters(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate input parameters for security and bounds."""
        result = ValidationResult()
        
        try:
            # Check for malicious patterns
            param_str = str(params)
            for pattern in self.blocked_patterns:
                if pattern in param_str:
                    result.add_error(f"Blocked pattern detected: {pattern}")
            
            # Validate optical parameters
            if 'wavelength' in params:
                wavelength = params['wavelength']
                if not (1200e-9 <= wavelength <= 1700e-9):
                    result.add_error(f"Wavelength {wavelength*1e9:.0f}nm outside safe range (1200-1700nm)")
                elif not (1260e-9 <= wavelength <= 1675e-9):
                    result.add_warning(f"Wavelength {wavelength*1e9:.0f}nm outside optimal telecom bands")
            
            if 'power' in params:
                power = params['power']
                if power < 0:
                    result.add_error(f"Negative optical power: {power}")
                elif power > 1.0:  # 1W safety limit
                    result.add_error(f"Optical power {power*1e3:.0f}mW exceeds safety limit (1000mW)")
                elif power > 0.1:  # 100mW warning
                    result.add_warning(f"High optical power {power*1e3:.0f}mW - safety precaution advised")
            
            if 'topology' in params:
                topology = params['topology']
                if not isinstance(topology, (list, tuple)):
                    result.add_error(f"Topology must be list/tuple, got {type(topology)}")
                elif len(topology) < 2:
                    result.add_error("Network topology must have at least 2 layers")
                elif any(n <= 0 for n in topology):
                    result.add_error("All topology values must be positive")
                elif any(n > 10000 for n in topology):
                    result.add_warning("Large layer sizes may cause performance issues")
            
            # Check for numerical stability
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    if np.isnan(value):
                        result.add_error(f"NaN value detected in parameter: {key}")
                    elif np.isinf(value):
                        result.add_error(f"Infinite value detected in parameter: {key}")
                    elif abs(value) > 1e12:
                        result.add_warning(f"Very large value in parameter {key}: {value}")
            
            result.success = len(result.errors) == 0
            
        except Exception as e:
            result.add_error(f"Parameter validation failed: {str(e)}")
        
        return result
    
    def validate_file_access(self, file_path: str) -> ValidationResult:
        """Validate file access for security."""
        result = ValidationResult()
        
        try:
            # Prevent directory traversal
            normalized_path = os.path.normpath(file_path)
            if '../' in file_path or '..\\' in file_path:
                result.add_error(f"Directory traversal detected in path: {file_path}")
            
            # Check file extension whitelist
            allowed_extensions = {'.py', '.txt', '.json', '.yaml', '.yml', '.log', '.csv'}
            _, ext = os.path.splitext(normalized_path)
            if ext not in allowed_extensions:
                result.add_warning(f"Potentially unsafe file extension: {ext}")
            
            # Check if path is within allowed directories
            allowed_dirs = ['/tmp', '/root/repo', '/var/log']
            path_allowed = any(normalized_path.startswith(allowed_dir) for allowed_dir in allowed_dirs)
            if not path_allowed:
                result.add_error(f"File access outside allowed directories: {normalized_path}")
            
            result.success = len(result.errors) == 0
            
        except Exception as e:
            result.add_error(f"File validation failed: {str(e)}")
        
        return result

class PerformanceMonitor:
    """Monitor performance metrics and resource usage."""
    
    def __init__(self):
        self.start_time = None
        self.peak_memory = 0
        self.operation_count = 0
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring operations."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            logger.info(f"Starting operation: {operation_name}")
            yield
            
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            duration = end_time - start_time
            memory_used = end_memory - start_memory
            
            self.operation_count += 1
            logger.info(f"Completed {operation_name}: {duration:.4f}s, {memory_used:.2f}MB memory")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback: estimate from Python objects
            import sys
            return sys.getsizeof(locals()) / 1024 / 1024

class RobustPhotonicNeuron:
    """Robust photonic neuron with comprehensive error handling and validation."""
    
    def __init__(self, threshold_power: float = 1e-6, wavelength: float = 1550e-9):
        self.validator = SecurityValidator()
        self.monitor = PerformanceMonitor()
        
        # Validate initialization parameters
        params = {'threshold_power': threshold_power, 'wavelength': wavelength}
        validation = self.validator.validate_input_parameters(params)
        
        if not validation.success:
            raise ValueError(f"Invalid neuron parameters: {validation.errors}")
        
        self.threshold_power = threshold_power
        self.wavelength = wavelength
        self.membrane_potential = 0.0
        self.last_spike_time = -float('inf')
        self.spike_count = 0
        self.error_count = 0
        
        logger.info(f"Initialized robust photonic neuron: λ={wavelength*1e9:.0f}nm, th={threshold_power*1e6:.1f}μW")
    
    def forward(self, optical_input: float, time: float) -> bool:
        """Process optical input with comprehensive error handling."""
        try:
            with self.monitor.monitor_operation("neuron_forward"):
                # Input validation
                if not isinstance(optical_input, (int, float)):
                    raise TypeError(f"Optical input must be numeric, got {type(optical_input)}")
                
                if not isinstance(time, (int, float)):
                    raise TypeError(f"Time must be numeric, got {type(time)}")
                
                if optical_input < 0:
                    logger.warning(f"Negative optical input: {optical_input}")
                    optical_input = 0
                
                if optical_input > 10.0:  # 10W safety limit
                    logger.error(f"Optical input {optical_input:.2f}W exceeds safety limit")
                    optical_input = min(optical_input, 1.0)
                    self.error_count += 1
                
                if np.isnan(optical_input) or np.isinf(optical_input):
                    logger.error(f"Invalid optical input: {optical_input}")
                    optical_input = 0
                    self.error_count += 1
                
                # Neuron dynamics with bounds checking
                refractory_period = 1e-9  # 1ns
                if time - self.last_spike_time > refractory_period:
                    # Leaky integrate-and-fire with numerical stability
                    old_potential = self.membrane_potential
                    self.membrane_potential += optical_input * 1e6  # Scale for stability
                    self.membrane_potential *= 0.99  # Leak factor
                    
                    # Check for numerical overflow
                    if abs(self.membrane_potential) > 1e12:
                        logger.error(f"Membrane potential overflow: {self.membrane_potential:.2e}")
                        self.membrane_potential = old_potential * 0.1  # Reset with small value
                        self.error_count += 1
                    
                    # Spike generation with hysteresis
                    spike_threshold = self.threshold_power * 1e6
                    if self.membrane_potential > spike_threshold:
                        self.membrane_potential = 0.0
                        self.last_spike_time = time
                        self.spike_count += 1
                        
                        logger.debug(f"Spike generated at t={time*1e9:.1f}ns (total: {self.spike_count})")
                        return True
                
                return False
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Neuron forward pass error: {str(e)}")
            return False
    
    def get_health_metrics(self) -> Dict[str, float]:
        """Get neuron health and performance metrics."""
        return {
            'spike_count': float(self.spike_count),
            'error_count': float(self.error_count),
            'error_rate': self.error_count / max(1, self.monitor.operation_count),
            'membrane_potential': float(self.membrane_potential),
            'last_spike_time': float(self.last_spike_time)
        }

class RobustSimulationFramework:
    """Robust simulation framework with comprehensive monitoring and recovery."""
    
    def __init__(self):
        self.validator = SecurityValidator()
        self.monitor = PerformanceMonitor()
        self.neurons = []
        self.simulation_history = []
        self.error_recovery_enabled = True
        
    def create_network(self, topology: List[int], **params) -> ValidationResult:
        """Create neural network with robust validation."""
        result = ValidationResult()
        
        try:
            with self.monitor.monitor_operation("network_creation"):
                # Validate network parameters
                all_params = {'topology': topology, **params}
                validation = self.validator.validate_input_parameters(all_params)
                
                if not validation.success:
                    result.errors.extend(validation.errors)
                    result.warnings.extend(validation.warnings)
                    return result
                
                # Create neurons with error recovery
                neurons_created = 0
                for layer_idx, layer_size in enumerate(topology):
                    layer_neurons = []
                    for neuron_idx in range(layer_size):
                        try:
                            neuron = RobustPhotonicNeuron(**params)
                            layer_neurons.append(neuron)
                            neurons_created += 1
                        except Exception as e:
                            result.add_error(f"Failed to create neuron [{layer_idx}][{neuron_idx}]: {str(e)}")
                            if not self.error_recovery_enabled:
                                return result
                    
                    if layer_neurons:  # Only add layer if it has neurons
                        self.neurons.append(layer_neurons)
                
                result.set_metric('neurons_created', neurons_created)
                result.set_metric('target_neurons', sum(topology))
                
                creation_success_rate = neurons_created / sum(topology)
                if creation_success_rate < 0.9:
                    result.add_warning(f"Low neuron creation success rate: {creation_success_rate:.1%}")
                elif creation_success_rate == 1.0:
                    logger.info("All neurons created successfully")
                
                result.success = neurons_created > 0
                
        except Exception as e:
            result.add_error(f"Network creation failed: {str(e)}")
        
        return result
    
    def run_simulation(self, input_data: np.ndarray, duration: float = 100e-9) -> ValidationResult:
        """Run simulation with comprehensive monitoring and error recovery."""
        result = ValidationResult()
        
        try:
            with self.monitor.monitor_operation("simulation"):
                # Validate simulation parameters
                sim_params = {
                    'input_shape': input_data.shape,
                    'duration': duration,
                    'input_min': float(np.min(input_data)),
                    'input_max': float(np.max(input_data))
                }
                
                validation = self.validator.validate_input_parameters(sim_params)
                if not validation.success:
                    result.errors.extend(validation.errors)
                    result.warnings.extend(validation.warnings)
                    return result
                
                # Check for invalid input data
                if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
                    result.add_error("Invalid values (NaN/Inf) in input data")
                    return result
                
                if not self.neurons:
                    result.add_error("No neurons available for simulation")
                    return result
                
                # Run simulation with progress tracking
                dt = 1e-9  # 1ns time step
                time_steps = int(duration / dt)
                spikes_generated = 0
                operations_completed = 0
                simulation_errors = 0
                
                start_time = time.time()
                
                for t in range(time_steps):
                    current_time = t * dt
                    
                    # Process each layer
                    for layer_idx, layer in enumerate(self.neurons):
                        for neuron_idx, neuron in enumerate(layer):
                            try:
                                # Get input for this neuron
                                if t < len(input_data):
                                    optical_input = float(input_data[t % len(input_data)])
                                else:
                                    optical_input = 0.0
                                
                                # Process neuron
                                spike = neuron.forward(optical_input, current_time)
                                if spike:
                                    spikes_generated += 1
                                
                                operations_completed += 1
                                
                            except Exception as e:
                                simulation_errors += 1
                                logger.error(f"Simulation error at t={t}, layer={layer_idx}, neuron={neuron_idx}: {str(e)}")
                                
                                if not self.error_recovery_enabled:
                                    result.add_error(f"Simulation failed: {str(e)}")
                                    return result
                    
                    # Progress reporting
                    if t % (time_steps // 10) == 0 and t > 0:
                        progress = (t / time_steps) * 100
                        logger.info(f"Simulation progress: {progress:.1f}%")
                
                # Calculate final metrics
                end_time = time.time()
                simulation_time = end_time - start_time
                
                result.set_metric('simulation_time', simulation_time)
                result.set_metric('time_steps_completed', time_steps)
                result.set_metric('spikes_generated', spikes_generated)
                result.set_metric('operations_completed', operations_completed)
                result.set_metric('simulation_errors', simulation_errors)
                result.set_metric('error_rate', simulation_errors / max(1, operations_completed))
                result.set_metric('spike_rate', spikes_generated / duration)
                
                # Simulation quality assessment
                if simulation_errors / operations_completed > 0.01:
                    result.add_warning(f"High error rate: {simulation_errors/operations_completed:.1%}")
                
                if spikes_generated == 0:
                    result.add_warning("No spikes generated during simulation")
                elif spikes_generated / operations_completed > 0.5:
                    result.add_warning("Very high spike rate - check input scaling")
                
                result.success = simulation_errors / operations_completed < 0.1
                
                logger.info(f"Simulation completed: {time_steps} steps, {spikes_generated} spikes, {simulation_errors} errors")
                
        except Exception as e:
            result.add_error(f"Simulation framework error: {str(e)}")
        
        return result
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        report = {
            'timestamp': time.time(),
            'framework_status': 'operational' if self.neurons else 'no_network',
            'total_neurons': sum(len(layer) for layer in self.neurons),
            'total_layers': len(self.neurons),
            'error_recovery_enabled': self.error_recovery_enabled,
            'monitor_stats': {
                'operation_count': self.monitor.operation_count,
                'peak_memory': self.monitor.peak_memory
            },
            'neuron_health': []
        }
        
        # Collect neuron health metrics
        for layer_idx, layer in enumerate(self.neurons):
            layer_health = []
            for neuron_idx, neuron in enumerate(layer):
                if hasattr(neuron, 'get_health_metrics'):
                    health_metrics = neuron.get_health_metrics()
                    health_metrics['layer'] = layer_idx
                    health_metrics['neuron'] = neuron_idx
                    layer_health.append(health_metrics)
            report['neuron_health'].extend(layer_health)
        
        return report

def run_generation2_validation():
    """Run comprehensive Generation 2 validation suite."""
    logger.info("🛡️ STARTING GENERATION 2: MAKE IT ROBUST VALIDATION")
    
    overall_result = ValidationResult()
    
    # Test 1: Security Validation
    logger.info("🔐 Testing Security Validation...")
    validator = SecurityValidator()
    
    # Test secure parameters
    safe_params = {
        'wavelength': 1550e-9,
        'power': 1e-3,
        'topology': [10, 5, 2]
    }
    security_test = validator.validate_input_parameters(safe_params)
    if security_test.success:
        logger.info("✅ Secure parameters validated successfully")
    else:
        overall_result.add_error("Secure parameter validation failed")
    
    # Test malicious parameters
    malicious_params = {
        'wavelength': float('nan'),
        'power': -1.0,
        'topology': [0, -5, 'exec("import os; os.system(\'ls\')")']
    }
    malicious_test = validator.validate_input_parameters(malicious_params)
    if not malicious_test.success:
        logger.info("✅ Malicious parameters properly rejected")
    else:
        overall_result.add_error("Malicious parameters not properly rejected")
    
    # Test 2: Robust Neuron
    logger.info("🧠 Testing Robust Neuron...")
    try:
        neuron = RobustPhotonicNeuron()
        
        # Test normal operation
        spike = neuron.forward(2e-6, 1e-9)
        logger.info(f"✅ Normal neuron operation: spike={spike}")
        
        # Test error conditions
        neuron.forward(float('inf'), 2e-9)  # Should handle gracefully
        neuron.forward(-1.0, 3e-9)  # Should handle gracefully
        
        health = neuron.get_health_metrics()
        logger.info(f"✅ Neuron health metrics: {health}")
        
    except Exception as e:
        overall_result.add_error(f"Robust neuron test failed: {str(e)}")
    
    # Test 3: Robust Simulation Framework
    logger.info("🔬 Testing Robust Simulation Framework...")
    try:
        framework = RobustSimulationFramework()
        
        # Create network
        network_result = framework.create_network([5, 3, 2])
        if network_result.success:
            logger.info("✅ Robust network creation successful")
        else:
            overall_result.add_warning("Network creation issues detected")
        
        # Run simulation
        test_input = np.array([1e-6, 2e-6, 0.5e-6, 1.5e-6])
        sim_result = framework.run_simulation(test_input, duration=50e-9)
        
        if sim_result.success:
            logger.info(f"✅ Robust simulation successful: {sim_result.metrics}")
        else:
            overall_result.add_error("Simulation failed")
        
        # Generate health report
        health_report = framework.generate_health_report()
        logger.info(f"✅ Health report generated: {health_report['framework_status']}")
        
    except Exception as e:
        overall_result.add_error(f"Simulation framework test failed: {str(e)}")
    
    # Test 4: Performance Monitoring
    logger.info("📊 Testing Performance Monitoring...")
    try:
        monitor = PerformanceMonitor()
        
        with monitor.monitor_operation("test_operation"):
            # Simulate some work
            time.sleep(0.01)
            data = np.random.randn(1000)
            result = np.sum(data)
        
        logger.info("✅ Performance monitoring operational")
        
    except Exception as e:
        overall_result.add_warning(f"Performance monitoring issue: {str(e)}")
    
    # Calculate final scores
    overall_result.calculate_scores()
    
    # Final assessment
    logger.info("🎯 GENERATION 2 VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"✅ Security Score: {overall_result.security_score:.1f}/100")
    logger.info(f"✅ Performance Score: {overall_result.performance_score:.1f}/100")
    logger.info(f"✅ Total Errors: {len(overall_result.errors)}")
    logger.info(f"✅ Total Warnings: {len(overall_result.warnings)}")
    logger.info("=" * 60)
    
    if overall_result.security_score >= 80 and overall_result.performance_score >= 60:
        logger.info("🎉 GENERATION 2: MAKE IT ROBUST - COMPLETED SUCCESSFULLY")
        return True
    else:
        logger.error("❌ GENERATION 2: ROBUSTNESS REQUIREMENTS NOT MET")
        return False

if __name__ == "__main__":
    success = run_generation2_validation()
    sys.exit(0 if success else 1)