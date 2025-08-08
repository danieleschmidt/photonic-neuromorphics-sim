"""
Resilience and Recovery Framework for Photonic Neuromorphic Systems.

This module provides comprehensive fault tolerance, error recovery, and
resilience mechanisms for photonic neural networks, including automated
recovery strategies, redundancy management, and graceful degradation.
"""

import time
import numpy as np
import torch
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, Future

from .core import PhotonicSNN, WaveguideNeuron
from .components import PhotonicComponent
from .architectures import PhotonicCrossbar, PhotonicReservoir
from .exceptions import OpticalModelError, NetworkTopologyError, handle_exception_with_recovery
from .monitoring import MetricsCollector


class FailureType(Enum):
    """Types of failures in photonic neuromorphic systems."""
    OPTICAL_LOSS = "optical_loss"
    COMPONENT_FAILURE = "component_failure"
    NETWORK_PARTITION = "network_partition"
    POWER_DEGRADATION = "power_degradation"
    THERMAL_DRIFT = "thermal_drift"
    WAVELENGTH_DRIFT = "wavelength_drift"
    CROSSTALK_INTERFERENCE = "crosstalk_interference"
    MEMORY_CORRUPTION = "memory_corruption"
    COMPUTATION_ERROR = "computation_error"
    COMMUNICATION_FAILURE = "communication_failure"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure modes."""
    RESTART = "restart"
    RECALIBRATE = "recalibrate"
    REROUTE = "reroute"
    REDUNDANCY_SWITCH = "redundancy_switch"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FALLBACK_MODE = "fallback_mode"
    ISOLATION = "isolation"


@dataclass
class FailureEvent:
    """Record of a failure event."""
    timestamp: float
    failure_type: FailureType
    severity: float  # 0.0 = minor, 1.0 = critical
    component_id: str
    description: str
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_success: Optional[bool] = None
    recovery_time: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResilienceConfig:
    """Configuration for resilience mechanisms."""
    enable_redundancy: bool = True
    redundancy_factor: int = 2  # Number of backup components
    enable_auto_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_timeout: float = 30.0  # seconds
    failure_threshold: float = 0.1  # 10% performance degradation triggers response
    enable_graceful_degradation: bool = True
    enable_predictive_maintenance: bool = True
    monitoring_interval: float = 1.0  # seconds
    enable_checkpointing: bool = True
    checkpoint_interval: float = 10.0  # seconds


class ResilienceManager:
    """
    Comprehensive resilience manager for photonic neuromorphic systems.
    
    Provides fault detection, automatic recovery, redundancy management,
    and graceful degradation capabilities.
    """
    
    def __init__(
        self, 
        config: Optional[ResilienceConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.config = config or ResilienceConfig()
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.failure_history: List[FailureEvent] = []
        self.recovery_strategies: Dict[FailureType, List[RecoveryStrategy]] = self._initialize_recovery_strategies()
        self.redundant_components: Dict[str, List[Any]] = {}
        self.active_components: Dict[str, Any] = {}
        self.failed_components: Dict[str, Any] = {}
        
        # Monitoring and recovery threads
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.recovery_executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance baselines
        self.performance_baselines: Dict[str, float] = {}
        self.current_performance: Dict[str, float] = {}
        
        # Checkpointing
        self.checkpoints: List[Dict[str, Any]] = []
        self.last_checkpoint_time = 0.0
        
        self.logger.info("Resilience Manager initialized")
    
    def _initialize_recovery_strategies(self) -> Dict[FailureType, List[RecoveryStrategy]]:
        """Initialize recovery strategies for different failure types."""
        return {
            FailureType.OPTICAL_LOSS: [
                RecoveryStrategy.PARAMETER_ADJUSTMENT,
                RecoveryStrategy.REROUTE,
                RecoveryStrategy.REDUNDANCY_SWITCH
            ],
            FailureType.COMPONENT_FAILURE: [
                RecoveryStrategy.REDUNDANCY_SWITCH,
                RecoveryStrategy.ISOLATION,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            FailureType.NETWORK_PARTITION: [
                RecoveryStrategy.REROUTE,
                RecoveryStrategy.RESTART,
                RecoveryStrategy.FALLBACK_MODE
            ],
            FailureType.POWER_DEGRADATION: [
                RecoveryStrategy.PARAMETER_ADJUSTMENT,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            FailureType.THERMAL_DRIFT: [
                RecoveryStrategy.RECALIBRATE,
                RecoveryStrategy.PARAMETER_ADJUSTMENT
            ],
            FailureType.WAVELENGTH_DRIFT: [
                RecoveryStrategy.RECALIBRATE,
                RecoveryStrategy.PARAMETER_ADJUSTMENT
            ],
            FailureType.CROSSTALK_INTERFERENCE: [
                RecoveryStrategy.REROUTE,
                RecoveryStrategy.ISOLATION,
                RecoveryStrategy.PARAMETER_ADJUSTMENT
            ],
            FailureType.MEMORY_CORRUPTION: [
                RecoveryStrategy.RESTART,
                RecoveryStrategy.FALLBACK_MODE
            ],
            FailureType.COMPUTATION_ERROR: [
                RecoveryStrategy.RECALIBRATE,
                RecoveryStrategy.RESTART,
                RecoveryStrategy.FALLBACK_MODE
            ],
            FailureType.COMMUNICATION_FAILURE: [
                RecoveryStrategy.REROUTE,
                RecoveryStrategy.RESTART
            ]
        }
    
    def register_component(self, component_id: str, component: Any, 
                          enable_redundancy: bool = True) -> None:
        """Register a component for monitoring and resilience management."""
        self.active_components[component_id] = component
        
        if enable_redundancy and self.config.enable_redundancy:
            # Create redundant components
            redundant_copies = []
            for i in range(self.config.redundancy_factor):
                try:
                    # Create copy of component (simplified)
                    if hasattr(component, 'copy'):
                        backup = component.copy()
                    elif hasattr(component, '__deepcopy__'):
                        import copy
                        backup = copy.deepcopy(component)
                    else:
                        # Fallback: create new instance of same type
                        backup = type(component)()
                    
                    redundant_copies.append(backup)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create backup for {component_id}: {e}")
            
            if redundant_copies:
                self.redundant_components[component_id] = redundant_copies
                self.logger.info(f"Created {len(redundant_copies)} backups for {component_id}")
        
        # Establish performance baseline
        baseline_performance = self._measure_component_performance(component)
        self.performance_baselines[component_id] = baseline_performance
        
        self.logger.info(f"Registered component {component_id} with resilience management")
    
    def start_monitoring(self, system: Any) -> None:
        """Start continuous system monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(system,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("Started resilience monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop continuous system monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.recovery_executor.shutdown(wait=False)
        
        self.logger.info("Stopped resilience monitoring")
    
    def _monitoring_loop(self, system: Any) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check system health
                health_issues = self._detect_health_issues(system)
                
                # Process any detected issues
                for issue in health_issues:
                    self._handle_failure_event(issue)
                
                # Checkpointing
                if (self.config.enable_checkpointing and 
                    time.time() - self.last_checkpoint_time > self.config.checkpoint_interval):
                    self._create_checkpoint(system)
                
                # Record monitoring metrics
                self.metrics_collector.record_metric(
                    "resilience_monitoring_cycle", time.time()
                )
                self.metrics_collector.record_metric(
                    "active_failure_events", len(self.failure_history)
                )
                
                # Sleep until next monitoring cycle
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _detect_health_issues(self, system: Any) -> List[FailureEvent]:
        """Detect health issues in the system."""
        issues = []
        
        try:
            # Check component performance
            for component_id, component in self.active_components.items():
                current_perf = self._measure_component_performance(component)
                baseline_perf = self.performance_baselines.get(component_id, 1.0)
                
                self.current_performance[component_id] = current_perf
                
                # Detect performance degradation
                if baseline_perf > 0:
                    degradation = (baseline_perf - current_perf) / baseline_perf
                    
                    if degradation > self.config.failure_threshold:
                        # Classify failure type based on degradation pattern
                        failure_type = self._classify_failure_type(degradation, component)
                        
                        issue = FailureEvent(
                            timestamp=time.time(),
                            failure_type=failure_type,
                            severity=min(degradation, 1.0),
                            component_id=component_id,
                            description=f"Performance degraded by {degradation:.1%}",
                            metrics={
                                "baseline_performance": baseline_perf,
                                "current_performance": current_perf,
                                "degradation": degradation
                            }
                        )
                        
                        issues.append(issue)
            
            # System-level health checks
            system_issues = self._detect_system_level_issues(system)
            issues.extend(system_issues)
            
        except Exception as e:
            self.logger.error(f"Health detection error: {e}")
        
        return issues
    
    def _measure_component_performance(self, component: Any) -> float:
        """Measure component performance (simplified metric)."""
        try:
            # For photonic components, test transfer function
            if hasattr(component, 'transfer_function'):
                transmission, phase = component.transfer_function(1550e-9, 1e-3)
                
                # Performance metric based on transmission and stability
                if not (0 <= transmission <= 1):
                    return 0.0  # Invalid transmission
                
                if np.isnan(transmission) or np.isinf(transmission):
                    return 0.0  # Numerical issues
                
                return transmission  # Simple performance metric
            
            # For neural networks, test forward pass
            elif hasattr(component, 'forward'):
                test_input = torch.randn(10, 100)  # Simple test input
                
                try:
                    start_time = time.time()
                    output = component.forward(test_input)
                    execution_time = time.time() - start_time
                    
                    # Performance based on speed and output quality
                    if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
                        return 0.0  # Invalid output
                    
                    # Faster execution = better performance (up to a limit)
                    speed_score = min(1.0 / max(execution_time, 0.001), 1.0)
                    
                    # Output magnitude as quality indicator
                    output_magnitude = torch.mean(torch.abs(output)).item()
                    magnitude_score = min(output_magnitude / 1.0, 1.0)  # Normalize to 1.0
                    
                    return (speed_score + magnitude_score) / 2.0
                    
                except Exception:
                    return 0.0  # Forward pass failed
            
            else:
                # Unknown component type, assume healthy
                return 1.0
                
        except Exception:
            return 0.0  # Measurement failed
    
    def _classify_failure_type(self, degradation: float, component: Any) -> FailureType:
        """Classify the type of failure based on degradation pattern."""
        # Simple classification based on component type and degradation severity
        if degradation > 0.8:  # > 80% degradation
            return FailureType.COMPONENT_FAILURE
        elif degradation > 0.5:  # > 50% degradation
            if hasattr(component, 'transfer_function'):
                return FailureType.OPTICAL_LOSS
            else:
                return FailureType.COMPUTATION_ERROR
        elif degradation > 0.2:  # > 20% degradation
            return FailureType.POWER_DEGRADATION
        else:
            return FailureType.THERMAL_DRIFT  # Minor degradation
    
    def _detect_system_level_issues(self, system: Any) -> List[FailureEvent]:
        """Detect system-level issues."""
        issues = []
        
        try:
            # Check for network connectivity issues
            if hasattr(system, 'topology') and hasattr(system, 'layers'):
                expected_layers = len(system.topology) - 1
                actual_layers = len(system.layers)
                
                if actual_layers != expected_layers:
                    issue = FailureEvent(
                        timestamp=time.time(),
                        failure_type=FailureType.NETWORK_PARTITION,
                        severity=0.8,
                        component_id="network_topology",
                        description=f"Layer count mismatch: {actual_layers} vs {expected_layers}"
                    )
                    issues.append(issue)
            
            # Check for memory issues
            if hasattr(system, 'layers'):
                for i, layer in enumerate(system.layers):
                    if hasattr(layer, 'data'):
                        if torch.any(torch.isnan(layer.data)):
                            issue = FailureEvent(
                                timestamp=time.time(),
                                failure_type=FailureType.MEMORY_CORRUPTION,
                                severity=1.0,
                                component_id=f"layer_{i}",
                                description="NaN values detected in weight matrix"
                            )
                            issues.append(issue)
        
        except Exception as e:
            self.logger.error(f"System-level issue detection failed: {e}")
        
        return issues
    
    def _handle_failure_event(self, failure_event: FailureEvent) -> None:
        """Handle a detected failure event."""
        self.failure_history.append(failure_event)
        
        self.logger.warning(
            f"Failure detected: {failure_event.failure_type.value} in {failure_event.component_id} "
            f"(severity: {failure_event.severity:.2f})"
        )
        
        if self.config.enable_auto_recovery:
            # Submit recovery task to executor
            recovery_future = self.recovery_executor.submit(
                self._attempt_recovery, failure_event
            )
            
            # Don't wait for recovery to complete (asynchronous)
            recovery_future.add_done_callback(
                lambda fut: self._handle_recovery_completion(failure_event, fut)
            )
        
        # Record failure metrics
        self.metrics_collector.increment_counter(f"failure_{failure_event.failure_type.value}")
        self.metrics_collector.record_metric("failure_severity", failure_event.severity)
    
    def _attempt_recovery(self, failure_event: FailureEvent) -> bool:
        """Attempt to recover from a failure event."""
        recovery_strategies = self.recovery_strategies.get(
            failure_event.failure_type, [RecoveryStrategy.RESTART]
        )
        
        for attempt in range(self.config.max_recovery_attempts):
            for strategy in recovery_strategies:
                try:
                    self.logger.info(
                        f"Attempting recovery strategy {strategy.value} for {failure_event.component_id} "
                        f"(attempt {attempt + 1}/{self.config.max_recovery_attempts})"
                    )
                    
                    start_time = time.time()
                    recovery_success = self._execute_recovery_strategy(
                        failure_event, strategy
                    )
                    recovery_time = time.time() - start_time
                    
                    # Update failure event with recovery info
                    failure_event.recovery_strategy = strategy
                    failure_event.recovery_success = recovery_success
                    failure_event.recovery_time = recovery_time
                    
                    if recovery_success:
                        self.logger.info(
                            f"Recovery successful using {strategy.value} "
                            f"(took {recovery_time:.2f}s)"
                        )
                        return True
                    
                except Exception as e:
                    self.logger.error(f"Recovery strategy {strategy.value} failed: {e}")
        
        self.logger.error(
            f"All recovery attempts failed for {failure_event.component_id}"
        )
        return False
    
    def _execute_recovery_strategy(
        self, failure_event: FailureEvent, strategy: RecoveryStrategy
    ) -> bool:
        """Execute a specific recovery strategy."""
        component_id = failure_event.component_id
        component = self.active_components.get(component_id)
        
        if strategy == RecoveryStrategy.RESTART:
            return self._restart_component(component_id, component)
        
        elif strategy == RecoveryStrategy.RECALIBRATE:
            return self._recalibrate_component(component_id, component)
        
        elif strategy == RecoveryStrategy.REROUTE:
            return self._reroute_component(component_id, component)
        
        elif strategy == RecoveryStrategy.REDUNDANCY_SWITCH:
            return self._switch_to_redundant_component(component_id)
        
        elif strategy == RecoveryStrategy.PARAMETER_ADJUSTMENT:
            return self._adjust_component_parameters(component_id, component, failure_event)
        
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._enable_graceful_degradation(component_id, component)
        
        elif strategy == RecoveryStrategy.FALLBACK_MODE:
            return self._enable_fallback_mode(component_id, component)
        
        elif strategy == RecoveryStrategy.ISOLATION:
            return self._isolate_component(component_id, component)
        
        else:
            self.logger.warning(f"Unknown recovery strategy: {strategy}")
            return False
    
    def _restart_component(self, component_id: str, component: Any) -> bool:
        """Restart a component."""
        try:
            # Reset component state if possible
            if hasattr(component, 'reset'):
                component.reset()
            elif hasattr(component, '__init__'):
                # Re-initialize component
                component.__init__()
            
            # Re-establish baseline performance
            new_performance = self._measure_component_performance(component)
            self.performance_baselines[component_id] = new_performance
            
            return new_performance > 0.5  # Consider successful if > 50% performance
            
        except Exception as e:
            self.logger.error(f"Component restart failed: {e}")
            return False
    
    def _recalibrate_component(self, component_id: str, component: Any) -> bool:
        """Recalibrate a component."""
        try:
            # For photonic components, adjust calibration parameters
            if hasattr(component, 'params'):
                # Simple calibration: adjust power level
                if hasattr(component.params, 'power'):
                    original_power = component.params.power
                    component.params.power *= 1.1  # Increase power by 10%
                    
                    # Test if calibration improved performance
                    new_performance = self._measure_component_performance(component)
                    if new_performance > 0.7:  # Good performance threshold
                        return True
                    else:
                        # Revert if no improvement
                        component.params.power = original_power
            
            return False
            
        except Exception as e:
            self.logger.error(f"Component recalibration failed: {e}")
            return False
    
    def _reroute_component(self, component_id: str, component: Any) -> bool:
        """Reroute connections around a component."""
        try:
            # This is a simplified rerouting implementation
            # In practice, would involve complex network topology changes
            
            self.logger.info(f"Rerouting traffic around {component_id}")
            
            # For demonstration, mark component as bypassed
            if hasattr(component, '_bypassed'):
                component._bypassed = True
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Component rerouting failed: {e}")
            return False
    
    def _switch_to_redundant_component(self, component_id: str) -> bool:
        """Switch to a redundant backup component."""
        try:
            redundant_components = self.redundant_components.get(component_id, [])
            
            if not redundant_components:
                self.logger.warning(f"No redundant components available for {component_id}")
                return False
            
            # Find a healthy backup component
            for backup in redundant_components:
                backup_performance = self._measure_component_performance(backup)
                
                if backup_performance > 0.7:  # Healthy backup
                    # Switch to backup
                    failed_component = self.active_components[component_id]
                    self.active_components[component_id] = backup
                    self.failed_components[f"{component_id}_failed"] = failed_component
                    
                    # Update performance baseline
                    self.performance_baselines[component_id] = backup_performance
                    
                    self.logger.info(f"Switched to redundant component for {component_id}")
                    return True
            
            self.logger.warning(f"No healthy redundant components for {component_id}")
            return False
            
        except Exception as e:
            self.logger.error(f"Redundancy switch failed: {e}")
            return False
    
    def _adjust_component_parameters(self, component_id: str, component: Any, 
                                   failure_event: FailureEvent) -> bool:
        """Adjust component parameters based on failure type."""
        try:
            if failure_event.failure_type == FailureType.OPTICAL_LOSS:
                # Increase optical power
                if hasattr(component, 'params') and hasattr(component.params, 'power'):
                    component.params.power = min(component.params.power * 1.2, 1.0)  # Max 1W
                    return True
            
            elif failure_event.failure_type == FailureType.THERMAL_DRIFT:
                # Adjust temperature compensation
                if hasattr(component, 'params') and hasattr(component.params, 'temperature'):
                    component.params.temperature = 300.0  # Reset to room temperature
                    return True
            
            elif failure_event.failure_type == FailureType.WAVELENGTH_DRIFT:
                # Reset wavelength to standard
                if hasattr(component, 'params') and hasattr(component.params, 'wavelength'):
                    component.params.wavelength = 1550e-9  # Standard C-band
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Parameter adjustment failed: {e}")
            return False
    
    def _enable_graceful_degradation(self, component_id: str, component: Any) -> bool:
        """Enable graceful degradation mode."""
        try:
            # Reduce component precision/quality to maintain basic functionality
            if hasattr(component, '_degraded_mode'):
                component._degraded_mode = True
                self.logger.info(f"Enabled graceful degradation for {component_id}")
                return True
            
            # For neural networks, reduce precision
            if hasattr(component, 'layers'):
                for layer in component.layers:
                    if hasattr(layer, 'data'):
                        # Quantize weights to lower precision
                        layer.data = torch.round(layer.data * 8) / 8  # 3-bit quantization
                
                self.logger.info(f"Applied precision reduction for {component_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Graceful degradation failed: {e}")
            return False
    
    def _enable_fallback_mode(self, component_id: str, component: Any) -> bool:
        """Enable fallback mode with basic functionality."""
        try:
            # Switch to simplified operation mode
            if hasattr(component, '_fallback_mode'):
                component._fallback_mode = True
                self.logger.info(f"Enabled fallback mode for {component_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Fallback mode activation failed: {e}")
            return False
    
    def _isolate_component(self, component_id: str, component: Any) -> bool:
        """Isolate a failed component from the system."""
        try:
            # Mark component as isolated
            if hasattr(component, '_isolated'):
                component._isolated = True
            
            # Move to failed components
            self.failed_components[component_id] = component
            
            # Remove from active components (if safe to do so)
            # In practice, would need more sophisticated isolation logic
            
            self.logger.info(f"Isolated component {component_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Component isolation failed: {e}")
            return False
    
    def _handle_recovery_completion(self, failure_event: FailureEvent, future: Future) -> None:
        """Handle completion of recovery attempt."""
        try:
            recovery_success = future.result()
            
            if recovery_success:
                self.metrics_collector.increment_counter("successful_recoveries")
                self.logger.info(f"Recovery completed successfully for {failure_event.component_id}")
            else:
                self.metrics_collector.increment_counter("failed_recoveries")
                self.logger.error(f"Recovery failed for {failure_event.component_id}")
        
        except Exception as e:
            self.logger.error(f"Recovery completion handling failed: {e}")
    
    def _create_checkpoint(self, system: Any) -> None:
        """Create a system checkpoint for recovery purposes."""
        try:
            checkpoint = {
                "timestamp": time.time(),
                "system_type": type(system).__name__,
                "component_states": {},
                "performance_baselines": self.performance_baselines.copy(),
                "active_components": list(self.active_components.keys())
            }
            
            # Capture component states (simplified)
            for component_id, component in self.active_components.items():
                try:
                    if hasattr(component, 'state_dict'):
                        checkpoint["component_states"][component_id] = component.state_dict()
                    elif hasattr(component, '__dict__'):
                        # Capture basic state
                        checkpoint["component_states"][component_id] = {
                            key: value for key, value in component.__dict__.items()
                            if not key.startswith('_') and isinstance(value, (int, float, str, bool))
                        }
                except Exception as e:
                    self.logger.warning(f"Failed to capture state for {component_id}: {e}")
            
            # Store checkpoint (keep only last N checkpoints)
            self.checkpoints.append(checkpoint)
            if len(self.checkpoints) > 10:  # Keep last 10 checkpoints
                self.checkpoints.pop(0)
            
            self.last_checkpoint_time = time.time()
            
            self.metrics_collector.increment_counter("checkpoints_created")
            self.logger.debug(f"Checkpoint created with {len(checkpoint['component_states'])} component states")
            
        except Exception as e:
            self.logger.error(f"Checkpoint creation failed: {e}")
    
    def restore_from_checkpoint(self, system: Any, checkpoint_index: int = -1) -> bool:
        """Restore system from a checkpoint."""
        try:
            if not self.checkpoints:
                self.logger.error("No checkpoints available for restore")
                return False
            
            checkpoint = self.checkpoints[checkpoint_index]
            
            self.logger.info(f"Restoring from checkpoint created at {time.ctime(checkpoint['timestamp'])}")
            
            # Restore component states
            restored_count = 0
            for component_id, component_state in checkpoint["component_states"].items():
                if component_id in self.active_components:
                    component = self.active_components[component_id]
                    
                    try:
                        if hasattr(component, 'load_state_dict') and isinstance(component_state, dict):
                            component.load_state_dict(component_state)
                            restored_count += 1
                        elif isinstance(component_state, dict):
                            # Restore basic attributes
                            for key, value in component_state.items():
                                if hasattr(component, key):
                                    setattr(component, key, value)
                            restored_count += 1
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to restore state for {component_id}: {e}")
            
            # Restore performance baselines
            self.performance_baselines.update(checkpoint["performance_baselines"])
            
            self.metrics_collector.increment_counter("checkpoint_restores")
            self.logger.info(f"Restored {restored_count} components from checkpoint")
            
            return restored_count > 0
            
        except Exception as e:
            self.logger.error(f"Checkpoint restore failed: {e}")
            return False
    
    def get_resilience_report(self) -> Dict[str, Any]:
        """Generate comprehensive resilience report."""
        current_time = time.time()
        
        # Analyze failure history
        recent_failures = [
            f for f in self.failure_history 
            if current_time - f.timestamp < 3600  # Last hour
        ]
        
        failure_types = {}
        for failure in recent_failures:
            failure_type = failure.failure_type.value
            if failure_type not in failure_types:
                failure_types[failure_type] = 0
            failure_types[failure_type] += 1
        
        # Calculate recovery success rate
        recovery_attempts = [f for f in self.failure_history if f.recovery_strategy is not None]
        successful_recoveries = [f for f in recovery_attempts if f.recovery_success]
        
        recovery_rate = len(successful_recoveries) / max(len(recovery_attempts), 1)
        
        # Component health summary
        component_health = {}
        for component_id in self.active_components:
            current_perf = self.current_performance.get(component_id, 0.0)
            baseline_perf = self.performance_baselines.get(component_id, 1.0)
            
            if baseline_perf > 0:
                health_ratio = current_perf / baseline_perf
            else:
                health_ratio = 0.0
            
            component_health[component_id] = min(health_ratio, 1.0)
        
        # Overall system health
        if component_health:
            overall_health = np.mean(list(component_health.values()))
        else:
            overall_health = 1.0
        
        report = {
            "timestamp": current_time,
            "overall_health": overall_health,
            "component_health": component_health,
            "total_failures": len(self.failure_history),
            "recent_failures": len(recent_failures),
            "failure_types": failure_types,
            "recovery_success_rate": recovery_rate,
            "active_components": len(self.active_components),
            "failed_components": len(self.failed_components),
            "redundant_components": len(self.redundant_components),
            "checkpoints_available": len(self.checkpoints),
            "monitoring_active": self.monitoring_active,
            "configuration": {
                "redundancy_enabled": self.config.enable_redundancy,
                "auto_recovery_enabled": self.config.enable_auto_recovery,
                "graceful_degradation_enabled": self.config.enable_graceful_degradation
            }
        }
        
        return report
    
    def get_failure_analysis(self) -> Dict[str, Any]:
        """Get detailed failure analysis."""
        if not self.failure_history:
            return {"message": "No failure events recorded"}
        
        # Failure frequency analysis
        failure_frequency = {}
        for failure in self.failure_history:
            failure_type = failure.failure_type.value
            if failure_type not in failure_frequency:
                failure_frequency[failure_type] = []
            failure_frequency[failure_type].append(failure.timestamp)
        
        # Recovery strategy effectiveness
        strategy_effectiveness = {}
        for failure in self.failure_history:
            if failure.recovery_strategy and failure.recovery_success is not None:
                strategy = failure.recovery_strategy.value
                if strategy not in strategy_effectiveness:
                    strategy_effectiveness[strategy] = {"success": 0, "total": 0}
                
                strategy_effectiveness[strategy]["total"] += 1
                if failure.recovery_success:
                    strategy_effectiveness[strategy]["success"] += 1
        
        # Calculate success rates
        for strategy in strategy_effectiveness:
            total = strategy_effectiveness[strategy]["total"]
            success = strategy_effectiveness[strategy]["success"]
            strategy_effectiveness[strategy]["success_rate"] = success / total if total > 0 else 0.0
        
        # Recent trends
        current_time = time.time()
        recent_24h = [f for f in self.failure_history if current_time - f.timestamp < 86400]
        recent_1h = [f for f in self.failure_history if current_time - f.timestamp < 3600]
        
        return {
            "total_failures": len(self.failure_history),
            "failures_last_24h": len(recent_24h),
            "failures_last_1h": len(recent_1h),
            "failure_frequency_by_type": failure_frequency,
            "recovery_strategy_effectiveness": strategy_effectiveness,
            "most_common_failure": max(failure_frequency.keys(), key=lambda k: len(failure_frequency[k])) if failure_frequency else None,
            "average_recovery_time": np.mean([f.recovery_time for f in self.failure_history if f.recovery_time]) if any(f.recovery_time for f in self.failure_history) else 0.0
        }


def create_resilient_system(
    base_system: Any, 
    config: Optional[ResilienceConfig] = None
) -> Tuple[Any, ResilienceManager]:
    """Create a resilient version of a photonic neuromorphic system."""
    resilience_manager = ResilienceManager(config)
    
    # Register system components
    if hasattr(base_system, 'neurons'):
        for layer_idx, layer_neurons in enumerate(base_system.neurons):
            for neuron_idx, neuron in enumerate(layer_neurons):
                component_id = f"neuron_{layer_idx}_{neuron_idx}"
                resilience_manager.register_component(component_id, neuron)
    
    elif hasattr(base_system, 'modulators'):
        for i, modulator_row in enumerate(base_system.modulators):
            for j, modulator in enumerate(modulator_row):
                component_id = f"modulator_{i}_{j}"
                resilience_manager.register_component(component_id, modulator)
    
    elif hasattr(base_system, 'node_components'):
        for i, component in enumerate(base_system.node_components):
            component_id = f"reservoir_node_{i}"
            resilience_manager.register_component(component_id, component)
    
    # Register the main system
    resilience_manager.register_component("main_system", base_system)
    
    # Start monitoring
    resilience_manager.start_monitoring(base_system)
    
    return base_system, resilience_manager
