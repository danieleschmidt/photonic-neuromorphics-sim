"""
Advanced Reliability and Resilience Framework for Photonic Neuromorphic Systems.

This module implements comprehensive reliability measures including fault tolerance,
self-healing mechanisms, graceful degradation, and system health monitoring
for production-ready photonic neural networks.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import queue
import json
from pathlib import Path

from .exceptions import (
    PhotonicNeuromorphicError, ErrorSeverity, ErrorContext,
    OpticalModelError, ValidationError, handle_exception_with_recovery
)
from .monitoring import MetricsCollector, PerformanceProfiler


class SystemHealthStatus(Enum):
    """Overall system health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"


class ComponentStatus(Enum):
    """Individual component status."""
    OPERATIONAL = "operational"
    WARNING = "warning"
    ERROR = "error" 
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


@dataclass
class HealthMetric:
    """Health metric for system monitoring."""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    status: ComponentStatus = ComponentStatus.OPERATIONAL
    
    def evaluate_status(self) -> ComponentStatus:
        """Evaluate status based on thresholds."""
        if self.value >= self.threshold_critical:
            self.status = ComponentStatus.ERROR
        elif self.value >= self.threshold_warning:
            self.status = ComponentStatus.WARNING
        else:
            self.status = ComponentStatus.OPERATIONAL
        return self.status


@dataclass
class ComponentHealth:
    """Health information for a system component."""
    component_id: str
    component_type: str
    status: ComponentStatus = ComponentStatus.OPERATIONAL
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    error_count: int = 0
    last_error: Optional[str] = None
    uptime: float = 0.0
    performance_score: float = 1.0
    recovery_attempts: int = 0
    
    def add_metric(self, metric: HealthMetric) -> None:
        """Add health metric."""
        self.metrics[metric.name] = metric
        # Update overall status based on worst metric
        metric_status = metric.evaluate_status()
        if metric_status.value > self.status.value:
            self.status = metric_status


class FaultDetector:
    """Advanced fault detection for photonic systems."""
    
    def __init__(
        self,
        detection_window: float = 60.0,  # 1 minute detection window
        anomaly_threshold: float = 2.0,  # Standard deviations
        min_samples: int = 10
    ):
        self.detection_window = detection_window
        self.anomaly_threshold = anomaly_threshold
        self.min_samples = min_samples
        
        # Historical data for anomaly detection
        self.metric_history: Dict[str, List[Tuple[float, float]]] = {}
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def add_measurement(self, metric_name: str, value: float, timestamp: float = None) -> None:
        """Add measurement for fault detection."""
        if timestamp is None:
            timestamp = time.time()
        
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
        
        # Add new measurement
        self.metric_history[metric_name].append((timestamp, value))
        
        # Cleanup old measurements
        cutoff_time = timestamp - self.detection_window
        self.metric_history[metric_name] = [
            (t, v) for t, v in self.metric_history[metric_name] if t >= cutoff_time
        ]
        
        # Update baseline statistics
        self._update_baseline_stats(metric_name)
    
    def _update_baseline_stats(self, metric_name: str) -> None:
        """Update baseline statistics for anomaly detection."""
        if metric_name not in self.metric_history:
            return
        
        values = [v for t, v in self.metric_history[metric_name]]
        
        if len(values) >= self.min_samples:
            self.baseline_stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
    
    def detect_anomaly(self, metric_name: str, value: float) -> Tuple[bool, float]:
        """Detect if value is anomalous."""
        if metric_name not in self.baseline_stats:
            return False, 0.0
        
        stats = self.baseline_stats[metric_name]
        if stats['std'] == 0:
            return False, 0.0
        
        # Z-score anomaly detection
        z_score = abs(value - stats['mean']) / stats['std']
        is_anomaly = z_score > self.anomaly_threshold
        
        return is_anomaly, z_score
    
    def detect_system_anomalies(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect anomalies across multiple system metrics."""
        anomalies = []
        
        for metric_name, value in metrics.items():
            is_anomaly, z_score = self.detect_anomaly(metric_name, value)
            
            if is_anomaly:
                anomalies.append({
                    'metric': metric_name,
                    'value': value,
                    'z_score': z_score,
                    'severity': self._assess_anomaly_severity(z_score),
                    'timestamp': time.time()
                })
        
        return anomalies
    
    def _assess_anomaly_severity(self, z_score: float) -> ErrorSeverity:
        """Assess severity of anomaly based on Z-score."""
        if z_score > 5.0:
            return ErrorSeverity.CRITICAL
        elif z_score > 3.0:
            return ErrorSeverity.HIGH
        elif z_score > 2.0:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW


class CircuitBreaker:
    """Circuit breaker pattern for fault isolation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout  
        self.success_threshold = success_threshold
        
        # State tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open
        
        self.logger = logging.getLogger(__name__)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker."""
        if self.state == "open":
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise PhotonicNeuromorphicError(
                    "Circuit breaker is open",
                    context=ErrorContext(
                        operation="circuit_breaker_call",
                        component="circuit_breaker",
                        parameters={'state': self.state, 'failures': self.failure_count}
                    )
                )
            else:
                self.state = "half-open"
                self.success_count = 0
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        self.failure_count = 0
        
        if self.state == "half-open":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "closed"
                self.logger.info("Circuit breaker closed after successful recovery")
    
    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time
        }


class SelfHealingSystem:
    """Self-healing capabilities for photonic systems."""
    
    def __init__(
        self,
        healing_strategies: Optional[Dict[str, Callable]] = None,
        max_healing_attempts: int = 3,
        healing_timeout: float = 300.0  # 5 minutes
    ):
        self.healing_strategies = healing_strategies or {}
        self.max_healing_attempts = max_healing_attempts
        self.healing_timeout = healing_timeout
        
        # Healing state tracking
        self.healing_attempts: Dict[str, int] = {}
        self.healing_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(__name__)
        
        # Default healing strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self) -> None:
        """Register default healing strategies."""
        self.healing_strategies.update({
            'memory_pressure': self._heal_memory_pressure,
            'processing_overload': self._heal_processing_overload,
            'optical_power_anomaly': self._heal_optical_anomaly,
            'thermal_issue': self._heal_thermal_issue,
            'communication_failure': self._heal_communication_failure
        })
    
    def trigger_healing(self, issue_type: str, context: Dict[str, Any]) -> bool:
        """Trigger self-healing for specific issue."""
        if issue_type not in self.healing_strategies:
            self.logger.warning(f"No healing strategy for issue type: {issue_type}")
            return False
        
        # Check healing attempt limits
        attempts = self.healing_attempts.get(issue_type, 0)
        if attempts >= self.max_healing_attempts:
            self.logger.error(f"Max healing attempts reached for {issue_type}")
            return False
        
        self.healing_attempts[issue_type] = attempts + 1
        
        self.logger.info(f"Starting self-healing for {issue_type} (attempt {attempts + 1})")
        
        start_time = time.time()
        
        try:
            # Execute healing strategy
            healing_func = self.healing_strategies[issue_type]
            success = healing_func(context)
            
            healing_duration = time.time() - start_time
            
            # Record healing attempt
            self.healing_history.append({
                'issue_type': issue_type,
                'success': success,
                'attempt_number': attempts + 1,
                'duration': healing_duration,
                'timestamp': start_time,
                'context': context.copy()
            })
            
            if success:
                self.logger.info(f"Self-healing successful for {issue_type} in {healing_duration:.2f}s")
                # Reset attempt counter on success
                self.healing_attempts[issue_type] = 0
            else:
                self.logger.warning(f"Self-healing failed for {issue_type}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Self-healing exception for {issue_type}: {e}")
            self.healing_history.append({
                'issue_type': issue_type,
                'success': False,
                'attempt_number': attempts + 1,
                'error': str(e),
                'timestamp': start_time,
                'context': context.copy()
            })
            return False
    
    def _heal_memory_pressure(self, context: Dict[str, Any]) -> bool:
        """Heal memory pressure issues."""
        try:
            # Clear caches
            import gc
            gc.collect()
            
            # Release unused tensor memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Reduce batch sizes if applicable
            if 'batch_size' in context and context['batch_size'] > 1:
                context['batch_size'] = max(1, context['batch_size'] // 2)
                self.logger.info(f"Reduced batch size to {context['batch_size']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Memory healing failed: {e}")
            return False
    
    def _heal_processing_overload(self, context: Dict[str, Any]) -> bool:
        """Heal processing overload issues."""
        try:
            # Reduce processing parallelism
            if 'num_workers' in context and context['num_workers'] > 1:
                context['num_workers'] = max(1, context['num_workers'] // 2)
                self.logger.info(f"Reduced worker count to {context['num_workers']}")
            
            # Add processing delays
            time.sleep(0.1)  # Brief pause to reduce load
            
            return True
            
        except Exception:
            return False
    
    def _heal_optical_anomaly(self, context: Dict[str, Any]) -> bool:
        """Heal optical power anomalies."""
        try:
            # Reset optical parameters to safe defaults
            if 'optical_power' in context:
                context['optical_power'] = min(context['optical_power'], 1e-3)  # Cap at 1 mW
                self.logger.info(f"Capped optical power to {context['optical_power']:.2e} W")
            
            # Recalibrate if calibration function available
            if 'recalibrate_func' in context:
                context['recalibrate_func']()
            
            return True
            
        except Exception:
            return False
    
    def _heal_thermal_issue(self, context: Dict[str, Any]) -> bool:
        """Heal thermal management issues."""
        try:
            # Reduce processing frequency
            if 'target_frequency' in context:
                context['target_frequency'] *= 0.8  # 20% reduction
                self.logger.info(f"Reduced frequency to {context['target_frequency']:.2e} Hz")
            
            # Enable thermal throttling
            context['thermal_throttling'] = True
            
            return True
            
        except Exception:
            return False
    
    def _heal_communication_failure(self, context: Dict[str, Any]) -> bool:
        """Heal communication failures."""
        try:
            # Reset communication parameters
            if 'timeout' in context:
                context['timeout'] = min(context['timeout'] * 2, 30.0)  # Increase timeout
            
            # Retry with exponential backoff
            retry_delay = context.get('retry_delay', 1.0)
            time.sleep(retry_delay)
            context['retry_delay'] = min(retry_delay * 1.5, 10.0)
            
            return True
            
        except Exception:
            return False
    
    def get_healing_statistics(self) -> Dict[str, Any]:
        """Get self-healing statistics."""
        if not self.healing_history:
            return {'total_attempts': 0, 'success_rate': 0.0}
        
        total_attempts = len(self.healing_history)
        successful_attempts = sum(1 for h in self.healing_history if h['success'])
        success_rate = successful_attempts / total_attempts
        
        # Group by issue type
        issue_stats = {}
        for healing in self.healing_history:
            issue_type = healing['issue_type']
            if issue_type not in issue_stats:
                issue_stats[issue_type] = {'attempts': 0, 'successes': 0}
            
            issue_stats[issue_type]['attempts'] += 1
            if healing['success']:
                issue_stats[issue_type]['successes'] += 1
        
        return {
            'total_attempts': total_attempts,
            'successful_attempts': successful_attempts,
            'success_rate': success_rate,
            'issue_breakdown': issue_stats,
            'current_attempts': dict(self.healing_attempts)
        }


class GracefulDegradation:
    """Graceful degradation system for maintaining service under stress."""
    
    def __init__(self, degradation_levels: Optional[List[str]] = None):
        self.degradation_levels = degradation_levels or [
            'full_performance',
            'reduced_precision', 
            'reduced_features',
            'minimal_service',
            'emergency_mode'
        ]
        
        self.current_level = 'full_performance'
        self.level_history = []
        self.feature_status: Dict[str, bool] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def degrade_performance(self, target_level: str, reason: str) -> bool:
        """Degrade system performance to target level."""
        if target_level not in self.degradation_levels:
            self.logger.error(f"Unknown degradation level: {target_level}")
            return False
        
        current_index = self.degradation_levels.index(self.current_level)
        target_index = self.degradation_levels.index(target_level)
        
        if target_index <= current_index:
            self.logger.info(f"Already at or below degradation level {target_level}")
            return True
        
        self.logger.warning(f"Degrading performance from {self.current_level} to {target_level}: {reason}")
        
        # Apply degradation actions
        degradation_success = self._apply_degradation(target_level)
        
        if degradation_success:
            # Record degradation
            self.level_history.append({
                'from_level': self.current_level,
                'to_level': target_level,
                'reason': reason,
                'timestamp': time.time()
            })
            
            self.current_level = target_level
            return True
        else:
            self.logger.error(f"Failed to apply degradation to {target_level}")
            return False
    
    def restore_performance(self, target_level: str = 'full_performance') -> bool:
        """Restore system performance."""
        if target_level not in self.degradation_levels:
            return False
        
        current_index = self.degradation_levels.index(self.current_level)
        target_index = self.degradation_levels.index(target_level)
        
        if target_index >= current_index:
            return True
        
        self.logger.info(f"Restoring performance from {self.current_level} to {target_level}")
        
        restoration_success = self._apply_restoration(target_level)
        
        if restoration_success:
            self.level_history.append({
                'from_level': self.current_level,
                'to_level': target_level,
                'reason': 'performance_restoration',
                'timestamp': time.time()
            })
            
            self.current_level = target_level
            return True
        else:
            return False
    
    def _apply_degradation(self, target_level: str) -> bool:
        """Apply specific degradation actions."""
        try:
            if target_level == 'reduced_precision':
                # Reduce numerical precision
                self.feature_status['high_precision'] = False
                self.feature_status['double_precision'] = False
                
            elif target_level == 'reduced_features':
                # Disable non-essential features
                self.feature_status['advanced_analytics'] = False
                self.feature_status['detailed_logging'] = False
                self.feature_status['optimization'] = False
                
            elif target_level == 'minimal_service':
                # Keep only core functionality
                self.feature_status['monitoring'] = False
                self.feature_status['caching'] = False
                self.feature_status['parallel_processing'] = False
                
            elif target_level == 'emergency_mode':
                # Disable everything except basic operation
                for feature in self.feature_status:
                    self.feature_status[feature] = False
                self.feature_status['basic_operation'] = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Degradation application failed: {e}")
            return False
    
    def _apply_restoration(self, target_level: str) -> bool:
        """Apply performance restoration."""
        try:
            if target_level == 'full_performance':
                # Restore all features
                for feature in self.feature_status:
                    self.feature_status[feature] = True
                    
            elif target_level == 'reduced_precision':
                # Restore most features except precision
                for feature in self.feature_status:
                    self.feature_status[feature] = True
                self.feature_status['high_precision'] = False
                self.feature_status['double_precision'] = False
            
            # Add other restoration levels as needed
            
            return True
            
        except Exception as e:
            self.logger.error(f"Restoration failed: {e}")
            return False
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is currently enabled."""
        return self.feature_status.get(feature, True)
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status."""
        return {
            'current_level': self.current_level,
            'available_levels': self.degradation_levels,
            'feature_status': dict(self.feature_status),
            'degradation_history': self.level_history[-10:]  # Last 10 changes
        }


class ReliabilityManager:
    """Comprehensive reliability management system."""
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        config_file: Optional[str] = None
    ):
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        # Initialize subsystems
        self.fault_detector = FaultDetector()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.self_healing = SelfHealingSystem()
        self.graceful_degradation = GracefulDegradation()
        
        # System health tracking
        self.system_health = SystemHealthStatus.HEALTHY
        self.component_health: Dict[str, ComponentHealth] = {}
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 30.0  # 30 seconds
        
        # Load configuration
        if config_file:
            self._load_configuration(config_file)
        
        # Health check queue
        self.health_check_queue = queue.Queue()
        
        self.logger.info("Reliability manager initialized")
    
    def start_monitoring(self) -> None:
        """Start continuous system monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Detect anomalies
                anomalies = self.fault_detector.detect_system_anomalies(metrics)
                
                # Process anomalies
                for anomaly in anomalies:
                    self._handle_anomaly(anomaly)
                
                # Update system health
                self._update_system_health()
                
                # Record metrics
                if self.metrics_collector:
                    self.metrics_collector.record_metric("system_health_score", self._calculate_health_score())
                    self.metrics_collector.record_metric("active_anomalies", len(anomalies))
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)  # Brief pause before retry
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}
        
        try:
            # Memory metrics
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            metrics['memory_usage_mb'] = memory_info.rss / 1024 / 1024
            metrics['memory_percent'] = process.memory_percent()
            
            # CPU metrics
            metrics['cpu_percent'] = process.cpu_percent()
            
            # Add fault detector measurements
            for metric_name, value in metrics.items():
                self.fault_detector.add_measurement(metric_name, value)
            
        except ImportError:
            # psutil not available, use basic metrics
            import resource
            memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            metrics['memory_usage_kb'] = memory_usage
            
        # Custom photonic system metrics
        metrics['optical_power_stability'] = self._measure_optical_stability()
        metrics['processing_latency'] = self._measure_processing_latency()
        metrics['error_rate'] = self._calculate_error_rate()
        
        return metrics
    
    def _measure_optical_stability(self) -> float:
        """Measure optical system stability (placeholder)."""
        # In real implementation, this would measure actual optical parameters
        return np.random.normal(1.0, 0.05)  # Simulated stability metric
    
    def _measure_processing_latency(self) -> float:
        """Measure processing latency (placeholder)."""
        # In real implementation, this would measure actual processing times
        return np.random.normal(10.0, 2.0)  # Simulated latency in ms
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        if self.metrics_collector:
            error_count = self.metrics_collector.get_counter("total_errors", 0)
            total_operations = self.metrics_collector.get_counter("total_operations", 1)
            return error_count / total_operations
        return 0.0
    
    def _handle_anomaly(self, anomaly: Dict[str, Any]) -> None:
        """Handle detected anomaly."""
        metric_name = anomaly['metric']
        severity = anomaly['severity']
        
        self.logger.warning(f"Anomaly detected in {metric_name}: z-score {anomaly['z_score']:.2f}")
        
        # Determine healing strategy based on metric
        healing_strategies = {
            'memory_usage_mb': 'memory_pressure',
            'memory_percent': 'memory_pressure',
            'cpu_percent': 'processing_overload',
            'optical_power_stability': 'optical_power_anomaly',
            'processing_latency': 'processing_overload'
        }
        
        healing_type = healing_strategies.get(metric_name)
        if healing_type:
            # Trigger self-healing
            success = self.self_healing.trigger_healing(healing_type, anomaly)
            
            if not success and severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                # Self-healing failed, trigger graceful degradation
                degradation_level = 'reduced_features' if severity == ErrorSeverity.HIGH else 'minimal_service'
                self.graceful_degradation.degrade_performance(
                    degradation_level, 
                    f"Anomaly in {metric_name}"
                )
    
    def _update_system_health(self) -> None:
        """Update overall system health status."""
        component_statuses = [comp.status for comp in self.component_health.values()]
        
        if not component_statuses:
            self.system_health = SystemHealthStatus.HEALTHY
            return
        
        # Determine overall health based on component statuses
        if any(status == ComponentStatus.ERROR for status in component_statuses):
            if self.graceful_degradation.current_level == 'emergency_mode':
                self.system_health = SystemHealthStatus.CRITICAL
            else:
                self.system_health = SystemHealthStatus.DEGRADED
        elif any(status == ComponentStatus.WARNING for status in component_statuses):
            self.system_health = SystemHealthStatus.DEGRADED
        else:
            self.system_health = SystemHealthStatus.HEALTHY
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score (0-1)."""
        if not self.component_health:
            return 1.0
        
        component_scores = []
        for component in self.component_health.values():
            # Convert status to score
            status_scores = {
                ComponentStatus.OPERATIONAL: 1.0,
                ComponentStatus.WARNING: 0.7,
                ComponentStatus.ERROR: 0.3,
                ComponentStatus.OFFLINE: 0.0
            }
            score = status_scores.get(component.status, 0.5)
            component_scores.append(score)
        
        return np.mean(component_scores)
    
    def register_component(self, component_id: str, component_type: str) -> None:
        """Register a system component for health monitoring."""
        self.component_health[component_id] = ComponentHealth(
            component_id=component_id,
            component_type=component_type
        )
        
        self.logger.info(f"Registered component: {component_id} ({component_type})")
    
    def update_component_health(
        self,
        component_id: str,
        metrics: Dict[str, float],
        thresholds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> None:
        """Update health metrics for a component."""
        if component_id not in self.component_health:
            self.logger.warning(f"Unknown component: {component_id}")
            return
        
        component = self.component_health[component_id]
        
        # Update metrics
        for metric_name, value in metrics.items():
            if thresholds and metric_name in thresholds:
                warning_threshold, critical_threshold = thresholds[metric_name]
                health_metric = HealthMetric(
                    name=metric_name,
                    value=value,
                    threshold_warning=warning_threshold,
                    threshold_critical=critical_threshold
                )
                component.add_metric(health_metric)
        
        # Update performance score
        component.performance_score = self._calculate_component_performance_score(component)
    
    def _calculate_component_performance_score(self, component: ComponentHealth) -> float:
        """Calculate performance score for component."""
        if not component.metrics:
            return 1.0
        
        scores = []
        for metric in component.metrics.values():
            # Convert metric status to score
            status_scores = {
                ComponentStatus.OPERATIONAL: 1.0,
                ComponentStatus.WARNING: 0.7,
                ComponentStatus.ERROR: 0.3,
                ComponentStatus.OFFLINE: 0.0
            }
            scores.append(status_scores.get(metric.status, 0.5))
        
        return np.mean(scores)
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker()
        return self.circuit_breakers[name]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'overall_health': self.system_health.value,
            'health_score': self._calculate_health_score(),
            'degradation_level': self.graceful_degradation.current_level,
            'component_count': len(self.component_health),
            'active_circuit_breakers': len([cb for cb in self.circuit_breakers.values() if cb.state != "closed"]),
            'self_healing_stats': self.self_healing.get_healing_statistics(),
            'monitoring_active': self.monitoring_active
        }
    
    def generate_health_report(self) -> str:
        """Generate comprehensive health report."""
        status = self.get_system_status()
        
        report = [
            "=== PHOTONIC NEUROMORPHIC SYSTEM HEALTH REPORT ===",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Overall Health: {status['overall_health'].upper()}",
            f"Health Score: {status['health_score']:.2f}/1.00",
            f"Degradation Level: {status['degradation_level']}",
            "",
            "Component Status:"
        ]
        
        for comp_id, component in self.component_health.items():
            report.append(f"  {comp_id}: {component.status.value} (score: {component.performance_score:.2f})")
        
        if status['active_circuit_breakers'] > 0:
            report.extend([
                "",
                "Active Circuit Breakers:"
            ])
            for name, cb in self.circuit_breakers.items():
                if cb.state != "closed":
                    report.append(f"  {name}: {cb.state} ({cb.failure_count} failures)")
        
        healing_stats = status['self_healing_stats']
        if healing_stats['total_attempts'] > 0:
            report.extend([
                "",
                f"Self-Healing: {healing_stats['total_attempts']} attempts, {healing_stats['success_rate']:.1%} success rate"
            ])
        
        return "\\n".join(report)
    
    def _load_configuration(self, config_file: str) -> None:
        """Load configuration from file."""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Apply configuration settings
                if 'monitoring_interval' in config:
                    self.monitoring_interval = config['monitoring_interval']
                
                if 'fault_detection' in config:
                    fd_config = config['fault_detection']
                    self.fault_detector = FaultDetector(
                        detection_window=fd_config.get('detection_window', 60.0),
                        anomaly_threshold=fd_config.get('anomaly_threshold', 2.0),
                        min_samples=fd_config.get('min_samples', 10)
                    )
                
                self.logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")


# Decorator for reliable function execution
def reliable_execution(
    retries: int = 3,
    timeout: float = 30.0,
    circuit_breaker_name: Optional[str] = None,
    fallback_func: Optional[Callable] = None
):
    """Decorator for reliable function execution with retries and circuit breaker."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get reliability manager from global state if available
            reliability_manager = getattr(wrapper, '_reliability_manager', None)
            
            circuit_breaker = None
            if circuit_breaker_name and reliability_manager:
                circuit_breaker = reliability_manager.get_circuit_breaker(circuit_breaker_name)
            
            for attempt in range(retries + 1):
                try:
                    if circuit_breaker:
                        return circuit_breaker.call(func, *args, **kwargs)
                    else:
                        # Execute with timeout
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(func, *args, **kwargs)
                            return future.result(timeout=timeout)
                    
                except TimeoutError:
                    if attempt < retries:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    elif fallback_func:
                        return fallback_func(*args, **kwargs)
                    else:
                        raise
                
                except Exception as e:
                    if attempt < retries:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    elif fallback_func:
                        return fallback_func(*args, **kwargs)
                    else:
                        raise
            
            return None  # Should not reach here
        
        return wrapper
    
    return decorator


# Context manager for reliability scope
@contextmanager
def reliability_scope(reliability_manager: ReliabilityManager):
    """Context manager for reliability-aware operations."""
    try:
        # Start monitoring if not already active
        if not reliability_manager.monitoring_active:
            reliability_manager.start_monitoring()
        
        yield reliability_manager
        
    except Exception as e:
        # Handle exceptions with self-healing
        healing_context = {
            'exception': str(e),
            'exception_type': type(e).__name__,
            'timestamp': time.time()
        }
        
        # Attempt self-healing
        healing_success = reliability_manager.self_healing.trigger_healing(
            'communication_failure', healing_context
        )
        
        if not healing_success:
            # Trigger graceful degradation
            reliability_manager.graceful_degradation.degrade_performance(
                'reduced_features', f"Exception: {e}"
            )
        
        raise
    
    finally:
        # Cleanup if needed
        pass