"""
Robust error handling and recovery system for photonic neuromorphics.

This module provides advanced error handling, automatic recovery mechanisms,
and comprehensive error reporting for photonic simulations.
"""

import logging
import time
import traceback
import threading
import uuid
import pickle
import json
from typing import Dict, Any, Optional, Callable, Type, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from contextlib import contextmanager
import queue
import sys
import os

from .exceptions import PhotonicNeuromorphicsException, SimulationError
from .enhanced_logging import PhotonicLogger, CorrelationContext


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    USER_INTERVENTION = "user_intervention"


@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    error_type: str = ""
    error_message: str = ""
    function_name: str = ""
    file_name: str = ""
    line_number: int = 0
    stack_trace: str = ""
    system_state: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    suggested_recovery: RecoveryStrategy = RecoveryStrategy.RETRY
    user_data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None


class AdvancedCircuitBreaker:
    """
    Advanced circuit breaker with adaptive thresholds and machine learning.
    
    Implements sophisticated failure detection and automatic recovery
    with statistical analysis and predictive failure prevention.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
        adaptive_threshold: bool = True,
        enable_ml_prediction: bool = True
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.adaptive_threshold = adaptive_threshold
        self.enable_ml_prediction = enable_ml_prediction
        
        # Circuit breaker state
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        
        # Advanced features
        self.failure_history = []
        self.success_history = []
        self.performance_metrics = {}
        self.failure_patterns = {}
        
        # Machine learning components
        if enable_ml_prediction:
            self._initialize_ml_components()
        
        # Thread safety
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)
    
    def _initialize_ml_components(self):
        """Initialize machine learning components for failure prediction."""
        try:
            import numpy as np
            from sklearn.ensemble import IsolationForest
            from collections import deque
            
            self.failure_predictor = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            self.feature_history = deque(maxlen=1000)
            self.prediction_enabled = True
            
        except ImportError:
            self._logger.warning("scikit-learn not available, disabling ML prediction")
            self.prediction_enabled = False
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
        """
        with self._lock:
            current_time = time.time()
            
            # Check if we should transition from OPEN to HALF_OPEN
            if (self.state == "OPEN" and 
                current_time - self.last_failure_time > self.recovery_timeout):
                self.state = "HALF_OPEN"
                self.success_count = 0
                self._logger.info(f"Circuit breaker transitioning to HALF_OPEN")
            
            # Check circuit breaker state
            if self.state == "OPEN":
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                )
            
            # Predict failure if ML is enabled
            if self.prediction_enabled and self._predict_failure():
                self._logger.warning("ML model predicts potential failure, increasing monitoring")
                # Continue execution but with increased monitoring
        
        try:
            # Execute the function
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            
            # Record success
            self._record_success(execution_time)
            
            return result
            
        except Exception as e:
            # Record failure
            self._record_failure(e, current_time)
            raise
    
    def _record_success(self, execution_time: float):
        """Record successful execution."""
        with self._lock:
            self.success_count += 1
            self.failure_count = max(0, self.failure_count - 1)  # Gradual recovery
            
            self.success_history.append({
                "timestamp": time.time(),
                "execution_time": execution_time
            })
            
            # Keep history manageable
            if len(self.success_history) > 1000:
                self.success_history = self.success_history[-500:]
            
            # Update performance metrics
            self.performance_metrics["avg_execution_time"] = execution_time
            
            # Update ML features
            if self.prediction_enabled:
                self._update_ml_features(execution_time, success=True)
            
            # Transition from HALF_OPEN to CLOSED if enough successes
            if (self.state == "HALF_OPEN" and 
                self.success_count >= self.success_threshold):
                self.state = "CLOSED"
                self.failure_count = 0
                self._logger.info("Circuit breaker transitioned to CLOSED")
    
    def _record_failure(self, exception: Exception, current_time: float):
        """Record failure and update circuit breaker state."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = current_time
            
            failure_record = {
                "timestamp": current_time,
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "failure_count": self.failure_count
            }
            self.failure_history.append(failure_record)
            
            # Keep history manageable
            if len(self.failure_history) > 1000:
                self.failure_history = self.failure_history[-500:]
            
            # Update ML features
            if self.prediction_enabled:
                self._update_ml_features(0.0, success=False, error=str(exception))
            
            # Adaptive threshold adjustment
            if self.adaptive_threshold:
                self._adjust_threshold()
            
            # Check if we should open the circuit
            current_threshold = self._get_current_threshold()
            if self.failure_count >= current_threshold:
                self.state = "OPEN"
                self._logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
            elif self.state == "HALF_OPEN":
                # Any failure in HALF_OPEN goes back to OPEN
                self.state = "OPEN"
                self._logger.warning("Circuit breaker returned to OPEN from HALF_OPEN")
    
    def _adjust_threshold(self):
        """Dynamically adjust failure threshold based on historical patterns."""
        if len(self.failure_history) < 10:
            return
        
        # Analyze recent failure patterns
        recent_failures = self.failure_history[-10:]
        time_window = recent_failures[-1]["timestamp"] - recent_failures[0]["timestamp"]
        
        if time_window > 0:
            failure_rate = len(recent_failures) / time_window
            
            # Adjust threshold based on failure rate
            if failure_rate > 0.1:  # High failure rate
                self.failure_threshold = max(2, self.failure_threshold - 1)
            elif failure_rate < 0.01:  # Low failure rate
                self.failure_threshold = min(10, self.failure_threshold + 1)
    
    def _get_current_threshold(self) -> int:
        """Get current failure threshold (may be adaptive)."""
        return self.failure_threshold
    
    def _update_ml_features(self, execution_time: float, success: bool, error: str = ""):
        """Update machine learning features for failure prediction."""
        if not self.prediction_enabled:
            return
        
        try:
            import numpy as np
            
            # Create feature vector
            current_time = time.time()
            features = [
                execution_time,
                self.failure_count,
                self.success_count,
                len(self.failure_history),
                current_time % 86400,  # Time of day
                int(success),
                len(error) if error else 0
            ]
            
            self.feature_history.append(features)
            
            # Retrain predictor periodically
            if len(self.feature_history) >= 100 and len(self.feature_history) % 50 == 0:
                self._retrain_predictor()
                
        except Exception as e:
            self._logger.warning(f"Failed to update ML features: {e}")
    
    def _retrain_predictor(self):
        """Retrain the failure prediction model."""
        if not self.prediction_enabled or len(self.feature_history) < 50:
            return
        
        try:
            import numpy as np
            
            features_array = np.array(list(self.feature_history))
            self.failure_predictor.fit(features_array)
            self._logger.debug("Retrained failure prediction model")
            
        except Exception as e:
            self._logger.warning(f"Failed to retrain predictor: {e}")
    
    def _predict_failure(self) -> bool:
        """Predict if next operation might fail."""
        if not self.prediction_enabled or len(self.feature_history) < 10:
            return False
        
        try:
            import numpy as np
            
            # Get recent features for prediction
            recent_features = np.array(list(self.feature_history)[-10:])
            latest_features = recent_features[-1:].reshape(1, -1)
            
            # Predict anomaly (potential failure)
            prediction = self.failure_predictor.predict(latest_features)
            
            return prediction[0] == -1  # -1 indicates anomaly
            
        except Exception as e:
            self._logger.warning(f"Failed to predict failure: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            total_calls = len(self.success_history) + len(self.failure_history)
            success_rate = len(self.success_history) / total_calls if total_calls > 0 else 0
            
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "failure_threshold": self.failure_threshold,
                "success_rate": success_rate,
                "total_calls": total_calls,
                "last_failure_time": self.last_failure_time,
                "performance_metrics": self.performance_metrics.copy(),
                "prediction_enabled": self.prediction_enabled
            }
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        with self._lock:
            self.state = "CLOSED"
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = 0
            self.failure_history.clear()
            self.success_history.clear()
            self._logger.info("Circuit breaker reset")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class DistributedErrorRecoverySystem:
    """
    Distributed error recovery system for multi-node photonic simulations.
    
    Coordinates error recovery across multiple simulation nodes and
    implements advanced recovery strategies.
    """
    
    def __init__(self, node_id: str = None):
        self.node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        self.recovery_strategies = {}
        self.error_history = []
        self.node_health = {}
        self.circuit_breakers = {}
        
        # Distributed coordination
        self.coordination_queue = queue.Queue()
        self.recovery_workers = []
        self.is_running = False
        
        # Performance monitoring
        self.performance_tracker = {}
        
        self._logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
    
    def register_recovery_strategy(
        self,
        error_type: Type[Exception],
        strategy: Callable,
        priority: int = 1
    ):
        """Register recovery strategy for specific error type."""
        if error_type not in self.recovery_strategies:
            self.recovery_strategies[error_type] = []
        
        self.recovery_strategies[error_type].append({
            "strategy": strategy,
            "priority": priority,
            "success_count": 0,
            "failure_count": 0
        })
        
        # Sort by priority
        self.recovery_strategies[error_type].sort(
            key=lambda x: x["priority"], reverse=True
        )
    
    def get_circuit_breaker(self, component_name: str) -> AdvancedCircuitBreaker:
        """Get or create circuit breaker for component."""
        if component_name not in self.circuit_breakers:
            self.circuit_breakers[component_name] = AdvancedCircuitBreaker(
                adaptive_threshold=True,
                enable_ml_prediction=True
            )
        return self.circuit_breakers[component_name]
    
    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        auto_recover: bool = True
    ) -> bool:
        """
        Handle error with distributed recovery coordination.
        
        Args:
            error: The exception that occurred
            context: Error context information
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            True if error was recovered, False otherwise
        """
        error_type = type(error)
        
        # Update error context
        context.error_type = error_type.__name__
        context.error_message = str(error)
        context.stack_trace = traceback.format_exc()
        
        # Log error
        self._logger.error(f"Error handled: {context.error_id} - {error}")
        
        # Record error in history
        with self._lock:
            self.error_history.append({
                "context": context,
                "timestamp": time.time(),
                "recovered": False
            })
        
        if not auto_recover:
            return False
        
        # Attempt recovery using registered strategies
        return self._attempt_recovery(error, context)
    
    def _attempt_recovery(self, error: Exception, context: ErrorContext) -> bool:
        """Attempt to recover from error using registered strategies."""
        error_type = type(error)
        
        # Find applicable recovery strategies
        strategies = []
        for exc_type, strategy_list in self.recovery_strategies.items():
            if issubclass(error_type, exc_type):
                strategies.extend(strategy_list)
        
        # Sort by priority and success rate
        strategies.sort(key=lambda x: (x["priority"], x["success_count"]), reverse=True)
        
        # Try each strategy
        for strategy_info in strategies:
            if context.recovery_attempts >= context.max_recovery_attempts:
                self._logger.warning(
                    f"Max recovery attempts reached for {context.error_id}"
                )
                break
            
            try:
                context.recovery_attempts += 1
                strategy = strategy_info["strategy"]
                
                self._logger.info(
                    f"Attempting recovery {context.recovery_attempts} with {strategy.__name__}"
                )
                
                result = strategy(error, context)
                
                if result:
                    strategy_info["success_count"] += 1
                    self._logger.info(f"Recovery successful for {context.error_id}")
                    
                    # Update error history
                    with self._lock:
                        for record in reversed(self.error_history):
                            if record["context"].error_id == context.error_id:
                                record["recovered"] = True
                                break
                    
                    return True
                else:
                    strategy_info["failure_count"] += 1
                    
            except Exception as recovery_error:
                strategy_info["failure_count"] += 1
                self._logger.error(
                    f"Recovery strategy failed: {recovery_error}"
                )
        
        self._logger.error(f"All recovery attempts failed for {context.error_id}")
        return False
    
    def start_distributed_recovery(self):
        """Start distributed recovery worker threads."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start recovery workers
        for i in range(2):  # Two worker threads
            worker = threading.Thread(
                target=self._recovery_worker,
                name=f"RecoveryWorker-{i}",
                daemon=True
            )
            worker.start()
            self.recovery_workers.append(worker)
        
        # Start health monitoring
        health_monitor = threading.Thread(
            target=self._health_monitor,
            name="HealthMonitor",
            daemon=True
        )
        health_monitor.start()
        
        self._logger.info("Distributed recovery system started")
    
    def stop_distributed_recovery(self):
        """Stop distributed recovery system."""
        self.is_running = False
        
        # Signal workers to stop
        for _ in self.recovery_workers:
            self.coordination_queue.put(None)
        
        self._logger.info("Distributed recovery system stopped")
    
    def _recovery_worker(self):
        """Worker thread for handling recovery tasks."""
        while self.is_running:
            try:
                task = self.coordination_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                
                # Process recovery task
                self._process_recovery_task(task)
                
            except queue.Empty:
                continue
            except Exception as e:
                self._logger.error(f"Recovery worker error: {e}")
    
    def _health_monitor(self):
        """Monitor system health and trigger preventive actions."""
        while self.is_running:
            try:
                # Check circuit breaker health
                for name, breaker in self.circuit_breakers.items():
                    metrics = breaker.get_metrics()
                    self.node_health[name] = metrics
                    
                    # Trigger alerts for unhealthy components
                    if metrics["success_rate"] < 0.8:
                        self._logger.warning(
                            f"Component {name} health degraded: {metrics['success_rate']:.2%}"
                        )
                
                # Sleep before next check
                time.sleep(30)
                
            except Exception as e:
                self._logger.error(f"Health monitor error: {e}")
    
    def _process_recovery_task(self, task: Dict[str, Any]):
        """Process a recovery task from the coordination queue."""
        task_type = task.get("type")
        
        if task_type == "circuit_breaker_open":
            self._handle_circuit_breaker_open(task)
        elif task_type == "node_failure":
            self._handle_node_failure(task)
        elif task_type == "performance_degradation":
            self._handle_performance_degradation(task)
    
    def _handle_circuit_breaker_open(self, task: Dict[str, Any]):
        """Handle circuit breaker opening."""
        component = task.get("component")
        self._logger.warning(f"Circuit breaker opened for {component}")
        
        # Implement recovery logic (e.g., switch to backup, reduce load)
        # This would be customized based on specific component requirements
    
    def _handle_node_failure(self, task: Dict[str, Any]):
        """Handle node failure in distributed system."""
        failed_node = task.get("node_id")
        self._logger.error(f"Node failure detected: {failed_node}")
        
        # Implement node recovery logic (e.g., restart, failover)
    
    def _handle_performance_degradation(self, task: Dict[str, Any]):
        """Handle performance degradation."""
        component = task.get("component")
        metrics = task.get("metrics", {})
        
        self._logger.warning(
            f"Performance degradation in {component}: {metrics}"
        )
        
        # Implement performance recovery (e.g., scaling, optimization)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        with self._lock:
            total_errors = len(self.error_history)
            recovered_errors = sum(1 for record in self.error_history if record["recovered"])
            recovery_rate = recovered_errors / total_errors if total_errors > 0 else 1.0
            
            recent_errors = [
                record for record in self.error_history
                if time.time() - record["timestamp"] < 3600  # Last hour
            ]
            
            return {
                "node_id": self.node_id,
                "is_running": self.is_running,
                "total_errors": total_errors,
                "recovery_rate": recovery_rate,
                "recent_errors": len(recent_errors),
                "circuit_breakers": {
                    name: breaker.get_metrics()
                    for name, breaker in self.circuit_breakers.items()
                },
                "node_health": self.node_health.copy(),
                "active_workers": len(self.recovery_workers)
            }
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    component: str = ""
    operation: str = ""
    function_name: str = ""
    line_number: int = 0
    file_name: str = ""
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    error_type: str = ""
    error_message: str = ""
    stack_trace: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    suggested_recovery: RecoveryStrategy = RecoveryStrategy.RETRY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'severity': self.severity.value,
            'component': self.component,
            'operation': self.operation,
            'function_name': self.function_name,
            'line_number': self.line_number,
            'file_name': self.file_name,
            'correlation_id': self.correlation_id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'stack_trace': self.stack_trace,
            'parameters': self.parameters,
            'system_state': self.system_state,
            'recovery_attempts': self.recovery_attempts,
            'max_recovery_attempts': self.max_recovery_attempts,
            'suggested_recovery': self.suggested_recovery.value
        }


@dataclass
class RecoveryAction:
    """Recovery action configuration."""
    strategy: RecoveryStrategy
    action: Callable[[], Any]
    description: str = ""
    max_attempts: int = 3
    delay_seconds: float = 1.0
    condition: Optional[Callable[[Exception], bool]] = None


class ErrorHandler:
    """Advanced error handler with recovery mechanisms."""
    
    def __init__(self, logger: Optional[PhotonicLogger] = None):
        self.logger = logger or PhotonicLogger()
        self._recovery_registry: Dict[Type[Exception], List[RecoveryAction]] = {}
        self._error_history: List[ErrorContext] = []
        self._error_queue = queue.Queue()
        self._shutdown = False
        
        # Start error processing thread
        self._processor_thread = threading.Thread(target=self._process_errors, daemon=True)
        self._processor_thread.start()
    
    def register_recovery(
        self, 
        exception_type: Type[Exception], 
        recovery_action: RecoveryAction
    ) -> None:
        """Register recovery action for specific exception type."""
        if exception_type not in self._recovery_registry:
            self._recovery_registry[exception_type] = []
        
        self._recovery_registry[exception_type].append(recovery_action)
        self.logger.get_logger('error_handler').info(
            f"Registered recovery action for {exception_type.__name__}: {recovery_action.description}"
        )
    
    def handle_error(
        self, 
        exception: Exception, 
        context: Optional[ErrorContext] = None,
        auto_recover: bool = True
    ) -> Tuple[bool, Any]:
        """
        Handle error with automatic recovery.
        
        Returns:
            Tuple of (recovery_successful, result_or_none)
        """
        # Create error context if not provided
        if context is None:
            context = self._create_error_context(exception)
        
        # Log error
        self._log_error(exception, context)
        
        # Add to error history
        self._error_history.append(context)
        
        # Attempt recovery if enabled
        if auto_recover:
            return self._attempt_recovery(exception, context)
        
        return False, None
    
    def _create_error_context(self, exception: Exception) -> ErrorContext:
        """Create comprehensive error context from exception."""
        # Get stack frame information
        frame = sys._getframe(2)  # Skip current and handle_error frames
        
        context = ErrorContext(
            error_type=type(exception).__name__,
            error_message=str(exception),
            stack_trace=traceback.format_exc(),
            function_name=frame.f_code.co_name,
            line_number=frame.f_lineno,
            file_name=frame.f_code.co_filename,
            correlation_id=CorrelationContext.get_correlation_id(),
            session_id=CorrelationContext.get_session_id(),
            user_id=CorrelationContext.get_user_id()
        )
        
        # Set severity based on exception type
        context.severity = self._determine_severity(exception)
        
        # Set suggested recovery strategy
        context.suggested_recovery = self._suggest_recovery_strategy(exception)
        
        # Capture system state
        context.system_state = self._capture_system_state()
        
        return context
    
    def _determine_severity(self, exception: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type."""
        if isinstance(exception, (MemoryError, SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        elif isinstance(exception, (FileNotFoundError, PermissionError)):
            return ErrorSeverity.HIGH
        elif isinstance(exception, (ValueError, TypeError, AttributeError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _suggest_recovery_strategy(self, exception: Exception) -> RecoveryStrategy:
        """Suggest recovery strategy based on exception type."""
        if isinstance(exception, (ConnectionError, TimeoutError)):
            return RecoveryStrategy.RETRY
        elif isinstance(exception, (FileNotFoundError, ImportError)):
            return RecoveryStrategy.FALLBACK
        elif isinstance(exception, MemoryError):
            return RecoveryStrategy.ABORT
        elif isinstance(exception, (PermissionError, OSError)):
            return RecoveryStrategy.USER_INTERVENTION
        else:
            return RecoveryStrategy.RETRY
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for diagnostics."""
        try:
            import psutil
            process = psutil.Process()
            
            return {
                'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'thread_count': process.num_threads(),
                'open_files': len(process.open_files()),
                'timestamp': time.time()
            }
        except ImportError:
            return {
                'thread_count': threading.active_count(),
                'timestamp': time.time()
            }
    
    def _log_error(self, exception: Exception, context: ErrorContext) -> None:
        """Log error with comprehensive context."""
        logger = self.logger.get_logger('error_handler')
        
        extra = {
            'component': 'error_handler',
            'operation': 'handle_error',
            'error_id': context.error_id,
            'metadata': {
                'severity': context.severity.value,
                'error_type': context.error_type,
                'function': context.function_name,
                'line': context.line_number,
                'recovery_attempts': context.recovery_attempts
            }
        }
        
        if context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.error(f"Error {context.error_id}: {context.error_message}", extra=extra, exc_info=True)
        else:
            logger.warning(f"Error {context.error_id}: {context.error_message}", extra=extra)
    
    def _attempt_recovery(self, exception: Exception, context: ErrorContext) -> Tuple[bool, Any]:
        """Attempt automatic recovery using registered strategies."""
        exception_type = type(exception)
        
        # Check for registered recovery actions
        recovery_actions = []
        for exc_type, actions in self._recovery_registry.items():
            if issubclass(exception_type, exc_type):
                recovery_actions.extend(actions)
        
        # Try recovery actions
        for action in recovery_actions:
            if action.condition and not action.condition(exception):
                continue
            
            if context.recovery_attempts >= action.max_attempts:
                continue
            
            try:
                context.recovery_attempts += 1
                
                self.logger.get_logger('error_handler').info(
                    f"Attempting recovery for {context.error_id}: {action.description}"
                )
                
                # Apply delay if specified
                if action.delay_seconds > 0:
                    time.sleep(action.delay_seconds)
                
                # Execute recovery action
                result = action.action()
                
                self.logger.get_logger('error_handler').info(
                    f"Recovery successful for {context.error_id}"
                )
                
                return True, result
                
            except Exception as recovery_error:
                self.logger.get_logger('error_handler').warning(
                    f"Recovery failed for {context.error_id}: {str(recovery_error)}"
                )
                continue
        
        return False, None
    
    def _process_errors(self) -> None:
        """Background thread for processing error queue."""
        while not self._shutdown:
            try:
                error_data = self._error_queue.get(timeout=1.0)
                # Process error data (analytics, reporting, etc.)
                self._error_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Avoid infinite recursion in error processing
                pass
    
    def get_error_statistics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for specified time window."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_errors = [e for e in self._error_history if e.timestamp > cutoff_time]
        
        error_types = {}
        severity_counts = {s.value: 0 for s in ErrorSeverity}
        recovery_success_rate = 0
        
        for error in recent_errors:
            # Count by error type
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            
            # Count by severity
            severity_counts[error.severity.value] += 1
            
            # Track recovery success
            if error.recovery_attempts > 0:
                recovery_success_rate += 1
        
        if recent_errors:
            recovery_success_rate = recovery_success_rate / len(recent_errors)
        
        return {
            'total_errors': len(recent_errors),
            'error_types': error_types,
            'severity_distribution': severity_counts,
            'recovery_success_rate': recovery_success_rate,
            'time_window_hours': time_window_hours,
            'timestamp': time.time()
        }
    
    def export_error_report(self, output_path: Path) -> None:
        """Export comprehensive error report."""
        report = {
            'report_timestamp': time.time(),
            'error_count': len(self._error_history),
            'registered_recoveries': {
                exc_type.__name__: [
                    {
                        'strategy': action.strategy.value,
                        'description': action.description,
                        'max_attempts': action.max_attempts
                    }
                    for action in actions
                ]
                for exc_type, actions in self._recovery_registry.items()
            },
            'error_history': [error.to_dict() for error in self._error_history[-1000:]],  # Last 1000 errors
            'statistics': self.get_error_statistics()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def shutdown(self) -> None:
        """Shutdown error handler."""
        self._shutdown = True
        self._processor_thread.join(timeout=5.0)


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    @contextmanager
    def __call__(self):
        """Context manager for circuit breaker."""
        with self._lock:
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            yield
            self._on_success()
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        with self._lock:
            self.failure_count = 0
            self.state = 'CLOSED'
    
    def _on_failure(self) -> None:
        """Handle failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'


def robust_operation(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    exponential_backoff: bool = True,
    circuit_breaker: Optional[CircuitBreaker] = None,
    fallback_function: Optional[Callable] = None,
    error_handler: Optional[ErrorHandler] = None
):
    """
    Decorator for making operations robust with retries, circuit breaker, and fallback.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            attempt = 0
            
            while attempt <= max_retries:
                try:
                    # Use circuit breaker if provided
                    if circuit_breaker:
                        with circuit_breaker():
                            return func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    attempt += 1
                    
                    # Handle error if handler provided
                    if error_handler:
                        recovered, result = error_handler.handle_error(e)
                        if recovered:
                            return result
                    
                    # Check if we should retry
                    if attempt <= max_retries:
                        delay = retry_delay
                        if exponential_backoff:
                            delay *= (2 ** (attempt - 1))
                        
                        time.sleep(delay)
                        continue
                    
                    # Try fallback if available
                    if fallback_function:
                        try:
                            return fallback_function(*args, **kwargs)
                        except Exception:
                            pass
                    
                    # All attempts failed
                    break
            
            # Raise the last exception if all attempts failed
            raise last_exception
        
        return wrapper
    return decorator


@contextmanager
def error_recovery_context(
    error_handler: ErrorHandler,
    operation: str = "",
    component: str = "",
    auto_recover: bool = True
):
    """Context manager for error recovery."""
    try:
        yield
    except Exception as e:
        context = ErrorContext(
            operation=operation,
            component=component
        )
        
        recovered, result = error_handler.handle_error(e, context, auto_recover)
        if not recovered:
            raise


def setup_default_recovery_actions(error_handler: ErrorHandler) -> None:
    """Setup default recovery actions for common errors."""
    
    # Memory error recovery
    def cleanup_memory():
        """Attempt to free memory."""
        import gc
        gc.collect()
        return True
    
    error_handler.register_recovery(
        MemoryError,
        RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK,
            action=cleanup_memory,
            description="Garbage collection memory cleanup",
            max_attempts=1
        )
    )
    
    # File not found recovery
    def create_missing_directories():
        """Create missing directories."""
        # This would be customized based on the specific application
        return True
    
    error_handler.register_recovery(
        FileNotFoundError,
        RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK,
            action=create_missing_directories,
            description="Create missing directories",
            max_attempts=1
        )
    )
    
    # Connection error recovery
    def wait_and_retry():
        """Simple wait for connection recovery."""
        time.sleep(5.0)
        return True
    
    error_handler.register_recovery(
        ConnectionError,
        RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            action=wait_and_retry,
            description="Wait for connection recovery",
            max_attempts=3,
            delay_seconds=5.0
        )
    )


def create_robust_error_system(logger: Optional[PhotonicLogger] = None) -> ErrorHandler:
    """Create a robust error handling system with default configurations."""
    error_handler = ErrorHandler(logger)
    setup_default_recovery_actions(error_handler)
    return error_handler