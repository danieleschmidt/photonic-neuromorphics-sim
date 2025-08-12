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