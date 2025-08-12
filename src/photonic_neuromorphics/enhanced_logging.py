"""
Enhanced logging system for photonic neuromorphics simulation framework.

Provides structured logging, performance tracking, error correlation,
and comprehensive observability for photonic simulations.
"""

import logging
import logging.handlers
import json
import time
import uuid
import threading
import queue
import traceback
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from contextlib import contextmanager
from functools import wraps
import sys
import os

from .monitoring import MetricsCollector


@dataclass
class LogEvent:
    """Structured log event."""
    timestamp: float = field(default_factory=time.time)
    level: str = "INFO"
    message: str = ""
    logger_name: str = ""
    module: str = ""
    function: str = ""
    line_number: int = 0
    thread_id: str = field(default_factory=lambda: str(threading.get_ident()))
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    component: str = ""
    operation: str = ""
    duration_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class CorrelationContext:
    """Thread-local correlation context for request tracing."""
    
    _local = threading.local()
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        """Set correlation ID for current thread."""
        cls._local.correlation_id = correlation_id
    
    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get correlation ID for current thread."""
        return getattr(cls._local, 'correlation_id', None)
    
    @classmethod
    def set_session_id(cls, session_id: str) -> None:
        """Set session ID for current thread."""
        cls._local.session_id = session_id
    
    @classmethod
    def get_session_id(cls) -> Optional[str]:
        """Get session ID for current thread."""
        return getattr(cls._local, 'session_id', None)
    
    @classmethod
    def set_user_id(cls, user_id: str) -> None:
        """Set user ID for current thread."""
        cls._local.user_id = user_id
    
    @classmethod
    def get_user_id(cls) -> Optional[str]:
        """Get user ID for current thread."""
        return getattr(cls._local, 'user_id', None)
    
    @classmethod
    def clear(cls) -> None:
        """Clear all context for current thread."""
        for attr in ['correlation_id', 'session_id', 'user_id']:
            if hasattr(cls._local, attr):
                delattr(cls._local, attr)


class StructuredFormatter(logging.Formatter):
    """Formatter for structured logging output."""
    
    def __init__(self, include_stack_trace: bool = True):
        super().__init__()
        self.include_stack_trace = include_stack_trace
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Extract caller information
        frame = sys._getframe()
        while frame:
            code = frame.f_code
            if not code.co_filename.endswith('logging/__init__.py'):
                break
            frame = frame.f_back
        
        # Create structured event
        event = LogEvent(
            timestamp=record.created,
            level=record.levelname,
            message=record.getMessage(),
            logger_name=record.name,
            module=record.module if hasattr(record, 'module') else '',
            function=record.funcName,
            line_number=record.lineno,
            correlation_id=CorrelationContext.get_correlation_id(),
            session_id=CorrelationContext.get_session_id(),
            user_id=CorrelationContext.get_user_id(),
            component=getattr(record, 'component', ''),
            operation=getattr(record, 'operation', ''),
            duration_ms=getattr(record, 'duration_ms', None),
            memory_mb=getattr(record, 'memory_mb', None),
            metadata=getattr(record, 'metadata', {})
        )
        
        # Add stack trace for errors
        if record.exc_info and self.include_stack_trace:
            event.stack_trace = self.formatException(record.exc_info)
        
        return event.to_json()


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler for high-performance logging."""
    
    def __init__(self, target_handler: logging.Handler, queue_size: int = 10000):
        super().__init__()
        self.target_handler = target_handler
        self.log_queue = queue.Queue(maxsize=queue_size)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self._shutdown = False
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record asynchronously."""
        try:
            self.log_queue.put_nowait(record)
        except queue.Full:
            # Drop log if queue is full to prevent blocking
            pass
    
    def _worker(self) -> None:
        """Worker thread for processing log records."""
        while not self._shutdown:
            try:
                record = self.log_queue.get(timeout=1.0)
                self.target_handler.emit(record)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                # Avoid infinite recursion in error handling
                pass
    
    def close(self) -> None:
        """Close handler and stop worker thread."""
        self._shutdown = True
        self.worker_thread.join(timeout=5.0)
        self.target_handler.close()
        super().close()


class PerformanceTracker:
    """Track performance metrics and log automatically."""
    
    def __init__(self, logger: logging.Logger, metrics_collector: Optional[MetricsCollector] = None):
        self.logger = logger
        self.metrics_collector = metrics_collector
        self._start_times: Dict[str, float] = {}
        self._start_memory: Dict[str, float] = {}
    
    @contextmanager
    def track_operation(self, operation: str, component: str = "", metadata: Dict[str, Any] = None):
        """Context manager for tracking operation performance."""
        operation_id = f"{operation}_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Log operation start
        self.logger.info(
            f"Starting operation: {operation}",
            extra={
                'component': component,
                'operation': operation,
                'operation_id': operation_id,
                'metadata': metadata or {}
            }
        )
        
        try:
            yield operation_id
            
            # Log successful completion
            duration_ms = (time.time() - start_time) * 1000
            memory_delta = self._get_memory_usage() - start_memory
            
            self.logger.info(
                f"Completed operation: {operation}",
                extra={
                    'component': component,
                    'operation': operation,
                    'operation_id': operation_id,
                    'duration_ms': duration_ms,
                    'memory_mb': memory_delta,
                    'metadata': metadata or {}
                }
            )
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_operation_time(operation, duration_ms / 1000)
                self.metrics_collector.record_memory_usage(memory_delta)
        
        except Exception as e:
            # Log operation failure
            duration_ms = (time.time() - start_time) * 1000
            memory_delta = self._get_memory_usage() - start_memory
            
            self.logger.error(
                f"Failed operation: {operation} - {str(e)}",
                extra={
                    'component': component,
                    'operation': operation,
                    'operation_id': operation_id,
                    'duration_ms': duration_ms,
                    'memory_mb': memory_delta,
                    'metadata': metadata or {},
                    'error': str(e),
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            
            # Record error metrics
            if self.metrics_collector:
                self.metrics_collector.record_error(operation, str(e))
            
            raise
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0


class LogAnalyzer:
    """Analyze logs for patterns, errors, and performance issues."""
    
    def __init__(self, log_file_path: Path):
        self.log_file_path = log_file_path
    
    def analyze_errors(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze error patterns in logs."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        error_patterns = {}
        total_errors = 0
        
        try:
            with open(self.log_file_path, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get('timestamp', 0) < cutoff_time:
                            continue
                        
                        if event.get('level') in ['ERROR', 'CRITICAL']:
                            total_errors += 1
                            
                            # Group by error type and operation
                            error_key = f"{event.get('operation', 'unknown')}:{event.get('metadata', {}).get('error_type', 'unknown')}"
                            
                            if error_key not in error_patterns:
                                error_patterns[error_key] = {
                                    'count': 0,
                                    'first_seen': event.get('timestamp'),
                                    'last_seen': event.get('timestamp'),
                                    'messages': []
                                }
                            
                            pattern = error_patterns[error_key]
                            pattern['count'] += 1
                            pattern['last_seen'] = max(pattern['last_seen'], event.get('timestamp', 0))
                            
                            if len(pattern['messages']) < 5:  # Keep sample messages
                                pattern['messages'].append(event.get('message', ''))
                    
                    except json.JSONDecodeError:
                        continue
        
        except FileNotFoundError:
            return {'error': 'Log file not found'}
        
        return {
            'total_errors': total_errors,
            'error_patterns': error_patterns,
            'analysis_window_hours': time_window_hours,
            'analysis_timestamp': time.time()
        }
    
    def analyze_performance(self, operation: str = None) -> Dict[str, Any]:
        """Analyze performance patterns for operations."""
        durations = []
        memory_usage = []
        operation_counts = {}
        
        try:
            with open(self.log_file_path, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        
                        if operation and event.get('operation') != operation:
                            continue
                        
                        event_operation = event.get('operation', 'unknown')
                        operation_counts[event_operation] = operation_counts.get(event_operation, 0) + 1
                        
                        if event.get('duration_ms'):
                            durations.append(event['duration_ms'])
                        
                        if event.get('memory_mb'):
                            memory_usage.append(event['memory_mb'])
                    
                    except json.JSONDecodeError:
                        continue
        
        except FileNotFoundError:
            return {'error': 'Log file not found'}
        
        # Calculate statistics
        stats = {}
        if durations:
            durations.sort()
            stats['duration_stats'] = {
                'count': len(durations),
                'min_ms': min(durations),
                'max_ms': max(durations),
                'avg_ms': sum(durations) / len(durations),
                'p50_ms': durations[len(durations) // 2],
                'p95_ms': durations[int(len(durations) * 0.95)],
                'p99_ms': durations[int(len(durations) * 0.99)]
            }
        
        if memory_usage:
            stats['memory_stats'] = {
                'count': len(memory_usage),
                'min_mb': min(memory_usage),
                'max_mb': max(memory_usage),
                'avg_mb': sum(memory_usage) / len(memory_usage)
            }
        
        return {
            'operation_counts': operation_counts,
            'performance_stats': stats,
            'analysis_timestamp': time.time()
        }


def logged_operation(operation: str, component: str = "", include_args: bool = False):
    """Decorator to automatically log function operations."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger from first argument if it has one
            logger = logging.getLogger(func.__module__)
            if args and hasattr(args[0], '_logger'):
                logger = args[0]._logger
            
            metadata = {'function': func.__name__}
            if include_args:
                metadata['args'] = str(args)
                metadata['kwargs'] = str(kwargs)
            
            # Use performance tracker if available
            tracker = None
            if args and hasattr(args[0], 'performance_tracker'):
                tracker = args[0].performance_tracker
            else:
                tracker = PerformanceTracker(logger)
            
            with tracker.track_operation(operation, component, metadata):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class PhotonicLogger:
    """Main logger class for photonic neuromorphics simulations."""
    
    def __init__(
        self,
        name: str = "photonic_neuromorphics",
        log_level: str = "INFO",
        log_dir: Path = Path("logs"),
        enable_console: bool = True,
        enable_file: bool = True,
        enable_async: bool = True,
        max_file_size: int = 100 * 1024 * 1024,  # 100 MB
        backup_count: int = 5,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.name = name
        self.log_dir = log_dir
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create log directory
        log_dir.mkdir(exist_ok=True)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup formatters
        self.structured_formatter = StructuredFormatter()
        self.console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup handlers
        if enable_console:
            self._setup_console_handler()
        
        if enable_file:
            self._setup_file_handlers(max_file_size, backup_count, enable_async)
        
        # Setup performance tracking
        self.performance_tracker = PerformanceTracker(self.logger, metrics_collector)
        
        # Setup log analyzer
        self.analyzer = LogAnalyzer(log_dir / f"{name}.log")
        
        self.logger.info("Photonic logger initialized", extra={'component': 'logging'})
    
    def _setup_console_handler(self) -> None:
        """Setup console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(self, max_size: int, backup_count: int, enable_async: bool) -> None:
        """Setup file logging handlers."""
        # Main application log
        app_log_path = self.log_dir / f"{self.name}.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_path, maxBytes=max_size, backupCount=backup_count
        )
        app_handler.setFormatter(self.structured_formatter)
        
        # Error log
        error_log_path = self.log_dir / f"{self.name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_path, maxBytes=max_size, backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.structured_formatter)
        
        # Wrap in async handlers if enabled
        if enable_async:
            app_handler = AsyncLogHandler(app_handler)
            error_handler = AsyncLogHandler(error_handler)
        
        self.logger.addHandler(app_handler)
        self.logger.addHandler(error_handler)
    
    def get_logger(self, component: str = "") -> logging.Logger:
        """Get logger with component context."""
        if component:
            logger_name = f"{self.name}.{component}"
            logger = logging.getLogger(logger_name)
            logger.parent = self.logger
            return logger
        return self.logger
    
    @contextmanager
    def correlation_context(self, correlation_id: str = None, session_id: str = None, user_id: str = None):
        """Context manager for correlation tracking."""
        # Set context
        if correlation_id:
            CorrelationContext.set_correlation_id(correlation_id)
        if session_id:
            CorrelationContext.set_session_id(session_id)
        if user_id:
            CorrelationContext.set_user_id(user_id)
        
        try:
            yield
        finally:
            # Clear context
            CorrelationContext.clear()
    
    def log_simulation_start(self, simulation_id: str, parameters: Dict[str, Any]) -> None:
        """Log simulation start with structured data."""
        self.logger.info(
            f"Starting simulation {simulation_id}",
            extra={
                'component': 'simulation',
                'operation': 'start',
                'simulation_id': simulation_id,
                'metadata': {
                    'parameter_count': len(parameters),
                    'simulation_type': parameters.get('type', 'unknown')
                }
            }
        )
    
    def log_simulation_end(self, simulation_id: str, success: bool, results: Dict[str, Any] = None) -> None:
        """Log simulation completion."""
        level = logging.INFO if success else logging.ERROR
        status = "completed" if success else "failed"
        
        metadata = {'simulation_id': simulation_id, 'success': success}
        if results:
            metadata['result_keys'] = list(results.keys())
        
        self.logger.log(
            level,
            f"Simulation {simulation_id} {status}",
            extra={
                'component': 'simulation',
                'operation': 'end',
                'metadata': metadata
            }
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive logging report."""
        error_analysis = self.analyzer.analyze_errors()
        performance_analysis = self.analyzer.analyze_performance()
        
        return {
            'logger_name': self.name,
            'log_directory': str(self.log_dir),
            'error_analysis': error_analysis,
            'performance_analysis': performance_analysis,
            'report_timestamp': time.time()
        }


def setup_photonic_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_async: bool = True,
    metrics_collector: Optional[MetricsCollector] = None
) -> PhotonicLogger:
    """Setup photonic neuromorphics logging system."""
    return PhotonicLogger(
        log_level=log_level,
        log_dir=Path(log_dir),
        enable_async=enable_async,
        metrics_collector=metrics_collector
    )