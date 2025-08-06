"""
Comprehensive monitoring and observability for photonic neuromorphics simulation.

Provides real-time monitoring, metrics collection, performance tracking,
and health diagnostics for photonic neural network simulations and RTL generation.
"""

import time
import threading
import logging
import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path
import psutil
import sys

from .exceptions import ResourceExhaustionError, PhotonicNeuromorphicsException


@dataclass
class MetricData:
    """Container for metric data points."""
    name: str
    value: Union[float, int]
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""


@dataclass
class HealthStatus:
    """System health status information."""
    status: str  # "healthy", "warning", "critical"
    timestamp: float
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)


class MetricsCollector:
    """
    Comprehensive metrics collection system for photonic simulations.
    
    Collects performance, resource usage, and custom application metrics
    with configurable sampling rates and storage backends.
    """
    
    def __init__(
        self,
        max_history: int = 10000,
        collection_interval: float = 1.0,
        enable_system_metrics: bool = True
    ):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of historical data points to retain
            collection_interval: Metric collection interval in seconds
            enable_system_metrics: Enable automatic system resource monitoring
        """
        self.max_history = max_history
        self.collection_interval = collection_interval
        self.enable_system_metrics = enable_system_metrics
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        
        # Collection control
        self._collecting = False
        self._collection_thread = None
        self._lock = threading.Lock()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Metric callbacks
        self.metric_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Start collection if system metrics enabled
        if enable_system_metrics:
            self.start_collection()
    
    def record_metric(
        self,
        name: str,
        value: Union[float, int],
        tags: Optional[Dict[str, str]] = None,
        unit: str = "",
        description: str = ""
    ):
        """
        Record a metric data point.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for grouping/filtering
            unit: Unit of measurement
            description: Metric description
        """
        with self._lock:
            metric = MetricData(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                unit=unit,
                description=description
            )
            
            self.metrics[name].append(metric)
            
            # Update gauge if it's a gauge-type metric
            if name.startswith("gauge_"):
                self.gauges[name] = value
            
            # Trigger callbacks
            for callback in self.metric_callbacks[name]:
                try:
                    callback(metric)
                except Exception as e:
                    self.logger.warning(f"Metric callback failed for {name}: {e}")
    
    def increment_counter(self, name: str, delta: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            self.counters[name] += delta
            self.record_metric(name, self.counters[name], tags, "count")
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        gauge_name = f"gauge_{name}"
        with self._lock:
            self.gauges[gauge_name] = value
            self.record_metric(gauge_name, value, tags, "gauge")
    
    def time_operation(self, name: str):
        """Context manager for timing operations."""
        return TimingContext(self, name)
    
    def get_metric_history(self, name: str, last_n: Optional[int] = None) -> List[MetricData]:
        """Get metric history."""
        with self._lock:
            history = list(self.metrics[name])
            if last_n:
                history = history[-last_n:]
            return history
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values."""
        with self._lock:
            current = {}
            
            # Get latest values for each metric
            for name, history in self.metrics.items():
                if history:
                    current[name] = history[-1].value
            
            # Add counters and gauges
            current.update(self.counters)
            current.update(self.gauges)
            
            return current
    
    def start_collection(self):
        """Start automatic metric collection."""
        if self._collecting:
            return
        
        self._collecting = True
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self._collection_thread.start()
        self.logger.info("Started automatic metric collection")
    
    def stop_collection(self):
        """Stop automatic metric collection."""
        self._collecting = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        self.logger.info("Stopped automatic metric collection")
    
    def _collection_loop(self):
        """Main collection loop for system metrics."""
        while self._collecting:
            try:
                if self.enable_system_metrics:
                    self._collect_system_metrics()
                
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error in metric collection: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.set_gauge("system_cpu_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.set_gauge("system_memory_percent", memory.percent)
            self.set_gauge("system_memory_available_gb", memory.available / 1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.set_gauge("system_disk_percent", disk.percent)
            self.set_gauge("system_disk_free_gb", disk.free / 1024**3)
            
            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                self.set_gauge("system_network_bytes_sent", network.bytes_sent)
                self.set_gauge("system_network_bytes_recv", network.bytes_recv)
            except Exception:
                pass  # Network stats may not be available
            
            # Process metrics
            process = psutil.Process(os.getpid())
            self.set_gauge("process_memory_mb", process.memory_info().rss / 1024**2)
            self.set_gauge("process_cpu_percent", process.cpu_percent())
            
        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {e}")
    
    def add_metric_callback(self, metric_name: str, callback: Callable[[MetricData], None]):
        """Add callback for specific metric updates."""
        self.metric_callbacks[metric_name].append(callback)
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        if format == "json":
            return self._export_json()
        elif format == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self) -> str:
        """Export metrics as JSON."""
        export_data = {
            "timestamp": time.time(),
            "metrics": {},
            "counters": dict(self.counters),
            "gauges": dict(self.gauges)
        }
        
        # Export recent metric history
        for name, history in self.metrics.items():
            if history:
                export_data["metrics"][name] = [
                    {
                        "value": m.value,
                        "timestamp": m.timestamp,
                        "tags": m.tags,
                        "unit": m.unit
                    }
                    for m in list(history)[-100:]  # Last 100 points
                ]
        
        return json.dumps(export_data, indent=2)
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Export counters
        for name, value in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        
        # Export gauges
        for name, value in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        return "\n".join(lines)


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, operation_name: str):
        self.collector = collector
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_metric(
                f"timing_{self.operation_name}",
                duration,
                unit="seconds",
                description=f"Execution time for {self.operation_name}"
            )


class HealthMonitor:
    """
    Comprehensive health monitoring system.
    
    Monitors system health, simulation stability, and performance
    with configurable alerts and automated diagnostics.
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        check_interval: float = 30.0
    ):
        """
        Initialize health monitor.
        
        Args:
            metrics_collector: Metrics collector instance
            check_interval: Health check interval in seconds
        """
        self.metrics_collector = metrics_collector
        self.check_interval = check_interval
        
        # Health checks registry
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.check_thresholds: Dict[str, Dict[str, float]] = {}
        
        # Health status history
        self.health_history: deque = deque(maxlen=1000)
        
        # Monitoring control
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[HealthStatus], None]] = []
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default system health checks."""
        self.register_health_check("cpu_usage", self._check_cpu_usage)
        self.register_health_check("memory_usage", self._check_memory_usage)
        self.register_health_check("disk_space", self._check_disk_space)
        self.register_health_check("simulation_stability", self._check_simulation_stability)
        
        # Set default thresholds
        self.set_threshold("cpu_usage", warning=80.0, critical=95.0)
        self.set_threshold("memory_usage", warning=85.0, critical=95.0)
        self.set_threshold("disk_space", warning=90.0, critical=98.0)
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a custom health check."""
        self.health_checks[name] = check_func
    
    def set_threshold(self, check_name: str, warning: float, critical: float):
        """Set thresholds for a health check."""
        self.check_thresholds[check_name] = {
            "warning": warning,
            "critical": critical
        }
    
    def perform_health_check(self) -> HealthStatus:
        """Perform comprehensive health check."""
        timestamp = time.time()
        checks = {}
        metrics = {}
        issues = []
        overall_status = "healthy"
        
        # Run all registered health checks
        for check_name, check_func in self.health_checks.items():
            try:
                result = check_func()
                checks[check_name] = result
                
                # Get metric value for threshold checking
                if check_name in self.check_thresholds:
                    metric_name = f"gauge_{check_name}"
                    current_value = self.metrics_collector.gauges.get(metric_name, 0.0)
                    metrics[check_name] = current_value
                    
                    thresholds = self.check_thresholds[check_name]
                    
                    if current_value >= thresholds["critical"]:
                        overall_status = "critical"
                        issues.append(f"{check_name}: {current_value:.1f} >= {thresholds['critical']:.1f} (critical)")
                    elif current_value >= thresholds["warning"] and overall_status != "critical":
                        overall_status = "warning" 
                        issues.append(f"{check_name}: {current_value:.1f} >= {thresholds['warning']:.1f} (warning)")
                
                if not result and check_name not in self.check_thresholds:
                    if overall_status != "critical":
                        overall_status = "warning"
                    issues.append(f"{check_name}: check failed")
                    
            except Exception as e:
                self.logger.error(f"Health check {check_name} failed: {e}")
                checks[check_name] = False
                issues.append(f"{check_name}: check error - {str(e)}")
                if overall_status != "critical":
                    overall_status = "warning"
        
        # Create health status
        health_status = HealthStatus(
            status=overall_status,
            timestamp=timestamp,
            checks=checks,
            metrics=metrics,
            issues=issues
        )
        
        # Store in history
        with self._lock:
            self.health_history.append(health_status)
        
        # Trigger alerts if needed
        if overall_status in ["warning", "critical"]:
            self._trigger_alerts(health_status)
        
        return health_status
    
    def _check_cpu_usage(self) -> bool:
        """Check CPU usage health."""
        cpu_percent = psutil.cpu_percent(interval=1.0)
        self.metrics_collector.set_gauge("cpu_usage", cpu_percent)
        return cpu_percent < 95.0
    
    def _check_memory_usage(self) -> bool:
        """Check memory usage health."""
        memory = psutil.virtual_memory()
        self.metrics_collector.set_gauge("memory_usage", memory.percent)
        return memory.percent < 95.0
    
    def _check_disk_space(self) -> bool:
        """Check disk space health."""
        disk = psutil.disk_usage('/')
        self.metrics_collector.set_gauge("disk_space", disk.percent)
        return disk.percent < 98.0
    
    def _check_simulation_stability(self) -> bool:
        """Check simulation stability based on error patterns."""
        # Check for recent errors in logs or metrics
        try:
            # This is a placeholder - in practice, you'd check error rates
            error_count = self.metrics_collector.counters.get("simulation_errors", 0)
            recent_errors = error_count - self.metrics_collector.counters.get("last_error_count", 0)
            
            if recent_errors > 10:  # More than 10 errors since last check
                return False
            
            self.metrics_collector.counters["last_error_count"] = error_count
            return True
        except Exception:
            return True  # Default to healthy if check fails
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("Started health monitoring")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("Stopped health monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self.perform_health_check()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                time.sleep(self.check_interval)
    
    def add_alert_callback(self, callback: Callable[[HealthStatus], None]):
        """Add callback for health alerts."""
        self.alert_callbacks.append(callback)
    
    def _trigger_alerts(self, health_status: HealthStatus):
        """Trigger health alerts."""
        for callback in self.alert_callbacks:
            try:
                callback(health_status)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def get_health_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get health summary for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            recent_history = [
                h for h in self.health_history
                if h.timestamp >= cutoff_time
            ]
        
        if not recent_history:
            return {"message": "No health data available"}
        
        # Calculate statistics
        status_counts = defaultdict(int)
        total_checks = defaultdict(int)
        failed_checks = defaultdict(int)
        
        for health in recent_history:
            status_counts[health.status] += 1
            for check_name, result in health.checks.items():
                total_checks[check_name] += 1
                if not result:
                    failed_checks[check_name] += 1
        
        # Calculate uptime percentage
        total_points = len(recent_history)
        healthy_points = status_counts["healthy"]
        uptime_percent = (healthy_points / total_points) * 100 if total_points > 0 else 0
        
        # Calculate check reliability
        check_reliability = {}
        for check_name in total_checks:
            success_rate = ((total_checks[check_name] - failed_checks[check_name]) / 
                          total_checks[check_name]) * 100
            check_reliability[check_name] = success_rate
        
        return {
            "time_period_hours": hours,
            "total_health_checks": total_points,
            "uptime_percent": uptime_percent,
            "status_distribution": dict(status_counts),
            "check_reliability": check_reliability,
            "current_status": recent_history[-1].status if recent_history else "unknown",
            "recent_issues": recent_history[-1].issues if recent_history else []
        }


class PerformanceProfiler:
    """
    Performance profiler for photonic simulation operations.
    
    Provides detailed performance analysis and bottleneck identification
    for optimization and tuning purposes.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize performance profiler."""
        self.metrics_collector = metrics_collector
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.active_profiles: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations."""
        return ProfilingContext(self, operation_name)
    
    def start_profile(self, name: str) -> str:
        """Start profiling an operation."""
        profile_id = f"{name}_{int(time.time() * 1000000)}"
        with self._lock:
            self.active_profiles[profile_id] = time.time()
        return profile_id
    
    def end_profile(self, profile_id: str, details: Optional[Dict[str, Any]] = None):
        """End profiling and record results."""
        with self._lock:
            if profile_id not in self.active_profiles:
                self.logger.warning(f"Profile {profile_id} not found")
                return
            
            start_time = self.active_profiles.pop(profile_id)
            duration = time.time() - start_time
            
            # Extract operation name
            operation_name = profile_id.rsplit('_', 1)[0]
            
            # Store profile data
            if operation_name not in self.profiles:
                self.profiles[operation_name] = {
                    "total_calls": 0,
                    "total_duration": 0.0,
                    "min_duration": float('inf'),
                    "max_duration": 0.0,
                    "recent_durations": deque(maxlen=100)
                }
            
            profile_data = self.profiles[operation_name]
            profile_data["total_calls"] += 1
            profile_data["total_duration"] += duration
            profile_data["min_duration"] = min(profile_data["min_duration"], duration)
            profile_data["max_duration"] = max(profile_data["max_duration"], duration)
            profile_data["recent_durations"].append(duration)
            
            if details:
                profile_data.setdefault("details", []).append({
                    "timestamp": time.time(),
                    "duration": duration,
                    **details
                })
            
            # Record metric
            self.metrics_collector.record_metric(
                f"profile_{operation_name}_duration",
                duration,
                unit="seconds"
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        with self._lock:
            report = {
                "timestamp": time.time(),
                "operations": {}
            }
            
            for operation_name, profile_data in self.profiles.items():
                if profile_data["total_calls"] > 0:
                    avg_duration = profile_data["total_duration"] / profile_data["total_calls"]
                    
                    # Calculate percentiles from recent data
                    recent = list(profile_data["recent_durations"])
                    if recent:
                        recent.sort()
                        n = len(recent)
                        p50 = recent[n // 2] if n > 0 else 0
                        p95 = recent[int(n * 0.95)] if n > 0 else 0
                        p99 = recent[int(n * 0.99)] if n > 0 else 0
                    else:
                        p50 = p95 = p99 = 0
                    
                    report["operations"][operation_name] = {
                        "total_calls": profile_data["total_calls"],
                        "total_duration": profile_data["total_duration"],
                        "avg_duration": avg_duration,
                        "min_duration": profile_data["min_duration"],
                        "max_duration": profile_data["max_duration"],
                        "p50_duration": p50,
                        "p95_duration": p95,
                        "p99_duration": p99,
                        "calls_per_second": profile_data["total_calls"] / max(profile_data["total_duration"], 0.001)
                    }
            
            return report


class ProfilingContext:
    """Context manager for performance profiling."""
    
    def __init__(self, profiler: PerformanceProfiler, operation_name: str, **details):
        self.profiler = profiler
        self.operation_name = operation_name
        self.details = details
        self.profile_id = None
    
    def __enter__(self):
        self.profile_id = self.profiler.start_profile(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profile_id:
            if exc_type:
                self.details["exception"] = str(exc_val)
                self.details["exception_type"] = exc_type.__name__
            
            self.profiler.end_profile(self.profile_id, self.details)


def create_monitoring_system(
    enable_system_monitoring: bool = True,
    enable_health_monitoring: bool = True,
    metrics_export_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive monitoring system.
    
    Args:
        enable_system_monitoring: Enable system resource monitoring
        enable_health_monitoring: Enable health monitoring
        metrics_export_path: Path to export metrics (optional)
        
    Returns:
        Dict containing monitoring components
    """
    # Create metrics collector
    metrics_collector = MetricsCollector(
        enable_system_metrics=enable_system_monitoring
    )
    
    # Create health monitor
    health_monitor = None
    if enable_health_monitoring:
        health_monitor = HealthMonitor(metrics_collector)
        health_monitor.start_monitoring()
    
    # Create performance profiler
    profiler = PerformanceProfiler(metrics_collector)
    
    # Setup metrics export if requested
    if metrics_export_path:
        def export_metrics():
            try:
                Path(metrics_export_path).parent.mkdir(parents=True, exist_ok=True)
                with open(metrics_export_path, 'w') as f:
                    f.write(metrics_collector.export_metrics())
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to export metrics: {e}")
        
        # Export metrics every 60 seconds
        import threading
        def periodic_export():
            while True:
                time.sleep(60)
                export_metrics()
        
        export_thread = threading.Thread(target=periodic_export, daemon=True)
        export_thread.start()
    
    return {
        "metrics_collector": metrics_collector,
        "health_monitor": health_monitor,
        "profiler": profiler
    }