"""
Production Health Monitoring System for Photonic Neuromorphic Computing.

This module provides comprehensive health monitoring, alerting, and diagnostics
for production photonic neuromorphic systems with real-time performance tracking.
"""

import time
import threading
import queue
import json
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import logging
import smtplib
from email.mime.text import MimeText
from collections import deque, defaultdict
import statistics

from .enhanced_logging import PhotonicLogger, PerformanceTracker
from .robust_research_framework import RobustQuantumPhotonicProcessor, RobustOpticalInterferenceProcessor


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Health metric data structure."""
    name: str
    value: float
    unit: str
    timestamp: float
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    description: str = ""


@dataclass
class HealthAlert:
    """Health alert data structure."""
    id: str
    severity: AlertSeverity
    component: str
    message: str
    metric_name: str
    metric_value: float
    timestamp: float
    resolved: bool = False
    resolved_timestamp: Optional[float] = None


@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""
    timestamp: float
    overall_status: HealthStatus
    component_statuses: Dict[str, HealthStatus]
    metrics: List[HealthMetric]
    active_alerts: List[HealthAlert]
    performance_summary: Dict[str, Any]
    recommendations: List[str]


class HealthMonitor:
    """
    Comprehensive health monitoring system for photonic neuromorphic components.
    
    Monitors:
    - Processing performance and latency
    - Memory usage and GPU utilization
    - Quantum coherence quality
    - Optical interference efficiency
    - Error rates and circuit breaker states
    - System resources and network connectivity
    """
    
    def __init__(self, monitoring_interval: float = 30.0, alert_cooldown: float = 300.0):
        self.monitoring_interval = monitoring_interval
        self.alert_cooldown = alert_cooldown
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts = {}
        self.alert_history = []
        
        # Component references
        self.quantum_processors = []
        self.optical_processors = []
        self.validators = []
        
        # Thresholds
        self.thresholds = {
            'cpu_usage_warning': 70.0,
            'cpu_usage_critical': 90.0,
            'memory_usage_warning': 80.0,
            'memory_usage_critical': 95.0,
            'processing_latency_warning': 1.0,
            'processing_latency_critical': 5.0,
            'error_rate_warning': 0.05,
            'error_rate_critical': 0.20,
            'coherence_quality_warning': 0.7,
            'coherence_quality_critical': 0.5,
            'interference_efficiency_warning': 0.6,
            'interference_efficiency_critical': 0.4
        }
        
        # Setup logging and alerting
        self.logger = PhotonicLogger(__name__)
        self.alert_callbacks = []
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Thread safety
        self._lock = threading.RLock()
    
    def register_component(self, component: Any, component_type: str):
        """Register a component for monitoring."""
        with self._lock:
            if component_type == "quantum_processor":
                self.quantum_processors.append(component)
            elif component_type == "optical_processor":
                self.optical_processors.append(component)
            elif component_type == "validator":
                self.validators.append(component)
            else:
                self.logger.warning(f"Unknown component type: {component_type}")
        
        self.logger.info(f"Registered {component_type} for monitoring")
    
    def add_alert_callback(self, callback: Callable[[HealthAlert], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics from all sources
                metrics = self._collect_all_metrics()
                
                # Evaluate health status
                alerts = self._evaluate_health_status(metrics)
                
                # Process alerts
                for alert in alerts:
                    self._process_alert(alert)
                
                # Update metrics history
                self._update_metrics_history(metrics)
                
                # Log health summary
                self._log_health_summary(metrics)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
            
            # Wait for next monitoring cycle
            time.sleep(self.monitoring_interval)
    
    def _collect_all_metrics(self) -> List[HealthMetric]:
        """Collect all health metrics from various sources."""
        metrics = []
        
        # System metrics
        metrics.extend(self._collect_system_metrics())
        
        # Quantum processor metrics
        metrics.extend(self._collect_quantum_processor_metrics())
        
        # Optical processor metrics
        metrics.extend(self._collect_optical_processor_metrics())
        
        # Performance metrics
        metrics.extend(self._collect_performance_metrics())
        
        return metrics
    
    def _collect_system_metrics(self) -> List[HealthMetric]:
        """Collect system-level metrics."""
        metrics = []
        timestamp = time.time()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = self._evaluate_threshold_status(
                cpu_percent,
                self.thresholds['cpu_usage_warning'],
                self.thresholds['cpu_usage_critical']
            )
            
            metrics.append(HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="%",
                timestamp=timestamp,
                status=cpu_status,
                threshold_warning=self.thresholds['cpu_usage_warning'],
                threshold_critical=self.thresholds['cpu_usage_critical'],
                description="System CPU usage percentage"
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_status = self._evaluate_threshold_status(
                memory.percent,
                self.thresholds['memory_usage_warning'],
                self.thresholds['memory_usage_critical']
            )
            
            metrics.append(HealthMetric(
                name="memory_usage",
                value=memory.percent,
                unit="%",
                timestamp=timestamp,
                status=memory_status,
                threshold_warning=self.thresholds['memory_usage_warning'],
                threshold_critical=self.thresholds['memory_usage_critical'],
                description="System memory usage percentage"
            ))
            
            # GPU metrics (if available)
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                
                metrics.append(HealthMetric(
                    name="gpu_memory_usage",
                    value=gpu_memory,
                    unit="%",
                    timestamp=timestamp,
                    status=self._evaluate_threshold_status(gpu_memory, 70.0, 90.0),
                    threshold_warning=70.0,
                    threshold_critical=90.0,
                    description="GPU memory usage percentage"
                ))
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def _collect_quantum_processor_metrics(self) -> List[HealthMetric]:
        """Collect quantum processor specific metrics."""
        metrics = []
        timestamp = time.time()
        
        for i, processor in enumerate(self.quantum_processors):
            try:
                # Processing performance
                if hasattr(processor, 'performance_metrics'):
                    perf_metrics = processor.performance_metrics
                    
                    if 'processing_latency' in perf_metrics and perf_metrics['processing_latency']:
                        avg_latency = statistics.mean(perf_metrics['processing_latency'][-10:])
                        latency_status = self._evaluate_threshold_status(
                            avg_latency,
                            self.thresholds['processing_latency_warning'],
                            self.thresholds['processing_latency_critical']
                        )
                        
                        metrics.append(HealthMetric(
                            name=f"quantum_processor_{i}_latency",
                            value=avg_latency,
                            unit="s",
                            timestamp=timestamp,
                            status=latency_status,
                            threshold_warning=self.thresholds['processing_latency_warning'],
                            threshold_critical=self.thresholds['processing_latency_critical'],
                            description=f"Quantum processor {i} average processing latency"
                        ))
                
                # Error rate
                if hasattr(processor, '_processing_errors'):
                    error_rate = processor._processing_errors / max(1, len(processor.performance_metrics.get('processing_latency', [1])))
                    error_status = self._evaluate_threshold_status(
                        error_rate,
                        self.thresholds['error_rate_warning'],
                        self.thresholds['error_rate_critical']
                    )
                    
                    metrics.append(HealthMetric(
                        name=f"quantum_processor_{i}_error_rate",
                        value=error_rate,
                        unit="ratio",
                        timestamp=timestamp,
                        status=error_status,
                        threshold_warning=self.thresholds['error_rate_warning'],
                        threshold_critical=self.thresholds['error_rate_critical'],
                        description=f"Quantum processor {i} error rate"
                    ))
                
                # Circuit breaker state
                if hasattr(processor, 'circuit_breaker'):
                    cb_state = 1.0 if processor.circuit_breaker.is_open else 0.0
                    cb_status = HealthStatus.CRITICAL if cb_state > 0 else HealthStatus.HEALTHY
                    
                    metrics.append(HealthMetric(
                        name=f"quantum_processor_{i}_circuit_breaker",
                        value=cb_state,
                        unit="state",
                        timestamp=timestamp,
                        status=cb_status,
                        description=f"Quantum processor {i} circuit breaker state (0=closed, 1=open)"
                    ))
            
            except Exception as e:
                self.logger.error(f"Error collecting quantum processor {i} metrics: {e}")
        
        return metrics
    
    def _collect_optical_processor_metrics(self) -> List[HealthMetric]:
        """Collect optical processor specific metrics."""
        metrics = []
        timestamp = time.time()
        
        for i, processor in enumerate(self.optical_processors):
            try:
                # Coherence quality
                if hasattr(processor, 'coherence_history') and processor.coherence_history:
                    recent_coherence = [entry['coherence_quality'] for entry in processor.coherence_history[-10:]]
                    avg_coherence = statistics.mean(recent_coherence)
                    
                    # Reverse thresholds for coherence (higher is better)
                    coherence_status = HealthStatus.HEALTHY
                    if avg_coherence < self.thresholds['coherence_quality_critical']:
                        coherence_status = HealthStatus.CRITICAL
                    elif avg_coherence < self.thresholds['coherence_quality_warning']:
                        coherence_status = HealthStatus.WARNING
                    
                    metrics.append(HealthMetric(
                        name=f"optical_processor_{i}_coherence_quality",
                        value=avg_coherence,
                        unit="quality",
                        timestamp=timestamp,
                        status=coherence_status,
                        threshold_warning=self.thresholds['coherence_quality_warning'],
                        threshold_critical=self.thresholds['coherence_quality_critical'],
                        description=f"Optical processor {i} average coherence quality"
                    ))
                
                # Interference efficiency
                if hasattr(processor, 'interference_efficiency') and processor.interference_efficiency:
                    recent_efficiency = processor.interference_efficiency[-10:]
                    avg_efficiency = statistics.mean(recent_efficiency)
                    
                    # Reverse thresholds for efficiency (higher is better)
                    efficiency_status = HealthStatus.HEALTHY
                    if avg_efficiency < self.thresholds['interference_efficiency_critical']:
                        efficiency_status = HealthStatus.CRITICAL
                    elif avg_efficiency < self.thresholds['interference_efficiency_warning']:
                        efficiency_status = HealthStatus.WARNING
                    
                    metrics.append(HealthMetric(
                        name=f"optical_processor_{i}_interference_efficiency",
                        value=avg_efficiency,
                        unit="efficiency",
                        timestamp=timestamp,
                        status=efficiency_status,
                        threshold_warning=self.thresholds['interference_efficiency_warning'],
                        threshold_critical=self.thresholds['interference_efficiency_critical'],
                        description=f"Optical processor {i} average interference efficiency"
                    ))
            
            except Exception as e:
                self.logger.error(f"Error collecting optical processor {i} metrics: {e}")
        
        return metrics
    
    def _collect_performance_metrics(self) -> List[HealthMetric]:
        """Collect general performance metrics."""
        metrics = []
        timestamp = time.time()
        
        try:
            # Get performance data from tracker
            if hasattr(self.performance_tracker, 'operation_metrics'):
                for operation, times in self.performance_tracker.operation_metrics.items():
                    if times:
                        avg_time = statistics.mean(times[-10:])
                        
                        metrics.append(HealthMetric(
                            name=f"performance_{operation}_latency",
                            value=avg_time,
                            unit="s",
                            timestamp=timestamp,
                            status=self._evaluate_threshold_status(avg_time, 0.5, 2.0),
                            threshold_warning=0.5,
                            threshold_critical=2.0,
                            description=f"Average latency for {operation} operations"
                        ))
        
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {e}")
        
        return metrics
    
    def _evaluate_threshold_status(self, value: float, warning_threshold: float, 
                                 critical_threshold: float) -> HealthStatus:
        """Evaluate health status based on thresholds."""
        if value >= critical_threshold:
            return HealthStatus.CRITICAL
        elif value >= warning_threshold:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _evaluate_health_status(self, metrics: List[HealthMetric]) -> List[HealthAlert]:
        """Evaluate overall health status and generate alerts."""
        alerts = []
        current_time = time.time()
        
        for metric in metrics:
            if metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                
                # Check if alert already exists and is in cooldown
                alert_key = f"{metric.name}_{metric.status.value}"
                
                if alert_key in self.active_alerts:
                    last_alert_time = self.active_alerts[alert_key].timestamp
                    if current_time - last_alert_time < self.alert_cooldown:
                        continue  # Skip due to cooldown
                
                # Create new alert
                severity = AlertSeverity.ERROR if metric.status == HealthStatus.CRITICAL else AlertSeverity.WARNING
                
                alert = HealthAlert(
                    id=f"alert_{alert_key}_{int(current_time)}",
                    severity=severity,
                    component=metric.name.split('_')[0],
                    message=self._generate_alert_message(metric),
                    metric_name=metric.name,
                    metric_value=metric.value,
                    timestamp=current_time
                )
                
                alerts.append(alert)
                self.active_alerts[alert_key] = alert
        
        return alerts
    
    def _generate_alert_message(self, metric: HealthMetric) -> str:
        """Generate human-readable alert message."""
        if metric.status == HealthStatus.CRITICAL:
            return f"CRITICAL: {metric.description} is {metric.value:.2f} {metric.unit}, exceeding critical threshold of {metric.threshold_critical}"
        elif metric.status == HealthStatus.WARNING:
            return f"WARNING: {metric.description} is {metric.value:.2f} {metric.unit}, exceeding warning threshold of {metric.threshold_warning}"
        else:
            return f"INFO: {metric.description} is {metric.value:.2f} {metric.unit}"
    
    def _process_alert(self, alert: HealthAlert):
        """Process and dispatch alert."""
        # Add to history
        self.alert_history.append(alert)
        
        # Log alert
        log_level = logging.ERROR if alert.severity == AlertSeverity.CRITICAL else logging.WARNING
        self.logger.log(log_level, alert.message, extra={
            'alert_id': alert.id,
            'component': alert.component,
            'metric_name': alert.metric_name,
            'metric_value': alert.metric_value
        })
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def _update_metrics_history(self, metrics: List[HealthMetric]):
        """Update metrics history for trending analysis."""
        for metric in metrics:
            self.metrics_history[metric.name].append({
                'value': metric.value,
                'timestamp': metric.timestamp,
                'status': metric.status
            })
    
    def _log_health_summary(self, metrics: List[HealthMetric]):
        """Log periodic health summary."""
        status_counts = defaultdict(int)
        for metric in metrics:
            status_counts[metric.status] += 1
        
        self.logger.info(
            "Health monitoring summary",
            extra={
                'total_metrics': len(metrics),
                'healthy': status_counts[HealthStatus.HEALTHY],
                'warning': status_counts[HealthStatus.WARNING],
                'critical': status_counts[HealthStatus.CRITICAL],
                'active_alerts': len(self.active_alerts)
            }
        )
    
    def get_health_report(self) -> SystemHealthReport:
        """Generate comprehensive health report."""
        current_metrics = self._collect_all_metrics()
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        if any(m.status == HealthStatus.CRITICAL for m in current_metrics):
            overall_status = HealthStatus.CRITICAL
        elif any(m.status == HealthStatus.WARNING for m in current_metrics):
            overall_status = HealthStatus.WARNING
        
        # Component statuses
        component_statuses = {}
        for metric in current_metrics:
            component = metric.name.split('_')[0]
            if component not in component_statuses or metric.status.value > component_statuses[component].value:
                component_statuses[component] = metric.status
        
        # Performance summary
        performance_summary = {
            'monitoring_uptime': time.time() - getattr(self, '_start_time', time.time()),
            'total_alerts_generated': len(self.alert_history),
            'active_alerts_count': len(self.active_alerts),
            'metrics_collected': len(current_metrics)
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(current_metrics)
        
        return SystemHealthReport(
            timestamp=time.time(),
            overall_status=overall_status,
            component_statuses=component_statuses,
            metrics=current_metrics,
            active_alerts=list(self.active_alerts.values()),
            performance_summary=performance_summary,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, metrics: List[HealthMetric]) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        
        critical_metrics = [m for m in metrics if m.status == HealthStatus.CRITICAL]
        warning_metrics = [m for m in metrics if m.status == HealthStatus.WARNING]
        
        if critical_metrics:
            recommendations.append("URGENT: Address critical issues immediately to prevent system failure")
            
            for metric in critical_metrics:
                if 'cpu_usage' in metric.name:
                    recommendations.append("• Reduce computational load or scale horizontally")
                elif 'memory_usage' in metric.name:
                    recommendations.append("• Free memory or increase available RAM")
                elif 'coherence_quality' in metric.name:
                    recommendations.append("• Recalibrate optical components and check environmental conditions")
                elif 'error_rate' in metric.name:
                    recommendations.append("• Review error logs and implement additional error handling")
        
        if warning_metrics:
            recommendations.append("Monitor warning conditions closely and prepare for potential escalation")
        
        if not critical_metrics and not warning_metrics:
            recommendations.append("System operating normally - continue regular monitoring")
        
        return recommendations


class AlertingSystem:
    """Advanced alerting system with multiple notification channels."""
    
    def __init__(self):
        self.logger = PhotonicLogger(__name__)
        self.notification_channels = []
    
    def add_email_notification(self, smtp_server: str, smtp_port: int, 
                             username: str, password: str, recipients: List[str]):
        """Add email notification channel."""
        def send_email_alert(alert: HealthAlert):
            try:
                subject = f"[{alert.severity.value.upper()}] Photonic System Alert: {alert.component}"
                body = f"""
Alert Details:
- Component: {alert.component}
- Severity: {alert.severity.value}
- Message: {alert.message}
- Metric: {alert.metric_name}
- Value: {alert.metric_value}
- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}

Please investigate and take appropriate action.
                """
                
                msg = MimeText(body)
                msg['Subject'] = subject
                msg['From'] = username
                msg['To'] = ', '.join(recipients)
                
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(username, password)
                    server.send_message(msg)
                
                self.logger.info(f"Email alert sent for {alert.id}")
                
            except Exception as e:
                self.logger.error(f"Failed to send email alert: {e}")
        
        self.notification_channels.append(send_email_alert)
    
    def add_webhook_notification(self, webhook_url: str):
        """Add webhook notification channel."""
        def send_webhook_alert(alert: HealthAlert):
            try:
                import requests
                
                payload = {
                    'alert_id': alert.id,
                    'severity': alert.severity.value,
                    'component': alert.component,
                    'message': alert.message,
                    'metric_name': alert.metric_name,
                    'metric_value': alert.metric_value,
                    'timestamp': alert.timestamp
                }
                
                response = requests.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                
                self.logger.info(f"Webhook alert sent for {alert.id}")
                
            except Exception as e:
                self.logger.error(f"Failed to send webhook alert: {e}")
        
        self.notification_channels.append(send_webhook_alert)
    
    def send_alert(self, alert: HealthAlert):
        """Send alert through all configured channels."""
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                self.logger.error(f"Alert channel failed: {e}")


def create_production_monitoring_system() -> Tuple[HealthMonitor, AlertingSystem]:
    """Create a complete production monitoring system."""
    
    # Create health monitor
    monitor = HealthMonitor(monitoring_interval=30.0, alert_cooldown=300.0)
    
    # Create alerting system
    alerting = AlertingSystem()
    
    # Connect alerting to monitoring
    monitor.add_alert_callback(alerting.send_alert)
    
    return monitor, alerting


def demonstrate_health_monitoring():
    """Demonstrate the health monitoring system."""
    print("🏥 PRODUCTION HEALTH MONITORING DEMONSTRATION")
    print("=" * 60)
    
    # Create monitoring system
    monitor, alerting = create_production_monitoring_system()
    
    # Create test components
    from .robust_research_framework import create_robust_research_environment
    
    environment = create_robust_research_environment()
    
    # Register components
    monitor.register_component(environment['quantum_processor'], "quantum_processor")
    monitor.register_component(environment['optical_processor'], "optical_processor")
    monitor.register_component(environment['validator'], "validator")
    
    print("✓ Registered components for monitoring")
    
    # Start monitoring
    monitor.start_monitoring()
    print("✓ Health monitoring started")
    
    # Generate test load
    print("\\n🔬 Generating test load...")
    test_data = torch.randn(4, 50, 16)
    
    for i in range(5):
        try:
            _ = environment['quantum_processor'](test_data)
            print(f"  Test {i+1} completed")
        except Exception as e:
            print(f"  Test {i+1} failed: {e}")
        
        time.sleep(2)
    
    # Wait for monitoring cycle
    time.sleep(35)
    
    # Get health report
    health_report = monitor.get_health_report()
    
    print("\\n📊 HEALTH REPORT:")
    print(f"Overall Status: {health_report.overall_status.value}")
    print(f"Components Monitored: {len(health_report.component_statuses)}")
    print(f"Metrics Collected: {len(health_report.metrics)}")
    print(f"Active Alerts: {len(health_report.active_alerts)}")
    
    if health_report.recommendations:
        print("\\n💡 RECOMMENDATIONS:")
        for rec in health_report.recommendations:
            print(f"  {rec}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("\\n✓ Health monitoring stopped")
    
    return health_report


if __name__ == "__main__":
    demonstrate_health_monitoring()