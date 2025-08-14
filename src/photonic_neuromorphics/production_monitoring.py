"""
Production Monitoring System for Photonic Neuromorphic Computing

Real-time monitoring, metrics collection, alerting, and observability for
production photonic neuromorphic systems with advanced analytics.
"""

import time
import json
import threading
import queue
import statistics
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import logging
from contextlib import contextmanager


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """Alert definition and state."""
    name: str
    condition: Callable[[float], bool]
    severity: AlertSeverity
    message: str
    cooldown_seconds: float = 300.0  # 5 minutes
    last_triggered: float = 0.0
    triggered_count: int = 0
    
    def should_trigger(self, value: float) -> bool:
        """Check if alert should trigger."""
        if not self.condition(value):
            return False
        
        # Check cooldown
        if time.time() - self.last_triggered < self.cooldown_seconds:
            return False
        
        return True
    
    def trigger(self) -> None:
        """Trigger the alert."""
        self.last_triggered = time.time()
        self.triggered_count += 1


class MetricsCollector:
    """Collects and stores metrics with configurable retention."""
    
    def __init__(self, max_points_per_metric: int = 10000):
        self.max_points_per_metric = max_points_per_metric
        self.metrics = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.metric_metadata = {}
        self.lock = threading.RLock()
        
    def record_metric(self, metric: MetricPoint) -> None:
        """Record a metric point."""
        with self.lock:
            self.metrics[metric.name].append(metric)
            
            # Update metadata
            if metric.name not in self.metric_metadata:
                self.metric_metadata[metric.name] = {
                    'metric_type': metric.metric_type,
                    'first_seen': metric.timestamp,
                    'labels': set()
                }
            
            # Track unique labels
            for label_key in metric.labels.keys():
                self.metric_metadata[metric.name]['labels'].add(label_key)
    
    def get_metric_values(self, metric_name: str, 
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None) -> List[MetricPoint]:
        """Get metric values within time range."""
        with self.lock:
            if metric_name not in self.metrics:
                return []
            
            points = list(self.metrics[metric_name])
            
            # Filter by time range
            if start_time is not None or end_time is not None:
                filtered_points = []
                for point in points:
                    if start_time is not None and point.timestamp < start_time:
                        continue
                    if end_time is not None and point.timestamp > end_time:
                        continue
                    filtered_points.append(point)
                return filtered_points
            
            return points
    
    def get_latest_value(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric."""
        with self.lock:
            if metric_name not in self.metrics or not self.metrics[metric_name]:
                return None
            return self.metrics[metric_name][-1].value
    
    def get_metric_statistics(self, metric_name: str, 
                            duration_seconds: float = 300) -> Dict[str, float]:
        """Get statistics for a metric over specified duration."""
        end_time = time.time()
        start_time = end_time - duration_seconds
        
        points = self.get_metric_values(metric_name, start_time, end_time)
        if not points:
            return {}
        
        values = [p.value for p in points]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'latest': values[-1],
            'rate_per_second': len(values) / duration_seconds
        }
    
    def clear_old_metrics(self, retention_seconds: float) -> None:
        """Clear metrics older than retention period."""
        cutoff_time = time.time() - retention_seconds
        
        with self.lock:
            for metric_name in list(self.metrics.keys()):
                metric_deque = self.metrics[metric_name]
                
                # Remove old points
                while metric_deque and metric_deque[0].timestamp < cutoff_time:
                    metric_deque.popleft()
                
                # Remove empty metrics
                if not metric_deque:
                    del self.metrics[metric_name]
                    if metric_name in self.metric_metadata:
                        del self.metric_metadata[metric_name]


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self):
        self.alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.notification_handlers = []
        self.lock = threading.RLock()
    
    def add_alert(self, alert: Alert) -> None:
        """Add an alert definition."""
        with self.lock:
            self.alerts[alert.name] = alert
    
    def remove_alert(self, alert_name: str) -> bool:
        """Remove an alert definition."""
        with self.lock:
            if alert_name in self.alerts:
                del self.alerts[alert_name]
                return True
            return False
    
    def check_alerts(self, metric_name: str, value: float) -> List[Alert]:
        """Check if any alerts should trigger for a metric value."""
        triggered_alerts = []
        
        with self.lock:
            for alert in self.alerts.values():
                if alert.should_trigger(value):
                    alert.trigger()
                    triggered_alerts.append(alert)
                    
                    # Add to history
                    alert_event = {
                        'alert_name': alert.name,
                        'metric_name': metric_name,
                        'value': value,
                        'severity': alert.severity.value,
                        'message': alert.message,
                        'timestamp': time.time()
                    }
                    self.alert_history.append(alert_event)
                    
                    # Notify handlers
                    for handler in self.notification_handlers:
                        try:
                            handler(alert_event)
                        except Exception as e:
                            logging.error(f"Error in alert handler: {e}")
        
        return triggered_alerts
    
    def add_notification_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Add a notification handler for alerts."""
        self.notification_handlers.append(handler)
    
    def get_alert_history(self, severity: Optional[AlertSeverity] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history with optional filtering."""
        with self.lock:
            history = list(self.alert_history)
            
            if severity:
                history = [h for h in history if h['severity'] == severity.value]
            
            return history[-limit:]  # Return most recent alerts
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        with self.lock:
            total_alerts = len(self.alerts)
            recent_alerts = len([h for h in self.alert_history 
                               if time.time() - h['timestamp'] < 3600])  # Last hour
            
            severity_counts = defaultdict(int)
            for event in self.alert_history:
                if time.time() - event['timestamp'] < 3600:
                    severity_counts[event['severity']] += 1
            
            return {
                'total_alert_definitions': total_alerts,
                'recent_alert_count': recent_alerts,
                'severity_breakdown': dict(severity_counts),
                'alert_rate_per_hour': recent_alerts
            }


class PhotonicSystemMonitor:
    """Comprehensive monitoring system for photonic neuromorphic systems."""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metric_queue = queue.Queue()
        
        # System metrics
        self.system_metrics = {
            'monitoring_uptime': 0.0,
            'metrics_collected': 0,
            'alerts_triggered': 0,
            'last_collection_time': 0.0
        }
        
        self._setup_default_alerts()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup monitoring logging."""
        self.logger = logging.getLogger('photonic_monitoring')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _setup_default_alerts(self):
        """Setup default alerts for photonic systems."""
        
        # High energy consumption alert
        self.alert_manager.add_alert(Alert(
            name="high_energy_consumption",
            condition=lambda x: x > 10e-3,  # > 10 mW
            severity=AlertSeverity.WARNING,
            message="Energy consumption is abnormally high",
            cooldown_seconds=60.0
        ))
        
        # Low efficiency alert
        self.alert_manager.add_alert(Alert(
            name="low_efficiency",
            condition=lambda x: x < 0.5,  # < 50% efficiency
            severity=AlertSeverity.WARNING,
            message="System efficiency is below threshold",
            cooldown_seconds=120.0
        ))
        
        # High error rate alert
        self.alert_manager.add_alert(Alert(
            name="high_error_rate",
            condition=lambda x: x > 0.1,  # > 10% error rate
            severity=AlertSeverity.CRITICAL,
            message="System error rate is critically high",
            cooldown_seconds=30.0
        ))
        
        # Temperature alert
        self.alert_manager.add_alert(Alert(
            name="high_temperature",
            condition=lambda x: x > 350,  # > 350K (77°C)
            severity=AlertSeverity.CRITICAL,
            message="System temperature is critically high",
            cooldown_seconds=60.0
        ))
    
    def start_monitoring(self) -> None:
        """Start the monitoring system."""
        if self.monitoring_active:
            self.logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Photonic system monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Photonic system monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        start_time = time.time()
        
        while self.monitoring_active:
            loop_start = time.time()
            
            # Process queued metrics
            self._process_metric_queue()
            
            # Update system metrics
            self.system_metrics['monitoring_uptime'] = time.time() - start_time
            self.system_metrics['last_collection_time'] = time.time()
            
            # Collect system health metrics
            self._collect_system_health_metrics()
            
            # Clean old metrics
            self.metrics_collector.clear_old_metrics(retention_seconds=3600)  # 1 hour retention
            
            # Sleep for remaining interval
            elapsed = time.time() - loop_start
            if elapsed < self.collection_interval:
                time.sleep(self.collection_interval - elapsed)
    
    def _process_metric_queue(self):
        """Process all metrics in the queue."""
        while not self.metric_queue.empty():
            try:
                metric = self.metric_queue.get_nowait()
                self.metrics_collector.record_metric(metric)
                self.system_metrics['metrics_collected'] += 1
                
                # Check alerts for this metric
                alerts = self.alert_manager.check_alerts(metric.name, metric.value)
                if alerts:
                    self.system_metrics['alerts_triggered'] += len(alerts)
                    for alert in alerts:
                        self.logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
                
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Error processing metric: {e}")
    
    def _collect_system_health_metrics(self):
        """Collect internal system health metrics."""
        current_time = time.time()
        
        # Memory usage (approximation)
        total_metric_points = sum(len(deque_obj) for deque_obj in self.metrics_collector.metrics.values())
        memory_metric = MetricPoint(
            name="system.memory.metric_points",
            value=total_metric_points,
            timestamp=current_time,
            metric_type=MetricType.GAUGE
        )
        self.metrics_collector.record_metric(memory_metric)
        
        # Collection rate
        rate_metric = MetricPoint(
            name="system.collection.rate_per_second",
            value=1.0 / self.collection_interval,
            timestamp=current_time,
            metric_type=MetricType.GAUGE
        )
        self.metrics_collector.record_metric(rate_metric)
        
        # Alert statistics
        alert_stats = self.alert_manager.get_alert_summary()
        alert_rate_metric = MetricPoint(
            name="system.alerts.rate_per_hour",
            value=alert_stats['alert_rate_per_hour'],
            timestamp=current_time,
            metric_type=MetricType.GAUGE
        )
        self.metrics_collector.record_metric(alert_rate_metric)
    
    def record_metric(self, name: str, value: float, 
                     labels: Optional[Dict[str, str]] = None,
                     metric_type: MetricType = MetricType.GAUGE) -> None:
        """Record a metric value."""
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels or {},
            metric_type=metric_type
        )
        
        # Add to queue for processing
        self.metric_queue.put(metric)
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        current_value = self.metrics_collector.get_latest_value(name) or 0
        self.record_metric(name, current_value + 1, labels, MetricType.COUNTER)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value."""
        self.record_metric(name, value, labels, MetricType.GAUGE)
    
    def record_timing(self, name: str, duration_seconds: float, 
                     labels: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric."""
        self.record_metric(name, duration_seconds, labels, MetricType.TIMING)
    
    @contextmanager
    def time_operation(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timing(metric_name, duration, labels)
    
    def get_metric_dashboard(self) -> Dict[str, Any]:
        """Get a dashboard view of all metrics."""
        dashboard = {
            'timestamp': time.time(),
            'system_health': {},
            'photonic_metrics': {},
            'alert_summary': {},
            'performance_summary': {}
        }
        
        # System health
        dashboard['system_health'] = {
            'monitoring_uptime': self.system_metrics['monitoring_uptime'],
            'metrics_collected': self.system_metrics['metrics_collected'],
            'alerts_triggered': self.system_metrics['alerts_triggered'],
            'collection_active': self.monitoring_active
        }
        
        # Recent photonic metrics
        photonic_metric_names = [
            'photonic.energy_consumption',
            'photonic.efficiency',
            'photonic.temperature',
            'photonic.accuracy',
            'photonic.processing_time'
        ]
        
        for metric_name in photonic_metric_names:
            stats = self.metrics_collector.get_metric_statistics(metric_name, duration_seconds=300)
            if stats:
                dashboard['photonic_metrics'][metric_name] = stats
        
        # Alert summary
        dashboard['alert_summary'] = self.alert_manager.get_alert_summary()
        
        # Performance summary
        dashboard['performance_summary'] = {
            'total_metrics_stored': sum(len(deque_obj) for deque_obj in self.metrics_collector.metrics.values()),
            'unique_metrics': len(self.metrics_collector.metrics),
            'collection_interval': self.collection_interval,
            'avg_collection_rate': self.system_metrics['metrics_collected'] / max(self.system_metrics['monitoring_uptime'], 1)
        }
        
        return dashboard
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        if format_type == "json":
            return self._export_json()
        elif format_type == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_json(self) -> str:
        """Export metrics as JSON."""
        export_data = {
            'timestamp': time.time(),
            'metrics': {},
            'metadata': self.metrics_collector.metric_metadata
        }
        
        for metric_name, metric_deque in self.metrics_collector.metrics.items():
            export_data['metrics'][metric_name] = [
                {
                    'value': point.value,
                    'timestamp': point.timestamp,
                    'labels': point.labels
                }
                for point in list(metric_deque)[-100:]  # Last 100 points
            ]
        
        return json.dumps(export_data, indent=2)
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for metric_name, metric_deque in self.metrics_collector.metrics.items():
            if not metric_deque:
                continue
            
            latest_point = metric_deque[-1]
            
            # Convert metric name to Prometheus format
            prom_name = metric_name.replace('.', '_').replace('-', '_')
            
            # Add help and type comments
            lines.append(f"# HELP {prom_name} Photonic neuromorphic metric")
            lines.append(f"# TYPE {prom_name} {latest_point.metric_type.value}")
            
            # Add metric line
            if latest_point.labels:
                label_str = ','.join([f'{k}="{v}"' for k, v in latest_point.labels.items()])
                lines.append(f"{prom_name}{{{label_str}}} {latest_point.value} {int(latest_point.timestamp * 1000)}")
            else:
                lines.append(f"{prom_name} {latest_point.value} {int(latest_point.timestamp * 1000)}")
        
        return '\n'.join(lines)


def demonstrate_production_monitoring():
    """Demonstrate production monitoring system."""
    print("📊 Demonstrating Production Monitoring System")
    print("=" * 60)
    
    # Create monitoring system
    monitor = PhotonicSystemMonitor(collection_interval=0.5)
    
    # Add custom alert
    custom_alert = Alert(
        name="demo_alert",
        condition=lambda x: x > 0.9,
        severity=AlertSeverity.INFO,
        message="Demo metric exceeded threshold",
        cooldown_seconds=5.0
    )
    monitor.alert_manager.add_alert(custom_alert)
    
    # Add notification handler
    def alert_handler(alert_event):
        print(f"   🚨 ALERT: {alert_event['alert_name']} - {alert_event['message']}")
    
    monitor.alert_manager.add_notification_handler(alert_handler)
    
    # Start monitoring
    print("\n1. Starting monitoring system...")
    monitor.start_monitoring()
    
    # Simulate photonic system operation
    print("\n2. Simulating photonic system metrics...")
    
    import random
    for i in range(20):
        # Simulate various metrics
        monitor.set_gauge("photonic.energy_consumption", random.uniform(1e-3, 15e-3))
        monitor.set_gauge("photonic.efficiency", random.uniform(0.3, 0.95))
        monitor.set_gauge("photonic.temperature", random.uniform(290, 360))
        monitor.set_gauge("photonic.accuracy", random.uniform(0.8, 0.98))
        
        # Timing metric example
        with monitor.time_operation("photonic.processing_time"):
            time.sleep(random.uniform(0.01, 0.05))  # Simulate processing
        
        # Counter metric
        monitor.increment_counter("photonic.spike_count")
        
        # Demo metric for alert testing
        demo_value = random.uniform(0.7, 1.1)
        monitor.set_gauge("demo.metric", demo_value)
        
        time.sleep(0.1)  # Brief pause
    
    # Wait for processing
    time.sleep(1.0)
    
    # Get dashboard
    print("\n3. System dashboard:")
    dashboard = monitor.get_metric_dashboard()
    
    print(f"   Monitoring uptime: {dashboard['system_health']['monitoring_uptime']:.1f}s")
    print(f"   Metrics collected: {dashboard['system_health']['metrics_collected']}")
    print(f"   Alerts triggered: {dashboard['system_health']['alerts_triggered']}")
    print(f"   Unique metrics: {dashboard['performance_summary']['unique_metrics']}")
    
    # Show recent metrics
    print("\n4. Recent photonic metrics:")
    for metric_name, stats in dashboard['photonic_metrics'].items():
        if stats:
            print(f"   {metric_name}: mean={stats['mean']:.4f}, latest={stats['latest']:.4f}")
    
    # Show alert summary
    print("\n5. Alert summary:")
    alert_summary = dashboard['alert_summary']
    print(f"   Alert definitions: {alert_summary['total_alert_definitions']}")
    print(f"   Recent alerts: {alert_summary['recent_alert_count']}")
    
    # Export metrics
    print("\n6. Exporting metrics...")
    json_export = monitor.export_metrics("json")
    print(f"   JSON export size: {len(json_export)} characters")
    
    prom_export = monitor.export_metrics("prometheus")
    print(f"   Prometheus export lines: {len(prom_export.split('\\n'))}")
    
    # Stop monitoring
    print("\n7. Stopping monitoring system...")
    monitor.stop_monitoring()
    
    return dashboard


if __name__ == "__main__":
    demonstrate_production_monitoring()