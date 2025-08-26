"""
Real-Time Adaptive Monitoring System for Photonic Neuromorphics

Advanced monitoring framework with adaptive thresholds, predictive analytics,
and autonomous response capabilities for photonic neural network systems.

Features:
- Real-time performance monitoring with microsecond resolution
- Predictive anomaly detection using machine learning
- Adaptive threshold adjustment based on system behavior
- Autonomous response and self-healing capabilities
- Multi-dimensional health scoring and optimization
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import time
import threading
from collections import deque, defaultdict
from enum import Enum
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .enhanced_logging import PhotonicLogger
from .monitoring import MetricsCollector
from .exceptions import MonitoringError, ValidationError


class HealthStatus(Enum):
    """System health status levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILURE = "failure"


class ResponseAction(Enum):
    """Autonomous response actions."""
    NONE = "none"
    PARAMETER_ADJUST = "parameter_adjust"
    LOAD_BALANCE = "load_balance"
    RESOURCE_SCALE = "resource_scale"
    FAILOVER = "failover"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class MonitoringMetric:
    """Individual monitoring metric with adaptive thresholds."""
    name: str
    current_value: float
    baseline_value: float = 0.0
    threshold_warning: float = 0.0
    threshold_critical: float = 0.0
    trend: float = 0.0  # Rate of change
    confidence: float = 1.0
    last_updated: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def update(self, new_value: float, timestamp: float = None) -> None:
        """Update metric with new value and calculate trend."""
        if timestamp is None:
            timestamp = time.time()
        
        self.history.append((timestamp, new_value))
        
        # Calculate trend from recent history
        if len(self.history) >= 2:
            recent_values = [v for _, v in list(self.history)[-10:]]
            if len(recent_values) >= 2:
                self.trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
        
        self.current_value = new_value
        self.last_updated = timestamp
        
        # Update baseline with exponential moving average
        alpha = 0.01  # Smoothing factor
        self.baseline_value = alpha * new_value + (1 - alpha) * self.baseline_value
    
    def get_status(self) -> HealthStatus:
        """Get current health status based on thresholds."""
        if self.current_value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.current_value >= self.threshold_warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.GOOD
    
    def predict_future_value(self, time_horizon: float) -> float:
        """Predict future value based on current trend."""
        return self.current_value + self.trend * time_horizon


class PredictiveAnomalyDetector(nn.Module):
    """Neural network for predictive anomaly detection."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)  # Anomaly score
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for anomaly detection."""
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take last timestep
        
        x = torch.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Anomaly score 0-1
        
        return x


@dataclass
class SystemComponent:
    """Monitored system component."""
    name: str
    component_type: str
    metrics: Dict[str, MonitoringMetric] = field(default_factory=dict)
    health_score: float = 1.0
    last_health_check: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    auto_response_enabled: bool = True
    
    def add_metric(self, metric: MonitoringMetric) -> None:
        """Add monitoring metric to component."""
        self.metrics[metric.name] = metric
    
    def update_health_score(self) -> float:
        """Calculate comprehensive health score."""
        if not self.metrics:
            return 1.0
        
        # Weight different metric types
        weights = {
            "performance": 0.4,
            "reliability": 0.3,
            "efficiency": 0.2,
            "security": 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric_name, metric in self.metrics.items():
            # Determine metric type from name
            metric_type = "performance"  # Default
            if "error" in metric_name.lower() or "failure" in metric_name.lower():
                metric_type = "reliability"
            elif "energy" in metric_name.lower() or "efficiency" in metric_name.lower():
                metric_type = "efficiency"
            elif "security" in metric_name.lower() or "threat" in metric_name.lower():
                metric_type = "security"
            
            weight = weights.get(metric_type, 0.1)
            
            # Calculate metric health (0-1 scale)
            status = metric.get_status()
            if status == HealthStatus.EXCELLENT:
                metric_health = 1.0
            elif status == HealthStatus.GOOD:
                metric_health = 0.8
            elif status == HealthStatus.WARNING:
                metric_health = 0.5
            elif status == HealthStatus.CRITICAL:
                metric_health = 0.2
            else:  # FAILURE
                metric_health = 0.0
            
            weighted_score += weight * metric_health * metric.confidence
            total_weight += weight
        
        self.health_score = weighted_score / total_weight if total_weight > 0 else 1.0
        self.last_health_check = time.time()
        
        return self.health_score


class RealTimeAdaptiveMonitor:
    """
    Real-time adaptive monitoring system with predictive capabilities.
    
    Monitors photonic neuromorphic systems with adaptive thresholds,
    predictive anomaly detection, and autonomous response capabilities.
    """
    
    def __init__(
        self,
        monitoring_interval: float = 0.001,  # 1ms monitoring
        prediction_horizon: float = 10.0,    # 10 second prediction
        adaptation_rate: float = 0.05
    ):
        self.monitoring_interval = monitoring_interval
        self.prediction_horizon = prediction_horizon
        self.adaptation_rate = adaptation_rate
        
        # System components
        self.components: Dict[str, SystemComponent] = {}
        self.global_metrics: Dict[str, MonitoringMetric] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Predictive models
        self.anomaly_detector = PredictiveAnomalyDetector()
        self.anomaly_detector.eval()
        self.model_trained = False
        self.training_data = deque(maxlen=10000)
        
        # Adaptive thresholds
        self.threshold_adaptation_enabled = True
        self.baseline_update_interval = 60.0  # 1 minute
        self.last_baseline_update = 0.0
        
        # Response system
        self.response_actions: Dict[str, Callable] = {}
        self.response_history: List[Dict[str, Any]] = []
        self.emergency_contacts = []
        
        # Logging and metrics
        self.logger = PhotonicLogger("AdaptiveMonitor")
        self.metrics_collector = MetricsCollector()
        
        # Performance tracking
        self.monitoring_stats = {
            "samples_processed": 0,
            "anomalies_detected": 0,
            "responses_triggered": 0,
            "false_positives": 0,
            "prediction_accuracy": 0.0
        }
        
        self.logger.info("Initialized real-time adaptive monitoring system")
    
    def register_component(
        self,
        component_name: str,
        component_type: str,
        dependencies: List[str] = None
    ) -> SystemComponent:
        """Register system component for monitoring."""
        component = SystemComponent(
            name=component_name,
            component_type=component_type,
            dependencies=dependencies or []
        )
        
        self.components[component_name] = component
        self.logger.info(f"Registered component: {component_name} ({component_type})")
        
        return component
    
    def add_metric(
        self,
        component_name: str,
        metric_name: str,
        initial_value: float = 0.0,
        warning_threshold: float = None,
        critical_threshold: float = None
    ) -> MonitoringMetric:
        """Add monitoring metric to component."""
        if component_name not in self.components:
            self.register_component(component_name, "unknown")
        
        metric = MonitoringMetric(
            name=metric_name,
            current_value=initial_value,
            baseline_value=initial_value,
            threshold_warning=warning_threshold or initial_value * 1.5,
            threshold_critical=critical_threshold or initial_value * 2.0,
            last_updated=time.time()
        )
        
        self.components[component_name].add_metric(metric)
        self.logger.debug(f"Added metric {metric_name} to {component_name}")
        
        return metric
    
    def update_metric(
        self,
        component_name: str,
        metric_name: str,
        value: float,
        timestamp: float = None
    ) -> None:
        """Update metric value."""
        if (component_name not in self.components or 
            metric_name not in self.components[component_name].metrics):
            return
        
        metric = self.components[component_name].metrics[metric_name]
        metric.update(value, timestamp)
        
        # Update component health score
        self.components[component_name].update_health_score()
        
        # Collect data for training
        if self.monitoring_active:
            self._collect_training_data(component_name, metric_name, value)
        
        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.record_metric(f"{component_name}_{metric_name}", value)
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Started real-time monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Stopped real-time monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            start_time = time.time()
            
            try:
                # Update adaptive thresholds
                if (time.time() - self.last_baseline_update > self.baseline_update_interval):
                    self._update_adaptive_thresholds()
                    self.last_baseline_update = time.time()
                
                # Check component health
                self._check_component_health()
                
                # Run predictive anomaly detection
                if self.model_trained:
                    self._run_predictive_analysis()
                
                # Update global system health
                self._update_global_health()
                
                # Autonomous responses
                self._check_autonomous_responses()
                
                self.monitoring_stats["samples_processed"] += 1
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
            
            # Maintain monitoring interval
            elapsed_time = time.time() - start_time
            sleep_time = max(0, self.monitoring_interval - elapsed_time)
            time.sleep(sleep_time)
    
    def _update_adaptive_thresholds(self) -> None:
        """Update adaptive thresholds based on system behavior."""
        if not self.threshold_adaptation_enabled:
            return
        
        for component in self.components.values():
            for metric in component.metrics.values():
                if len(metric.history) < 100:  # Need sufficient history
                    continue
                
                # Calculate statistical thresholds
                recent_values = [v for _, v in list(metric.history)[-100:]]
                mean_val = np.mean(recent_values)
                std_val = np.std(recent_values)
                
                # Adaptive threshold adjustment
                new_warning = mean_val + 2 * std_val
                new_critical = mean_val + 3 * std_val
                
                # Smooth adaptation to prevent oscillation
                metric.threshold_warning = (
                    self.adaptation_rate * new_warning + 
                    (1 - self.adaptation_rate) * metric.threshold_warning
                )
                metric.threshold_critical = (
                    self.adaptation_rate * new_critical + 
                    (1 - self.adaptation_rate) * metric.threshold_critical
                )
        
        self.logger.debug("Updated adaptive thresholds")
    
    def _check_component_health(self) -> None:
        """Check health of all components."""
        for component_name, component in self.components.items():
            old_health = component.health_score
            new_health = component.update_health_score()
            
            # Log significant health changes
            if abs(new_health - old_health) > 0.1:
                if new_health < old_health:
                    self.logger.warning(f"Component {component_name} health decreased: "
                                      f"{old_health:.2f} -> {new_health:.2f}")
                else:
                    self.logger.info(f"Component {component_name} health improved: "
                                   f"{old_health:.2f} -> {new_health:.2f}")
    
    def _run_predictive_analysis(self) -> None:
        """Run predictive anomaly detection."""
        if not self.model_trained or len(self.training_data) < 50:
            return
        
        try:
            # Prepare input data
            recent_data = list(self.training_data)[-50:]  # Last 50 samples
            input_tensor = torch.FloatTensor([
                [sample[2] for sample in recent_data[-10:]]  # Last 10 values
            ])
            
            # Run prediction
            with torch.no_grad():
                anomaly_score = self.anomaly_detector(input_tensor.unsqueeze(0)).item()
            
            # Check for anomalies
            if anomaly_score > 0.7:  # Anomaly threshold
                self.monitoring_stats["anomalies_detected"] += 1
                self._handle_predicted_anomaly(anomaly_score, recent_data[-1])
            
        except Exception as e:
            self.logger.error(f"Error in predictive analysis: {e}")
    
    def _update_global_health(self) -> None:
        """Update global system health score."""
        if not self.components:
            return
        
        # Calculate weighted global health
        total_health = 0.0
        total_weight = 0.0
        
        for component in self.components.values():
            weight = self._get_component_weight(component)
            total_health += component.health_score * weight
            total_weight += weight
        
        global_health = total_health / total_weight if total_weight > 0 else 1.0
        
        # Update global metric
        if "global_health" not in self.global_metrics:
            self.global_metrics["global_health"] = MonitoringMetric(
                name="global_health",
                current_value=global_health
            )
        
        self.global_metrics["global_health"].update(global_health)
        
        # Record global health
        if self.metrics_collector:
            self.metrics_collector.record_metric("global_health_score", global_health)
    
    def _get_component_weight(self, component: SystemComponent) -> float:
        """Get component weight for global health calculation."""
        # Weight based on component type and criticality
        weights = {
            "core": 1.0,
            "compute": 0.8,
            "network": 0.6,
            "storage": 0.4,
            "auxiliary": 0.2
        }
        
        return weights.get(component.component_type, 0.5)
    
    def _check_autonomous_responses(self) -> None:
        """Check and execute autonomous responses."""
        for component_name, component in self.components.items():
            if not component.auto_response_enabled:
                continue
            
            # Check if response is needed
            response_action = self._determine_response_action(component)
            
            if response_action != ResponseAction.NONE:
                self._execute_response(component_name, response_action)
    
    def _determine_response_action(self, component: SystemComponent) -> ResponseAction:
        """Determine appropriate response action for component."""
        if component.health_score < 0.2:
            return ResponseAction.EMERGENCY_SHUTDOWN
        elif component.health_score < 0.4:
            return ResponseAction.FAILOVER
        elif component.health_score < 0.6:
            return ResponseAction.RESOURCE_SCALE
        elif component.health_score < 0.8:
            return ResponseAction.PARAMETER_ADJUST
        else:
            return ResponseAction.NONE
    
    def _execute_response(self, component_name: str, action: ResponseAction) -> None:
        """Execute autonomous response action."""
        self.monitoring_stats["responses_triggered"] += 1
        
        response_record = {
            "timestamp": time.time(),
            "component": component_name,
            "action": action.value,
            "health_score": self.components[component_name].health_score,
            "success": False
        }
        
        try:
            if action == ResponseAction.PARAMETER_ADJUST:
                self._adjust_parameters(component_name)
            elif action == ResponseAction.LOAD_BALANCE:
                self._rebalance_load(component_name)
            elif action == ResponseAction.RESOURCE_SCALE:
                self._scale_resources(component_name)
            elif action == ResponseAction.FAILOVER:
                self._initiate_failover(component_name)
            elif action == ResponseAction.EMERGENCY_SHUTDOWN:
                self._emergency_shutdown(component_name)
            
            response_record["success"] = True
            self.logger.info(f"Executed {action.value} for {component_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute {action.value} for {component_name}: {e}")
        
        self.response_history.append(response_record)
    
    def _adjust_parameters(self, component_name: str) -> None:
        """Adjust component parameters for optimization."""
        # Implementation would depend on specific component type
        if component_name in self.response_actions:
            self.response_actions[component_name]("parameter_adjust")
    
    def _rebalance_load(self, component_name: str) -> None:
        """Rebalance load across system components."""
        if component_name in self.response_actions:
            self.response_actions[component_name]("load_balance")
    
    def _scale_resources(self, component_name: str) -> None:
        """Scale resources for component."""
        if component_name in self.response_actions:
            self.response_actions[component_name]("resource_scale")
    
    def _initiate_failover(self, component_name: str) -> None:
        """Initiate failover to backup component."""
        if component_name in self.response_actions:
            self.response_actions[component_name]("failover")
    
    def _emergency_shutdown(self, component_name: str) -> None:
        """Emergency shutdown of component."""
        self.logger.critical(f"Emergency shutdown initiated for {component_name}")
        if component_name in self.response_actions:
            self.response_actions[component_name]("emergency_shutdown")
    
    def _collect_training_data(
        self,
        component_name: str,
        metric_name: str,
        value: float
    ) -> None:
        """Collect data for training predictive models."""
        self.training_data.append((
            time.time(),
            f"{component_name}_{metric_name}",
            value
        ))
        
        # Train model when sufficient data available
        if len(self.training_data) >= 1000 and not self.model_trained:
            self._train_anomaly_detector()
    
    def _train_anomaly_detector(self) -> None:
        """Train predictive anomaly detector."""
        try:
            # Prepare training data
            data = np.array([sample[2] for sample in self.training_data])
            
            # Create sequences for LSTM training
            sequence_length = 10
            sequences = []
            labels = []
            
            for i in range(len(data) - sequence_length):
                seq = data[i:i+sequence_length]
                # Label as anomaly if value is > 2 std devs from mean
                label = float(abs(data[i+sequence_length] - np.mean(seq)) > 2 * np.std(seq))
                
                sequences.append(seq)
                labels.append(label)
            
            # Convert to tensors
            X = torch.FloatTensor(sequences).unsqueeze(-1)
            y = torch.FloatTensor(labels).unsqueeze(-1)
            
            # Training setup
            self.anomaly_detector.train()
            optimizer = torch.optim.Adam(self.anomaly_detector.parameters(), lr=0.001)
            criterion = nn.BCELoss()
            
            # Training loop
            for epoch in range(50):  # Quick training
                optimizer.zero_grad()
                predictions = self.anomaly_detector(X)
                loss = criterion(predictions, y)
                loss.backward()
                optimizer.step()
            
            self.anomaly_detector.eval()
            self.model_trained = True
            
            self.logger.info("Trained anomaly detection model")
            
        except Exception as e:
            self.logger.error(f"Failed to train anomaly detector: {e}")
    
    def _handle_predicted_anomaly(self, anomaly_score: float, last_sample: Tuple) -> None:
        """Handle predicted anomaly."""
        self.logger.warning(f"Predicted anomaly detected: score={anomaly_score:.3f}, "
                          f"metric={last_sample[1]}, value={last_sample[2]}")
        
        # Increase monitoring frequency temporarily
        original_interval = self.monitoring_interval
        self.monitoring_interval = original_interval * 0.1  # 10x faster monitoring
        
        # Schedule return to normal monitoring
        def restore_interval():
            time.sleep(30)  # Monitor intensively for 30 seconds
            self.monitoring_interval = original_interval
        
        threading.Thread(target=restore_interval, daemon=True).start()
    
    def register_response_action(
        self,
        component_name: str,
        response_function: Callable[[str], None]
    ) -> None:
        """Register custom response action for component."""
        self.response_actions[component_name] = response_function
        self.logger.debug(f"Registered response action for {component_name}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        component_status = {}
        for name, component in self.components.items():
            component_status[name] = {
                "health_score": component.health_score,
                "metrics": {
                    metric_name: {
                        "value": metric.current_value,
                        "baseline": metric.baseline_value,
                        "trend": metric.trend,
                        "status": metric.get_status().value
                    }
                    for metric_name, metric in component.metrics.items()
                }
            }
        
        global_health = (
            self.global_metrics["global_health"].current_value
            if "global_health" in self.global_metrics else 1.0
        )
        
        return {
            "global_health": global_health,
            "monitoring_active": self.monitoring_active,
            "model_trained": self.model_trained,
            "components": component_status,
            "monitoring_stats": self.monitoring_stats.copy(),
            "recent_responses": self.response_history[-10:]
        }
    
    def get_predictions(self, time_horizon: float = None) -> Dict[str, Dict[str, float]]:
        """Get predictions for metrics based on current trends."""
        if time_horizon is None:
            time_horizon = self.prediction_horizon
        
        predictions = {}
        
        for component_name, component in self.components.items():
            component_predictions = {}
            
            for metric_name, metric in component.metrics.items():
                predicted_value = metric.predict_future_value(time_horizon)
                component_predictions[metric_name] = predicted_value
            
            predictions[component_name] = component_predictions
        
        return predictions
    
    def optimize_monitoring_parameters(self) -> Dict[str, float]:
        """Optimize monitoring parameters based on historical performance."""
        optimization_results = {}
        
        # Analyze response effectiveness
        if len(self.response_history) > 10:
            successful_responses = [r for r in self.response_history if r["success"]]
            success_rate = len(successful_responses) / len(self.response_history)
            
            # Adjust adaptation rate based on success
            if success_rate > 0.8:
                self.adaptation_rate = min(0.1, self.adaptation_rate * 1.1)
            else:
                self.adaptation_rate = max(0.01, self.adaptation_rate * 0.9)
            
            optimization_results["adaptation_rate"] = self.adaptation_rate
            optimization_results["response_success_rate"] = success_rate
        
        # Optimize monitoring interval based on system activity
        if self.monitoring_stats["samples_processed"] > 1000:
            anomaly_rate = (
                self.monitoring_stats["anomalies_detected"] / 
                self.monitoring_stats["samples_processed"]
            )
            
            # Adjust monitoring frequency based on anomaly rate
            if anomaly_rate > 0.1:  # High anomaly rate
                self.monitoring_interval = max(0.0005, self.monitoring_interval * 0.9)
            elif anomaly_rate < 0.01:  # Low anomaly rate
                self.monitoring_interval = min(0.01, self.monitoring_interval * 1.1)
            
            optimization_results["monitoring_interval"] = self.monitoring_interval
            optimization_results["anomaly_rate"] = anomaly_rate
        
        self.logger.info(f"Optimized monitoring parameters: {optimization_results}")
        
        return optimization_results


def create_photonic_monitoring_demo(
    num_components: int = 5
) -> Tuple[RealTimeAdaptiveMonitor, Dict[str, Any]]:
    """Create demonstration of real-time adaptive monitoring."""
    
    # Create monitoring system
    monitor = RealTimeAdaptiveMonitor(
        monitoring_interval=0.01,  # 10ms for demo
        prediction_horizon=30.0,   # 30 second prediction
        adaptation_rate=0.1        # Faster adaptation for demo
    )
    
    # Register photonic system components
    components = {
        "quantum_processor": {
            "type": "core",
            "metrics": {
                "quantum_coherence": (0.95, 0.85, 0.75),  # value, warning, critical
                "entanglement_fidelity": (0.99, 0.90, 0.80),
                "decoherence_rate": (0.01, 0.05, 0.10),
                "gate_fidelity": (0.999, 0.995, 0.990)
            }
        },
        "photonic_network": {
            "type": "network",
            "metrics": {
                "optical_loss": (0.1, 0.3, 0.5),
                "crosstalk": (-30, -20, -15),
                "bandwidth_utilization": (0.6, 0.8, 0.9),
                "latency": (1e-9, 5e-9, 10e-9)
            }
        },
        "neural_processor": {
            "type": "compute",
            "metrics": {
                "spike_rate": (100e3, 500e3, 1e6),
                "processing_accuracy": (0.95, 0.85, 0.75),
                "energy_efficiency": (1e-12, 5e-12, 10e-12),
                "thermal_load": (300, 350, 400)
            }
        },
        "memory_system": {
            "type": "storage",
            "metrics": {
                "read_latency": (1e-9, 10e-9, 50e-9),
                "write_throughput": (1e9, 5e8, 1e8),
                "error_rate": (1e-12, 1e-9, 1e-6),
                "capacity_utilization": (0.7, 0.85, 0.95)
            }
        },
        "cooling_system": {
            "type": "auxiliary",
            "metrics": {
                "temperature": (4.0, 10.0, 20.0),  # Kelvin
                "cooling_power": (100, 200, 500),   # Watts
                "vibration": (1e-9, 1e-8, 1e-7),   # meters
                "stability": (0.99, 0.95, 0.90)
            }
        }
    }
    
    # Register components and metrics
    for comp_name, comp_config in components.items():
        component = monitor.register_component(comp_name, comp_config["type"])
        
        for metric_name, (init_val, warn_thresh, crit_thresh) in comp_config["metrics"].items():
            monitor.add_metric(
                comp_name,
                metric_name,
                initial_value=init_val,
                warning_threshold=warn_thresh,
                critical_threshold=crit_thresh
            )
    
    # Create demo response actions
    def create_response_action(component_name: str) -> Callable[[str], None]:
        def response_action(action_type: str) -> None:
            print(f"Executing {action_type} for {component_name}")
            # Simulate response delay
            time.sleep(0.1)
        return response_action
    
    for comp_name in components.keys():
        monitor.register_response_action(comp_name, create_response_action(comp_name))
    
    demo_config = {
        "components": list(components.keys()),
        "total_metrics": sum(len(comp["metrics"]) for comp in components.values()),
        "monitoring_interval": monitor.monitoring_interval,
        "prediction_horizon": monitor.prediction_horizon
    }
    
    return monitor, demo_config


async def run_monitoring_simulation(
    monitor: RealTimeAdaptiveMonitor,
    duration: float = 60.0,
    anomaly_probability: float = 0.05
) -> Dict[str, Any]:
    """Run monitoring simulation with synthetic data."""
    
    start_time = time.time()
    simulation_results = {
        "total_samples": 0,
        "anomalies_generated": 0,
        "health_trajectory": [],
        "component_performance": {},
        "response_actions": []
    }
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # Generate synthetic data for all components
            for component_name, component in monitor.components.items():
                for metric_name, metric in component.metrics.items():
                    
                    # Generate realistic data with trend and noise
                    base_value = metric.baseline_value
                    trend = np.sin((current_time - start_time) * 0.1) * 0.1 * base_value
                    noise = np.random.normal(0, 0.05 * base_value)
                    
                    # Occasionally inject anomalies
                    if np.random.random() < anomaly_probability:
                        anomaly = np.random.choice([-1, 1]) * 0.5 * base_value
                        simulation_results["anomalies_generated"] += 1
                    else:
                        anomaly = 0
                    
                    new_value = max(0, base_value + trend + noise + anomaly)
                    
                    # Update metric
                    monitor.update_metric(component_name, metric_name, new_value, current_time)
                    simulation_results["total_samples"] += 1
            
            # Record system status periodically
            if simulation_results["total_samples"] % 100 == 0:
                status = monitor.get_system_status()
                simulation_results["health_trajectory"].append({
                    "time": current_time - start_time,
                    "global_health": status["global_health"],
                    "monitoring_stats": status["monitoring_stats"].copy()
                })
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.01)
    
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
    
    # Collect final results
    final_status = monitor.get_system_status()
    simulation_results["final_status"] = final_status
    simulation_results["response_actions"] = monitor.response_history[-20:]
    
    return simulation_results


def validate_monitoring_effectiveness(
    monitor: RealTimeAdaptiveMonitor,
    test_duration: float = 30.0
) -> Dict[str, Any]:
    """Validate effectiveness of monitoring system."""
    
    validation_results = {
        "detection_accuracy": 0.0,
        "response_time": 0.0,
        "false_positive_rate": 0.0,
        "system_stability": 0.0
    }
    
    # Run validation simulation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        simulation_results = loop.run_until_complete(
            run_monitoring_simulation(
                monitor, 
                duration=test_duration,
                anomaly_probability=0.1  # Higher anomaly rate for testing
            )
        )
        
        # Calculate detection accuracy
        if simulation_results["anomalies_generated"] > 0:
            detected_anomalies = monitor.monitoring_stats["anomalies_detected"]
            validation_results["detection_accuracy"] = (
                detected_anomalies / simulation_results["anomalies_generated"]
            )
        
        # Calculate average response time
        response_times = []
        for response in monitor.response_history:
            # Simulate response time based on action type
            if response["action"] == "parameter_adjust":
                response_times.append(0.1)
            elif response["action"] == "failover":
                response_times.append(1.0)
            else:
                response_times.append(0.5)
        
        if response_times:
            validation_results["response_time"] = np.mean(response_times)
        
        # Calculate system stability (health variance)
        health_values = [h["global_health"] for h in simulation_results["health_trajectory"]]
        if health_values:
            validation_results["system_stability"] = 1.0 - np.std(health_values)
        
        # Estimate false positive rate (simplified)
        total_detections = monitor.monitoring_stats["anomalies_detected"]
        true_anomalies = simulation_results["anomalies_generated"]
        if total_detections > 0:
            false_positives = max(0, total_detections - true_anomalies)
            validation_results["false_positive_rate"] = false_positives / total_detections
    
    finally:
        loop.close()
    
    return validation_results