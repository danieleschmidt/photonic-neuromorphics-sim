"""
XR Reliability and Fault Tolerance for Photonic Neuromorphic Systems.

This module provides comprehensive reliability, fault tolerance, and self-healing
capabilities for XR agent mesh systems, ensuring robust operation in production
environments.
"""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import json
from collections import defaultdict, deque
import threading
import uuid

from .xr_agent_mesh import XRAgent, XRAgentMesh, XRMessage, XRCoordinate
from .monitoring import MetricsCollector
from .exceptions import ValidationError, OpticalModelError


class FailureType(Enum):
    """Types of failures in XR systems."""
    AGENT_DISCONNECT = "agent_disconnect"
    PROCESSING_ERROR = "processing_error"
    NETWORK_PARTITION = "network_partition"
    OPTICAL_DEGRADATION = "optical_degradation"
    MEMORY_OVERFLOW = "memory_overflow"
    LATENCY_VIOLATION = "latency_violation"
    ACCURACY_DEGRADATION = "accuracy_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class FailureEvent:
    """Represents a failure event in the XR system."""
    failure_id: str
    failure_type: FailureType
    affected_component: str
    severity: int  # 1-10, 10 being most severe
    timestamp: float
    details: Dict[str, Any]
    recovery_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[float] = None
    
    def duration(self) -> float:
        """Get failure duration."""
        end_time = self.resolution_time if self.resolved else time.time()
        return end_time - self.timestamp


@dataclass
class HealthMetrics:
    """Health metrics for system components."""
    component_id: str
    status: HealthStatus
    uptime: float
    error_rate: float
    performance_score: float
    resource_utilization: Dict[str, float]
    last_update: float
    alerts: List[str] = field(default_factory=list)
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return (self.status == HealthStatus.HEALTHY and 
                self.error_rate < 0.05 and 
                self.performance_score > 0.7)


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, 
                 recovery_timeout: float = 30.0,
                 half_open_max_calls: int = 3):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half_open
        self.half_open_calls = 0
        
        self._logger = logging.getLogger(__name__)
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                self.half_open_calls = 0
                self._logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        if self.state == "half_open":
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = "open"
                self._logger.warning("Circuit breaker returned to OPEN state")
                raise Exception("Circuit breaker is OPEN - service unavailable")
            
            self.half_open_calls += 1
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                self._logger.info("Circuit breaker returned to CLOSED state")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self._logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
            
            raise e
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time,
            'time_until_half_open': max(0, self.recovery_timeout - (time.time() - self.last_failure_time))
        }


class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_factor: float = 2.0):
        """Initialize retry manager."""
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        
        self._logger = logging.getLogger(__name__)
    
    async def retry_async(self, func: Callable, *args, **kwargs):
        """Retry async function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
                    self._logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    self._logger.error(f"All {self.max_retries + 1} attempts failed")
        
        raise last_exception
    
    def retry_sync(self, func: Callable, *args, **kwargs):
        """Retry sync function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
                    self._logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
                else:
                    self._logger.error(f"All {self.max_retries + 1} attempts failed")
        
        raise last_exception


class HealthMonitor:
    """Monitors health of XR system components."""
    
    def __init__(self, check_interval: float = 5.0):
        """Initialize health monitor."""
        self.check_interval = check_interval
        self.health_metrics: Dict[str, HealthMetrics] = {}
        self.failure_history: List[FailureEvent] = []
        self.alert_thresholds = {
            'error_rate': 0.1,
            'performance_score': 0.5,
            'memory_usage': 0.9,
            'cpu_usage': 0.8
        }
        
        self.is_monitoring = False
        self._logger = logging.getLogger(__name__)
        self._metrics_collector = None
        
    def set_metrics_collector(self, collector: MetricsCollector):
        """Set metrics collector."""
        self._metrics_collector = collector
    
    def register_component(self, component_id: str, component_type: str = "generic"):
        """Register component for health monitoring."""
        self.health_metrics[component_id] = HealthMetrics(
            component_id=component_id,
            status=HealthStatus.HEALTHY,
            uptime=0.0,
            error_rate=0.0,
            performance_score=1.0,
            resource_utilization={'memory': 0.0, 'cpu': 0.0},
            last_update=time.time()
        )
        
        self._logger.info(f"Registered {component_type} component: {component_id}")
    
    def update_component_health(self, component_id: str, metrics: Dict[str, Any]):
        """Update component health metrics."""
        if component_id not in self.health_metrics:
            self.register_component(component_id)
        
        health = self.health_metrics[component_id]
        current_time = time.time()
        
        # Update metrics
        health.uptime = metrics.get('uptime', health.uptime)
        health.error_rate = metrics.get('error_rate', health.error_rate)
        health.performance_score = metrics.get('performance_score', health.performance_score)
        health.resource_utilization.update(metrics.get('resource_utilization', {}))
        health.last_update = current_time
        
        # Determine health status
        old_status = health.status
        health.status = self._calculate_health_status(health)
        
        # Generate alerts if status changed
        if health.status != old_status:
            self._logger.info(f"Component {component_id} status changed: {old_status} -> {health.status}")
            
            if health.status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                self._trigger_alert(component_id, health)
        
        # Check for threshold violations
        self._check_alert_thresholds(component_id, health)
        
        if self._metrics_collector:
            self._metrics_collector.record_metric(f"health_score_{component_id}", health.performance_score)
            self._metrics_collector.record_metric(f"error_rate_{component_id}", health.error_rate)
    
    def record_failure(self, component_id: str, failure_type: FailureType, 
                      severity: int, details: Dict[str, Any] = None):
        """Record a failure event."""
        failure = FailureEvent(
            failure_id=str(uuid.uuid4()),
            failure_type=failure_type,
            affected_component=component_id,
            severity=severity,
            timestamp=time.time(),
            details=details or {}
        )
        
        self.failure_history.append(failure)
        
        # Keep failure history manageable
        if len(self.failure_history) > 1000:
            self.failure_history = self.failure_history[-800:]  # Keep most recent 800
        
        self._logger.error(f"Failure recorded: {failure_type.value} in {component_id} (severity: {severity})")
        
        if self._metrics_collector:
            self._metrics_collector.increment_counter(f"failures_{failure_type.value}")
            self._metrics_collector.record_metric("failure_severity", severity)
        
        return failure.failure_id
    
    def resolve_failure(self, failure_id: str, recovery_actions: List[str] = None):
        """Mark failure as resolved."""
        for failure in self.failure_history:
            if failure.failure_id == failure_id and not failure.resolved:
                failure.resolved = True
                failure.resolution_time = time.time()
                failure.recovery_actions = recovery_actions or []
                
                duration = failure.duration()
                self._logger.info(f"Failure {failure_id} resolved after {duration:.1f}s")
                
                if self._metrics_collector:
                    self._metrics_collector.record_metric("failure_resolution_time", duration)
                
                return True
        
        return False
    
    def get_component_health(self, component_id: str) -> Optional[HealthMetrics]:
        """Get health metrics for component."""
        return self.health_metrics.get(component_id)
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        if not self.health_metrics:
            return {'status': 'unknown', 'components': 0}
        
        status_counts = defaultdict(int)
        total_performance = 0.0
        total_error_rate = 0.0
        
        for health in self.health_metrics.values():
            status_counts[health.status.value] += 1
            total_performance += health.performance_score
            total_error_rate += health.error_rate
        
        component_count = len(self.health_metrics)
        avg_performance = total_performance / component_count
        avg_error_rate = total_error_rate / component_count
        
        # Determine overall system status
        if status_counts['failed'] > 0:
            overall_status = 'failed'
        elif status_counts['critical'] > 0:
            overall_status = 'critical'
        elif status_counts['degraded'] > component_count * 0.3:  # >30% degraded
            overall_status = 'degraded'
        elif status_counts['recovering'] > 0:
            overall_status = 'recovering'
        else:
            overall_status = 'healthy'
        
        recent_failures = [f for f in self.failure_history 
                          if time.time() - f.timestamp < 3600]  # Last hour
        
        return {
            'status': overall_status,
            'components': component_count,
            'healthy_components': status_counts['healthy'],
            'degraded_components': status_counts['degraded'],
            'critical_components': status_counts['critical'],
            'failed_components': status_counts['failed'],
            'avg_performance': avg_performance,
            'avg_error_rate': avg_error_rate,
            'recent_failures': len(recent_failures),
            'unresolved_failures': len([f for f in self.failure_history if not f.resolved])
        }
    
    async def start_monitoring(self):
        """Start health monitoring loop."""
        self.is_monitoring = True
        self._logger.info("Health monitoring started")
        
        asyncio.create_task(self._monitoring_loop())
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        self._logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # Check for stale components
                for component_id, health in self.health_metrics.items():
                    time_since_update = current_time - health.last_update
                    
                    if time_since_update > 60.0:  # No update for 1 minute
                        self._logger.warning(f"Component {component_id} hasn't updated in {time_since_update:.1f}s")
                        health.status = HealthStatus.DEGRADED
                        
                        if time_since_update > 300.0:  # No update for 5 minutes
                            health.status = HealthStatus.FAILED
                            self.record_failure(component_id, FailureType.AGENT_DISCONNECT, 7,
                                             {'reason': 'no_health_updates', 'duration': time_since_update})
                
                # Update system-wide metrics
                if self._metrics_collector:
                    summary = self.get_system_health_summary()
                    self._metrics_collector.record_metric("system_health_score", summary['avg_performance'])
                    self._metrics_collector.record_metric("system_error_rate", summary['avg_error_rate'])
                    self._metrics_collector.record_metric("healthy_components", summary['healthy_components'])
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self._logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval * 2)  # Longer sleep on error
    
    def _calculate_health_status(self, health: HealthMetrics) -> HealthStatus:
        """Calculate health status based on metrics."""
        error_rate = health.error_rate
        performance = health.performance_score
        memory_usage = health.resource_utilization.get('memory', 0.0)
        
        if error_rate > 0.5 or performance < 0.2 or memory_usage > 0.95:
            return HealthStatus.FAILED
        elif error_rate > 0.2 or performance < 0.4 or memory_usage > 0.9:
            return HealthStatus.CRITICAL
        elif error_rate > 0.1 or performance < 0.6 or memory_usage > 0.8:
            return HealthStatus.DEGRADED
        elif health.status == HealthStatus.FAILED and performance > 0.7:
            return HealthStatus.RECOVERING
        else:
            return HealthStatus.HEALTHY
    
    def _trigger_alert(self, component_id: str, health: HealthMetrics):
        """Trigger alert for component."""
        alert_msg = f"ALERT: Component {component_id} status is {health.status.value}"
        health.alerts.append(alert_msg)
        
        # Keep only recent alerts
        if len(health.alerts) > 10:
            health.alerts = health.alerts[-5:]
        
        self._logger.warning(alert_msg)
        
        if self._metrics_collector:
            self._metrics_collector.increment_counter(f"alerts_{health.status.value}")
    
    def _check_alert_thresholds(self, component_id: str, health: HealthMetrics):
        """Check if metrics exceed alert thresholds."""
        for metric, threshold in self.alert_thresholds.items():
            if metric == 'error_rate' and health.error_rate > threshold:
                self._trigger_alert(component_id, health)
            elif metric == 'performance_score' and health.performance_score < threshold:
                self._trigger_alert(component_id, health)
            elif metric in health.resource_utilization:
                if health.resource_utilization[metric] > threshold:
                    self._trigger_alert(component_id, health)


class SelfHealingManager:
    """Manages self-healing capabilities for XR systems."""
    
    def __init__(self, mesh: XRAgentMesh, health_monitor: HealthMonitor):
        """Initialize self-healing manager."""
        self.mesh = mesh
        self.health_monitor = health_monitor
        self.healing_strategies: Dict[FailureType, List[Callable]] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        
        self._setup_default_strategies()
        self._logger = logging.getLogger(__name__)
        self._metrics_collector = None
        
    def set_metrics_collector(self, collector: MetricsCollector):
        """Set metrics collector."""
        self._metrics_collector = collector
        
    def register_healing_strategy(self, failure_type: FailureType, strategy: Callable):
        """Register healing strategy for failure type."""
        if failure_type not in self.healing_strategies:
            self.healing_strategies[failure_type] = []
        
        self.healing_strategies[failure_type].append(strategy)
        self._logger.info(f"Registered healing strategy for {failure_type.value}")
    
    async def attempt_healing(self, failure: FailureEvent) -> bool:
        """Attempt to heal from failure."""
        self._logger.info(f"Attempting healing for failure {failure.failure_id}")
        
        if failure.failure_type not in self.healing_strategies:
            self._logger.warning(f"No healing strategies for {failure.failure_type.value}")
            return False
        
        strategies = self.healing_strategies[failure.failure_type]
        recovery_actions = []
        
        for strategy in strategies:
            try:
                success = await strategy(failure, self.mesh)
                action_name = strategy.__name__
                recovery_actions.append(action_name)
                
                if success:
                    self._logger.info(f"Healing successful with strategy: {action_name}")
                    
                    # Record recovery
                    self.recovery_history.append({
                        'failure_id': failure.failure_id,
                        'failure_type': failure.failure_type.value,
                        'recovery_strategy': action_name,
                        'timestamp': time.time(),
                        'success': True
                    })
                    
                    # Mark failure as resolved
                    self.health_monitor.resolve_failure(failure.failure_id, recovery_actions)
                    
                    if self._metrics_collector:
                        self._metrics_collector.increment_counter("successful_recoveries")
                    
                    return True
                    
            except Exception as e:
                self._logger.error(f"Healing strategy {strategy.__name__} failed: {e}")
                recovery_actions.append(f"{strategy.__name__}_failed")
        
        # All strategies failed
        self._logger.error(f"All healing strategies failed for failure {failure.failure_id}")
        
        self.recovery_history.append({
            'failure_id': failure.failure_id,
            'failure_type': failure.failure_type.value,
            'recovery_strategies': recovery_actions,
            'timestamp': time.time(),
            'success': False
        })
        
        if self._metrics_collector:
            self._metrics_collector.increment_counter("failed_recoveries")
        
        return False
    
    def _setup_default_strategies(self):
        """Setup default healing strategies."""
        self.register_healing_strategy(FailureType.AGENT_DISCONNECT, self._restart_agent)
        self.register_healing_strategy(FailureType.NETWORK_PARTITION, self._repair_network)
        self.register_healing_strategy(FailureType.MEMORY_OVERFLOW, self._cleanup_memory)
        self.register_healing_strategy(FailureType.PROCESSING_ERROR, self._reset_processor)
        self.register_healing_strategy(FailureType.LATENCY_VIOLATION, self._optimize_routing)
    
    async def _restart_agent(self, failure: FailureEvent, mesh: XRAgentMesh) -> bool:
        """Restart failed agent."""
        agent_id = failure.affected_component
        
        if agent_id in mesh.agents:
            agent = mesh.agents[agent_id]
            
            try:
                # Stop and restart agent
                await agent.stop()
                await asyncio.sleep(1.0)  # Brief pause
                await agent.start()
                
                self._logger.info(f"Successfully restarted agent {agent_id}")
                return True
                
            except Exception as e:
                self._logger.error(f"Failed to restart agent {agent_id}: {e}")
                return False
        
        return False
    
    async def _repair_network(self, failure: FailureEvent, mesh: XRAgentMesh) -> bool:
        """Repair network partition."""
        try:
            # Re-establish connections based on proximity
            mesh.auto_connect_by_proximity(max_distance=15.0)  # Increase connection radius
            
            # Verify connectivity
            active_agents = [agent_id for agent_id, agent in mesh.agents.items() if agent.is_active]
            
            if len(active_agents) > 1:
                self._logger.info("Network partition repaired")
                return True
            
        except Exception as e:
            self._logger.error(f"Network repair failed: {e}")
        
        return False
    
    async def _cleanup_memory(self, failure: FailureEvent, mesh: XRAgentMesh) -> bool:
        """Cleanup memory overflow."""
        agent_id = failure.affected_component
        
        if agent_id in mesh.agents:
            agent = mesh.agents[agent_id]
            
            try:
                # Clear message queue
                while not agent.message_queue.empty():
                    try:
                        agent.message_queue.get_nowait()
                    except:
                        break
                
                # Clear performance history
                agent.performance_metrics = {
                    'messages_processed': 0,
                    'avg_processing_time': 0.0,
                    'total_energy_consumed': 0.0,
                    'error_count': 0
                }
                
                self._logger.info(f"Memory cleanup completed for agent {agent_id}")
                return True
                
            except Exception as e:
                self._logger.error(f"Memory cleanup failed for {agent_id}: {e}")
        
        return False
    
    async def _reset_processor(self, failure: FailureEvent, mesh: XRAgentMesh) -> bool:
        """Reset processing components."""
        agent_id = failure.affected_component
        
        if agent_id in mesh.agents:
            agent = mesh.agents[agent_id]
            
            try:
                # Reset processor state
                if hasattr(agent, 'processor'):
                    agent.processor.processing_time = 0.0
                    agent.processor.energy_consumption = 0.0
                
                self._logger.info(f"Processor reset completed for agent {agent_id}")
                return True
                
            except Exception as e:
                self._logger.error(f"Processor reset failed for {agent_id}: {e}")
        
        return False
    
    async def _optimize_routing(self, failure: FailureEvent, mesh: XRAgentMesh) -> bool:
        """Optimize routing to reduce latency."""
        try:
            # Rebuild routing table based on current network state
            mesh.message_routing_table.clear()
            
            # Update topology to prefer low-latency connections
            for agent_id, neighbors in mesh.topology.items():
                if agent_id in mesh.agents and mesh.agents[agent_id].is_active:
                    # Sort neighbors by distance (proxy for latency)
                    if neighbors:
                        agent_pos = mesh.agents[agent_id].position
                        sorted_neighbors = sorted(neighbors, 
                                                key=lambda n: agent_pos.distance_to(mesh.agents[n].position) 
                                                if n in mesh.agents else float('inf'))
                        mesh.topology[agent_id] = sorted_neighbors[:3]  # Keep only closest 3
            
            self._logger.info("Routing optimization completed")
            return True
            
        except Exception as e:
            self._logger.error(f"Routing optimization failed: {e}")
            return False


class ReliableXRAgent(XRAgent):
    """XR Agent with enhanced reliability features."""
    
    def __init__(self, agent_id: str, agent_type, position: XRCoordinate,
                 processing_capability=None):
        """Initialize reliable XR agent."""
        super().__init__(agent_id, agent_type, position, processing_capability)
        
        # Reliability components
        self.circuit_breaker = CircuitBreaker()
        self.retry_manager = RetryManager()
        self.health_monitor = None
        self.last_health_update = time.time()
        self.health_check_interval = 5.0
        
        # Enhanced metrics
        self.reliability_metrics = {
            'circuit_breaker_trips': 0,
            'retry_attempts': 0,
            'successful_recoveries': 0,
            'health_checks': 0
        }
    
    def set_health_monitor(self, health_monitor: HealthMonitor):
        """Set health monitor for this agent."""
        self.health_monitor = health_monitor
        self.health_monitor.register_component(self.agent_id, "xr_agent")
    
    async def reliable_send_message(self, message: XRMessage):
        """Send message with reliability features."""
        try:
            await self.circuit_breaker.call(self.send_message, message)
        except Exception as e:
            self.reliability_metrics['circuit_breaker_trips'] += 1
            self._logger.error(f"Circuit breaker prevented message send: {e}")
            
            if self.health_monitor:
                self.health_monitor.record_failure(
                    self.agent_id, FailureType.PROCESSING_ERROR, 5,
                    {'operation': 'send_message', 'error': str(e)}
                )
            raise
    
    async def reliable_receive_message(self, message: XRMessage):
        """Receive message with retry capability."""
        try:
            await self.retry_manager.retry_async(self.receive_message, message)
            self.reliability_metrics['successful_recoveries'] += 1
        except Exception as e:
            self.reliability_metrics['retry_attempts'] += 1
            self._logger.error(f"Failed to receive message after retries: {e}")
            
            if self.health_monitor:
                self.health_monitor.record_failure(
                    self.agent_id, FailureType.PROCESSING_ERROR, 4,
                    {'operation': 'receive_message', 'error': str(e)}
                )
            raise
    
    async def start(self):
        """Start agent with health monitoring."""
        await super().start()
        
        # Start health monitoring
        if self.health_monitor:
            asyncio.create_task(self._health_monitoring_loop())
    
    async def _health_monitoring_loop(self):
        """Monitor and report agent health."""
        while self.is_active:
            try:
                current_time = time.time()
                
                # Calculate health metrics
                error_rate = (self.performance_metrics['error_count'] / 
                             max(self.performance_metrics['messages_processed'], 1))
                
                # Performance score based on processing efficiency
                if self.performance_metrics['messages_processed'] > 0:
                    avg_processing_time = (self.performance_metrics['avg_processing_time'] / 
                                         self.performance_metrics['messages_processed'])
                    performance_score = max(0.0, 1.0 - min(avg_processing_time * 1000, 1.0))  # Normalize to 0-1
                else:
                    performance_score = 1.0
                
                # Resource utilization (simulated)
                memory_usage = len(str(self.performance_metrics)) / 1000.0  # Rough estimate
                cpu_usage = min(error_rate * 5, 1.0)  # Higher error rate = higher CPU usage
                
                # Update health monitor
                self.health_monitor.update_component_health(self.agent_id, {
                    'uptime': current_time - self.last_health_update,
                    'error_rate': error_rate,
                    'performance_score': performance_score,
                    'resource_utilization': {
                        'memory': memory_usage,
                        'cpu': cpu_usage
                    }
                })
                
                self.reliability_metrics['health_checks'] += 1
                self.last_health_update = current_time
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self._logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval * 2)
    
    def get_reliability_summary(self) -> Dict[str, Any]:
        """Get reliability metrics summary."""
        circuit_state = self.circuit_breaker.get_state()
        
        return {
            'agent_id': self.agent_id,
            'circuit_breaker_state': circuit_state['state'],
            'circuit_breaker_failures': circuit_state['failure_count'],
            'reliability_metrics': self.reliability_metrics,
            'performance_metrics': self.performance_metrics,
            'health_status': (self.health_monitor.get_component_health(self.agent_id).status.value 
                            if self.health_monitor else 'unknown')
        }