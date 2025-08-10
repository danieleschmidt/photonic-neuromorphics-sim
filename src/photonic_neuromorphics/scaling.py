"""
Advanced Scaling and Performance Optimization for Photonic Neuromorphic Systems.

This module implements comprehensive scaling capabilities including auto-scaling,
load balancing, distributed processing, and performance optimization for
production-ready photonic neural networks.
"""

import time
import threading
import multiprocessing
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import lru_cache, wraps
import psutil
import gc

from .core import PhotonicSNN
from .monitoring import MetricsCollector, PerformanceProfiler
from .exceptions import PhotonicNeuromorphicError, handle_exception_with_recovery
from .optimization import AdaptiveCache, MemoryPool, BatchProcessor


class ScalingStrategy(Enum):
    """Scaling strategies for photonic systems."""
    HORIZONTAL = "horizontal"    # Scale out (more instances)
    VERTICAL = "vertical"        # Scale up (more resources per instance)
    HYBRID = "hybrid"           # Combination of both
    ELASTIC = "elastic"         # Auto-scaling based on demand


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_AWARE = "resource_aware"
    OPTICAL_POWER_AWARE = "optical_power_aware"


@dataclass
class ScalingConfig:
    """Configuration for scaling system."""
    strategy: ScalingStrategy = ScalingStrategy.HYBRID
    min_instances: int = 1
    max_instances: int = 16
    target_cpu_utilization: float = 70.0  # Percent
    target_memory_utilization: float = 80.0  # Percent
    target_optical_utilization: float = 60.0  # Percent
    scale_up_threshold: float = 80.0  # Percent
    scale_down_threshold: float = 30.0  # Percent
    scale_up_cooldown: float = 300.0  # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes
    load_balancing: LoadBalancingAlgorithm = LoadBalancingAlgorithm.RESOURCE_AWARE
    enable_auto_scaling: bool = True
    enable_predictive_scaling: bool = True


@dataclass
class InstanceMetrics:
    """Metrics for a scaling instance."""
    instance_id: str
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    optical_power_usage: float = 0.0
    active_connections: int = 0
    response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    last_update: float = field(default_factory=time.time)
    
    def is_healthy(self) -> bool:
        """Check if instance is healthy."""
        return (
            self.cpu_utilization < 95.0 and
            self.memory_utilization < 90.0 and
            self.error_rate < 5.0 and
            time.time() - self.last_update < 60.0  # Updated within last minute
        )
    
    def get_load_score(self) -> float:
        """Calculate load score for load balancing."""
        # Weighted combination of utilization metrics
        cpu_weight = 0.4
        memory_weight = 0.3
        optical_weight = 0.2
        response_weight = 0.1
        
        normalized_response = min(self.response_time / 1000.0, 1.0)  # Normalize to 1s
        
        load_score = (
            cpu_weight * (self.cpu_utilization / 100.0) +
            memory_weight * (self.memory_utilization / 100.0) +
            optical_weight * (self.optical_power_usage / 100.0) +
            response_weight * normalized_response
        )
        
        return min(load_score, 1.0)


class PhotonicLoadBalancer:
    """Advanced load balancer for photonic neuromorphic systems."""
    
    def __init__(
        self,
        algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.RESOURCE_AWARE,
        sticky_sessions: bool = False,
        health_check_interval: float = 30.0
    ):
        self.algorithm = algorithm
        self.sticky_sessions = sticky_sessions
        self.health_check_interval = health_check_interval
        
        # Instance tracking
        self.instances: Dict[str, InstanceMetrics] = {}
        self.instance_weights: Dict[str, float] = {}
        self.session_affinity: Dict[str, str] = {}  # session_id -> instance_id
        
        # Round-robin state
        self.round_robin_index = 0
        
        # Health monitoring
        self.health_monitor_active = False
        self.health_monitor_thread: Optional[threading.Thread] = None
        
        self.logger = logging.getLogger(__name__)
    
    def register_instance(self, instance_id: str, weight: float = 1.0) -> None:
        """Register a new instance for load balancing."""
        self.instances[instance_id] = InstanceMetrics(instance_id=instance_id)
        self.instance_weights[instance_id] = weight
        
        self.logger.info(f"Registered instance: {instance_id} (weight: {weight})")
    
    def unregister_instance(self, instance_id: str) -> None:
        """Unregister an instance."""
        if instance_id in self.instances:
            del self.instances[instance_id]
            del self.instance_weights[instance_id]
            
            # Remove session affinities
            sessions_to_remove = [
                session for session, instance in self.session_affinity.items()
                if instance == instance_id
            ]
            for session in sessions_to_remove:
                del self.session_affinity[session]
            
            self.logger.info(f"Unregistered instance: {instance_id}")
    
    def update_instance_metrics(self, instance_id: str, metrics: Dict[str, float]) -> None:
        """Update metrics for an instance."""
        if instance_id not in self.instances:
            return
        
        instance = self.instances[instance_id]
        instance.cpu_utilization = metrics.get('cpu_utilization', 0.0)
        instance.memory_utilization = metrics.get('memory_utilization', 0.0)
        instance.optical_power_usage = metrics.get('optical_power_usage', 0.0)
        instance.active_connections = int(metrics.get('active_connections', 0))
        instance.response_time = metrics.get('response_time', 0.0)
        instance.throughput = metrics.get('throughput', 0.0)
        instance.error_rate = metrics.get('error_rate', 0.0)
        instance.last_update = time.time()
    
    def select_instance(self, session_id: Optional[str] = None) -> Optional[str]:
        """Select best instance for request based on load balancing algorithm."""
        healthy_instances = [
            instance_id for instance_id, instance in self.instances.items()
            if instance.is_healthy()
        ]
        
        if not healthy_instances:
            self.logger.warning("No healthy instances available")
            return None
        
        # Handle sticky sessions
        if session_id and self.sticky_sessions:
            if session_id in self.session_affinity:
                preferred_instance = self.session_affinity[session_id]
                if preferred_instance in healthy_instances:
                    return preferred_instance
        
        # Select instance based on algorithm
        if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            selected = self._round_robin_selection(healthy_instances)
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            selected = self._weighted_round_robin_selection(healthy_instances)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            selected = self._least_connections_selection(healthy_instances)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            selected = self._least_response_time_selection(healthy_instances)
        elif self.algorithm == LoadBalancingAlgorithm.RESOURCE_AWARE:
            selected = self._resource_aware_selection(healthy_instances)
        elif self.algorithm == LoadBalancingAlgorithm.OPTICAL_POWER_AWARE:
            selected = self._optical_power_aware_selection(healthy_instances)
        else:
            selected = healthy_instances[0]  # Default to first healthy
        
        # Update session affinity
        if session_id and selected and self.sticky_sessions:
            self.session_affinity[session_id] = selected
        
        return selected
    
    def _round_robin_selection(self, instances: List[str]) -> str:
        """Simple round-robin selection."""
        if not instances:
            return None
        
        selected = instances[self.round_robin_index % len(instances)]
        self.round_robin_index += 1
        return selected
    
    def _weighted_round_robin_selection(self, instances: List[str]) -> str:
        """Weighted round-robin selection."""
        if not instances:
            return None
        
        # Create weighted list
        weighted_instances = []
        for instance_id in instances:
            weight = self.instance_weights.get(instance_id, 1.0)
            weighted_instances.extend([instance_id] * int(weight * 10))
        
        if not weighted_instances:
            return instances[0]
        
        selected = weighted_instances[self.round_robin_index % len(weighted_instances)]
        self.round_robin_index += 1
        return selected
    
    def _least_connections_selection(self, instances: List[str]) -> str:
        """Select instance with least active connections."""
        if not instances:
            return None
        
        return min(instances, key=lambda x: self.instances[x].active_connections)
    
    def _least_response_time_selection(self, instances: List[str]) -> str:
        """Select instance with least response time."""
        if not instances:
            return None
        
        return min(instances, key=lambda x: self.instances[x].response_time)
    
    def _resource_aware_selection(self, instances: List[str]) -> str:
        """Select instance based on overall resource utilization."""
        if not instances:
            return None
        
        return min(instances, key=lambda x: self.instances[x].get_load_score())
    
    def _optical_power_aware_selection(self, instances: List[str]) -> str:
        """Select instance based on optical power usage."""
        if not instances:
            return None
        
        return min(instances, key=lambda x: self.instances[x].optical_power_usage)
    
    def get_load_distribution(self) -> Dict[str, float]:
        """Get current load distribution across instances."""
        total_load = sum(instance.get_load_score() for instance in self.instances.values())
        
        if total_load == 0:
            return {instance_id: 0.0 for instance_id in self.instances}
        
        return {
            instance_id: instance.get_load_score() / total_load
            for instance_id, instance in self.instances.items()
        }
    
    def start_health_monitoring(self) -> None:
        """Start health monitoring for instances."""
        if self.health_monitor_active:
            return
        
        self.health_monitor_active = True
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True
        )
        self.health_monitor_thread.start()
        
        self.logger.info("Load balancer health monitoring started")
    
    def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        self.health_monitor_active = False
        
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            self.health_monitor_thread.join(timeout=5.0)
    
    def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        while self.health_monitor_active:
            try:
                # Check instance health
                unhealthy_instances = []
                for instance_id, instance in self.instances.items():
                    if not instance.is_healthy():
                        unhealthy_instances.append(instance_id)
                        self.logger.warning(f"Instance {instance_id} is unhealthy")
                
                # Log load distribution
                load_dist = self.get_load_distribution()
                self.logger.debug(f"Load distribution: {load_dist}")
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(5.0)


class AutoScaler:
    """Intelligent auto-scaling for photonic systems."""
    
    def __init__(
        self,
        config: ScalingConfig,
        load_balancer: PhotonicLoadBalancer,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.config = config
        self.load_balancer = load_balancer
        self.metrics_collector = metrics_collector
        
        # Scaling state
        self.current_instances = config.min_instances
        self.last_scale_up_time = 0.0
        self.last_scale_down_time = 0.0
        
        # Instance lifecycle callbacks
        self.scale_up_callback: Optional[Callable[[int], bool]] = None
        self.scale_down_callback: Optional[Callable[[List[str]], bool]] = None
        
        # Predictive scaling
        self.demand_history: List[Tuple[float, float]] = []  # (timestamp, demand)
        self.prediction_window = 1800.0  # 30 minutes
        
        # Monitoring
        self.scaling_active = False
        self.scaling_thread: Optional[threading.Thread] = None
        self.scaling_interval = 60.0  # 1 minute
        
        self.logger = logging.getLogger(__name__)
    
    def set_scale_up_callback(self, callback: Callable[[int], bool]) -> None:
        """Set callback for scaling up instances."""
        self.scale_up_callback = callback
    
    def set_scale_down_callback(self, callback: Callable[[List[str]], bool]) -> None:
        """Set callback for scaling down instances."""
        self.scale_down_callback = callback
    
    def start_auto_scaling(self) -> None:
        """Start auto-scaling monitoring."""
        if not self.config.enable_auto_scaling:
            return
        
        if self.scaling_active:
            return
        
        self.scaling_active = True
        self.scaling_thread = threading.Thread(
            target=self._scaling_loop,
            daemon=True
        )
        self.scaling_thread.start()
        
        self.logger.info("Auto-scaling started")
    
    def stop_auto_scaling(self) -> None:
        """Stop auto-scaling."""
        self.scaling_active = False
        
        if self.scaling_thread and self.scaling_thread.is_alive():
            self.scaling_thread.join(timeout=5.0)
        
        self.logger.info("Auto-scaling stopped")
    
    def _scaling_loop(self) -> None:
        """Main scaling loop."""
        while self.scaling_active:
            try:
                # Collect metrics
                metrics = self._collect_scaling_metrics()
                
                # Make scaling decision
                scaling_decision = self._make_scaling_decision(metrics)
                
                # Execute scaling action
                if scaling_decision['action'] == 'scale_up':
                    self._execute_scale_up(scaling_decision['target_instances'])
                elif scaling_decision['action'] == 'scale_down':
                    self._execute_scale_down(scaling_decision['instances_to_remove'])
                
                # Record demand for predictive scaling
                if self.config.enable_predictive_scaling:
                    demand = self._calculate_current_demand(metrics)
                    self.demand_history.append((time.time(), demand))
                    
                    # Cleanup old history
                    cutoff_time = time.time() - self.prediction_window
                    self.demand_history = [
                        (t, d) for t, d in self.demand_history if t >= cutoff_time
                    ]
                
                # Record metrics
                if self.metrics_collector:
                    self.metrics_collector.record_metric("current_instances", self.current_instances)
                    self.metrics_collector.record_metric("average_cpu_utilization", metrics.get('avg_cpu', 0))
                    self.metrics_collector.record_metric("average_memory_utilization", metrics.get('avg_memory', 0))
                
                time.sleep(self.scaling_interval)
                
            except Exception as e:
                self.logger.error(f"Scaling loop error: {e}")
                time.sleep(10.0)  # Longer pause on error
    
    def _collect_scaling_metrics(self) -> Dict[str, float]:
        """Collect metrics for scaling decisions."""
        instances = self.load_balancer.instances
        
        if not instances:
            return {'avg_cpu': 0, 'avg_memory': 0, 'avg_optical': 0, 'healthy_instances': 0}
        
        healthy_instances = [i for i in instances.values() if i.is_healthy()]
        
        if not healthy_instances:
            return {'avg_cpu': 100, 'avg_memory': 100, 'avg_optical': 100, 'healthy_instances': 0}
        
        avg_cpu = np.mean([i.cpu_utilization for i in healthy_instances])
        avg_memory = np.mean([i.memory_utilization for i in healthy_instances])
        avg_optical = np.mean([i.optical_power_usage for i in healthy_instances])
        avg_response_time = np.mean([i.response_time for i in healthy_instances])
        total_throughput = sum(i.throughput for i in healthy_instances)
        
        return {
            'avg_cpu': avg_cpu,
            'avg_memory': avg_memory,
            'avg_optical': avg_optical,
            'avg_response_time': avg_response_time,
            'total_throughput': total_throughput,
            'healthy_instances': len(healthy_instances),
            'total_instances': len(instances)
        }
    
    def _make_scaling_decision(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Make intelligent scaling decision."""
        decision = {'action': 'no_change', 'reason': ''}
        
        # Check if we have minimum instances
        if metrics['healthy_instances'] < self.config.min_instances:
            decision = {
                'action': 'scale_up',
                'target_instances': self.config.min_instances,
                'reason': 'Below minimum instance count'
            }
        
        # Check for scale up conditions
        elif self._should_scale_up(metrics):
            target_instances = min(
                self.current_instances + 1,
                self.config.max_instances
            )
            decision = {
                'action': 'scale_up',
                'target_instances': target_instances,
                'reason': self._get_scale_up_reason(metrics)
            }
        
        # Check for scale down conditions
        elif self._should_scale_down(metrics):
            instances_to_remove = self._select_instances_to_remove()
            decision = {
                'action': 'scale_down',
                'instances_to_remove': instances_to_remove,
                'reason': self._get_scale_down_reason(metrics)
            }
        
        # Predictive scaling
        if self.config.enable_predictive_scaling and decision['action'] == 'no_change':
            predictive_decision = self._predictive_scaling_decision(metrics)
            if predictive_decision['action'] != 'no_change':
                decision = predictive_decision
        
        return decision
    
    def _should_scale_up(self, metrics: Dict[str, float]) -> bool:
        """Check if system should scale up."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scale_up_time < self.config.scale_up_cooldown:
            return False
        
        # Check if at maximum instances
        if self.current_instances >= self.config.max_instances:
            return False
        
        # Check utilization thresholds
        scale_up_reasons = []
        
        if metrics['avg_cpu'] > self.config.scale_up_threshold:
            scale_up_reasons.append('high_cpu')
        
        if metrics['avg_memory'] > self.config.scale_up_threshold:
            scale_up_reasons.append('high_memory')
        
        if metrics['avg_optical'] > self.config.target_optical_utilization * 1.2:
            scale_up_reasons.append('high_optical_usage')
        
        if metrics['avg_response_time'] > 1000.0:  # 1 second threshold
            scale_up_reasons.append('high_response_time')
        
        return len(scale_up_reasons) > 0
    
    def _should_scale_down(self, metrics: Dict[str, float]) -> bool:
        """Check if system should scale down."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scale_down_time < self.config.scale_down_cooldown:
            return False
        
        # Check if at minimum instances
        if self.current_instances <= self.config.min_instances:
            return False
        
        # Check utilization thresholds (all must be low)
        return (
            metrics['avg_cpu'] < self.config.scale_down_threshold and
            metrics['avg_memory'] < self.config.scale_down_threshold and
            metrics['avg_optical'] < self.config.target_optical_utilization * 0.5
        )
    
    def _get_scale_up_reason(self, metrics: Dict[str, float]) -> str:
        """Get reason for scaling up."""
        reasons = []
        
        if metrics['avg_cpu'] > self.config.scale_up_threshold:
            reasons.append(f"CPU: {metrics['avg_cpu']:.1f}%")
        
        if metrics['avg_memory'] > self.config.scale_up_threshold:
            reasons.append(f"Memory: {metrics['avg_memory']:.1f}%")
        
        if metrics['avg_optical'] > self.config.target_optical_utilization * 1.2:
            reasons.append(f"Optical: {metrics['avg_optical']:.1f}%")
        
        return "High utilization: " + ", ".join(reasons)
    
    def _get_scale_down_reason(self, metrics: Dict[str, float]) -> str:
        """Get reason for scaling down."""
        return f"Low utilization: CPU {metrics['avg_cpu']:.1f}%, Memory {metrics['avg_memory']:.1f}%"
    
    def _select_instances_to_remove(self) -> List[str]:
        """Select instances to remove when scaling down."""
        instances = self.load_balancer.instances
        
        # Select instance with lowest load
        if instances:
            instance_loads = [
                (instance_id, instance.get_load_score())
                for instance_id, instance in instances.items()
            ]
            instance_loads.sort(key=lambda x: x[1])  # Sort by load score
            
            # Remove instance with lowest load
            return [instance_loads[0][0]]
        
        return []
    
    def _calculate_current_demand(self, metrics: Dict[str, float]) -> float:
        """Calculate current system demand."""
        # Weighted combination of utilization metrics
        demand = (
            0.4 * metrics['avg_cpu'] / 100.0 +
            0.3 * metrics['avg_memory'] / 100.0 +
            0.2 * metrics['avg_optical'] / 100.0 +
            0.1 * min(metrics['avg_response_time'] / 1000.0, 1.0)
        )
        
        return min(demand, 1.0)
    
    def _predictive_scaling_decision(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Make predictive scaling decision based on demand trends."""
        if len(self.demand_history) < 10:  # Need sufficient history
            return {'action': 'no_change', 'reason': 'Insufficient demand history'}
        
        # Simple trend analysis
        recent_demands = [d for t, d in self.demand_history[-10:]]
        demand_trend = np.polyfit(range(len(recent_demands)), recent_demands, 1)[0]
        
        current_demand = self._calculate_current_demand(current_metrics)
        
        # Predict demand in next scaling interval
        predicted_demand = current_demand + demand_trend * (self.scaling_interval / 60.0)
        
        # Make decision based on predicted demand
        if predicted_demand > 0.8 and self.current_instances < self.config.max_instances:
            return {
                'action': 'scale_up',
                'target_instances': self.current_instances + 1,
                'reason': f'Predicted demand increase: {predicted_demand:.2f}'
            }
        elif predicted_demand < 0.3 and self.current_instances > self.config.min_instances:
            instances_to_remove = self._select_instances_to_remove()
            return {
                'action': 'scale_down',
                'instances_to_remove': instances_to_remove,
                'reason': f'Predicted demand decrease: {predicted_demand:.2f}'
            }
        
        return {'action': 'no_change', 'reason': f'Predicted demand stable: {predicted_demand:.2f}'}
    
    def _execute_scale_up(self, target_instances: int) -> bool:
        """Execute scale up action."""
        if target_instances <= self.current_instances:
            return True
        
        instances_to_add = target_instances - self.current_instances
        
        self.logger.info(f"Scaling up: adding {instances_to_add} instances")
        
        if self.scale_up_callback:
            success = self.scale_up_callback(instances_to_add)
            if success:
                self.current_instances = target_instances
                self.last_scale_up_time = time.time()
                
                if self.metrics_collector:
                    self.metrics_collector.increment_counter("scale_up_events")
                
                return True
            else:
                self.logger.error("Scale up callback failed")
                return False
        else:
            self.logger.warning("No scale up callback configured")
            return False
    
    def _execute_scale_down(self, instances_to_remove: List[str]) -> bool:
        """Execute scale down action."""
        if not instances_to_remove:
            return True
        
        self.logger.info(f"Scaling down: removing {len(instances_to_remove)} instances")
        
        if self.scale_down_callback:
            success = self.scale_down_callback(instances_to_remove)
            if success:
                self.current_instances -= len(instances_to_remove)
                self.last_scale_down_time = time.time()
                
                if self.metrics_collector:
                    self.metrics_collector.increment_counter("scale_down_events")
                
                return True
            else:
                self.logger.error("Scale down callback failed")
                return False
        else:
            self.logger.warning("No scale down callback configured")
            return False
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics."""
        return {
            'current_instances': self.current_instances,
            'min_instances': self.config.min_instances,
            'max_instances': self.config.max_instances,
            'scaling_strategy': self.config.strategy.value,
            'auto_scaling_enabled': self.config.enable_auto_scaling,
            'predictive_scaling_enabled': self.config.enable_predictive_scaling,
            'last_scale_up': self.last_scale_up_time,
            'last_scale_down': self.last_scale_down_time,
            'demand_history_length': len(self.demand_history),
            'load_balancing_algorithm': self.load_balancer.algorithm.value
        }


class DistributedPhotonics:
    """Distributed processing for photonic neuromorphic systems."""
    
    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        backend: str = "nccl",
        init_method: str = "env://"
    ):
        self.world_size = world_size
        self.rank = rank
        self.backend = backend
        self.init_method = init_method
        
        self.is_distributed = world_size > 1
        self.is_main_process = rank == 0
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize distributed processing if needed
        if self.is_distributed:
            self._initialize_distributed()
    
    def _initialize_distributed(self) -> None:
        """Initialize distributed processing."""
        try:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.backend,
                    init_method=self.init_method,
                    world_size=self.world_size,
                    rank=self.rank
                )
                
                self.logger.info(f"Initialized distributed processing: rank {self.rank}/{self.world_size}")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed processing: {e}")
            self.is_distributed = False
    
    def distribute_model(self, model: PhotonicSNN) -> PhotonicSNN:
        """Distribute model across processes."""
        if not self.is_distributed:
            return model
        
        try:
            # Wrap model for distributed training
            if hasattr(torch.nn.parallel, 'DistributedDataParallel'):
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.rank] if torch.cuda.is_available() else None
                )
                
                self.logger.info("Model wrapped for distributed processing")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to distribute model: {e}")
            return model
    
    def all_reduce(self, tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
        """Perform all-reduce operation across processes."""
        if not self.is_distributed:
            return tensor
        
        try:
            if op == "mean":
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor /= self.world_size
            elif op == "sum":
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            elif op == "max":
                dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            elif op == "min":
                dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            
            return tensor
            
        except Exception as e:
            self.logger.error(f"All-reduce failed: {e}")
            return tensor
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from source process to all processes."""
        if not self.is_distributed:
            return tensor
        
        try:
            dist.broadcast(tensor, src=src)
            return tensor
            
        except Exception as e:
            self.logger.error(f"Broadcast failed: {e}")
            return tensor
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.is_distributed:
            try:
                dist.barrier()
            except Exception as e:
                self.logger.error(f"Barrier failed: {e}")
    
    def cleanup(self) -> None:
        """Cleanup distributed resources."""
        if self.is_distributed and dist.is_initialized():
            try:
                dist.destroy_process_group()
                self.logger.info("Distributed processing cleaned up")
            except Exception as e:
                self.logger.error(f"Cleanup failed: {e}")


class PhotonicScalingManager:
    """Comprehensive scaling manager for photonic neuromorphic systems."""
    
    def __init__(
        self,
        config: ScalingConfig,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.config = config
        self.metrics_collector = metrics_collector
        
        # Initialize components
        self.load_balancer = PhotonicLoadBalancer(
            algorithm=config.load_balancing,
            health_check_interval=30.0
        )
        
        self.auto_scaler = AutoScaler(
            config=config,
            load_balancer=self.load_balancer,
            metrics_collector=metrics_collector
        )
        
        self.distributed_manager = DistributedPhotonics()
        
        # Instance management
        self.active_instances: Dict[str, Dict[str, Any]] = {}
        self.instance_processes: Dict[str, multiprocessing.Process] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def start(self) -> None:
        """Start the scaling system."""
        self.logger.info("Starting photonic scaling manager")
        
        # Start load balancer health monitoring
        self.load_balancer.start_health_monitoring()
        
        # Start auto-scaling
        self.auto_scaler.start_auto_scaling()
        
        # Set up scaling callbacks
        self.auto_scaler.set_scale_up_callback(self._create_instances)
        self.auto_scaler.set_scale_down_callback(self._terminate_instances)
        
        # Create initial instances
        self._create_instances(self.config.min_instances)
    
    def stop(self) -> None:
        """Stop the scaling system."""
        self.logger.info("Stopping photonic scaling manager")
        
        # Stop auto-scaling
        self.auto_scaler.stop_auto_scaling()
        
        # Stop load balancer monitoring
        self.load_balancer.stop_health_monitoring()
        
        # Terminate all instances
        instance_ids = list(self.active_instances.keys())
        if instance_ids:
            self._terminate_instances(instance_ids)
        
        # Cleanup distributed resources
        self.distributed_manager.cleanup()
    
    def _create_instances(self, count: int) -> bool:
        """Create new instances."""
        try:
            created_count = 0
            
            for i in range(count):
                instance_id = f"photonic_instance_{len(self.active_instances)}_{int(time.time())}"
                
                # Create instance configuration
                instance_config = {
                    'instance_id': instance_id,
                    'port': 8080 + len(self.active_instances),
                    'optical_power_limit': 10e-3,  # 10 mW
                    'max_connections': 100
                }
                
                # Create and start instance process
                process = multiprocessing.Process(
                    target=self._run_instance,
                    args=(instance_config,),
                    daemon=True
                )
                
                process.start()
                
                # Register with load balancer
                self.load_balancer.register_instance(instance_id, weight=1.0)
                
                # Track instance
                self.active_instances[instance_id] = instance_config
                self.instance_processes[instance_id] = process
                
                created_count += 1
                
                self.logger.info(f"Created instance: {instance_id}")
            
            return created_count == count
            
        except Exception as e:
            self.logger.error(f"Failed to create instances: {e}")
            return False
    
    def _terminate_instances(self, instance_ids: List[str]) -> bool:
        """Terminate specified instances."""
        try:
            terminated_count = 0
            
            for instance_id in instance_ids:
                if instance_id in self.active_instances:
                    # Unregister from load balancer
                    self.load_balancer.unregister_instance(instance_id)
                    
                    # Terminate process
                    if instance_id in self.instance_processes:
                        process = self.instance_processes[instance_id]
                        process.terminate()
                        process.join(timeout=10.0)  # Wait up to 10 seconds
                        
                        if process.is_alive():
                            process.kill()  # Force kill if necessary
                        
                        del self.instance_processes[instance_id]
                    
                    # Remove from tracking
                    del self.active_instances[instance_id]
                    
                    terminated_count += 1
                    
                    self.logger.info(f"Terminated instance: {instance_id}")
            
            return terminated_count == len(instance_ids)
            
        except Exception as e:
            self.logger.error(f"Failed to terminate instances: {e}")
            return False
    
    def _run_instance(self, config: Dict[str, Any]) -> None:
        """Run a photonic instance process."""
        try:
            instance_id = config['instance_id']
            
            # Create photonic model for this instance
            model = PhotonicSNN(
                topology=[784, 256, 128, 10],  # MNIST example
                wavelength=1550e-9
            )
            
            # Instance main loop
            self.logger.info(f"Instance {instance_id} started")
            
            while True:
                # Simulate processing
                time.sleep(0.1)
                
                # Update instance metrics
                metrics = self._collect_instance_metrics(instance_id)
                self.load_balancer.update_instance_metrics(instance_id, metrics)
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.logger.error(f"Instance {config['instance_id']} error: {e}")
    
    def _collect_instance_metrics(self, instance_id: str) -> Dict[str, float]:
        """Collect metrics for a specific instance."""
        try:
            process = psutil.Process()
            
            return {
                'cpu_utilization': process.cpu_percent(),
                'memory_utilization': process.memory_percent(),
                'optical_power_usage': np.random.uniform(10, 80),  # Simulated
                'active_connections': np.random.randint(0, 50),
                'response_time': np.random.uniform(50, 200),  # ms
                'throughput': np.random.uniform(100, 1000),  # requests/s
                'error_rate': np.random.uniform(0, 2)  # %
            }
            
        except Exception:
            return {
                'cpu_utilization': 0,
                'memory_utilization': 0,
                'optical_power_usage': 0,
                'active_connections': 0,
                'response_time': 1000,
                'throughput': 0,
                'error_rate': 100
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling system status."""
        return {
            'active_instances': len(self.active_instances),
            'target_instances': self.auto_scaler.current_instances,
            'load_distribution': self.load_balancer.get_load_distribution(),
            'scaling_statistics': self.auto_scaler.get_scaling_statistics(),
            'health_status': {
                instance_id: instance.is_healthy()
                for instance_id, instance in self.load_balancer.instances.items()
            }
        }


# Performance optimization decorators
def distributed_processing(world_size: int = None):
    """Decorator for distributed processing."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if world_size and world_size > 1:
                # Set up distributed processing
                distributed_manager = DistributedPhotonics(world_size=world_size)
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    distributed_manager.cleanup()
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def auto_scaling(config: ScalingConfig = None):
    """Decorator for auto-scaling capabilities."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if config and config.enable_auto_scaling:
                scaling_manager = PhotonicScalingManager(config)
                scaling_manager.start()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    scaling_manager.stop()
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Factory functions
def create_scaling_config(
    strategy: ScalingStrategy = ScalingStrategy.HYBRID,
    min_instances: int = 1,
    max_instances: int = 8
) -> ScalingConfig:
    """Create optimized scaling configuration."""
    return ScalingConfig(
        strategy=strategy,
        min_instances=min_instances,
        max_instances=max_instances,
        target_cpu_utilization=70.0,
        target_memory_utilization=80.0,
        target_optical_utilization=60.0,
        enable_auto_scaling=True,
        enable_predictive_scaling=True,
        load_balancing=LoadBalancingAlgorithm.RESOURCE_AWARE
    )


def create_high_performance_scaling() -> PhotonicScalingManager:
    """Create high-performance scaling manager."""
    config = create_scaling_config(
        strategy=ScalingStrategy.ELASTIC,
        min_instances=2,
        max_instances=16
    )
    
    return PhotonicScalingManager(config)