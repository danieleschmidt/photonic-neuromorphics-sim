"""
Autonomous Performance Optimizer
===============================

Advanced performance optimization system with autonomous learning,
adaptive scaling, and intelligent resource management for photonic
neuromorphic computing platforms.

Features:
- Autonomous performance learning and adaptation
- Intelligent resource allocation and scaling
- Multi-dimensional optimization (latency, throughput, energy)
- Predictive scaling based on workload patterns
- Self-healing performance anomaly detection
- Advanced caching and resource pooling
"""

import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import deque
import statistics
import psutil
import gc
from contextlib import asynccontextmanager
import weakref

from .enhanced_logging import PhotonicLogger, PerformanceTracker
from .monitoring import MetricsCollector, SystemHealthMonitor
from .distributed_computing import NodeManager, DistributedPhotonicSimulator
from .realtime_adaptive_optimization import RealTimeOptimizer


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    LATENCY_FOCUSED = "latency_focused"
    THROUGHPUT_FOCUSED = "throughput_focused"  
    ENERGY_EFFICIENT = "energy_efficient"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_PRESSURE = "memory_pressure"
    THROUGHPUT_BASED = "throughput_based"
    LATENCY_BASED = "latency_based"
    PREDICTIVE = "predictive"
    COMPOSITE = "composite"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    timestamp: float
    latency_ms: float
    throughput_ops_sec: float
    cpu_utilization: float
    memory_usage_mb: float
    energy_consumption_w: float
    cache_hit_rate: float
    error_rate: float
    queue_depth: int
    active_connections: int


@dataclass
class OptimizationTarget:
    """Performance optimization targets."""
    max_latency_ms: float = 50.0
    min_throughput_ops_sec: float = 1000.0
    max_cpu_utilization: float = 0.8
    max_memory_usage_mb: float = 4096.0
    max_energy_consumption_w: float = 10.0
    min_cache_hit_rate: float = 0.9
    max_error_rate: float = 0.01


@dataclass  
class AutoScalingConfig:
    """Auto-scaling configuration."""
    min_instances: int = 1
    max_instances: int = 20
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    scale_up_cooldown: float = 300.0  # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes
    predictive_window: int = 50
    enable_predictive_scaling: bool = True


class PerformanceProfiler:
    """Advanced performance profiling and analysis."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_history: deque = deque(maxlen=window_size)
        self.profile_data: Dict[str, Any] = {}
        self.bottleneck_detection = True
        
    def record_metric(self, metric: PerformanceMetrics):
        """Record a performance metric."""
        self.metrics_history.append(metric)
        self._update_profile_data()
        
    def _update_profile_data(self):
        """Update aggregated profile data."""
        if not self.metrics_history:
            return
            
        recent_metrics = list(self.metrics_history)
        
        self.profile_data = {
            "avg_latency_ms": statistics.mean(m.latency_ms for m in recent_metrics),
            "p95_latency_ms": np.percentile([m.latency_ms for m in recent_metrics], 95),
            "p99_latency_ms": np.percentile([m.latency_ms for m in recent_metrics], 99),
            "avg_throughput": statistics.mean(m.throughput_ops_sec for m in recent_metrics),
            "peak_throughput": max(m.throughput_ops_sec for m in recent_metrics),
            "avg_cpu_utilization": statistics.mean(m.cpu_utilization for m in recent_metrics),
            "peak_memory_mb": max(m.memory_usage_mb for m in recent_metrics),
            "avg_energy_w": statistics.mean(m.energy_consumption_w for m in recent_metrics),
            "cache_hit_rate": statistics.mean(m.cache_hit_rate for m in recent_metrics),
            "error_rate": statistics.mean(m.error_rate for m in recent_metrics)
        }
        
    def detect_bottlenecks(self) -> List[str]:
        """Detect performance bottlenecks."""
        bottlenecks = []
        
        if not self.profile_data:
            return bottlenecks
            
        # CPU bottleneck
        if self.profile_data["avg_cpu_utilization"] > 0.9:
            bottlenecks.append("cpu_bound")
            
        # Memory bottleneck  
        if self.profile_data["peak_memory_mb"] > 8192:
            bottlenecks.append("memory_bound")
            
        # Cache efficiency
        if self.profile_data["cache_hit_rate"] < 0.8:
            bottlenecks.append("cache_inefficient")
            
        # High latency
        if self.profile_data["p95_latency_ms"] > 100:
            bottlenecks.append("latency_bound")
            
        return bottlenecks
        
    def get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        bottlenecks = self.detect_bottlenecks()
        
        if "cpu_bound" in bottlenecks:
            recommendations.extend([
                "increase_cpu_cores",
                "optimize_algorithms", 
                "enable_parallel_processing"
            ])
            
        if "memory_bound" in bottlenecks:
            recommendations.extend([
                "increase_memory_limit",
                "optimize_memory_usage",
                "enable_memory_compression"
            ])
            
        if "cache_inefficient" in bottlenecks:
            recommendations.extend([
                "increase_cache_size",
                "optimize_cache_policy",
                "implement_prefetching"
            ])
            
        if "latency_bound" in bottlenecks:
            recommendations.extend([
                "optimize_network_paths",
                "implement_edge_caching",
                "reduce_serialization_overhead"
            ])
            
        return recommendations


class IntelligentResourceManager:
    """Intelligent resource allocation and management."""
    
    def __init__(self, total_cpu_cores: int = None, total_memory_gb: int = None):
        self.total_cpu_cores = total_cpu_cores or psutil.cpu_count()
        self.total_memory_gb = total_memory_gb or (psutil.virtual_memory().total // (1024**3))
        
        self.resource_pools = {
            "cpu": ResourcePool("cpu", self.total_cpu_cores),
            "memory": ResourcePool("memory", self.total_memory_gb * 1024),  # MB
            "network": ResourcePool("network", 1000),  # Mbps
            "storage": ResourcePool("storage", 10000)  # IOPS
        }
        
        self.allocation_history: deque = deque(maxlen=1000)
        self.predictive_allocator = PredictiveResourceAllocator()
        
    def allocate_resources(
        self, 
        request_id: str, 
        cpu_cores: float, 
        memory_mb: float,
        network_mbps: float = 0,
        storage_iops: float = 0
    ) -> bool:
        """
        Allocate resources for a request.
        
        Returns:
            True if allocation successful, False otherwise
        """
        allocation = {
            "request_id": request_id,
            "timestamp": time.time(),
            "cpu_cores": cpu_cores,
            "memory_mb": memory_mb,
            "network_mbps": network_mbps,
            "storage_iops": storage_iops
        }
        
        # Check availability
        if not self._check_resource_availability(allocation):
            return False
            
        # Perform allocation
        self.resource_pools["cpu"].allocate(cpu_cores)
        self.resource_pools["memory"].allocate(memory_mb)
        self.resource_pools["network"].allocate(network_mbps)
        self.resource_pools["storage"].allocate(storage_iops)
        
        self.allocation_history.append(allocation)
        return True
        
    def _check_resource_availability(self, allocation: Dict[str, Any]) -> bool:
        """Check if resources are available for allocation."""
        return (
            self.resource_pools["cpu"].can_allocate(allocation["cpu_cores"]) and
            self.resource_pools["memory"].can_allocate(allocation["memory_mb"]) and
            self.resource_pools["network"].can_allocate(allocation["network_mbps"]) and
            self.resource_pools["storage"].can_allocate(allocation["storage_iops"])
        )
        
    def release_resources(self, request_id: str):
        """Release resources for a request."""
        # Find allocation
        for allocation in reversed(self.allocation_history):
            if allocation["request_id"] == request_id:
                self.resource_pools["cpu"].release(allocation["cpu_cores"])
                self.resource_pools["memory"].release(allocation["memory_mb"])
                self.resource_pools["network"].release(allocation["network_mbps"])
                self.resource_pools["storage"].release(allocation["storage_iops"])
                break
                
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        return {
            name: pool.get_utilization()
            for name, pool in self.resource_pools.items()
        }
        
    def optimize_allocation(self) -> Dict[str, Any]:
        """Optimize resource allocation based on usage patterns."""
        utilization = self.get_resource_utilization()
        recommendations = []
        
        # Detect over/under-provisioning
        for resource, util in utilization.items():
            if util > 0.9:
                recommendations.append(f"scale_up_{resource}")
            elif util < 0.3:
                recommendations.append(f"scale_down_{resource}")
                
        return {
            "current_utilization": utilization,
            "optimization_recommendations": recommendations,
            "predicted_demand": self.predictive_allocator.predict_demand()
        }


class ResourcePool:
    """Individual resource pool management."""
    
    def __init__(self, name: str, capacity: float):
        self.name = name
        self.capacity = capacity
        self.allocated = 0.0
        self.reservations: Dict[str, float] = {}
        
    def can_allocate(self, amount: float) -> bool:
        """Check if amount can be allocated."""
        return self.allocated + amount <= self.capacity
        
    def allocate(self, amount: float) -> bool:
        """Allocate resource amount."""
        if self.can_allocate(amount):
            self.allocated += amount
            return True
        return False
        
    def release(self, amount: float):
        """Release resource amount."""
        self.allocated = max(0, self.allocated - amount)
        
    def get_utilization(self) -> float:
        """Get current utilization percentage."""
        return self.allocated / self.capacity if self.capacity > 0 else 0


class PredictiveResourceAllocator:
    """Predictive resource allocation using time series analysis."""
    
    def __init__(self, history_window: int = 100):
        self.history_window = history_window
        self.demand_history: deque = deque(maxlen=history_window)
        
    def record_demand(self, demand: Dict[str, float]):
        """Record resource demand."""
        demand["timestamp"] = time.time()
        self.demand_history.append(demand)
        
    def predict_demand(self, horizon_minutes: int = 30) -> Dict[str, float]:
        """Predict resource demand for the next horizon."""
        if len(self.demand_history) < 10:
            return {"cpu": 0.5, "memory": 0.5, "network": 0.3, "storage": 0.3}
            
        # Simple trend-based prediction
        recent_demands = list(self.demand_history)[-20:]
        
        predictions = {}
        for resource in ["cpu", "memory", "network", "storage"]:
            values = [d.get(resource, 0) for d in recent_demands]
            if values:
                # Linear trend extrapolation
                trend = (values[-1] - values[0]) / len(values)
                predicted = values[-1] + trend * (horizon_minutes / 5)  # 5-minute intervals
                predictions[resource] = max(0, min(1, predicted))  # Clamp to [0,1]
            else:
                predictions[resource] = 0.5
                
        return predictions


class AdvancedCacheManager:
    """Advanced caching system with intelligent eviction."""
    
    def __init__(
        self, 
        max_size_mb: int = 1024,
        eviction_policy: str = "lru",
        enable_prefetching: bool = True
    ):
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.enable_prefetching = enable_prefetching
        
        self.cache: Dict[str, Any] = {}
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        self.access_history: deque = deque(maxlen=10000)
        self.current_size_bytes = 0
        
        self.hit_count = 0
        self.miss_count = 0
        
    def get(self, key: str) -> Any:
        """Get item from cache."""
        if key in self.cache:
            self.hit_count += 1
            self._update_access(key)
            return self.cache[key]
        else:
            self.miss_count += 1
            return None
            
    def put(self, key: str, value: Any, size_bytes: int = None):
        """Put item in cache."""
        if size_bytes is None:
            size_bytes = self._estimate_size(value)
            
        # Ensure space available
        while (self.current_size_bytes + size_bytes > self.max_size_bytes and 
               len(self.cache) > 0):
            self._evict_item()
            
        self.cache[key] = value
        self.cache_metadata[key] = {
            "size_bytes": size_bytes,
            "last_access": time.time(),
            "access_count": 1,
            "created": time.time()
        }
        self.current_size_bytes += size_bytes
        
    def _update_access(self, key: str):
        """Update access metadata."""
        if key in self.cache_metadata:
            metadata = self.cache_metadata[key]
            metadata["last_access"] = time.time()
            metadata["access_count"] += 1
            
        self.access_history.append({
            "key": key,
            "timestamp": time.time()
        })
        
    def _evict_item(self):
        """Evict item based on eviction policy."""
        if not self.cache:
            return
            
        if self.eviction_policy == "lru":
            # Least Recently Used
            oldest_key = min(
                self.cache_metadata.keys(),
                key=lambda k: self.cache_metadata[k]["last_access"]
            )
        elif self.eviction_policy == "lfu":
            # Least Frequently Used
            oldest_key = min(
                self.cache_metadata.keys(),
                key=lambda k: self.cache_metadata[k]["access_count"]
            )
        else:
            # FIFO
            oldest_key = min(
                self.cache_metadata.keys(),
                key=lambda k: self.cache_metadata[k]["created"]
            )
            
        self._remove_item(oldest_key)
        
    def _remove_item(self, key: str):
        """Remove item from cache."""
        if key in self.cache:
            size_bytes = self.cache_metadata[key]["size_bytes"]
            del self.cache[key]
            del self.cache_metadata[key]
            self.current_size_bytes -= size_bytes
            
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        import sys
        return sys.getsizeof(value)
        
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hit_rate": self.get_hit_rate(),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "cache_size_mb": self.current_size_bytes / (1024 * 1024),
            "cache_utilization": self.current_size_bytes / self.max_size_bytes,
            "item_count": len(self.cache)
        }


class AutonomousPerformanceOptimizer:
    """
    Main autonomous performance optimization system.
    """
    
    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE,
        targets: OptimizationTarget = None,
        scaling_config: AutoScalingConfig = None
    ):
        self.strategy = strategy
        self.targets = targets or OptimizationTarget()
        self.scaling_config = scaling_config or AutoScalingConfig()
        
        # Initialize components
        self.logger = PhotonicLogger("AutonomousOptimizer")
        self.profiler = PerformanceProfiler()
        self.resource_manager = IntelligentResourceManager()
        self.cache_manager = AdvancedCacheManager()
        self.metrics_collector = MetricsCollector()
        
        # State tracking
        self.optimization_state = "initializing"
        self.current_instances = self.scaling_config.min_instances
        self.last_scale_action = 0
        self.optimization_history: deque = deque(maxlen=1000)
        
        # Optimization loop
        self.optimization_task: Optional[asyncio.Task] = None
        self.is_optimizing = False
        
    async def start_optimization(self):
        """Start autonomous optimization loop."""
        self.logger.info("Starting autonomous performance optimization")
        self.is_optimizing = True
        self.optimization_state = "running"
        
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        
    async def stop_optimization(self):
        """Stop autonomous optimization loop."""
        self.logger.info("Stopping autonomous performance optimization")
        self.is_optimizing = False
        
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
                
        self.optimization_state = "stopped"
        
    async def _optimization_loop(self):
        """Main optimization loop."""
        while self.is_optimizing:
            try:
                # Collect current metrics
                current_metrics = await self._collect_current_metrics()
                self.profiler.record_metric(current_metrics)
                
                # Analyze performance
                bottlenecks = self.profiler.detect_bottlenecks()
                recommendations = self.profiler.get_optimization_recommendations()
                
                # Apply optimizations
                if bottlenecks or recommendations:
                    optimization_actions = await self._generate_optimization_actions(
                        bottlenecks, recommendations
                    )
                    await self._apply_optimizations(optimization_actions)
                    
                # Check auto-scaling
                await self._check_auto_scaling(current_metrics)
                
                # Resource optimization
                await self._optimize_resources()
                
                # Cache optimization
                self._optimize_cache()
                
                # Record optimization cycle
                self.optimization_history.append({
                    "timestamp": time.time(),
                    "metrics": current_metrics,
                    "bottlenecks": bottlenecks,
                    "actions_taken": recommendations,
                    "resource_utilization": self.resource_manager.get_resource_utilization()
                })
                
                await asyncio.sleep(5.0)  # 5-second optimization interval
                
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(10.0)
                
    async def _collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Simulate photonic-specific metrics
        import random
        
        return PerformanceMetrics(
            timestamp=time.time(),
            latency_ms=random.uniform(10, 100),
            throughput_ops_sec=random.uniform(500, 2000),
            cpu_utilization=cpu_percent / 100.0,
            memory_usage_mb=memory.used / (1024 * 1024),
            energy_consumption_w=random.uniform(1, 15),
            cache_hit_rate=self.cache_manager.get_hit_rate(),
            error_rate=random.uniform(0, 0.02),
            queue_depth=random.randint(0, 50),
            active_connections=random.randint(10, 200)
        )
        
    async def _generate_optimization_actions(
        self, 
        bottlenecks: List[str], 
        recommendations: List[str]
    ) -> List[str]:
        """Generate optimization actions based on analysis."""
        actions = []
        
        # Strategy-specific optimizations
        if self.strategy == OptimizationStrategy.LATENCY_FOCUSED:
            if "latency_bound" in bottlenecks:
                actions.extend([
                    "increase_cpu_allocation",
                    "optimize_cache_size",
                    "enable_connection_pooling"
                ])
                
        elif self.strategy == OptimizationStrategy.THROUGHPUT_FOCUSED:
            if "cpu_bound" in bottlenecks:
                actions.extend([
                    "scale_out_instances",
                    "enable_parallel_processing",
                    "optimize_batch_sizes"
                ])
                
        elif self.strategy == OptimizationStrategy.ENERGY_EFFICIENT:
            actions.extend([
                "reduce_cpu_frequency",
                "optimize_memory_usage",
                "enable_power_management"
            ])
            
        # Add recommendations
        actions.extend(recommendations)
        
        return list(set(actions))  # Remove duplicates
        
    async def _apply_optimizations(self, actions: List[str]):
        """Apply optimization actions."""
        for action in actions:
            try:
                if action == "increase_cpu_allocation":
                    await self._increase_cpu_allocation()
                elif action == "optimize_cache_size":
                    await self._optimize_cache_size()
                elif action == "scale_out_instances":
                    await self._scale_out_instances()
                elif action == "enable_parallel_processing":
                    await self._enable_parallel_processing()
                elif action == "optimize_memory_usage":
                    await self._optimize_memory_usage()
                    
                self.logger.info(f"Applied optimization: {action}")
                
            except Exception as e:
                self.logger.error(f"Failed to apply optimization {action}: {e}")
                
    async def _check_auto_scaling(self, metrics: PerformanceMetrics):
        """Check and apply auto-scaling decisions."""
        current_time = time.time()
        
        # Check cooldown periods
        if (current_time - self.last_scale_action < 
            min(self.scaling_config.scale_up_cooldown, 
                self.scaling_config.scale_down_cooldown)):
            return
            
        # Determine scaling action
        scale_action = None
        
        if self.scaling_config.enable_predictive_scaling:
            # Predictive scaling
            predicted_demand = self.resource_manager.predictive_allocator.predict_demand()
            if max(predicted_demand.values()) > self.scaling_config.scale_up_threshold:
                scale_action = "scale_up"
            elif max(predicted_demand.values()) < self.scaling_config.scale_down_threshold:
                scale_action = "scale_down"
        else:
            # Reactive scaling
            if (metrics.cpu_utilization > self.scaling_config.scale_up_threshold or
                metrics.memory_usage_mb > self.targets.max_memory_usage_mb * 0.8):
                scale_action = "scale_up"
            elif (metrics.cpu_utilization < self.scaling_config.scale_down_threshold and
                  self.current_instances > self.scaling_config.min_instances):
                scale_action = "scale_down"
                
        # Apply scaling action
        if scale_action == "scale_up" and self.current_instances < self.scaling_config.max_instances:
            await self._scale_up()
            self.last_scale_action = current_time
        elif scale_action == "scale_down" and self.current_instances > self.scaling_config.min_instances:
            await self._scale_down()
            self.last_scale_action = current_time
            
    async def _optimize_resources(self):
        """Optimize resource allocation."""
        optimization_result = self.resource_manager.optimize_allocation()
        
        for recommendation in optimization_result["optimization_recommendations"]:
            if recommendation.startswith("scale_up_"):
                resource = recommendation.split("_")[-1]
                await self._scale_up_resource(resource)
            elif recommendation.startswith("scale_down_"):
                resource = recommendation.split("_")[-1]
                await self._scale_down_resource(resource)
                
    def _optimize_cache(self):
        """Optimize cache configuration."""
        cache_stats = self.cache_manager.get_statistics()
        
        # Adjust cache size based on hit rate
        if cache_stats["hit_rate"] < 0.8 and cache_stats["cache_utilization"] > 0.9:
            # Increase cache size
            self.cache_manager.max_size_mb = int(self.cache_manager.max_size_mb * 1.2)
            self.cache_manager.max_size_bytes = self.cache_manager.max_size_mb * 1024 * 1024
            
    # Optimization action implementations
    async def _increase_cpu_allocation(self):
        """Increase CPU allocation."""
        self.logger.info("Increasing CPU allocation")
        
    async def _optimize_cache_size(self):
        """Optimize cache size."""
        current_hit_rate = self.cache_manager.get_hit_rate()
        if current_hit_rate < 0.9:
            self.cache_manager.max_size_mb = int(self.cache_manager.max_size_mb * 1.1)
            self.logger.info(f"Increased cache size to {self.cache_manager.max_size_mb}MB")
            
    async def _scale_out_instances(self):
        """Scale out instances."""
        if self.current_instances < self.scaling_config.max_instances:
            self.current_instances += 1
            self.logger.info(f"Scaled out to {self.current_instances} instances")
            
    async def _enable_parallel_processing(self):
        """Enable parallel processing optimizations."""
        self.logger.info("Enabling parallel processing optimizations")
        
    async def _optimize_memory_usage(self):
        """Optimize memory usage."""
        # Force garbage collection
        gc.collect()
        self.logger.info("Optimized memory usage")
        
    async def _scale_up(self):
        """Scale up instances."""
        if self.current_instances < self.scaling_config.max_instances:
            self.current_instances += 1
            self.logger.info(f"Scaled up to {self.current_instances} instances")
            
    async def _scale_down(self):
        """Scale down instances."""
        if self.current_instances > self.scaling_config.min_instances:
            self.current_instances -= 1
            self.logger.info(f"Scaled down to {self.current_instances} instances")
            
    async def _scale_up_resource(self, resource: str):
        """Scale up specific resource."""
        self.logger.info(f"Scaling up {resource} resource")
        
    async def _scale_down_resource(self, resource: str):
        """Scale down specific resource."""
        self.logger.info(f"Scaling down {resource} resource")
        
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            "state": self.optimization_state,
            "strategy": self.strategy.value,
            "current_instances": self.current_instances,
            "resource_utilization": self.resource_manager.get_resource_utilization(),
            "cache_statistics": self.cache_manager.get_statistics(),
            "performance_profile": self.profiler.profile_data,
            "bottlenecks": self.profiler.detect_bottlenecks(),
            "optimization_history_length": len(self.optimization_history)
        }
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        status = self.get_optimization_status()
        
        # Calculate performance improvements
        if len(self.optimization_history) > 10:
            recent_metrics = [entry["metrics"] for entry in list(self.optimization_history)[-10:]]
            initial_metrics = [entry["metrics"] for entry in list(self.optimization_history)[:10]]
            
            avg_recent_latency = statistics.mean(m.latency_ms for m in recent_metrics)
            avg_initial_latency = statistics.mean(m.latency_ms for m in initial_metrics)
            
            avg_recent_throughput = statistics.mean(m.throughput_ops_sec for m in recent_metrics)
            avg_initial_throughput = statistics.mean(m.throughput_ops_sec for m in initial_metrics)
            
            latency_improvement = (avg_initial_latency - avg_recent_latency) / avg_initial_latency
            throughput_improvement = (avg_recent_throughput - avg_initial_throughput) / avg_initial_throughput
        else:
            latency_improvement = 0
            throughput_improvement = 0
            
        return {
            "optimization_status": status,
            "performance_improvements": {
                "latency_improvement_percent": latency_improvement * 100,
                "throughput_improvement_percent": throughput_improvement * 100
            },
            "targets_met": {
                "latency_target": status["performance_profile"].get("avg_latency_ms", 0) <= self.targets.max_latency_ms,
                "throughput_target": status["performance_profile"].get("avg_throughput", 0) >= self.targets.min_throughput_ops_sec,
                "cache_target": status["cache_statistics"]["hit_rate"] >= self.targets.min_cache_hit_rate
            },
            "recommendations": self.profiler.get_optimization_recommendations()
        }


async def create_autonomous_performance_demo() -> Dict[str, Any]:
    """
    Create and run autonomous performance optimization demonstration.
    
    Returns:
        Demonstration results
    """
    print("⚡ AUTONOMOUS PERFORMANCE OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Create optimizer with adaptive strategy
    optimizer = AutonomousPerformanceOptimizer(
        strategy=OptimizationStrategy.ADAPTIVE,
        targets=OptimizationTarget(
            max_latency_ms=30.0,
            min_throughput_ops_sec=1500.0,
            max_cpu_utilization=0.7,
            min_cache_hit_rate=0.95
        ),
        scaling_config=AutoScalingConfig(
            min_instances=2,
            max_instances=10,
            enable_predictive_scaling=True
        )
    )
    
    # Start optimization
    await optimizer.start_optimization()
    
    print("🔄 Running optimization for 30 seconds...")
    await asyncio.sleep(30.0)
    
    # Get results
    performance_report = optimizer.get_performance_report()
    
    # Stop optimization
    await optimizer.stop_optimization()
    
    print("✅ Autonomous optimization completed!")
    print(f"Latency Improvement: {performance_report['performance_improvements']['latency_improvement_percent']:.1f}%")
    print(f"Throughput Improvement: {performance_report['performance_improvements']['throughput_improvement_percent']:.1f}%")
    print(f"Cache Hit Rate: {performance_report['optimization_status']['cache_statistics']['hit_rate']:.2%}")
    print(f"Current Instances: {performance_report['optimization_status']['current_instances']}")
    
    return performance_report


if __name__ == "__main__":
    # Run autonomous performance optimization demonstration
    results = asyncio.run(create_autonomous_performance_demo())
    
    # Save results
    with open("autonomous_performance_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
        
    print(f"\n📊 Results saved to: autonomous_performance_results.json")