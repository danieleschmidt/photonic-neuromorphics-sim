"""
Performance optimization and scaling for photonic neuromorphic systems.

Provides advanced optimization techniques including adaptive caching, 
parallel processing, memory pooling, and auto-scaling capabilities
for high-performance photonic neural network simulation.
"""

import numpy as np
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import time
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from functools import lru_cache, wraps
import logging
import psutil
import gc

from .exceptions import ResourceExhaustionError, PhotonicNeuromorphicsException
from .monitoring import MetricsCollector, PerformanceProfiler


@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations."""
    enable_caching: bool = True
    cache_size: int = 1000
    enable_parallel: bool = True
    max_workers: int = 0  # 0 = auto-detect
    enable_memory_pooling: bool = True
    memory_pool_size: int = 1024  # MB
    enable_auto_scaling: bool = True
    scaling_threshold_cpu: float = 80.0  # %
    scaling_threshold_memory: float = 85.0  # %
    enable_gpu_acceleration: bool = False
    batch_size: int = 32
    prefetch_factor: int = 2


class AdaptiveCache:
    """
    Adaptive caching system with intelligent eviction policies.
    
    Implements LRU, LFU, and time-based caching strategies with
    automatic adaptation based on access patterns and memory pressure.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 3600.0,  # 1 hour default TTL
        enable_adaptive: bool = True
    ):
        """
        Initialize adaptive cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cached items in seconds
            enable_adaptive: Enable adaptive cache management
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_adaptive = enable_adaptive
        
        # Cache storage
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        self.creation_times: Dict[str, float] = {}
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Threading
        self._lock = threading.RLock()
        
        # Adaptive parameters
        self.hit_ratio_window = []  # Rolling window for hit ratio
        self.current_strategy = "lru"  # lru, lfu, ttl
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with adaptive tracking."""
        with self._lock:
            current_time = time.time()
            
            if key in self.cache:
                # Check TTL
                if current_time - self.creation_times[key] > self.ttl_seconds:
                    self._evict_key(key)
                    self.misses += 1
                    return None
                
                # Update access statistics
                self.access_times[key] = current_time
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                self.hits += 1
                
                return self.cache[key]
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with intelligent eviction."""
        with self._lock:
            current_time = time.time()
            
            # If key already exists, update it
            if key in self.cache:
                self.cache[key] = value
                self.access_times[key] = current_time
                self.creation_times[key] = current_time
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                return
            
            # Check if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_item()
            
            # Add new item
            self.cache[key] = value
            self.access_times[key] = current_time
            self.creation_times[key] = current_time
            self.access_counts[key] = 1
            
            # Adapt cache strategy if enabled
            if self.enable_adaptive:
                self._adapt_strategy()
    
    def _evict_item(self) -> None:
        """Evict item based on current strategy."""
        if not self.cache:
            return
        
        current_time = time.time()
        
        if self.current_strategy == "lru":
            # Evict least recently used
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        elif self.current_strategy == "lfu":
            # Evict least frequently used
            oldest_key = min(self.access_counts.items(), key=lambda x: x[1])[0]
        else:  # ttl strategy
            # Evict oldest created
            oldest_key = min(self.creation_times.items(), key=lambda x: x[1])[0]
        
        self._evict_key(oldest_key)
    
    def _evict_key(self, key: str) -> None:
        """Evict specific key."""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            del self.access_counts[key]
            del self.creation_times[key]
            self.evictions += 1
    
    def _adapt_strategy(self) -> None:
        """Adapt caching strategy based on performance."""
        # Calculate current hit ratio
        total_requests = self.hits + self.misses
        if total_requests < 100:  # Need minimum data
            return
        
        current_hit_ratio = self.hits / total_requests
        self.hit_ratio_window.append(current_hit_ratio)
        
        # Keep rolling window
        if len(self.hit_ratio_window) > 10:
            self.hit_ratio_window.pop(0)
        
        # Adapt strategy every 1000 requests
        if total_requests % 1000 == 0 and len(self.hit_ratio_window) >= 5:
            avg_hit_ratio = sum(self.hit_ratio_window) / len(self.hit_ratio_window)
            
            # Switch strategy if performance is poor
            if avg_hit_ratio < 0.5:
                if self.current_strategy == "lru":
                    self.current_strategy = "lfu"
                elif self.current_strategy == "lfu":
                    self.current_strategy = "ttl"
                else:
                    self.current_strategy = "lru"
                
                self.logger.info(f"Adapted cache strategy to {self.current_strategy} (hit ratio: {avg_hit_ratio:.3f})")
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.creation_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_ratio = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_ratio": hit_ratio,
                "current_strategy": self.current_strategy,
                "memory_usage_mb": self._estimate_memory_usage() / 1024**2
            }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of cache in bytes."""
        try:
            import sys
            total_size = 0
            for key, value in self.cache.items():
                total_size += sys.getsizeof(key) + sys.getsizeof(value)
            return total_size
        except Exception:
            return len(self.cache) * 1024  # Rough estimate


class MemoryPool:
    """
    High-performance memory pool for reducing allocation overhead.
    
    Provides pre-allocated memory pools for frequently used data structures
    with automatic garbage collection and memory pressure management.
    """
    
    def __init__(self, pool_size_mb: int = 1024):
        """
        Initialize memory pool.
        
        Args:
            pool_size_mb: Total pool size in megabytes
        """
        self.pool_size_bytes = pool_size_mb * 1024 * 1024
        self.pools: Dict[str, List[Any]] = {}
        self.allocated_size = 0
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.allocations = 0
        self.deallocations = 0
        self.pool_hits = 0
        self.pool_misses = 0
    
    def get_array(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """Get pre-allocated array from pool or create new one."""
        key = f"array_{shape}_{dtype}"
        
        with self._lock:
            if key in self.pools and self.pools[key]:
                array = self.pools[key].pop()
                array.fill(0)  # Reset to zeros
                self.pool_hits += 1
                return array
            else:
                # Create new array
                array = np.zeros(shape, dtype=dtype)
                self.pool_misses += 1
                self.allocations += 1
                
                # Track memory usage
                array_size = array.nbytes
                if self.allocated_size + array_size > self.pool_size_bytes:
                    self._cleanup_pools()
                
                self.allocated_size += array_size
                return array
    
    def return_array(self, array: np.ndarray) -> None:
        """Return array to pool for reuse."""
        if array is None:
            return
        
        dtype = array.dtype
        shape = array.shape
        key = f"array_{shape}_{dtype}"
        
        with self._lock:
            if key not in self.pools:
                self.pools[key] = []
            
            # Limit pool size per type
            if len(self.pools[key]) < 100:  # Max 100 arrays per pool
                self.pools[key].append(array)
                self.deallocations += 1
    
    def get_tensor_buffer(self, size: int) -> List[float]:
        """Get pre-allocated tensor buffer."""
        key = f"tensor_buffer_{size}"
        
        with self._lock:
            if key in self.pools and self.pools[key]:
                buffer = self.pools[key].pop()
                self.pool_hits += 1
                return buffer
            else:
                buffer = [0.0] * size
                self.pool_misses += 1
                self.allocations += 1
                return buffer
    
    def return_tensor_buffer(self, buffer: List[float]) -> None:
        """Return tensor buffer to pool."""
        if buffer is None:
            return
        
        key = f"tensor_buffer_{len(buffer)}"
        
        with self._lock:
            if key not in self.pools:
                self.pools[key] = []
            
            if len(self.pools[key]) < 50:  # Limit buffer pool size
                # Clear buffer
                for i in range(len(buffer)):
                    buffer[i] = 0.0
                self.pools[key].append(buffer)
                self.deallocations += 1
    
    def _cleanup_pools(self) -> None:
        """Clean up memory pools to free space."""
        total_freed = 0
        
        for key, pool in list(self.pools.items()):
            if pool:
                # Remove half of the pooled items
                removed_count = len(pool) // 2
                for _ in range(removed_count):
                    if pool:
                        item = pool.pop()
                        if hasattr(item, 'nbytes'):
                            total_freed += item.nbytes
                        del item
        
        self.allocated_size = max(0, self.allocated_size - total_freed)
        gc.collect()  # Force garbage collection
        
        self.logger.info(f"Cleaned up memory pools, freed {total_freed / 1024**2:.2f} MB")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            total_pooled_items = sum(len(pool) for pool in self.pools.values())
            pool_efficiency = self.pool_hits / (self.pool_hits + self.pool_misses) if (self.pool_hits + self.pool_misses) > 0 else 0
            
            return {
                "allocated_size_mb": self.allocated_size / 1024**2,
                "pool_size_limit_mb": self.pool_size_bytes / 1024**2,
                "total_pools": len(self.pools),
                "total_pooled_items": total_pooled_items,
                "allocations": self.allocations,
                "deallocations": self.deallocations,
                "pool_hits": self.pool_hits,
                "pool_misses": self.pool_misses,
                "pool_efficiency": pool_efficiency
            }


class ParallelProcessor:
    """
    High-performance parallel processing for photonic simulations.
    
    Provides intelligent workload distribution, dynamic load balancing,
    and adaptive scaling based on system resources and workload characteristics.
    """
    
    def __init__(
        self,
        max_workers: int = 0,
        use_processes: bool = False,
        chunk_size_factor: float = 1.0
    ):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of workers (0 = auto-detect)
            use_processes: Use processes instead of threads
            chunk_size_factor: Factor for automatic chunk size calculation
        """
        self.max_workers = max_workers or min(mp.cpu_count(), 32)  # Reasonable upper limit
        self.use_processes = use_processes
        self.chunk_size_factor = chunk_size_factor
        
        self.logger = logging.getLogger(__name__)
        self._current_executor = None
        self._lock = threading.Lock()
        
        # Performance tracking
        self.task_times: List[float] = []
        self.throughput_history: List[float] = []
        
        # Adaptive parameters
        self.optimal_chunk_size = 1
        self.optimal_worker_count = self.max_workers
    
    def process_parallel(
        self,
        func: Callable,
        data_chunks: List[Any],
        **kwargs
    ) -> List[Any]:
        """
        Process data chunks in parallel with adaptive optimization.
        
        Args:
            func: Function to execute on each chunk
            data_chunks: List of data chunks to process
            **kwargs: Additional arguments for the function
            
        Returns:
            List of results from processing each chunk
        """
        if not data_chunks:
            return []
        
        start_time = time.time()
        
        # Determine optimal parameters
        chunk_count = len(data_chunks)
        worker_count = min(self.optimal_worker_count, chunk_count, self.max_workers)
        
        self.logger.debug(f"Processing {chunk_count} chunks with {worker_count} workers")
        
        try:
            # Choose executor type
            executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
            
            with executor_class(max_workers=worker_count) as executor:
                # Submit all tasks
                future_to_chunk = {
                    executor.submit(func, chunk, **kwargs): i 
                    for i, chunk in enumerate(data_chunks)
                }
                
                # Collect results in order
                results = [None] * len(data_chunks)
                completed_tasks = 0
                
                for future in as_completed(future_to_chunk):
                    chunk_index = future_to_chunk[future]
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout
                        results[chunk_index] = result
                        completed_tasks += 1
                    except Exception as e:
                        self.logger.error(f"Chunk {chunk_index} failed: {e}")
                        results[chunk_index] = None  # or some default value
                        completed_tasks += 1
                    
                    # Progress logging
                    if completed_tasks % max(1, chunk_count // 10) == 0:
                        progress = (completed_tasks / chunk_count) * 100
                        self.logger.debug(f"Progress: {progress:.1f}% ({completed_tasks}/{chunk_count})")
            
            # Performance tracking and adaptation
            total_time = time.time() - start_time
            self.task_times.append(total_time)
            
            throughput = chunk_count / total_time if total_time > 0 else 0
            self.throughput_history.append(throughput)
            
            # Adaptive optimization
            self._adapt_parameters(chunk_count, worker_count, total_time)
            
            self.logger.info(f"Processed {chunk_count} chunks in {total_time:.3f}s "
                           f"(throughput: {throughput:.2f} chunks/s)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Parallel processing failed: {e}")
            # Fallback to sequential processing
            self.logger.warning("Falling back to sequential processing")
            return [func(chunk, **kwargs) for chunk in data_chunks]
    
    def _adapt_parameters(self, chunk_count: int, worker_count: int, execution_time: float):
        """Adapt processing parameters based on performance."""
        if len(self.throughput_history) < 5:
            return  # Need more data
        
        # Calculate recent average throughput
        recent_throughput = sum(self.throughput_history[-5:]) / 5
        
        # Adapt worker count based on throughput trends
        if len(self.throughput_history) >= 2:
            throughput_trend = self.throughput_history[-1] - self.throughput_history[-2]
            
            if throughput_trend > 0 and self.optimal_worker_count < self.max_workers:
                # Performance improving, try more workers
                self.optimal_worker_count = min(self.max_workers, self.optimal_worker_count + 1)
            elif throughput_trend < -0.1 and self.optimal_worker_count > 1:
                # Performance degrading, try fewer workers
                self.optimal_worker_count = max(1, self.optimal_worker_count - 1)
        
        # Adapt chunk size based on execution time
        if execution_time > 0:
            # Target: ~0.1-1.0 seconds per chunk for good granularity
            target_chunk_time = 0.5  # seconds
            current_chunk_time = execution_time / chunk_count
            
            if current_chunk_time < 0.1:  # Too fine-grained
                self.optimal_chunk_size = min(10, int(self.optimal_chunk_size * 1.5))
            elif current_chunk_time > 2.0:  # Too coarse-grained
                self.optimal_chunk_size = max(1, int(self.optimal_chunk_size * 0.8))
    
    def get_optimal_chunk_size(self, total_items: int) -> int:
        """Calculate optimal chunk size for given number of items."""
        base_chunk_size = max(1, total_items // (self.optimal_worker_count * 4))
        return int(base_chunk_size * self.chunk_size_factor * self.optimal_chunk_size)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parallel processing statistics."""
        avg_time = sum(self.task_times) / len(self.task_times) if self.task_times else 0
        avg_throughput = sum(self.throughput_history) / len(self.throughput_history) if self.throughput_history else 0
        
        return {
            "max_workers": self.max_workers,
            "optimal_worker_count": self.optimal_worker_count,
            "optimal_chunk_size": self.optimal_chunk_size,
            "use_processes": self.use_processes,
            "total_tasks_processed": len(self.task_times),
            "average_execution_time": avg_time,
            "average_throughput": avg_throughput,
            "recent_throughput": self.throughput_history[-1] if self.throughput_history else 0
        }


class AutoScaler:
    """
    Automatic scaling system for photonic neural network processing.
    
    Monitors system resources and automatically adjusts processing parameters
    to maintain optimal performance while avoiding resource exhaustion.
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 85.0,
        scaling_cooldown: float = 60.0  # seconds
    ):
        """
        Initialize auto-scaler.
        
        Args:
            metrics_collector: Metrics collector for monitoring
            cpu_threshold: CPU usage threshold for scaling
            memory_threshold: Memory usage threshold for scaling
            scaling_cooldown: Minimum time between scaling actions
        """
        self.metrics_collector = metrics_collector
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.scaling_cooldown = scaling_cooldown
        
        self.logger = logging.getLogger(__name__)
        self.last_scaling_time = 0.0
        
        # Scaling history
        self.scaling_actions: List[Dict[str, Any]] = []
        
        # Current scaling parameters
        self.current_batch_size = 32
        self.current_worker_count = mp.cpu_count()
        self.current_memory_limit = 2048  # MB
    
    def check_and_scale(self) -> Dict[str, Any]:
        """
        Check system resources and perform scaling if needed.
        
        Returns:
            Dict containing scaling decisions and new parameters
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_scaling_time < self.scaling_cooldown:
            return {"action": "no_change", "reason": "cooldown_active"}
        
        # Get current resource usage
        try:
            cpu_percent = psutil.cpu_percent(interval=1.0)
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            # Record metrics
            self.metrics_collector.set_gauge("scaling_cpu_percent", cpu_percent)
            self.metrics_collector.set_gauge("scaling_memory_percent", memory_percent)
            
        except Exception as e:
            self.logger.error(f"Failed to get system resources: {e}")
            return {"action": "error", "reason": str(e)}
        
        scaling_action = {"action": "no_change", "timestamp": current_time}
        
        # Check if scaling is needed
        if memory_percent > self.memory_threshold:
            # Memory pressure - scale down
            scaling_action = self._scale_down_memory(memory_percent)
            
        elif cpu_percent > self.cpu_threshold:
            # CPU pressure - optimize for CPU efficiency
            scaling_action = self._scale_for_cpu(cpu_percent)
            
        elif memory_percent < 60 and cpu_percent < 50:
            # Resources available - scale up if beneficial
            scaling_action = self._scale_up_resources(cpu_percent, memory_percent)
        
        # Apply scaling if action was taken
        if scaling_action["action"] != "no_change":
            self.last_scaling_time = current_time
            self.scaling_actions.append(scaling_action)
            
            # Keep history limited
            if len(self.scaling_actions) > 100:
                self.scaling_actions = self.scaling_actions[-100:]
            
            self.logger.info(f"Auto-scaling action: {scaling_action}")
        
        return scaling_action
    
    def _scale_down_memory(self, memory_percent: float) -> Dict[str, Any]:
        """Scale down due to memory pressure."""
        action = {
            "action": "scale_down_memory",
            "reason": f"memory_pressure_{memory_percent:.1f}%",
            "old_batch_size": self.current_batch_size,
            "old_worker_count": self.current_worker_count
        }
        
        # Reduce batch size first
        if self.current_batch_size > 8:
            self.current_batch_size = max(8, self.current_batch_size // 2)
            action["new_batch_size"] = self.current_batch_size
        
        # Then reduce worker count if memory is still critical
        if memory_percent > 90 and self.current_worker_count > 2:
            self.current_worker_count = max(2, self.current_worker_count - 2)
            action["new_worker_count"] = self.current_worker_count
        
        return action
    
    def _scale_for_cpu(self, cpu_percent: float) -> Dict[str, Any]:
        """Optimize for CPU efficiency."""
        action = {
            "action": "optimize_cpu",
            "reason": f"cpu_pressure_{cpu_percent:.1f}%",
            "old_worker_count": self.current_worker_count
        }
        
        # Reduce worker count to avoid CPU oversubscription
        if self.current_worker_count > 4:
            self.current_worker_count = max(4, int(self.current_worker_count * 0.8))
            action["new_worker_count"] = self.current_worker_count
        
        return action
    
    def _scale_up_resources(self, cpu_percent: float, memory_percent: float) -> Dict[str, Any]:
        """Scale up when resources are available."""
        action = {
            "action": "scale_up",
            "reason": f"resources_available_cpu_{cpu_percent:.1f}%_mem_{memory_percent:.1f}%",
            "old_batch_size": self.current_batch_size,
            "old_worker_count": self.current_worker_count
        }
        
        # Increase batch size for better efficiency
        if self.current_batch_size < 128:
            self.current_batch_size = min(128, int(self.current_batch_size * 1.5))
            action["new_batch_size"] = self.current_batch_size
        
        # Increase worker count if CPU is underutilized
        if cpu_percent < 30 and self.current_worker_count < mp.cpu_count():
            self.current_worker_count = min(mp.cpu_count(), self.current_worker_count + 2)
            action["new_worker_count"] = self.current_worker_count
        
        return action
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current scaling parameters."""
        return {
            "batch_size": self.current_batch_size,
            "worker_count": self.current_worker_count,
            "memory_limit_mb": self.current_memory_limit,
            "cpu_threshold": self.cpu_threshold,
            "memory_threshold": self.memory_threshold
        }
    
    def get_scaling_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get recent scaling history."""
        return self.scaling_actions[-last_n:] if self.scaling_actions else []


def create_performance_optimizer(
    config: Optional[OptimizationConfig] = None,
    metrics_collector: Optional[MetricsCollector] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive performance optimization system.
    
    Args:
        config: Optimization configuration
        metrics_collector: Metrics collector for monitoring
        
    Returns:
        Dict containing optimization components
    """
    config = config or OptimizationConfig()
    
    # Create cache
    cache = None
    if config.enable_caching:
        cache = AdaptiveCache(
            max_size=config.cache_size,
            enable_adaptive=True
        )
    
    # Create memory pool
    memory_pool = None
    if config.enable_memory_pooling:
        memory_pool = MemoryPool(config.memory_pool_size)
    
    # Create parallel processor
    parallel_processor = None
    if config.enable_parallel:
        parallel_processor = ParallelProcessor(
            max_workers=config.max_workers,
            use_processes=False  # Start with threads for better memory sharing
        )
    
    # Create auto-scaler
    auto_scaler = None
    if config.enable_auto_scaling and metrics_collector:
        auto_scaler = AutoScaler(
            metrics_collector=metrics_collector,
            cpu_threshold=config.scaling_threshold_cpu,
            memory_threshold=config.scaling_threshold_memory
        )
    
    return {
        "config": config,
        "cache": cache,
        "memory_pool": memory_pool,
        "parallel_processor": parallel_processor,
        "auto_scaler": auto_scaler
    }


def cached_computation(cache: AdaptiveCache, key_func: Optional[Callable] = None):
    """
    Decorator for caching expensive computations.
    
    Args:
        cache: Cache instance to use
        key_func: Function to generate cache key (optional)
        
    Returns:
        Decorated function with caching
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                cache_key = f"{func.__name__}_{hash(str(args))}_{hash(str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


class BatchProcessor:
    """
    High-performance batch processor with adaptive batching strategies.
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        memory_pool: Optional[MemoryPool] = None,
        parallel_processor: Optional[ParallelProcessor] = None
    ):
        self.batch_size = batch_size
        self.memory_pool = memory_pool
        self.parallel_processor = parallel_processor
        self.logger = logging.getLogger(__name__)
    
    def process_batches(
        self,
        data: List[Any],
        process_func: Callable,
        **kwargs
    ) -> List[Any]:
        """Process data in optimized batches."""
        if not data:
            return []
        
        # Create batches
        batches = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batches.append(batch)
        
        self.logger.debug(f"Processing {len(data)} items in {len(batches)} batches")
        
        # Process batches
        if self.parallel_processor and len(batches) > 1:
            # Parallel batch processing
            batch_results = self.parallel_processor.process_parallel(
                self._process_single_batch,
                batches,
                process_func=process_func,
                **kwargs
            )
        else:
            # Sequential batch processing
            batch_results = []
            for batch in batches:
                result = self._process_single_batch(batch, process_func, **kwargs)
                batch_results.append(result)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            if batch_result:
                results.extend(batch_result)
        
        return results
    
    def _process_single_batch(
        self,
        batch: List[Any],
        process_func: Callable,
        **kwargs
    ) -> List[Any]:
        """Process a single batch of data."""
        try:
            return [process_func(item, **kwargs) for item in batch]
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            return [None] * len(batch)  # Return placeholder results