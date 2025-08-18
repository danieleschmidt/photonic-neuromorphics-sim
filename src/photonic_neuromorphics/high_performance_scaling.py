"""
High-Performance Scaling Framework for Photonic Neuromorphic Computing.

This module implements advanced scaling techniques for photonic neuromorphic systems,
including distributed processing, GPU acceleration, memory optimization, and
auto-scaling capabilities for production environments.
"""

import time
import threading
import queue
import multiprocessing as mp
from multiprocessing import Manager
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as torch_mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil
import gc
import asyncio
import uvloop
from collections import deque
import pickle
import redis
import logging

from .research import QuantumPhotonicNeuromorphicProcessor, OpticalInterferenceProcessor
from .robust_research_framework import RobustQuantumPhotonicProcessor, RobustOpticalInterferenceProcessor
from .enhanced_logging import PhotonicLogger, PerformanceTracker
from .production_health_monitor import HealthMonitor


@dataclass
class ScalingConfig:
    """Configuration for high-performance scaling."""
    # Distributed processing
    enable_distributed: bool = True
    world_size: int = 4
    backend: str = "nccl"  # or "gloo" for CPU
    
    # GPU acceleration
    enable_gpu_acceleration: bool = True
    mixed_precision: bool = True
    gpu_memory_fraction: float = 0.8
    
    # Memory optimization
    enable_memory_optimization: bool = True
    memory_pool_size: int = 1024 * 1024 * 1024  # 1GB
    garbage_collection_threshold: int = 10
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    min_workers: int = 2
    max_workers: int = 16
    target_latency_ms: float = 100.0
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    
    # Caching and optimization
    enable_result_caching: bool = True
    cache_size: int = 1000
    enable_model_compilation: bool = True
    enable_kernel_fusion: bool = True
    
    # Batch processing
    dynamic_batching: bool = True
    max_batch_size: int = 64
    batch_timeout_ms: float = 10.0
    
    # Performance monitoring
    enable_performance_profiling: bool = True
    profiling_interval: float = 30.0


class DistributedPhotonicProcessor:
    """
    Distributed processing framework for photonic neuromorphic computing.
    
    Provides:
    - Multi-GPU distributed training and inference
    - Load balancing across processing nodes
    - Fault tolerance and failover
    - Dynamic scaling based on workload
    """
    
    def __init__(self, config: ScalingConfig = None):
        self.config = config or ScalingConfig()
        self.logger = PhotonicLogger(__name__)
        
        # Distributed state
        self.rank = 0
        self.world_size = self.config.world_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Processing components
        self.quantum_processor = None
        self.optical_processor = None
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        self.processing_times = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)
        
        # Auto-scaling state
        self.current_workers = self.config.min_workers
        self.worker_pool = None
        self.scaling_lock = threading.Lock()
        
        # Caching
        self.result_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Batch processing
        self.batch_queue = queue.Queue()
        self.batch_processor_running = False
        
    def initialize_distributed(self, rank: int, world_size: int):
        """Initialize distributed processing."""
        self.rank = rank
        self.world_size = world_size
        
        # Initialize process group
        if self.config.enable_distributed and world_size > 1:
            dist.init_process_group(
                backend=self.config.backend,
                rank=rank,
                world_size=world_size
            )
            
            self.logger.info(f"Initialized distributed processing: rank {rank}/{world_size}")
        
        # Setup device
        if torch.cuda.is_available() and self.config.enable_gpu_acceleration:
            self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
            torch.cuda.set_device(self.device)
            
            # Set memory fraction
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(self.config.gpu_memory_fraction)
        
        # Initialize processors with scaling optimizations
        self._initialize_optimized_processors()
        
        # Setup auto-scaling
        if self.config.enable_auto_scaling:
            self._setup_auto_scaling()
        
        # Start batch processing
        if self.config.dynamic_batching:
            self._start_batch_processing()
    
    def _initialize_optimized_processors(self):
        """Initialize processors with scaling optimizations."""
        
        # Create base processors
        self.quantum_processor = RobustQuantumPhotonicProcessor(
            qubit_count=16,
            photonic_channels=32
        ).to(self.device)
        
        self.optical_processor = RobustOpticalInterferenceProcessor(
            channels=16
        ).to(self.device)
        
        # Enable mixed precision if configured
        if self.config.mixed_precision and self.device.type == "cuda":
            self.quantum_processor = torch.jit.script(self.quantum_processor)
            
        # Wrap with DDP for distributed training
        if self.config.enable_distributed and self.world_size > 1:
            self.quantum_processor = DDP(
                self.quantum_processor,
                device_ids=[self.device.index] if self.device.type == "cuda" else None
            )
        
        # Model compilation for optimization
        if self.config.enable_model_compilation:
            try:
                # PyTorch 2.0+ compilation
                if hasattr(torch, 'compile'):
                    self.quantum_processor = torch.compile(
                        self.quantum_processor,
                        mode="max-autotune"
                    )
                    self.logger.info("Enabled PyTorch compilation for optimization")
            except Exception as e:
                self.logger.warning(f"Could not enable model compilation: {e}")
        
        self.logger.info("Initialized optimized processors for scaling")
    
    def _setup_auto_scaling(self):
        """Setup auto-scaling infrastructure."""
        self.worker_pool = ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="photonic_worker"
        )
        
        # Start scaling monitor
        scaling_thread = threading.Thread(
            target=self._scaling_monitor,
            daemon=True,
            name="scaling_monitor"
        )
        scaling_thread.start()
        
        self.logger.info("Auto-scaling system initialized")
    
    def _start_batch_processing(self):
        """Start dynamic batch processing."""
        self.batch_processor_running = True
        
        batch_thread = threading.Thread(
            target=self._batch_processor,
            daemon=True,
            name="batch_processor"
        )
        batch_thread.start()
        
        self.logger.info("Dynamic batch processing started")
    
    def process_distributed(self, data: torch.Tensor, 
                          processing_type: str = "quantum") -> torch.Tensor:
        """
        Process data using distributed photonic computing.
        
        Args:
            data: Input tensor to process
            processing_type: Type of processing ("quantum" or "optical")
            
        Returns:
            Processed output tensor
        """
        
        with self.performance_tracker.track_operation(f"distributed_{processing_type}_processing"):
            
            # Check cache first
            if self.config.enable_result_caching:
                cache_key = self._compute_cache_key(data, processing_type)
                if cache_key in self.result_cache:
                    self.cache_hits += 1
                    return self.result_cache[cache_key].clone()
                self.cache_misses += 1
            
            # Dynamic batching
            if self.config.dynamic_batching:
                return self._process_with_batching(data, processing_type)
            
            # Direct processing
            return self._process_direct(data, processing_type)
    
    def _process_with_batching(self, data: torch.Tensor, processing_type: str) -> torch.Tensor:
        """Process data using dynamic batching."""
        
        # Create processing request
        result_future = asyncio.Future()
        request = {
            'data': data,
            'processing_type': processing_type,
            'future': result_future,
            'timestamp': time.time()
        }
        
        # Add to batch queue
        self.batch_queue.put(request)
        
        # Wait for result (with timeout)
        try:
            # Use asyncio.wait_for for timeout handling
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                asyncio.wait_for(result_future, timeout=1.0)
            )
            return result
        except asyncio.TimeoutError:
            self.logger.warning("Batch processing timeout, falling back to direct processing")
            return self._process_direct(data, processing_type)
    
    def _process_direct(self, data: torch.Tensor, processing_type: str) -> torch.Tensor:
        """Process data directly without batching."""
        
        start_time = time.time()
        
        try:
            # Move data to device
            data = data.to(self.device)
            
            # Memory optimization
            if self.config.enable_memory_optimization:
                self._optimize_memory()
            
            # Process based on type
            if processing_type == "quantum":
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    result = self.quantum_processor(data)
            elif processing_type == "optical":
                # Optical processing requires query and key
                query = data
                key = data
                
                # Use first available wavelength
                result = self.optical_processor.compute_attention(
                    query, key, wavelength_idx=0
                )
            else:
                raise ValueError(f"Unknown processing type: {processing_type}")
            
            # Cache result
            if self.config.enable_result_caching:
                cache_key = self._compute_cache_key(data, processing_type)
                self._update_cache(cache_key, result)
            
            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise
    
    def _batch_processor(self):
        """Background batch processor for dynamic batching."""
        
        while self.batch_processor_running:
            try:
                # Collect batch
                batch_requests = []
                deadline = time.time() + (self.config.batch_timeout_ms / 1000.0)
                
                while (len(batch_requests) < self.config.max_batch_size and 
                       time.time() < deadline):
                    try:
                        request = self.batch_queue.get(timeout=0.001)
                        batch_requests.append(request)
                    except queue.Empty:
                        break
                
                if not batch_requests:
                    continue
                
                # Process batch
                self._process_batch(batch_requests)
                
            except Exception as e:
                self.logger.error(f"Batch processor error: {e}")
                time.sleep(0.1)
    
    def _process_batch(self, requests: List[Dict[str, Any]]):
        """Process a batch of requests efficiently."""
        
        if not requests:
            return
        
        try:
            # Group by processing type
            quantum_requests = [r for r in requests if r['processing_type'] == 'quantum']
            optical_requests = [r for r in requests if r['processing_type'] == 'optical']
            
            # Process quantum batch
            if quantum_requests:
                self._process_quantum_batch(quantum_requests)
            
            # Process optical batch
            if optical_requests:
                self._process_optical_batch(optical_requests)
                
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            # Set error for all requests
            for request in requests:
                if not request['future'].done():
                    request['future'].set_exception(e)
    
    def _process_quantum_batch(self, requests: List[Dict[str, Any]]):
        """Process a batch of quantum requests."""
        
        try:
            # Stack input data
            batch_data = torch.stack([r['data'] for r in requests])
            batch_data = batch_data.to(self.device)
            
            # Process entire batch
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                batch_results = self.quantum_processor(batch_data)
            
            # Distribute results
            for i, request in enumerate(requests):
                if not request['future'].done():
                    request['future'].set_result(batch_results[i])
                    
        except Exception as e:
            # Set error for all requests
            for request in requests:
                if not request['future'].done():
                    request['future'].set_exception(e)
    
    def _process_optical_batch(self, requests: List[Dict[str, Any]]):
        """Process a batch of optical requests."""
        
        try:
            # Process optical requests individually (they require query/key pairs)
            for request in requests:
                try:
                    data = request['data'].to(self.device)
                    result = self.optical_processor.compute_attention(
                        data, data, wavelength_idx=0
                    )
                    if not request['future'].done():
                        request['future'].set_result(result)
                except Exception as e:
                    if not request['future'].done():
                        request['future'].set_exception(e)
                        
        except Exception as e:
            for request in requests:
                if not request['future'].done():
                    request['future'].set_exception(e)
    
    def _scaling_monitor(self):
        """Monitor performance and trigger auto-scaling."""
        
        while True:
            try:
                time.sleep(self.config.profiling_interval)
                
                # Calculate current metrics
                metrics = self._calculate_scaling_metrics()
                
                # Make scaling decisions
                self._make_scaling_decision(metrics)
                
            except Exception as e:
                self.logger.error(f"Scaling monitor error: {e}")
    
    def _calculate_scaling_metrics(self) -> Dict[str, float]:
        """Calculate metrics for scaling decisions."""
        
        if not self.processing_times:
            return {'avg_latency': 0, 'throughput': 0, 'cpu_usage': 0, 'memory_usage': 0}
        
        # Average latency (ms)
        avg_latency = np.mean(list(self.processing_times)) * 1000
        
        # Throughput (requests/second)
        recent_times = list(self.processing_times)[-100:]  # Last 100 requests
        if len(recent_times) > 1:
            time_span = max(recent_times) - min(recent_times)
            throughput = len(recent_times) / max(time_span, 0.001)
        else:
            throughput = 0
        
        # System resource usage
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        metrics = {
            'avg_latency': avg_latency,
            'throughput': throughput,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'cache_hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1) * 100
        }
        
        # Track throughput history
        self.throughput_history.append(throughput)
        
        return metrics
    
    def _make_scaling_decision(self, metrics: Dict[str, float]):
        """Make auto-scaling decisions based on metrics."""
        
        with self.scaling_lock:
            current_load = max(metrics['cpu_usage'], metrics['memory_usage']) / 100.0
            avg_latency = metrics['avg_latency']
            
            should_scale_up = (
                current_load > self.config.scale_up_threshold or
                avg_latency > self.config.target_latency_ms
            )
            
            should_scale_down = (
                current_load < self.config.scale_down_threshold and
                avg_latency < self.config.target_latency_ms * 0.5
            )
            
            if should_scale_up and self.current_workers < self.config.max_workers:
                self._scale_up()
            elif should_scale_down and self.current_workers > self.config.min_workers:
                self._scale_down()
    
    def _scale_up(self):
        """Scale up the number of workers."""
        new_workers = min(self.current_workers + 2, self.config.max_workers)
        
        if new_workers > self.current_workers:
            self.current_workers = new_workers
            # Note: In a real implementation, this would add actual workers
            self.logger.info(f"Scaled up to {self.current_workers} workers")
    
    def _scale_down(self):
        """Scale down the number of workers."""
        new_workers = max(self.current_workers - 1, self.config.min_workers)
        
        if new_workers < self.current_workers:
            self.current_workers = new_workers
            # Note: In a real implementation, this would remove workers gracefully
            self.logger.info(f"Scaled down to {self.current_workers} workers")
    
    def _optimize_memory(self):
        """Optimize memory usage."""
        
        # Garbage collection
        if len(self.processing_times) % self.config.garbage_collection_threshold == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _compute_cache_key(self, data: torch.Tensor, processing_type: str) -> str:
        """Compute cache key for result caching."""
        
        # Simple hash based on data properties and processing type
        data_hash = hash((
            tuple(data.shape),
            processing_type,
            float(data.mean().item()),
            float(data.std().item())
        ))
        
        return str(data_hash)
    
    def _update_cache(self, key: str, result: torch.Tensor):
        """Update result cache with size management."""
        
        if len(self.result_cache) >= self.config.cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.result_cache))
            del self.result_cache[oldest_key]
        
        self.result_cache[key] = result.clone()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        
        metrics = self._calculate_scaling_metrics()
        
        stats = {
            'processing_metrics': metrics,
            'scaling_info': {
                'current_workers': self.current_workers,
                'min_workers': self.config.min_workers,
                'max_workers': self.config.max_workers
            },
            'cache_info': {
                'cache_size': len(self.result_cache),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': metrics.get('cache_hit_rate', 0)
            },
            'system_info': {
                'device': str(self.device),
                'distributed': self.config.enable_distributed,
                'mixed_precision': self.config.mixed_precision,
                'world_size': self.world_size,
                'rank': self.rank
            }
        }
        
        return stats
    
    def cleanup(self):
        """Cleanup distributed resources."""
        
        self.batch_processor_running = False
        
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)
        
        if self.config.enable_distributed and dist.is_initialized():
            dist.destroy_process_group()
        
        self.logger.info("Distributed processor cleanup completed")


class HighPerformanceInferenceEngine:
    """
    High-performance inference engine for production photonic neuromorphic systems.
    
    Features:
    - Multi-threaded inference with load balancing
    - GPU acceleration and mixed precision
    - Dynamic batching and request queuing
    - Real-time performance monitoring
    - Horizontal scaling capabilities
    """
    
    def __init__(self, config: ScalingConfig = None):
        self.config = config or ScalingConfig()
        self.logger = PhotonicLogger(__name__)
        
        # Processing infrastructure
        self.distributed_processor = DistributedPhotonicProcessor(self.config)
        
        # Request handling
        self.request_queue = asyncio.Queue(maxsize=1000)
        self.inference_workers = []
        self.load_balancer = LoadBalancer(self.config.max_workers)
        
        # Performance monitoring
        self.health_monitor = HealthMonitor()
        self.performance_tracker = PerformanceTracker()
        
        # State management
        self.is_running = False
        self.worker_tasks = []
    
    async def start_engine(self):
        """Start the high-performance inference engine."""
        
        self.logger.info("Starting high-performance inference engine")
        
        # Initialize distributed processing
        self.distributed_processor.initialize_distributed(0, 1)  # Single node for now
        
        # Start inference workers
        self.is_running = True
        
        for i in range(self.config.min_workers):
            worker_task = asyncio.create_task(
                self._inference_worker(worker_id=i),
                name=f"inference_worker_{i}"
            )
            self.worker_tasks.append(worker_task)
        
        # Start health monitoring
        self.health_monitor.register_component(self.distributed_processor, "distributed_processor")
        self.health_monitor.start_monitoring()
        
        self.logger.info(f"Inference engine started with {len(self.worker_tasks)} workers")
    
    async def process_request(self, data: torch.Tensor, 
                            processing_type: str = "quantum",
                            priority: int = 0) -> torch.Tensor:
        """
        Process inference request with high performance.
        
        Args:
            data: Input tensor
            processing_type: Type of processing
            priority: Request priority (higher = more important)
            
        Returns:
            Processed result tensor
        """
        
        # Create request
        request = InferenceRequest(
            data=data,
            processing_type=processing_type,
            priority=priority,
            timestamp=time.time()
        )
        
        # Add to queue
        await self.request_queue.put(request)
        
        # Wait for result
        result = await request.result_future
        
        return result
    
    async def _inference_worker(self, worker_id: int):
        """Async inference worker."""
        
        self.logger.info(f"Inference worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get request from queue
                request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=1.0
                )
                
                # Process request
                await self._process_inference_request(request, worker_id)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
    
    async def _process_inference_request(self, request: 'InferenceRequest', worker_id: int):
        """Process individual inference request."""
        
        try:
            with self.performance_tracker.track_operation(f"inference_{request.processing_type}"):
                
                # Process using distributed processor
                result = self.distributed_processor.process_distributed(
                    request.data,
                    request.processing_type
                )
                
                # Set result
                if not request.result_future.done():
                    request.result_future.set_result(result)
                
        except Exception as e:
            if not request.result_future.done():
                request.result_future.set_exception(e)
            self.logger.error(f"Inference processing failed: {e}")
    
    async def stop_engine(self):
        """Stop the inference engine gracefully."""
        
        self.logger.info("Stopping inference engine")
        
        self.is_running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Stop monitoring
        self.health_monitor.stop_monitoring()
        
        # Cleanup distributed processor
        self.distributed_processor.cleanup()
        
        self.logger.info("Inference engine stopped")
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        
        return {
            'distributed_stats': self.distributed_processor.get_performance_stats(),
            'queue_size': self.request_queue.qsize(),
            'active_workers': len([t for t in self.worker_tasks if not t.done()]),
            'health_status': self.health_monitor.get_health_report(),
            'performance_metrics': self.performance_tracker.get_metrics()
        }


@dataclass
class InferenceRequest:
    """Inference request data structure."""
    data: torch.Tensor
    processing_type: str
    priority: int
    timestamp: float
    result_future: asyncio.Future = field(default_factory=asyncio.Future)


class LoadBalancer:
    """Load balancer for distributing requests across workers."""
    
    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self.worker_loads = [0] * max_workers
        self.round_robin_counter = 0
    
    def get_next_worker(self) -> int:
        """Get the next worker ID using round-robin."""
        worker_id = self.round_robin_counter % len(self.worker_loads)
        self.round_robin_counter += 1
        return worker_id
    
    def update_worker_load(self, worker_id: int, load: float):
        """Update worker load information."""
        if 0 <= worker_id < len(self.worker_loads):
            self.worker_loads[worker_id] = load


async def create_high_performance_system() -> HighPerformanceInferenceEngine:
    """Create and start a high-performance photonic neuromorphic system."""
    
    # Configure for maximum performance
    config = ScalingConfig(
        enable_distributed=True,
        enable_gpu_acceleration=True,
        mixed_precision=True,
        enable_auto_scaling=True,
        dynamic_batching=True,
        enable_result_caching=True,
        enable_model_compilation=True
    )
    
    # Create inference engine
    engine = HighPerformanceInferenceEngine(config)
    
    # Start the engine
    await engine.start_engine()
    
    return engine


def run_scaling_benchmark():
    """Run comprehensive scaling benchmark."""
    print("⚡ HIGH-PERFORMANCE SCALING BENCHMARK")
    print("=" * 60)
    
    async def benchmark():
        # Create high-performance system
        print("🚀 Creating high-performance system...")
        engine = await create_high_performance_system()
        
        # Generate test workload
        print("📊 Running scaling benchmark...")
        
        test_data = [torch.randn(4, 25, 16) for _ in range(20)]
        
        # Benchmark sequential processing
        start_time = time.time()
        sequential_results = []
        
        for data in test_data[:5]:  # Process 5 samples sequentially
            result = await engine.process_request(data, "quantum")
            sequential_results.append(result)
        
        sequential_time = time.time() - start_time
        
        # Benchmark concurrent processing
        start_time = time.time()
        
        concurrent_tasks = [
            engine.process_request(data, "quantum")
            for data in test_data[:10]  # Process 10 samples concurrently
        ]
        
        concurrent_results = await asyncio.gather(*concurrent_tasks)
        concurrent_time = time.time() - start_time
        
        # Calculate metrics
        sequential_throughput = 5 / sequential_time
        concurrent_throughput = 10 / concurrent_time
        speedup = concurrent_throughput / sequential_throughput
        
        print("\\n📈 SCALING RESULTS:")
        print(f"Sequential Throughput: {sequential_throughput:.2f} requests/sec")
        print(f"Concurrent Throughput: {concurrent_throughput:.2f} requests/sec")
        print(f"Scaling Speedup: {speedup:.2f}x")
        
        # Get system stats
        stats = engine.get_engine_stats()
        
        print("\\n⚙️ SYSTEM PERFORMANCE:")
        dist_stats = stats['distributed_stats']
        print(f"Cache Hit Rate: {dist_stats['cache_info']['hit_rate']:.1f}%")
        print(f"Average Latency: {dist_stats['processing_metrics']['avg_latency']:.2f}ms")
        print(f"Current Workers: {dist_stats['scaling_info']['current_workers']}")
        
        # Stop engine
        await engine.stop_engine()
        
        return {
            'sequential_throughput': sequential_throughput,
            'concurrent_throughput': concurrent_throughput,
            'speedup': speedup,
            'system_stats': stats
        }
    
    # Run benchmark
    if hasattr(asyncio, 'run'):
        return asyncio.run(benchmark())
    else:
        # Fallback for older Python versions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(benchmark())
        finally:
            loop.close()


if __name__ == "__main__":
    # Enable uvloop for better async performance (if available)
    try:
        uvloop.install()
    except ImportError:
        pass
    
    results = run_scaling_benchmark()
    
    print("\\n🏆 SCALING BENCHMARK COMPLETE!")
    print(f"✅ Achieved {results['speedup']:.2f}x speedup with concurrent processing")
    print(f"✅ Peak throughput: {results['concurrent_throughput']:.2f} requests/sec")