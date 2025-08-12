"""
Advanced scaling and distributed computing for photonic neuromorphics.

This module provides horizontal and vertical scaling capabilities, distributed
simulation, and cloud-native deployment features for large-scale photonic networks.
"""

import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
import time
import uuid
import json
import pickle
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue, Empty
import logging
import numpy as np
import torch

from .enhanced_logging import PhotonicLogger, logged_operation
from .robust_error_handling import ErrorHandler, robust_operation
from .monitoring import MetricsCollector


@dataclass
class ScalingConfig:
    """Configuration for scaling operations."""
    max_workers: int = mp.cpu_count()
    batch_size: int = 1000
    chunk_size: int = 100
    memory_limit_gb: float = 8.0
    gpu_enabled: bool = False
    distributed_enabled: bool = False
    auto_scaling: bool = True
    scaling_factor: float = 1.5
    min_workers: int = 2
    max_workers_limit: int = 64
    load_threshold: float = 0.8
    scale_down_delay: float = 300.0  # 5 minutes


@dataclass
class WorkUnit:
    """Unit of work for distributed processing."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    estimated_duration: float = 1.0
    memory_requirement: float = 100.0  # MB
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    status: str = "pending"  # pending, running, completed, failed
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    worker_id: Optional[str] = None
    error_message: Optional[str] = None
    
    def duration(self) -> Optional[float]:
        """Calculate actual duration if completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


class WorkerPool:
    """Adaptive worker pool for photonic simulations."""
    
    def __init__(
        self,
        config: ScalingConfig,
        worker_function: Callable[[WorkUnit], Any],
        logger: Optional[PhotonicLogger] = None
    ):
        self.config = config
        self.worker_function = worker_function
        self.logger = logger or PhotonicLogger()
        
        self._work_queue: Queue = Queue()
        self._result_queue: Queue = Queue()
        self._workers: List[threading.Thread] = []
        self._worker_stats: Dict[str, Dict[str, Any]] = {}
        self._shutdown = False
        self._metrics_collector = MetricsCollector()
        
        # Auto-scaling state
        self._current_workers = 0
        self._load_history: List[float] = []
        self._last_scale_time = time.time()
        
        # Start initial workers
        self._scale_workers(self.config.min_workers)
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self._monitor_thread.start()
    
    def submit_work(self, work_unit: WorkUnit) -> str:
        """Submit work unit to the pool."""
        self._work_queue.put(work_unit)
        self.logger.get_logger('worker_pool').debug(f"Submitted work unit {work_unit.id}")
        return work_unit.id
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[WorkUnit]:
        """Get completed work unit."""
        try:
            return self._result_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get pool status information."""
        queue_size = self._work_queue.qsize()
        completed_results = self._result_queue.qsize()
        
        # Calculate current load
        current_load = queue_size / max(self._current_workers, 1)
        
        return {
            'active_workers': self._current_workers,
            'queue_size': queue_size,
            'completed_results': completed_results,
            'current_load': current_load,
            'worker_stats': self._worker_stats.copy(),
            'auto_scaling_enabled': self.config.auto_scaling
        }
    
    def _worker_loop(self, worker_id: str) -> None:
        """Main worker loop."""
        self._worker_stats[worker_id] = {
            'tasks_completed': 0,
            'total_duration': 0.0,
            'last_activity': time.time(),
            'status': 'idle'
        }
        
        logger = self.logger.get_logger(f'worker_{worker_id}')
        
        while not self._shutdown:
            try:
                # Get work unit with timeout
                work_unit = self._work_queue.get(timeout=1.0)
                
                # Update worker stats
                self._worker_stats[worker_id]['status'] = 'busy'
                self._worker_stats[worker_id]['last_activity'] = time.time()
                
                # Process work unit
                work_unit.worker_id = worker_id
                work_unit.started_at = time.time()
                work_unit.status = "running"
                
                logger.debug(f"Processing work unit {work_unit.id}")
                
                try:
                    # Execute work function
                    result = self.worker_function(work_unit)
                    work_unit.result = result
                    work_unit.status = "completed"
                    work_unit.completed_at = time.time()
                    
                    # Update metrics
                    duration = work_unit.duration()
                    if duration:
                        self._worker_stats[worker_id]['tasks_completed'] += 1
                        self._worker_stats[worker_id]['total_duration'] += duration
                        self._metrics_collector.record_operation_time(work_unit.operation, duration)
                    
                    logger.debug(f"Completed work unit {work_unit.id} in {duration:.3f}s")
                
                except Exception as e:
                    work_unit.status = "failed"
                    work_unit.error_message = str(e)
                    work_unit.completed_at = time.time()
                    
                    logger.error(f"Work unit {work_unit.id} failed: {e}")
                    self._metrics_collector.record_error(work_unit.operation, str(e))
                
                # Return result
                self._result_queue.put(work_unit)
                self._work_queue.task_done()
                
                # Update worker status
                self._worker_stats[worker_id]['status'] = 'idle'
                
            except Empty:
                # No work available, continue loop
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                continue
    
    def _scale_workers(self, target_count: int) -> None:
        """Scale worker pool to target count."""
        current_count = len(self._workers)
        
        if target_count > current_count:
            # Scale up
            for i in range(target_count - current_count):
                worker_id = f"worker_{len(self._workers)}"
                worker = threading.Thread(
                    target=self._worker_loop,
                    args=(worker_id,),
                    daemon=True
                )
                worker.start()
                self._workers.append(worker)
                
                self.logger.get_logger('worker_pool').info(f"Started worker {worker_id}")
        
        elif target_count < current_count:
            # Scale down by setting shutdown flag
            # Workers will naturally finish and exit
            pass
        
        self._current_workers = len(self._workers)
        self._last_scale_time = time.time()
    
    def _monitor_performance(self) -> None:
        """Monitor performance and auto-scale if enabled."""
        while not self._shutdown:
            try:
                # Calculate current load
                queue_size = self._work_queue.qsize()
                current_load = queue_size / max(self._current_workers, 1)
                
                self._load_history.append(current_load)
                if len(self._load_history) > 60:  # Keep last 60 measurements
                    self._load_history.pop(0)
                
                # Auto-scaling decisions
                if self.config.auto_scaling and len(self._load_history) >= 10:
                    avg_load = sum(self._load_history[-10:]) / 10
                    time_since_last_scale = time.time() - self._last_scale_time
                    
                    if avg_load > self.config.load_threshold and time_since_last_scale > 30:
                        # Scale up
                        new_worker_count = min(
                            int(self._current_workers * self.config.scaling_factor),
                            self.config.max_workers_limit
                        )
                        if new_worker_count > self._current_workers:
                            self._scale_workers(new_worker_count)
                            self.logger.get_logger('worker_pool').info(
                                f"Auto-scaled up to {new_worker_count} workers (load: {avg_load:.2f})"
                            )
                    
                    elif avg_load < 0.3 and time_since_last_scale > self.config.scale_down_delay:
                        # Scale down
                        new_worker_count = max(
                            int(self._current_workers / self.config.scaling_factor),
                            self.config.min_workers
                        )
                        if new_worker_count < self._current_workers:
                            self._scale_workers(new_worker_count)
                            self.logger.get_logger('worker_pool').info(
                                f"Auto-scaled down to {new_worker_count} workers (load: {avg_load:.2f})"
                            )
                
                time.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                self.logger.get_logger('worker_pool').error(f"Monitor error: {e}")
                time.sleep(10.0)
    
    def shutdown(self, timeout: float = 30.0) -> None:
        """Shutdown worker pool."""
        self.logger.get_logger('worker_pool').info("Shutting down worker pool")
        
        self._shutdown = True
        
        # Wait for workers to finish
        start_time = time.time()
        for worker in self._workers:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time > 0:
                worker.join(timeout=remaining_time)
        
        # Monitor thread will exit naturally
        self._monitor_thread.join(timeout=5.0)


class DistributedSimulation:
    """Distributed photonic simulation across multiple nodes."""
    
    def __init__(
        self,
        config: ScalingConfig,
        logger: Optional[PhotonicLogger] = None
    ):
        self.config = config
        self.logger = logger or PhotonicLogger()
        self._work_pools: Dict[str, WorkerPool] = {}
        self._node_registry: Dict[str, Dict[str, Any]] = {}
        self._job_registry: Dict[str, Dict[str, Any]] = {}
        
    def register_node(
        self,
        node_id: str,
        capabilities: Dict[str, Any],
        worker_function: Callable[[WorkUnit], Any]
    ) -> None:
        """Register a compute node."""
        self._node_registry[node_id] = {
            'capabilities': capabilities,
            'status': 'available',
            'last_heartbeat': time.time()
        }
        
        # Create worker pool for this node
        node_config = ScalingConfig(
            max_workers=capabilities.get('max_workers', self.config.max_workers),
            memory_limit_gb=capabilities.get('memory_gb', self.config.memory_limit_gb)
        )
        
        self._work_pools[node_id] = WorkerPool(
            node_config,
            worker_function,
            self.logger
        )
        
        self.logger.get_logger('distributed').info(f"Registered node {node_id}")
    
    def submit_distributed_job(
        self,
        job_name: str,
        work_units: List[WorkUnit],
        requirements: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit distributed job."""
        job_id = str(uuid.uuid4())
        
        self._job_registry[job_id] = {
            'name': job_name,
            'status': 'submitted',
            'total_units': len(work_units),
            'completed_units': 0,
            'failed_units': 0,
            'requirements': requirements or {},
            'submitted_at': time.time(),
            'work_unit_ids': [unit.id for unit in work_units]
        }
        
        # Distribute work units across nodes
        self._distribute_work(job_id, work_units, requirements)
        
        return job_id
    
    def _distribute_work(
        self,
        job_id: str,
        work_units: List[WorkUnit],
        requirements: Optional[Dict[str, Any]] = None
    ) -> None:
        """Distribute work units across available nodes."""
        available_nodes = [
            node_id for node_id, info in self._node_registry.items()
            if info['status'] == 'available'
        ]
        
        if not available_nodes:
            raise RuntimeError("No available nodes for distributed execution")
        
        # Simple round-robin distribution for now
        for i, work_unit in enumerate(work_units):
            node_id = available_nodes[i % len(available_nodes)]
            
            # Check if node meets requirements
            if requirements and not self._node_meets_requirements(node_id, requirements):
                continue
            
            # Submit to node's worker pool
            pool = self._work_pools[node_id]
            pool.submit_work(work_unit)
        
        self.logger.get_logger('distributed').info(
            f"Distributed {len(work_units)} work units across {len(available_nodes)} nodes"
        )
    
    def _node_meets_requirements(self, node_id: str, requirements: Dict[str, Any]) -> bool:
        """Check if node meets job requirements."""
        node_info = self._node_registry[node_id]
        capabilities = node_info['capabilities']
        
        # Check memory requirement
        if 'memory_gb' in requirements:
            if capabilities.get('memory_gb', 0) < requirements['memory_gb']:
                return False
        
        # Check GPU requirement
        if requirements.get('gpu_required', False):
            if not capabilities.get('gpu_available', False):
                return False
        
        return True
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of distributed job."""
        if job_id not in self._job_registry:
            return None
        
        job_info = self._job_registry[job_id].copy()
        
        # Update completion status by checking worker pools
        completed_count = 0
        failed_count = 0
        
        for pool in self._work_pools.values():
            # Count completed results (simplified)
            while True:
                result = pool.get_result(timeout=0.1)
                if result is None:
                    break
                
                if result.id in job_info['work_unit_ids']:
                    if result.status == 'completed':
                        completed_count += 1
                    elif result.status == 'failed':
                        failed_count += 1
        
        job_info['completed_units'] = completed_count
        job_info['failed_units'] = failed_count
        
        # Update overall status
        if completed_count + failed_count == job_info['total_units']:
            job_info['status'] = 'completed'
        elif completed_count + failed_count > 0:
            job_info['status'] = 'running'
        
        return job_info
    
    def shutdown(self) -> None:
        """Shutdown distributed simulation."""
        for pool in self._work_pools.values():
            pool.shutdown()


class GPUAccelerator:
    """GPU acceleration for photonic simulations."""
    
    def __init__(self, logger: Optional[PhotonicLogger] = None):
        self.logger = logger or PhotonicLogger()
        self.device = self._detect_gpu_device()
        self._memory_pool = {}
        
    def _detect_gpu_device(self) -> torch.device:
        """Detect available GPU device."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.get_logger('gpu').info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            self.logger.get_logger('gpu').info("Using Apple Metal Performance Shaders")
        else:
            device = torch.device('cpu')
            self.logger.get_logger('gpu').info("Using CPU (no GPU acceleration)")
        
        return device
    
    @logged_operation("gpu_simulation", "gpu_accelerator")
    def accelerate_simulation(
        self,
        simulation_function: Callable,
        parameters: Dict[str, Any],
        batch_size: int = 1000
    ) -> Any:
        """Accelerate simulation using GPU."""
        if self.device.type == 'cpu':
            # No GPU available, run on CPU
            return simulation_function(parameters)
        
        try:
            # Move data to GPU
            gpu_parameters = self._move_to_device(parameters)
            
            # Enable GPU optimizations
            with torch.cuda.device(self.device):
                if hasattr(torch.cuda, 'amp'):
                    # Use automatic mixed precision if available
                    with torch.cuda.amp.autocast():
                        result = simulation_function(gpu_parameters)
                else:
                    result = simulation_function(gpu_parameters)
            
            # Move result back to CPU
            cpu_result = self._move_to_cpu(result)
            
            return cpu_result
            
        except Exception as e:
            self.logger.get_logger('gpu').warning(f"GPU acceleration failed: {e}, falling back to CPU")
            return simulation_function(parameters)
    
    def _move_to_device(self, data: Any) -> Any:
        """Move data to GPU device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._move_to_device(item) for item in data]
        else:
            return data
    
    def _move_to_cpu(self, data: Any) -> Any:
        """Move data back to CPU."""
        if isinstance(data, torch.Tensor):
            return data.cpu()
        elif isinstance(data, dict):
            return {k: self._move_to_cpu(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._move_to_cpu(item) for item in data]
        else:
            return data
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get GPU memory statistics."""
        if self.device.type == 'cuda':
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
                'device_name': torch.cuda.get_device_name(),
                'device_count': torch.cuda.device_count()
            }
        else:
            return {'message': 'GPU memory stats not available'}


class HorizontalScaler:
    """Horizontal scaling for photonic simulations."""
    
    def __init__(
        self,
        config: ScalingConfig,
        logger: Optional[PhotonicLogger] = None
    ):
        self.config = config
        self.logger = logger or PhotonicLogger()
        self.distributed_sim = DistributedSimulation(config, logger)
        self.gpu_accelerator = GPUAccelerator(logger)
        
    @robust_operation(max_retries=3)
    def scale_simulation(
        self,
        simulation_function: Callable,
        parameter_sets: List[Dict[str, Any]],
        use_gpu: bool = True,
        distributed: bool = False
    ) -> List[Any]:
        """Scale simulation across multiple workers/nodes."""
        total_sets = len(parameter_sets)
        
        self.logger.get_logger('scaler').info(
            f"Scaling simulation for {total_sets} parameter sets"
        )
        
        if distributed and len(parameter_sets) > 1000:
            # Use distributed simulation for large workloads
            return self._distributed_scale(simulation_function, parameter_sets)
        elif use_gpu and self.gpu_accelerator.device.type != 'cpu':
            # Use GPU acceleration
            return self._gpu_scale(simulation_function, parameter_sets)
        else:
            # Use local parallel processing
            return self._local_scale(simulation_function, parameter_sets)
    
    def _local_scale(
        self,
        simulation_function: Callable,
        parameter_sets: List[Dict[str, Any]]
    ) -> List[Any]:
        """Scale using local parallel processing."""
        results = []
        
        # Determine optimal number of workers
        optimal_workers = min(self.config.max_workers, len(parameter_sets))
        
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(simulation_function, params): params
                for params in parameter_sets
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_params):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                except Exception as e:
                    self.logger.get_logger('scaler').error(f"Simulation failed: {e}")
                    results.append(None)
        
        return results
    
    def _gpu_scale(
        self,
        simulation_function: Callable,
        parameter_sets: List[Dict[str, Any]]
    ) -> List[Any]:
        """Scale using GPU acceleration."""
        results = []
        batch_size = self.config.batch_size
        
        # Process in batches to manage GPU memory
        for i in range(0, len(parameter_sets), batch_size):
            batch = parameter_sets[i:i + batch_size]
            
            # Create batched parameters
            batched_params = self._create_batch_parameters(batch)
            
            # Run on GPU
            batch_result = self.gpu_accelerator.accelerate_simulation(
                simulation_function,
                batched_params,
                len(batch)
            )
            
            # Split batch result back to individual results
            individual_results = self._split_batch_result(batch_result, len(batch))
            results.extend(individual_results)
        
        return results
    
    def _distributed_scale(
        self,
        simulation_function: Callable,
        parameter_sets: List[Dict[str, Any]]
    ) -> List[Any]:
        """Scale using distributed simulation."""
        # Create work units
        work_units = []
        for i, params in enumerate(parameter_sets):
            work_unit = WorkUnit(
                operation="photonic_simulation",
                parameters=params,
                priority=1,
                estimated_duration=10.0  # Estimate based on complexity
            )
            work_units.append(work_unit)
        
        # Submit distributed job
        job_id = self.distributed_sim.submit_distributed_job(
            "scaled_simulation",
            work_units
        )
        
        # Wait for completion and collect results
        results = []
        while True:
            status = self.distributed_sim.get_job_status(job_id)
            if not status:
                break
            
            if status['status'] == 'completed':
                # Collect all results
                for work_unit in work_units:
                    if work_unit.result is not None:
                        results.append(work_unit.result)
                    else:
                        results.append(None)
                break
            
            time.sleep(1.0)  # Poll every second
        
        return results
    
    def _create_batch_parameters(self, parameter_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create batched parameters for GPU processing."""
        # This would depend on the specific simulation function
        # For now, return first parameter set as example
        if parameter_sets:
            return parameter_sets[0]
        return {}
    
    def _split_batch_result(self, batch_result: Any, count: int) -> List[Any]:
        """Split batch result into individual results."""
        # This would depend on the specific result format
        # For now, replicate result for each item
        return [batch_result] * count


def create_scaling_system(
    max_workers: Optional[int] = None,
    enable_gpu: bool = True,
    enable_distributed: bool = False,
    logger: Optional[PhotonicLogger] = None
) -> HorizontalScaler:
    """Create a comprehensive scaling system."""
    config = ScalingConfig(
        max_workers=max_workers or mp.cpu_count(),
        gpu_enabled=enable_gpu,
        distributed_enabled=enable_distributed,
        auto_scaling=True
    )
    
    return HorizontalScaler(config, logger)


@logged_operation("batch_simulation", "scaling")
def batch_simulate_photonic_networks(
    networks: List[Any],
    simulation_params: Dict[str, Any],
    scaler: Optional[HorizontalScaler] = None
) -> List[Any]:
    """Batch simulate multiple photonic networks with scaling."""
    if scaler is None:
        scaler = create_scaling_system()
    
    # Create parameter sets for each network
    parameter_sets = []
    for network in networks:
        params = simulation_params.copy()
        params['network'] = network
        parameter_sets.append(params)
    
    # Define simulation function
    def simulate_single_network(params):
        network = params['network']
        # Run actual simulation (would call network.simulate() or similar)
        return {'network_id': id(network), 'simulation_result': 'completed'}
    
    # Scale the simulation
    results = scaler.scale_simulation(
        simulate_single_network,
        parameter_sets,
        use_gpu=simulation_params.get('use_gpu', True),
        distributed=simulation_params.get('distributed', False)
    )
    
    return results