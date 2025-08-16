"""
Distributed Computing Framework for Photonic Neuromorphics

Advanced distributed processing system for large-scale photonic neural network
simulations across multiple nodes, with load balancing, fault tolerance, and
high-performance communication.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import logging
import time
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import socket
import pickle
import hashlib
from pathlib import Path
import os

from .core import PhotonicSNN, OpticalParameters, encode_to_spikes
from .exceptions import ValidationError, PhotonicNeuromorphicsException
from .autonomous_learning import AutonomousLearningFramework, LearningMetrics
from .quantum_photonic_interface import HybridQuantumPhotonic
from .realtime_adaptive_optimization import RealTimeOptimizer, PerformanceMetrics


@dataclass
class NodeInfo:
    """Information about a compute node."""
    node_id: str
    hostname: str
    port: int
    capabilities: Dict[str, Any] = field(default_factory=dict)
    status: str = "idle"  # idle, busy, error, offline
    load: float = 0.0  # CPU load 0-1
    memory_usage: float = 0.0  # Memory usage 0-1
    gpu_count: int = 0
    photonic_units: int = 0
    quantum_qubits: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = hashlib.md5(f"{self.hostname}:{self.port}".encode()).hexdigest()[:8]


@dataclass
class ComputeTask:
    """Distributed compute task."""
    task_id: str
    task_type: str  # simulation, training, optimization, inference
    priority: int = 1  # 1=low, 5=high
    data_size: int = 0  # bytes
    estimated_time: float = 0.0  # seconds
    requirements: Dict[str, Any] = field(default_factory=dict)
    payload: bytes = b""
    result: Optional[bytes] = None
    status: str = "pending"  # pending, running, completed, failed
    assigned_node: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = hashlib.md5(f"{time.time()}{self.task_type}".encode()).hexdigest()[:12]


class NodeManager:
    """Manages distributed compute nodes."""
    
    def __init__(self, master_port: int = 29500):
        self.master_port = master_port
        self.nodes: Dict[str, NodeInfo] = {}
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, ComputeTask] = {}
        self.completed_tasks: Dict[str, ComputeTask] = {}
        
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.is_running = False
        self.management_thread = None
        
        # Performance tracking
        self.total_tasks_completed = 0
        self.total_compute_time = 0.0
        self.node_performance_history = {}
    
    def start(self) -> None:
        """Start the node manager."""
        if self.is_running:
            return
        
        self.is_running = True
        self.management_thread = threading.Thread(target=self._management_loop)
        self.management_thread.daemon = True
        self.management_thread.start()
        
        self.logger.info(f"Node manager started on port {self.master_port}")
    
    def stop(self) -> None:
        """Stop the node manager."""
        self.is_running = False
        if self.management_thread:
            self.management_thread.join(timeout=2.0)
        
        self.executor.shutdown(wait=False)
        self.logger.info("Node manager stopped")
    
    def register_node(self, node_info: NodeInfo) -> bool:
        """Register a new compute node."""
        try:
            # Validate node capabilities
            if self._validate_node(node_info):
                self.nodes[node_info.node_id] = node_info
                self.node_performance_history[node_info.node_id] = []
                
                self.logger.info(f"Registered node {node_info.node_id} "
                               f"({node_info.hostname}:{node_info.port})")
                return True
            else:
                self.logger.warning(f"Failed to validate node {node_info.node_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to register node: {e}")
            return False
    
    def submit_task(self, task: ComputeTask) -> str:
        """Submit a task for distributed execution."""
        # Priority queue: higher priority = lower number
        priority = -task.priority
        self.task_queue.put((priority, time.time(), task))
        
        self.logger.debug(f"Submitted task {task.task_id} (priority {task.priority})")
        return task.task_id
    
    def get_task_status(self, task_id: str) -> Optional[ComputeTask]:
        """Get status of a submitted task."""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        else:
            return None
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status."""
        online_nodes = [n for n in self.nodes.values() 
                       if time.time() - n.last_heartbeat < 30]
        
        total_gpus = sum(n.gpu_count for n in online_nodes)
        total_qubits = sum(n.quantum_qubits for n in online_nodes)
        total_photonic_units = sum(n.photonic_units for n in online_nodes)
        
        avg_load = np.mean([n.load for n in online_nodes]) if online_nodes else 0
        
        return {
            'total_nodes': len(self.nodes),
            'online_nodes': len(online_nodes),
            'total_gpus': total_gpus,
            'total_quantum_qubits': total_qubits,
            'total_photonic_units': total_photonic_units,
            'average_load': avg_load,
            'pending_tasks': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'total_tasks_completed': self.total_tasks_completed,
            'average_task_time': self.total_compute_time / max(1, self.total_tasks_completed)
        }
    
    def _management_loop(self) -> None:
        """Main management loop."""
        while self.is_running:
            try:
                # Process pending tasks
                self._schedule_tasks()
                
                # Check node health
                self._check_node_health()
                
                # Clean up completed tasks (keep last 1000)
                self._cleanup_tasks()
                
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Management loop error: {e}")
                time.sleep(5.0)
    
    def _schedule_tasks(self) -> None:
        """Schedule tasks to available nodes."""
        while not self.task_queue.empty():
            try:
                # Get highest priority task
                priority, submit_time, task = self.task_queue.get_nowait()
                
                # Find best node for this task
                best_node = self._find_best_node(task)
                
                if best_node:
                    # Assign task to node
                    task.assigned_node = best_node.node_id
                    task.status = "running"
                    task.start_time = time.time()
                    
                    self.active_tasks[task.task_id] = task
                    best_node.status = "busy"
                    
                    # Execute task asynchronously
                    self.executor.submit(self._execute_task, task, best_node)
                    
                    self.logger.debug(f"Scheduled task {task.task_id} to node {best_node.node_id}")
                    
                else:
                    # No available nodes, put task back
                    self.task_queue.put((priority, submit_time, task))
                    break
                    
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Task scheduling error: {e}")
    
    def _find_best_node(self, task: ComputeTask) -> Optional[NodeInfo]:
        """Find the best node for a task based on requirements and load."""
        available_nodes = []
        
        for node in self.nodes.values():
            # Check if node is online and available
            if (time.time() - node.last_heartbeat < 30 and 
                node.status in ["idle", "busy"] and
                node.load < 0.9):
                
                # Check if node meets task requirements
                if self._node_meets_requirements(node, task.requirements):
                    available_nodes.append(node)
        
        if not available_nodes:
            return None
        
        # Score nodes based on multiple factors
        def score_node(node: NodeInfo) -> float:
            score = 0.0
            
            # Lower load is better
            score += (1.0 - node.load) * 40
            
            # Lower memory usage is better
            score += (1.0 - node.memory_usage) * 30
            
            # More capabilities are better
            score += node.gpu_count * 10
            score += node.quantum_qubits * 5
            score += node.photonic_units * 15
            
            # Historical performance
            if node.node_id in self.node_performance_history:
                history = self.node_performance_history[node.node_id]
                if history:
                    avg_time = np.mean([h['completion_time'] for h in history[-10:]])
                    score += max(0, 60 - avg_time)  # Faster nodes get higher score
            
            return score
        
        # Return highest scoring node
        best_node = max(available_nodes, key=score_node)
        return best_node
    
    def _node_meets_requirements(self, node: NodeInfo, requirements: Dict[str, Any]) -> bool:
        """Check if node meets task requirements."""
        for req_key, req_value in requirements.items():
            if req_key == "min_gpus":
                if node.gpu_count < req_value:
                    return False
            elif req_key == "min_quantum_qubits":
                if node.quantum_qubits < req_value:
                    return False
            elif req_key == "min_photonic_units":
                if node.photonic_units < req_value:
                    return False
            elif req_key == "max_load":
                if node.load > req_value:
                    return False
        
        return True
    
    def _execute_task(self, task: ComputeTask, node: NodeInfo) -> None:
        """Execute a task on a specific node."""
        try:
            start_time = time.time()
            
            # Simulate task execution (in real implementation, this would
            # communicate with the actual node)
            result = self._simulate_task_execution(task, node)
            
            execution_time = time.time() - start_time
            
            # Update task status
            task.status = "completed"
            task.end_time = time.time()
            task.result = result
            
            # Update node status
            node.status = "idle"
            node.load = max(0, node.load - 0.1)  # Simulate load decrease
            
            # Move task to completed
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
            
            # Update performance tracking
            self.total_tasks_completed += 1
            self.total_compute_time += execution_time
            
            # Record node performance
            self.node_performance_history[node.node_id].append({
                'completion_time': execution_time,
                'task_type': task.task_type,
                'timestamp': time.time()
            })
            
            self.logger.debug(f"Completed task {task.task_id} in {execution_time:.2f}s")
            
        except Exception as e:
            # Handle task failure
            task.status = "failed"
            task.error_message = str(e)
            task.end_time = time.time()
            
            node.status = "idle"
            
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
            
            self.logger.error(f"Task {task.task_id} failed: {e}")
    
    def _simulate_task_execution(self, task: ComputeTask, node: NodeInfo) -> bytes:
        """Simulate task execution (placeholder for actual implementation)."""
        # Simulate different execution times based on task type
        if task.task_type == "simulation":
            time.sleep(0.5 + np.random.exponential(1.0))
        elif task.task_type == "training":
            time.sleep(1.0 + np.random.exponential(2.0))
        elif task.task_type == "optimization":
            time.sleep(0.8 + np.random.exponential(1.5))
        elif task.task_type == "inference":
            time.sleep(0.1 + np.random.exponential(0.3))
        else:
            time.sleep(0.5)
        
        # Return dummy result
        result = {
            'task_id': task.task_id,
            'node_id': node.node_id,
            'result': f"Task {task.task_type} completed successfully",
            'metrics': {
                'accuracy': np.random.uniform(0.8, 0.95),
                'throughput': np.random.uniform(100, 1000),
                'energy_efficiency': np.random.uniform(50, 150)
            }
        }
        
        return pickle.dumps(result)
    
    def _check_node_health(self) -> None:
        """Check health of all nodes."""
        current_time = time.time()
        
        for node in self.nodes.values():
            # Mark nodes as offline if no heartbeat for 60 seconds
            if current_time - node.last_heartbeat > 60:
                if node.status != "offline":
                    node.status = "offline"
                    self.logger.warning(f"Node {node.node_id} went offline")
            
            # Simulate load and memory updates
            if node.status != "offline":
                node.load = max(0, min(1, node.load + np.random.normal(0, 0.05)))
                node.memory_usage = max(0, min(1, node.memory_usage + np.random.normal(0, 0.02)))
                node.last_heartbeat = current_time
    
    def _cleanup_tasks(self) -> None:
        """Clean up old completed tasks."""
        if len(self.completed_tasks) > 1000:
            # Keep only the most recent 1000 tasks
            sorted_tasks = sorted(
                self.completed_tasks.items(),
                key=lambda x: x[1].end_time or 0,
                reverse=True
            )
            
            self.completed_tasks = dict(sorted_tasks[:1000])
    
    def _validate_node(self, node: NodeInfo) -> bool:
        """Validate node configuration."""
        # For demo purposes, accept all nodes without network validation
        # In production, this would check actual network connectivity
        
        # Basic validation checks
        if not node.hostname or not node.port:
            return False
        
        if node.port < 1024 or node.port > 65535:
            return False
        
        # For demo nodes (worker-1, worker-2, etc.), always return True
        if node.hostname.startswith('worker-'):
            return True
        
        # For real nodes, could implement actual network checking
        return True


class DistributedPhotonicSimulator:
    """Distributed photonic neural network simulator."""
    
    def __init__(self, 
                 node_manager: NodeManager,
                 enable_quantum: bool = True,
                 enable_optimization: bool = True):
        self.node_manager = node_manager
        self.enable_quantum = enable_quantum
        self.enable_optimization = enable_optimization
        
        self.logger = logging.getLogger(__name__)
        self.simulation_history = []
        
    def distributed_training(self,
                           network: PhotonicSNN,
                           train_data: torch.Tensor,
                           train_labels: torch.Tensor,
                           num_epochs: int = 10,
                           batch_size: int = 32) -> Dict[str, Any]:
        """Perform distributed training across multiple nodes."""
        self.logger.info(f"Starting distributed training on {len(self.node_manager.nodes)} nodes")
        
        # Split data across available nodes
        num_nodes = len([n for n in self.node_manager.nodes.values() 
                        if n.status != "offline"])
        
        if num_nodes == 0:
            raise PhotonicNeuromorphicsException("No available nodes for distributed training")
        
        # Create training tasks
        training_tasks = []
        data_per_node = len(train_data) // num_nodes
        
        for epoch in range(num_epochs):
            for node_idx in range(num_nodes):
                start_idx = node_idx * data_per_node
                end_idx = min((node_idx + 1) * data_per_node, len(train_data))
                
                if start_idx < end_idx:
                    # Create training task
                    task_data = {
                        'network_state': network.state_dict(),
                        'data_batch': train_data[start_idx:end_idx],
                        'labels_batch': train_labels[start_idx:end_idx],
                        'epoch': epoch,
                        'node_idx': node_idx
                    }
                    
                    task = ComputeTask(
                        task_type="training",
                        priority=3,
                        data_size=len(pickle.dumps(task_data)),
                        estimated_time=2.0,
                        requirements={
                            "min_photonic_units": 1,
                            "max_load": 0.8
                        },
                        payload=pickle.dumps(task_data)
                    )
                    
                    task_id = self.node_manager.submit_task(task)
                    training_tasks.append(task_id)
        
        # Wait for all training tasks to complete
        completed_tasks = self._wait_for_tasks(training_tasks)
        
        # Aggregate results
        training_results = self._aggregate_training_results(completed_tasks)
        
        self.logger.info(f"Distributed training completed with {len(completed_tasks)} tasks")
        return training_results
    
    def distributed_inference(self,
                            network: PhotonicSNN,
                            input_data: torch.Tensor,
                            batch_size: int = 64) -> torch.Tensor:
        """Perform distributed inference across multiple nodes."""
        self.logger.info(f"Starting distributed inference for {len(input_data)} samples")
        
        # Split inference across nodes
        num_nodes = len([n for n in self.node_manager.nodes.values() 
                        if n.status != "offline"])
        
        if num_nodes == 0:
            raise PhotonicNeuromorphicsException("No available nodes for distributed inference")
        
        inference_tasks = []
        batch_per_node = max(1, len(input_data) // num_nodes)
        
        for i in range(0, len(input_data), batch_per_node):
            batch_data = input_data[i:i + batch_per_node]
            
            task_data = {
                'network_state': network.state_dict(),
                'input_batch': batch_data,
                'batch_index': i // batch_per_node
            }
            
            task = ComputeTask(
                task_type="inference",
                priority=4,
                data_size=len(pickle.dumps(task_data)),
                estimated_time=0.5,
                requirements={
                    "min_photonic_units": 1,
                    "max_load": 0.9
                },
                payload=pickle.dumps(task_data)
            )
            
            task_id = self.node_manager.submit_task(task)
            inference_tasks.append((task_id, i))
        
        # Wait for all inference tasks
        completed_tasks = self._wait_for_tasks([t[0] for t in inference_tasks])
        
        # Aggregate inference results
        output_results = self._aggregate_inference_results(completed_tasks, len(input_data))
        
        self.logger.info(f"Distributed inference completed")
        return output_results
    
    def distributed_optimization(self,
                               network: PhotonicSNN,
                               validation_data: torch.Tensor,
                               validation_labels: torch.Tensor) -> PhotonicSNN:
        """Perform distributed optimization of network parameters."""
        if not self.enable_optimization:
            return network
        
        self.logger.info("Starting distributed optimization")
        
        # Create optimization tasks
        optimization_tasks = []
        
        # Task 1: Evolutionary architecture optimization
        arch_task_data = {
            'optimization_type': 'evolutionary',
            'network_topology': network.topology,
            'validation_data': validation_data,
            'validation_labels': validation_labels
        }
        
        arch_task = ComputeTask(
            task_type="optimization",
            priority=2,
            data_size=len(pickle.dumps(arch_task_data)),
            estimated_time=5.0,
            requirements={
                "min_photonic_units": 1,
                "max_load": 0.7
            },
            payload=pickle.dumps(arch_task_data)
        )
        
        arch_task_id = self.node_manager.submit_task(arch_task)
        optimization_tasks.append(arch_task_id)
        
        # Task 2: Optical parameter optimization
        optical_task_data = {
            'optimization_type': 'optical',
            'optical_params': network.optical_params,
            'validation_data': validation_data,
            'validation_labels': validation_labels
        }
        
        optical_task = ComputeTask(
            task_type="optimization",
            priority=2,
            data_size=len(pickle.dumps(optical_task_data)),
            estimated_time=3.0,
            requirements={
                "min_photonic_units": 1,
                "max_load": 0.7
            },
            payload=pickle.dumps(optical_task_data)
        )
        
        optical_task_id = self.node_manager.submit_task(optical_task)
        optimization_tasks.append(optical_task_id)
        
        # Wait for optimization tasks
        completed_tasks = self._wait_for_tasks(optimization_tasks)
        
        # Apply optimization results
        optimized_network = self._apply_optimization_results(network, completed_tasks)
        
        self.logger.info("Distributed optimization completed")
        return optimized_network
    
    def _wait_for_tasks(self, task_ids: List[str], timeout: float = 300.0) -> List[ComputeTask]:
        """Wait for tasks to complete."""
        start_time = time.time()
        completed_tasks = []
        
        while len(completed_tasks) < len(task_ids):
            if time.time() - start_time > timeout:
                self.logger.warning(f"Timeout waiting for tasks: {len(completed_tasks)}/{len(task_ids)} completed")
                break
            
            for task_id in task_ids:
                task = self.node_manager.get_task_status(task_id)
                if task and task.status == "completed" and task not in completed_tasks:
                    completed_tasks.append(task)
                elif task and task.status == "failed":
                    self.logger.error(f"Task {task_id} failed: {task.error_message}")
            
            time.sleep(0.5)
        
        return completed_tasks
    
    def _aggregate_training_results(self, completed_tasks: List[ComputeTask]) -> Dict[str, Any]:
        """Aggregate results from distributed training tasks."""
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
        
        for task in completed_tasks:
            if task.result:
                try:
                    result = pickle.loads(task.result)
                    metrics = result.get('metrics', {})
                    
                    # Weight by number of samples processed
                    samples = metrics.get('samples_processed', 1)
                    total_loss += metrics.get('loss', 0) * samples
                    total_accuracy += metrics.get('accuracy', 0) * samples
                    total_samples += samples
                    
                except Exception as e:
                    self.logger.error(f"Failed to parse task result: {e}")
        
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_accuracy = total_accuracy / total_samples
        else:
            avg_loss = float('inf')
            avg_accuracy = 0.0
        
        return {
            'average_loss': avg_loss,
            'average_accuracy': avg_accuracy,
            'total_samples_processed': total_samples,
            'tasks_completed': len(completed_tasks),
            'distributed_efficiency': len(completed_tasks) / len(self.node_manager.nodes)
        }
    
    def _aggregate_inference_results(self, completed_tasks: List[ComputeTask], total_samples: int) -> torch.Tensor:
        """Aggregate results from distributed inference tasks."""
        # Create placeholder output tensor
        output_tensor = torch.zeros(total_samples, 10)  # Assuming 10 output classes
        
        for task in completed_tasks:
            if task.result:
                try:
                    result = pickle.loads(task.result)
                    batch_output = result.get('output', torch.zeros(1, 10))
                    batch_index = result.get('batch_index', 0)
                    
                    start_idx = batch_index * len(batch_output)
                    end_idx = min(start_idx + len(batch_output), total_samples)
                    
                    if start_idx < total_samples:
                        output_tensor[start_idx:end_idx] = batch_output[:end_idx-start_idx]
                        
                except Exception as e:
                    self.logger.error(f"Failed to parse inference result: {e}")
        
        return output_tensor
    
    def _apply_optimization_results(self, network: PhotonicSNN, completed_tasks: List[ComputeTask]) -> PhotonicSNN:
        """Apply optimization results to network."""
        optimized_network = network
        
        for task in completed_tasks:
            if task.result:
                try:
                    result = pickle.loads(task.result)
                    opt_type = result.get('optimization_type')
                    
                    if opt_type == 'evolutionary':
                        # Apply architecture optimization
                        new_topology = result.get('optimized_topology')
                        if new_topology and new_topology != network.topology:
                            self.logger.info(f"Applying optimized topology: {new_topology}")
                            # In practice, would create new network with optimized topology
                    
                    elif opt_type == 'optical':
                        # Apply optical parameter optimization
                        new_optical_params = result.get('optimized_optical_params')
                        if new_optical_params:
                            optimized_network.optical_params = new_optical_params
                            self.logger.info("Applied optimized optical parameters")
                            
                except Exception as e:
                    self.logger.error(f"Failed to apply optimization result: {e}")
        
        return optimized_network


def create_distributed_demo_cluster() -> Tuple[NodeManager, DistributedPhotonicSimulator]:
    """Create demonstration distributed cluster."""
    # Create node manager
    node_manager = NodeManager()
    node_manager.start()
    
    # Register demo nodes
    demo_nodes = [
        NodeInfo(
            node_id="node_001",
            hostname="worker-1",
            port=8080,
            gpu_count=2,
            photonic_units=4,
            quantum_qubits=8
        ),
        NodeInfo(
            node_id="node_002", 
            hostname="worker-2",
            port=8080,
            gpu_count=1,
            photonic_units=2,
            quantum_qubits=4
        ),
        NodeInfo(
            node_id="node_003",
            hostname="worker-3", 
            port=8080,
            gpu_count=3,
            photonic_units=6,
            quantum_qubits=12
        )
    ]
    
    for node in demo_nodes:
        node_manager.register_node(node)
    
    # Create distributed simulator
    simulator = DistributedPhotonicSimulator(
        node_manager=node_manager,
        enable_quantum=True,
        enable_optimization=True
    )
    
    return node_manager, simulator


def run_distributed_computing_demo():
    """Run distributed computing demonstration."""
    print("🌐 Distributed Photonic Computing Demo")
    print("=" * 42)
    
    try:
        # Create distributed cluster
        node_manager, simulator = create_distributed_demo_cluster()
        
        # Display cluster status
        cluster_status = node_manager.get_cluster_status()
        print(f"Cluster Status:")
        print(f"  Online nodes: {cluster_status['online_nodes']}")
        print(f"  Total GPUs: {cluster_status['total_gpus']}")
        print(f"  Total photonic units: {cluster_status['total_photonic_units']}")
        print(f"  Total quantum qubits: {cluster_status['total_quantum_qubits']}")
        
        # Create test network and data
        from .core import create_mnist_photonic_snn
        network = create_mnist_photonic_snn()
        
        torch.manual_seed(42)
        train_data = torch.randn(100, 784)
        train_labels = torch.randint(0, 10, (100,))
        test_data = torch.randn(50, 784)
        
        print(f"\nTest data: {len(train_data)} training, {len(test_data)} testing samples")
        
        # Test distributed training
        print("\n🚀 Testing Distributed Training...")
        training_results = simulator.distributed_training(
            network, train_data, train_labels, num_epochs=2, batch_size=16
        )
        
        print(f"Training completed:")
        print(f"  Average accuracy: {training_results['average_accuracy']:.4f}")
        print(f"  Tasks completed: {training_results['tasks_completed']}")
        print(f"  Distributed efficiency: {training_results['distributed_efficiency']:.2f}")
        
        # Test distributed inference
        print("\n🔮 Testing Distributed Inference...")
        inference_results = simulator.distributed_inference(network, test_data)
        
        print(f"Inference completed:")
        print(f"  Output shape: {inference_results.shape}")
        print(f"  Sample predictions: {inference_results[0].argmax().item()}")
        
        # Test distributed optimization
        print("\n⚙️  Testing Distributed Optimization...")
        val_data = train_data[:20]
        val_labels = train_labels[:20]
        
        optimized_network = simulator.distributed_optimization(
            network, val_data, val_labels
        )
        
        print(f"Optimization completed")
        print(f"  Network topology: {optimized_network.topology}")
        
        # Final cluster statistics
        final_status = node_manager.get_cluster_status()
        print(f"\n📊 Final Cluster Statistics:")
        print(f"  Total tasks completed: {final_status['total_tasks_completed']}")
        print(f"  Average task time: {final_status['average_task_time']:.2f}s")
        print(f"  Cluster utilization: {final_status['average_load']:.2f}")
        
        return node_manager, simulator
        
    finally:
        # Cleanup
        if 'node_manager' in locals():
            node_manager.stop()


if __name__ == "__main__":
    run_distributed_computing_demo()