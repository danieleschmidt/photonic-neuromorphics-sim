"""
Scalability Framework for Photonic Neuromorphic Systems

Advanced scalability features including distributed computing, load balancing,
auto-scaling, resource management, and horizontal scaling capabilities.
"""

import time
import threading
import json
import socket
import hashlib
import uuid
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import queue
import logging


class ScalingMode(Enum):
    """Scaling modes for different scenarios."""
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    ELASTIC = "elastic"
    PREDICTIVE = "predictive"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RANDOM = "weighted_random"
    PERFORMANCE_BASED = "performance_based"
    GEOGRAPHIC = "geographic"


@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system."""
    node_id: str
    host: str
    port: int
    capacity: int = 100
    current_load: int = 0
    health_status: str = "healthy"
    last_heartbeat: float = field(default_factory=time.time)
    performance_score: float = 1.0
    specializations: List[str] = field(default_factory=list)
    
    @property
    def load_percentage(self) -> float:
        """Calculate current load as percentage of capacity."""
        return (self.current_load / self.capacity) * 100 if self.capacity > 0 else 0
    
    @property
    def is_available(self) -> bool:
        """Check if node is available for new tasks."""
        return (self.health_status == "healthy" and 
                self.current_load < self.capacity and
                time.time() - self.last_heartbeat < 30)  # 30 second timeout


@dataclass
class WorkloadTask:
    """Represents a computational task."""
    task_id: str
    task_type: str
    priority: int = 1
    estimated_duration: float = 1.0
    resource_requirements: Dict[str, int] = field(default_factory=dict)
    data_size: int = 0
    created_time: float = field(default_factory=time.time)
    assigned_node: Optional[str] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    
    @property
    def execution_time(self) -> Optional[float]:
        """Calculate actual execution time."""
        if self.start_time and self.completion_time:
            return self.completion_time - self.start_time
        return None
    
    @property
    def wait_time(self) -> float:
        """Calculate time spent waiting in queue."""
        start = self.start_time or time.time()
        return start - self.created_time


class NodeManager:
    """Manages compute nodes in the distributed system."""
    
    def __init__(self):
        self.nodes: Dict[str, ComputeNode] = {}
        self.lock = threading.RLock()
        self.heartbeat_interval = 10.0  # seconds
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Node discovery and health monitoring
        self.failed_nodes = set()
        self.node_performance_history = defaultdict(list)
    
    def register_node(self, node: ComputeNode) -> bool:
        """Register a new compute node."""
        with self.lock:
            if node.node_id in self.nodes:
                # Update existing node
                existing_node = self.nodes[node.node_id]
                existing_node.host = node.host
                existing_node.port = node.port
                existing_node.capacity = node.capacity
                existing_node.specializations = node.specializations
                existing_node.last_heartbeat = time.time()
                return True
            else:
                # Add new node
                node.last_heartbeat = time.time()
                self.nodes[node.node_id] = node
                logging.info(f"Registered new compute node: {node.node_id}")
                return True
    
    def unregister_node(self, node_id: str) -> bool:
        """Unregister a compute node."""
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                self.failed_nodes.discard(node_id)
                logging.info(f"Unregistered compute node: {node_id}")
                return True
            return False
    
    def update_node_heartbeat(self, node_id: str, 
                            load: Optional[int] = None,
                            performance_score: Optional[float] = None) -> bool:
        """Update node heartbeat and status."""
        with self.lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.last_heartbeat = time.time()
                
                if load is not None:
                    node.current_load = load
                
                if performance_score is not None:
                    node.performance_score = performance_score
                    self.node_performance_history[node_id].append({
                        'timestamp': time.time(),
                        'score': performance_score
                    })
                    
                    # Keep only recent history
                    if len(self.node_performance_history[node_id]) > 100:
                        self.node_performance_history[node_id].pop(0)
                
                # Remove from failed nodes if back online
                if node_id in self.failed_nodes:
                    self.failed_nodes.remove(node_id)
                    node.health_status = "healthy"
                    logging.info(f"Node {node_id} back online")
                
                return True
            return False
    
    def get_available_nodes(self, 
                          task_type: Optional[str] = None,
                          min_capacity: int = 1) -> List[ComputeNode]:
        """Get list of available nodes for task assignment."""
        with self.lock:
            available_nodes = []
            
            for node in self.nodes.values():
                if not node.is_available:
                    continue
                
                if node.capacity - node.current_load < min_capacity:
                    continue
                
                # Check specialization if required
                if task_type and task_type not in node.specializations and node.specializations:
                    continue
                
                available_nodes.append(node)
            
            return available_nodes
    
    def start_monitoring(self):
        """Start node health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logging.info("Node monitoring started")
    
    def stop_monitoring(self):
        """Stop node health monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logging.info("Node monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for node health."""
        while self.monitoring_active:
            current_time = time.time()
            
            with self.lock:
                # Check for failed nodes
                for node_id, node in list(self.nodes.items()):
                    if current_time - node.last_heartbeat > 30:  # 30 second timeout
                        if node_id not in self.failed_nodes:
                            self.failed_nodes.add(node_id)
                            node.health_status = "unhealthy"
                            logging.warning(f"Node {node_id} marked as unhealthy")
            
            time.sleep(self.heartbeat_interval)
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get comprehensive cluster statistics."""
        with self.lock:
            total_nodes = len(self.nodes)
            healthy_nodes = sum(1 for node in self.nodes.values() if node.is_available)
            total_capacity = sum(node.capacity for node in self.nodes.values())
            total_load = sum(node.current_load for node in self.nodes.values())
            
            avg_performance = 0.0
            if self.nodes:
                avg_performance = sum(node.performance_score for node in self.nodes.values()) / len(self.nodes)
            
            return {
                'total_nodes': total_nodes,
                'healthy_nodes': healthy_nodes,
                'failed_nodes': len(self.failed_nodes),
                'total_capacity': total_capacity,
                'total_load': total_load,
                'utilization': (total_load / total_capacity * 100) if total_capacity > 0 else 0,
                'average_performance': avg_performance,
                'specializations': self._get_specialization_summary()
            }
    
    def _get_specialization_summary(self) -> Dict[str, int]:
        """Get summary of node specializations."""
        specialization_counts = defaultdict(int)
        for node in self.nodes.values():
            for spec in node.specializations:
                specialization_counts[spec] += 1
        return dict(specialization_counts)


class LoadBalancer:
    """Advanced load balancer for distributing tasks across nodes."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.PERFORMANCE_BASED):
        self.strategy = strategy
        self.node_manager = NodeManager()
        
        # Load balancing state
        self.round_robin_index = 0
        self.connection_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        
        # Performance tracking
        self.assignment_history = deque(maxlen=1000)
        self.load_balancing_stats = {
            'total_assignments': 0,
            'failed_assignments': 0,
            'average_assignment_time': 0.0
        }
    
    def assign_task(self, task: WorkloadTask) -> Optional[ComputeNode]:
        """Assign task to optimal compute node."""
        start_time = time.time()
        
        # Get available nodes
        available_nodes = self.node_manager.get_available_nodes(
            task_type=task.task_type,
            min_capacity=task.resource_requirements.get('cpu', 1)
        )
        
        if not available_nodes:
            self.load_balancing_stats['failed_assignments'] += 1
            return None
        
        # Select node based on strategy
        selected_node = self._select_node(available_nodes, task)
        
        if selected_node:
            # Update node load
            selected_node.current_load += task.resource_requirements.get('cpu', 1)
            task.assigned_node = selected_node.node_id
            
            # Update connection count
            self.connection_counts[selected_node.node_id] += 1
            
            # Record assignment
            assignment_time = time.time() - start_time
            self.assignment_history.append({
                'task_id': task.task_id,
                'node_id': selected_node.node_id,
                'assignment_time': assignment_time,
                'strategy': self.strategy.value,
                'timestamp': time.time()
            })
            
            # Update stats
            self.load_balancing_stats['total_assignments'] += 1
            total_time = self.load_balancing_stats['average_assignment_time']
            total_assignments = self.load_balancing_stats['total_assignments']
            self.load_balancing_stats['average_assignment_time'] = (
                (total_time * (total_assignments - 1) + assignment_time) / total_assignments
            )
        
        return selected_node
    
    def _select_node(self, available_nodes: List[ComputeNode], task: WorkloadTask) -> Optional[ComputeNode]:
        """Select optimal node based on load balancing strategy."""
        if not available_nodes:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            # Simple round-robin selection
            selected_node = available_nodes[self.round_robin_index % len(available_nodes)]
            self.round_robin_index += 1
            return selected_node
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            # Select node with least connections
            return min(available_nodes, key=lambda node: self.connection_counts[node.node_id])
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_RANDOM:
            # Weighted random based on capacity
            import random
            weights = [node.capacity - node.current_load for node in available_nodes]
            total_weight = sum(weights)
            
            if total_weight <= 0:
                return random.choice(available_nodes)
            
            r = random.uniform(0, total_weight)
            cumulative = 0
            for i, weight in enumerate(weights):
                cumulative += weight
                if r <= cumulative:
                    return available_nodes[i]
            
            return available_nodes[-1]
        
        elif self.strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            # Select based on performance score and load
            def score_function(node):
                load_factor = 1.0 - (node.current_load / node.capacity)
                return node.performance_score * load_factor
            
            return max(available_nodes, key=score_function)
        
        else:  # Default to least loaded
            return min(available_nodes, key=lambda node: node.load_percentage)
    
    def complete_task(self, task: WorkloadTask) -> None:
        """Mark task as completed and update node load."""
        if task.assigned_node:
            node = self.node_manager.nodes.get(task.assigned_node)
            if node:
                # Reduce node load
                cpu_requirement = task.resource_requirements.get('cpu', 1)
                node.current_load = max(0, node.current_load - cpu_requirement)
                
                # Update connection count
                self.connection_counts[node.node_id] = max(0, self.connection_counts[node.node_id] - 1)
                
                # Record response time if available
                if task.execution_time:
                    self.response_times[node.node_id].append(task.execution_time)
                    
                    # Keep only recent response times
                    if len(self.response_times[node.node_id]) > 100:
                        self.response_times[node.node_id].pop(0)
                    
                    # Update node performance score
                    avg_response_time = sum(self.response_times[node.node_id]) / len(self.response_times[node.node_id])
                    # Lower response time = higher performance score
                    node.performance_score = max(0.1, 2.0 - avg_response_time)
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing performance statistics."""
        recent_assignments = [a for a in self.assignment_history if time.time() - a['timestamp'] < 300]  # Last 5 minutes
        
        strategy_distribution = defaultdict(int)
        for assignment in self.assignment_history:
            strategy_distribution[assignment['strategy']] += 1
        
        node_assignment_counts = defaultdict(int)
        for assignment in recent_assignments:
            node_assignment_counts[assignment['node_id']] += 1
        
        return {
            'current_strategy': self.strategy.value,
            'total_assignments': self.load_balancing_stats['total_assignments'],
            'failed_assignments': self.load_balancing_stats['failed_assignments'],
            'success_rate': 1.0 - (self.load_balancing_stats['failed_assignments'] / 
                                 max(self.load_balancing_stats['total_assignments'], 1)),
            'average_assignment_time': self.load_balancing_stats['average_assignment_time'],
            'recent_assignments': len(recent_assignments),
            'node_distribution': dict(node_assignment_counts),
            'strategy_distribution': dict(strategy_distribution)
        }


class AutoScaler:
    """Automatic scaling system for dynamic resource management."""
    
    def __init__(self, target_utilization: float = 70.0):
        self.target_utilization = target_utilization
        self.scaling_policies = {
            'scale_up_threshold': 80.0,
            'scale_down_threshold': 30.0,
            'cooldown_period': 300.0,  # 5 minutes
            'min_nodes': 1,
            'max_nodes': 10
        }
        
        self.scaling_history = deque(maxlen=100)
        self.last_scaling_action = 0.0
        self.node_manager = None  # Set externally
        
    def evaluate_scaling_need(self, cluster_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if scaling action is needed."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_action < self.scaling_policies['cooldown_period']:
            return {'action': 'none', 'reason': 'cooldown_period'}
        
        current_utilization = cluster_stats['utilization']
        healthy_nodes = cluster_stats['healthy_nodes']
        
        scaling_decision = {'action': 'none', 'reason': 'within_target'}
        
        # Scale up decision
        if (current_utilization > self.scaling_policies['scale_up_threshold'] and
            healthy_nodes < self.scaling_policies['max_nodes']):
            
            scaling_decision = {
                'action': 'scale_up',
                'reason': f'utilization_{current_utilization:.1f}%_above_threshold',
                'current_utilization': current_utilization,
                'target_nodes': min(healthy_nodes + 1, self.scaling_policies['max_nodes'])
            }
        
        # Scale down decision
        elif (current_utilization < self.scaling_policies['scale_down_threshold'] and
              healthy_nodes > self.scaling_policies['min_nodes']):
            
            scaling_decision = {
                'action': 'scale_down',
                'reason': f'utilization_{current_utilization:.1f}%_below_threshold',
                'current_utilization': current_utilization,
                'target_nodes': max(healthy_nodes - 1, self.scaling_policies['min_nodes'])
            }
        
        return scaling_decision
    
    def execute_scaling_action(self, scaling_decision: Dict[str, Any]) -> bool:
        """Execute scaling action."""
        if scaling_decision['action'] == 'none':
            return True
        
        current_time = time.time()
        
        if scaling_decision['action'] == 'scale_up':
            success = self._scale_up(scaling_decision.get('target_nodes', 1))
        elif scaling_decision['action'] == 'scale_down':
            success = self._scale_down(scaling_decision.get('target_nodes', 1))
        else:
            return False
        
        if success:
            # Record scaling action
            self.last_scaling_action = current_time
            self.scaling_history.append({
                'timestamp': current_time,
                'action': scaling_decision['action'],
                'reason': scaling_decision['reason'],
                'utilization': scaling_decision.get('current_utilization', 0),
                'target_nodes': scaling_decision.get('target_nodes', 0)
            })
            
            logging.info(f"Scaling action executed: {scaling_decision['action']} "
                        f"to {scaling_decision.get('target_nodes', 0)} nodes")
        
        return success
    
    def _scale_up(self, target_nodes: int) -> bool:
        """Scale up by adding compute nodes."""
        if not self.node_manager:
            return False
        
        current_nodes = len(self.node_manager.nodes)
        nodes_to_add = target_nodes - current_nodes
        
        for i in range(nodes_to_add):
            # Create new virtual node
            new_node = ComputeNode(
                node_id=f"auto_node_{uuid.uuid4().hex[:8]}",
                host="localhost",
                port=8000 + len(self.node_manager.nodes),
                capacity=100,
                specializations=["photonic_simulation", "neural_computation"]
            )
            
            self.node_manager.register_node(new_node)
        
        return True
    
    def _scale_down(self, target_nodes: int) -> bool:
        """Scale down by removing compute nodes."""
        if not self.node_manager:
            return False
        
        current_nodes = len(self.node_manager.nodes)
        nodes_to_remove = current_nodes - target_nodes
        
        # Remove nodes with lowest load first
        nodes_by_load = sorted(self.node_manager.nodes.values(), 
                             key=lambda node: node.current_load)
        
        removed = 0
        for node in nodes_by_load:
            if removed >= nodes_to_remove:
                break
            
            # Only remove nodes with low load
            if node.current_load <= 10:  # 10% threshold
                self.node_manager.unregister_node(node.node_id)
                removed += 1
        
        return removed > 0
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get auto-scaling metrics and history."""
        recent_actions = [action for action in self.scaling_history 
                         if time.time() - action['timestamp'] < 3600]  # Last hour
        
        action_counts = defaultdict(int)
        for action in self.scaling_history:
            action_counts[action['action']] += 1
        
        return {
            'scaling_policies': self.scaling_policies,
            'target_utilization': self.target_utilization,
            'total_scaling_actions': len(self.scaling_history),
            'recent_scaling_actions': len(recent_actions),
            'action_distribution': dict(action_counts),
            'last_scaling_action': self.last_scaling_action,
            'cooldown_remaining': max(0, self.scaling_policies['cooldown_period'] - 
                                    (time.time() - self.last_scaling_action))
        }


class DistributedTaskScheduler:
    """Distributed task scheduler with advanced queuing and prioritization."""
    
    def __init__(self):
        self.task_queue = queue.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = deque(maxlen=1000)
        
        self.load_balancer = LoadBalancer()
        self.auto_scaler = AutoScaler()
        self.auto_scaler.node_manager = self.load_balancer.node_manager
        
        # Scheduler state
        self.scheduler_active = False
        self.scheduler_thread = None
        self.processing_executor = ThreadPoolExecutor(max_workers=10)
        
        # Performance metrics
        self.scheduler_metrics = {
            'tasks_scheduled': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_queue_time': 0.0,
            'average_execution_time': 0.0
        }
    
    def submit_task(self, task: WorkloadTask) -> str:
        """Submit task to the distributed scheduler."""
        # Use negative priority for max-heap behavior (higher priority = lower number)
        priority = -task.priority
        self.task_queue.put((priority, time.time(), task))
        
        self.scheduler_metrics['tasks_scheduled'] += 1
        logging.info(f"Task {task.task_id} submitted with priority {task.priority}")
        
        return task.task_id
    
    def start_scheduler(self):
        """Start the distributed task scheduler."""
        if self.scheduler_active:
            return
        
        self.scheduler_active = True
        self.load_balancer.node_manager.start_monitoring()
        
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logging.info("Distributed task scheduler started")
    
    def stop_scheduler(self):
        """Stop the distributed task scheduler."""
        self.scheduler_active = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        
        self.load_balancer.node_manager.stop_monitoring()
        self.processing_executor.shutdown(wait=True)
        
        logging.info("Distributed task scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.scheduler_active:
            try:
                # Get cluster stats for auto-scaling
                cluster_stats = self.load_balancer.node_manager.get_cluster_stats()
                
                # Evaluate auto-scaling
                scaling_decision = self.auto_scaler.evaluate_scaling_need(cluster_stats)
                if scaling_decision['action'] != 'none':
                    self.auto_scaler.execute_scaling_action(scaling_decision)
                
                # Process pending tasks
                self._process_pending_tasks()
                
                # Check for completed tasks
                self._check_completed_tasks()
                
                time.sleep(1.0)  # Scheduler loop interval
                
            except Exception as e:
                logging.error(f"Error in scheduler loop: {e}")
                time.sleep(5.0)  # Wait before retrying
    
    def _process_pending_tasks(self):
        """Process pending tasks in the queue."""
        processed_count = 0
        max_batch_size = 10
        
        while processed_count < max_batch_size and not self.task_queue.empty():
            try:
                # Get task from queue with timeout
                priority, queued_time, task = self.task_queue.get(timeout=0.1)
                
                # Assign task to node
                assigned_node = self.load_balancer.assign_task(task)
                
                if assigned_node:
                    # Start task execution
                    task.start_time = time.time()
                    self.active_tasks[task.task_id] = task
                    
                    # Submit to executor for processing
                    future = self.processing_executor.submit(self._execute_task, task, assigned_node)
                    task.future = future
                    
                    # Update queue time metric
                    queue_time = time.time() - queued_time
                    self._update_average_metric('average_queue_time', queue_time)
                    
                    processed_count += 1
                else:
                    # No available nodes, put task back in queue
                    self.task_queue.put((priority, queued_time, task))
                    break
                
            except queue.Empty:
                break
            except Exception as e:
                logging.error(f"Error processing task: {e}")
    
    def _execute_task(self, task: WorkloadTask, node: ComputeNode) -> Any:
        """Execute task on assigned node."""
        try:
            # Simulate task execution
            import random
            execution_time = random.uniform(0.1, task.estimated_duration)
            time.sleep(execution_time)
            
            # Mark task as completed
            task.completion_time = time.time()
            result = f"Task {task.task_id} completed on node {node.node_id}"
            
            return result
            
        except Exception as e:
            logging.error(f"Task {task.task_id} failed: {e}")
            raise e
    
    def _check_completed_tasks(self):
        """Check for completed tasks and update metrics."""
        completed_task_ids = []
        
        for task_id, task in list(self.active_tasks.items()):
            if hasattr(task, 'future') and task.future.done():
                try:
                    result = task.future.result()
                    
                    # Task completed successfully
                    self.completed_tasks.append(task)
                    self.load_balancer.complete_task(task)
                    
                    # Update metrics
                    self.scheduler_metrics['tasks_completed'] += 1
                    if task.execution_time:
                        self._update_average_metric('average_execution_time', task.execution_time)
                    
                    completed_task_ids.append(task_id)
                    logging.info(f"Task {task_id} completed successfully")
                    
                except Exception as e:
                    # Task failed
                    self.scheduler_metrics['tasks_failed'] += 1
                    self.load_balancer.complete_task(task)
                    
                    completed_task_ids.append(task_id)
                    logging.error(f"Task {task_id} failed: {e}")
        
        # Remove completed tasks from active list
        for task_id in completed_task_ids:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    def _update_average_metric(self, metric_name: str, new_value: float):
        """Update running average for a metric."""
        current_avg = self.scheduler_metrics[metric_name]
        total_tasks = self.scheduler_metrics['tasks_completed'] + self.scheduler_metrics['tasks_failed']
        
        if total_tasks > 0:
            self.scheduler_metrics[metric_name] = (current_avg * (total_tasks - 1) + new_value) / total_tasks
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status."""
        cluster_stats = self.load_balancer.node_manager.get_cluster_stats()
        load_balancing_stats = self.load_balancer.get_load_balancing_stats()
        scaling_metrics = self.auto_scaler.get_scaling_metrics()
        
        return {
            'scheduler_active': self.scheduler_active,
            'queue_size': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'scheduler_metrics': self.scheduler_metrics,
            'cluster_stats': cluster_stats,
            'load_balancing': load_balancing_stats,
            'auto_scaling': scaling_metrics,
            'timestamp': time.time()
        }


def demonstrate_scalability_framework():
    """Demonstrate scalability framework capabilities."""
    print("⚡ Demonstrating Scalability Framework")
    print("=" * 50)
    
    # Create distributed scheduler
    scheduler = DistributedTaskScheduler()
    
    # Register some initial compute nodes
    print("\n1. Setting up compute cluster...")
    
    for i in range(3):
        node = ComputeNode(
            node_id=f"node_{i}",
            host="localhost",
            port=8000 + i,
            capacity=100,
            specializations=["photonic_simulation", "neural_computation"]
        )
        scheduler.load_balancer.node_manager.register_node(node)
    
    print(f"   Registered 3 initial compute nodes")
    
    # Start scheduler
    print("\n2. Starting distributed scheduler...")
    scheduler.start_scheduler()
    
    # Submit test tasks
    print("\n3. Submitting test tasks...")
    
    task_ids = []
    for i in range(20):
        task = WorkloadTask(
            task_id=f"task_{i}",
            task_type="photonic_simulation",
            priority=i % 5 + 1,  # Priorities 1-5
            estimated_duration=0.5,
            resource_requirements={'cpu': 10},
            data_size=1024 * (i + 1)
        )
        
        task_id = scheduler.submit_task(task)
        task_ids.append(task_id)
    
    print(f"   Submitted {len(task_ids)} tasks")
    
    # Monitor execution
    print("\n4. Monitoring execution...")
    
    for i in range(15):  # Monitor for 15 seconds
        status = scheduler.get_scheduler_status()
        
        print(f"   T+{i+1}s: Queue: {status['queue_size']}, "
              f"Active: {status['active_tasks']}, "
              f"Completed: {status['completed_tasks']}, "
              f"Nodes: {status['cluster_stats']['healthy_nodes']}, "
              f"Utilization: {status['cluster_stats']['utilization']:.1f}%")
        
        time.sleep(1.0)
    
    # Get final status
    print("\n5. Final scheduler status:")
    final_status = scheduler.get_scheduler_status()
    
    # Scheduler metrics
    metrics = final_status['scheduler_metrics']
    print(f"   Tasks scheduled: {metrics['tasks_scheduled']}")
    print(f"   Tasks completed: {metrics['tasks_completed']}")
    print(f"   Tasks failed: {metrics['tasks_failed']}")
    print(f"   Average queue time: {metrics['average_queue_time']:.3f}s")
    print(f"   Average execution time: {metrics['average_execution_time']:.3f}s")
    
    # Cluster stats
    cluster = final_status['cluster_stats']
    print(f"   Cluster utilization: {cluster['utilization']:.1f}%")
    print(f"   Healthy nodes: {cluster['healthy_nodes']}/{cluster['total_nodes']}")
    print(f"   Total capacity: {cluster['total_capacity']}")
    
    # Load balancing stats
    load_balancing = final_status['load_balancing']
    print(f"   Load balancing success rate: {load_balancing['success_rate']:.2%}")
    print(f"   Average assignment time: {load_balancing['average_assignment_time']:.4f}s")
    
    # Auto-scaling stats
    auto_scaling = final_status['auto_scaling']
    print(f"   Scaling actions: {auto_scaling['total_scaling_actions']}")
    print(f"   Target utilization: {auto_scaling['target_utilization']:.1f}%")
    
    # Test auto-scaling
    print("\n6. Testing auto-scaling...")
    
    # Submit high-load tasks to trigger scale-up
    for i in range(30):
        task = WorkloadTask(
            task_id=f"load_task_{i}",
            task_type="neural_computation",
            priority=5,
            estimated_duration=2.0,
            resource_requirements={'cpu': 20}
        )
        scheduler.submit_task(task)
    
    print("   Submitted high-load tasks to trigger scaling...")
    
    # Monitor scaling
    for i in range(10):
        status = scheduler.get_scheduler_status()
        cluster_stats = status['cluster_stats']
        
        print(f"   Scaling T+{i+1}s: Nodes: {cluster_stats['healthy_nodes']}, "
              f"Utilization: {cluster_stats['utilization']:.1f}%, "
              f"Queue: {status['queue_size']}")
        
        time.sleep(1.0)
    
    # Stop scheduler
    print("\n7. Stopping scheduler...")
    scheduler.stop_scheduler()
    
    return final_status


if __name__ == "__main__":
    demonstrate_scalability_framework()