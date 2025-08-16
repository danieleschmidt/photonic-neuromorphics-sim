"""
Real-Time Adaptive Optimization for Photonic Neuromorphics

Advanced real-time optimization system that continuously adapts network parameters,
architecture, and optical settings for optimal performance in changing environments.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import logging
import time
import threading
from collections import deque
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import json

from .core import PhotonicSNN, OpticalParameters, encode_to_spikes
from .exceptions import ValidationError, OpticalModelError, PhotonicNeuromorphicsException
from .autonomous_learning import LearningMetrics, AutonomousLearningFramework
from .quantum_photonic_interface import HybridQuantumPhotonic, create_quantum_photonic_demo


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    accuracy: float = 0.0
    throughput: float = 0.0  # samples/second
    latency: float = 0.0  # seconds
    energy_efficiency: float = 0.0  # TOPS/W
    optical_efficiency: float = 0.0
    memory_usage: float = 0.0  # MB
    gpu_utilization: float = 0.0
    temperature: float = 0.0  # Celsius (simulated)
    power_consumption: float = 0.0  # Watts
    error_rate: float = 0.0
    
    def __post_init__(self):
        self.timestamp = time.time()
    
    def composite_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted composite performance score."""
        if weights is None:
            weights = {
                'accuracy': 0.25,
                'throughput': 0.20,
                'energy_efficiency': 0.20,
                'latency': -0.15,  # Lower is better
                'optical_efficiency': 0.10
            }
        
        score = (
            weights.get('accuracy', 0) * self.accuracy +
            weights.get('throughput', 0) * min(self.throughput / 1000, 1.0) +
            weights.get('energy_efficiency', 0) * min(self.energy_efficiency / 100, 1.0) +
            weights.get('latency', 0) * max(1.0 - self.latency, 0) +
            weights.get('optical_efficiency', 0) * self.optical_efficiency
        )
        
        return max(0.0, min(1.0, score))


class RealTimeProfiler:
    """Real-time performance profiler for photonic systems."""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.metrics_history = deque(maxlen=1000)
        self.is_profiling = False
        self.profiling_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Performance counters
        self.total_inferences = 0
        self.total_time = 0.0
        self.energy_consumption = 0.0
        
    def start_profiling(self) -> None:
        """Start real-time profiling."""
        if self.is_profiling:
            return
        
        self.is_profiling = True
        self.profiling_thread = threading.Thread(target=self._profiling_loop)
        self.profiling_thread.daemon = True
        self.profiling_thread.start()
        self.logger.info("Started real-time profiling")
    
    def stop_profiling(self) -> None:
        """Stop real-time profiling."""
        self.is_profiling = False
        if self.profiling_thread:
            self.profiling_thread.join(timeout=1.0)
        self.logger.info("Stopped real-time profiling")
    
    def _profiling_loop(self) -> None:
        """Main profiling loop."""
        while self.is_profiling:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.sampling_interval)
            except Exception as e:
                self.logger.error(f"Profiling error: {e}")
                time.sleep(self.sampling_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        import psutil
        
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        
        # Calculate derived metrics
        throughput = self.total_inferences / (self.total_time + 1e-6)
        energy_efficiency = throughput / (self.energy_consumption + 1e-6)
        
        return PerformanceMetrics(
            throughput=throughput,
            memory_usage=memory_info.used / (1024**2),  # MB
            gpu_utilization=cpu_percent / 100.0,  # Simplified
            power_consumption=self.energy_consumption,
            temperature=20 + cpu_percent * 0.5,  # Simulated temperature
        )
    
    def record_inference(self, 
                        accuracy: float,
                        latency: float,
                        energy_used: float,
                        optical_efficiency: float = 0.0) -> None:
        """Record metrics from an inference."""
        self.total_inferences += 1
        self.total_time += latency
        self.energy_consumption += energy_used
        
        # Create immediate metrics
        metrics = PerformanceMetrics(
            accuracy=accuracy,
            latency=latency,
            energy_efficiency=1.0 / (energy_used + 1e-6),
            optical_efficiency=optical_efficiency
        )
        
        self.metrics_history.append(metrics)
    
    def get_recent_metrics(self, window_size: int = 10) -> List[PerformanceMetrics]:
        """Get recent performance metrics."""
        return list(self.metrics_history)[-window_size:]
    
    def get_performance_trends(self) -> Dict[str, List[float]]:
        """Get performance trends over time."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 samples
        
        trends = {
            'accuracy': [m.accuracy for m in recent_metrics],
            'throughput': [m.throughput for m in recent_metrics],
            'latency': [m.latency for m in recent_metrics],
            'energy_efficiency': [m.energy_efficiency for m in recent_metrics],
            'optical_efficiency': [m.optical_efficiency for m in recent_metrics]
        }
        
        return trends


class AdaptiveParameterTuner:
    """Adaptive parameter tuning using reinforcement learning principles."""
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 exploration_rate: float = 0.1,
                 reward_smoothing: float = 0.9):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.reward_smoothing = reward_smoothing
        
        # Parameter spaces to explore
        self.parameter_ranges = {
            'wavelength': (1260e-9, 1675e-9),
            'power': (1e-6, 10e-3),
            'coupling_efficiency': (0.1, 0.99),
            'detector_efficiency': (0.1, 0.95),
            'learning_rate': (1e-5, 1e-1),
            'batch_size': (8, 128)
        }
        
        # Q-table for parameter values (simplified)
        self.q_table = {}
        self.state_history = deque(maxlen=1000)
        self.action_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        
        self.logger = logging.getLogger(__name__)
    
    def get_adaptive_parameters(self, 
                              current_metrics: PerformanceMetrics,
                              target_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Get adaptively tuned parameters based on current performance."""
        state = self._encode_state(current_metrics)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.exploration_rate:
            # Explore: random parameter adjustment
            action = self._get_random_action()
        else:
            # Exploit: use learned policy
            action = self._get_best_action(state)
        
        # Convert action to parameter adjustments
        parameter_adjustments = self._action_to_parameters(action)
        
        # Record state and action
        self.state_history.append(state)
        self.action_history.append(action)
        
        return parameter_adjustments
    
    def update_policy(self, reward: float) -> None:
        """Update the policy based on observed reward."""
        if not self.state_history or not self.action_history:
            return
        
        state = self.state_history[-1]
        action = self.action_history[-1]
        
        # Q-learning update
        state_key = tuple(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
        
        # Simple Q-update (without future state estimation)
        self.q_table[state_key][action] += self.learning_rate * reward
        
        self.reward_history.append(reward)
        
        # Decay exploration rate
        self.exploration_rate *= 0.9995
        self.exploration_rate = max(0.01, self.exploration_rate)
    
    def _encode_state(self, metrics: PerformanceMetrics) -> List[float]:
        """Encode performance metrics as state vector."""
        return [
            metrics.accuracy,
            min(metrics.throughput / 1000, 1.0),
            metrics.latency,
            metrics.energy_efficiency / 100,
            metrics.optical_efficiency
        ]
    
    def _get_random_action(self) -> str:
        """Get random action for exploration."""
        actions = [
            'increase_power', 'decrease_power',
            'increase_wavelength', 'decrease_wavelength',
            'increase_coupling', 'decrease_coupling',
            'increase_lr', 'decrease_lr'
        ]
        return np.random.choice(actions)
    
    def _get_best_action(self, state: List[float]) -> str:
        """Get best action based on learned policy."""
        state_key = tuple(state)
        
        if state_key not in self.q_table or not self.q_table[state_key]:
            return self._get_random_action()
        
        # Return action with highest Q-value
        best_action = max(self.q_table[state_key], key=self.q_table[state_key].get)
        return best_action
    
    def _action_to_parameters(self, action: str) -> Dict[str, Any]:
        """Convert action to parameter adjustments."""
        adjustments = {}
        
        if action == 'increase_power':
            adjustments['power_multiplier'] = 1.1
        elif action == 'decrease_power':
            adjustments['power_multiplier'] = 0.9
        elif action == 'increase_wavelength':
            adjustments['wavelength_offset'] = 10e-9
        elif action == 'decrease_wavelength':
            adjustments['wavelength_offset'] = -10e-9
        elif action == 'increase_coupling':
            adjustments['coupling_multiplier'] = 1.05
        elif action == 'decrease_coupling':
            adjustments['coupling_multiplier'] = 0.95
        elif action == 'increase_lr':
            adjustments['learning_rate_multiplier'] = 1.2
        elif action == 'decrease_lr':
            adjustments['learning_rate_multiplier'] = 0.8
        
        return adjustments


class RealTimeOptimizer:
    """Real-time optimization orchestrator."""
    
    def __init__(self,
                 photonic_system: Union[PhotonicSNN, HybridQuantumPhotonic],
                 optimization_interval: float = 1.0,
                 max_optimization_threads: int = 4):
        self.photonic_system = photonic_system
        self.optimization_interval = optimization_interval
        self.max_threads = max_optimization_threads
        
        # Components
        self.profiler = RealTimeProfiler()
        self.parameter_tuner = AdaptiveParameterTuner()
        self.autonomous_learner = AutonomousLearningFramework()
        
        # Control
        self.is_optimizing = False
        self.optimization_thread = None
        self.executor = ThreadPoolExecutor(max_workers=max_optimization_threads)
        
        # Optimization history
        self.optimization_history = deque(maxlen=1000)
        self.performance_targets = {
            'accuracy': 0.95,
            'throughput': 1000.0,
            'latency': 0.001,
            'energy_efficiency': 100.0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def start_optimization(self) -> None:
        """Start real-time optimization."""
        if self.is_optimizing:
            return
        
        self.is_optimizing = True
        self.profiler.start_profiling()
        
        self.optimization_thread = threading.Thread(target=self._optimization_loop)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        
        self.logger.info("Started real-time optimization")
    
    def stop_optimization(self) -> None:
        """Stop real-time optimization."""
        self.is_optimizing = False
        self.profiler.stop_profiling()
        
        if self.optimization_thread:
            self.optimization_thread.join(timeout=2.0)
        
        self.executor.shutdown(wait=False)
        self.logger.info("Stopped real-time optimization")
    
    def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while self.is_optimizing:
            try:
                start_time = time.time()
                
                # Get current performance metrics
                recent_metrics = self.profiler.get_recent_metrics(window_size=5)
                if not recent_metrics:
                    time.sleep(self.optimization_interval)
                    continue
                
                current_metrics = recent_metrics[-1]
                
                # Check if optimization is needed
                if self._needs_optimization(current_metrics):
                    self._perform_optimization(current_metrics)
                
                # Sleep for remaining interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.optimization_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                time.sleep(self.optimization_interval)
    
    def _needs_optimization(self, current_metrics: PerformanceMetrics) -> bool:
        """Determine if optimization is needed."""
        # Check if performance is below targets
        below_accuracy_target = current_metrics.accuracy < self.performance_targets['accuracy']
        below_throughput_target = current_metrics.throughput < self.performance_targets['throughput']
        above_latency_target = current_metrics.latency > self.performance_targets['latency']
        
        # Check for performance degradation
        recent_metrics = self.profiler.get_recent_metrics(window_size=10)
        if len(recent_metrics) >= 5:
            recent_accuracy = np.mean([m.accuracy for m in recent_metrics[-5:]])
            older_accuracy = np.mean([m.accuracy for m in recent_metrics[-10:-5]])
            performance_degrading = recent_accuracy < older_accuracy - 0.05
        else:
            performance_degrading = False
        
        return (below_accuracy_target or below_throughput_target or 
                above_latency_target or performance_degrading)
    
    def _perform_optimization(self, current_metrics: PerformanceMetrics) -> None:
        """Perform optimization based on current metrics."""
        self.logger.info("Performing real-time optimization")
        
        # Get adaptive parameter adjustments
        adjustments = self.parameter_tuner.get_adaptive_parameters(
            current_metrics, self.performance_targets
        )
        
        # Apply parameter adjustments
        if adjustments:
            self._apply_parameter_adjustments(adjustments)
        
        # Submit asynchronous optimization tasks
        future_architectural = self.executor.submit(self._optimize_architecture)
        future_optical = self.executor.submit(self._optimize_optical_parameters)
        
        # Record optimization attempt
        self.optimization_history.append({
            'timestamp': time.time(),
            'metrics_before': current_metrics,
            'adjustments': adjustments,
            'reason': 'performance_optimization'
        })
    
    def _apply_parameter_adjustments(self, adjustments: Dict[str, Any]) -> None:
        """Apply parameter adjustments to the system."""
        try:
            # Get current optical parameters
            if hasattr(self.photonic_system, 'optical_params'):
                optical_params = self.photonic_system.optical_params
            elif hasattr(self.photonic_system, 'photonic_network'):
                optical_params = self.photonic_system.photonic_network.optical_params
            else:
                self.logger.warning("Could not find optical parameters to adjust")
                return
            
            # Apply optical parameter adjustments
            if 'power_multiplier' in adjustments:
                optical_params.power *= adjustments['power_multiplier']
                optical_params.power = np.clip(optical_params.power, 1e-6, 10e-3)
            
            if 'wavelength_offset' in adjustments:
                optical_params.wavelength += adjustments['wavelength_offset']
                optical_params.wavelength = np.clip(optical_params.wavelength, 1260e-9, 1675e-9)
            
            if 'coupling_multiplier' in adjustments:
                optical_params.coupling_efficiency *= adjustments['coupling_multiplier']
                optical_params.coupling_efficiency = np.clip(optical_params.coupling_efficiency, 0.1, 0.99)
            
            self.logger.debug(f"Applied parameter adjustments: {adjustments}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply parameter adjustments: {e}")
    
    def _optimize_architecture(self) -> None:
        """Optimize network architecture (background task)."""
        try:
            self.logger.debug("Running background architecture optimization")
            
            # Simplified architecture optimization
            # In a real system, this would run more sophisticated algorithms
            
            if hasattr(self.photonic_system, 'topology'):
                current_topology = self.photonic_system.topology
                
                # Try slight topology modifications
                if len(current_topology) > 2:
                    for i in range(1, len(current_topology) - 1):
                        # Small adjustments to layer sizes
                        if current_topology[i] > 20:
                            current_topology[i] = max(10, current_topology[i] - 5)
                        elif current_topology[i] < 200:
                            current_topology[i] = min(500, current_topology[i] + 5)
            
        except Exception as e:
            self.logger.error(f"Architecture optimization failed: {e}")
    
    def _optimize_optical_parameters(self) -> None:
        """Optimize optical parameters (background task)."""
        try:
            self.logger.debug("Running background optical optimization")
            
            # Get current metrics for optimization guidance
            recent_metrics = self.profiler.get_recent_metrics(window_size=3)
            if not recent_metrics:
                return
            
            avg_optical_efficiency = np.mean([m.optical_efficiency for m in recent_metrics])
            
            # Optimize wavelength for better efficiency
            if avg_optical_efficiency < 0.8:
                # Move wavelength closer to optimal 1550nm
                if hasattr(self.photonic_system, 'optical_params'):
                    optical_params = self.photonic_system.optical_params
                elif hasattr(self.photonic_system, 'photonic_network'):
                    optical_params = self.photonic_system.photonic_network.optical_params
                else:
                    return
                
                optimal_wavelength = 1550e-9
                current_wavelength = optical_params.wavelength
                
                # Gradual adjustment towards optimal
                adjustment = (optimal_wavelength - current_wavelength) * 0.1
                optical_params.wavelength += adjustment
                
        except Exception as e:
            self.logger.error(f"Optical optimization failed: {e}")
    
    async def async_inference(self, 
                            input_data: torch.Tensor,
                            labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, PerformanceMetrics]:
        """Perform inference with real-time optimization."""
        start_time = time.time()
        
        try:
            # Perform inference
            if hasattr(self.photonic_system, 'quantum_enhanced_forward'):
                # Quantum-photonic system
                output = self.photonic_system.quantum_enhanced_forward(input_data)
            else:
                # Regular photonic system
                output = self.photonic_system.forward(input_data)
            
            # Calculate metrics
            inference_time = time.time() - start_time
            
            accuracy = 0.0
            if labels is not None:
                accuracy = (output.argmax(1) == labels).float().mean().item()
            
            # Estimate energy consumption (simplified)
            energy_used = inference_time * 0.1  # 100mW average
            
            # Estimate optical efficiency
            if hasattr(self.photonic_system, 'optical_params'):
                wavelength_nm = self.photonic_system.optical_params.wavelength * 1e9
            elif hasattr(self.photonic_system, 'photonic_network'):
                wavelength_nm = self.photonic_system.photonic_network.optical_params.wavelength * 1e9
            else:
                wavelength_nm = 1550
            
            optical_efficiency = max(0.1, 1.0 - abs(wavelength_nm - 1550) / 1550)
            
            # Record metrics
            self.profiler.record_inference(
                accuracy=accuracy,
                latency=inference_time,
                energy_used=energy_used,
                optical_efficiency=optical_efficiency
            )
            
            # Update parameter tuner with reward
            metrics = PerformanceMetrics(
                accuracy=accuracy,
                latency=inference_time,
                energy_efficiency=1.0 / (energy_used + 1e-6),
                optical_efficiency=optical_efficiency
            )
            
            reward = metrics.composite_score()
            self.parameter_tuner.update_policy(reward)
            
            return output, metrics
            
        except Exception as e:
            self.logger.error(f"Async inference failed: {e}")
            error_metrics = PerformanceMetrics(error_rate=1.0)
            return torch.zeros_like(input_data[:, :3]), error_metrics  # Dummy output
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        recent_history = list(self.optimization_history)[-100:]
        
        if not recent_history:
            return {}
        
        return {
            'total_optimizations': len(self.optimization_history),
            'recent_optimizations': len(recent_history),
            'optimization_frequency': len(recent_history) / (time.time() - recent_history[0]['timestamp'] + 1e-6),
            'parameter_exploration_rate': self.parameter_tuner.exploration_rate,
            'average_reward': np.mean(list(self.parameter_tuner.reward_history)),
            'q_table_size': len(self.parameter_tuner.q_table),
            'performance_trends': self.profiler.get_performance_trends()
        }


def create_realtime_optimization_demo() -> RealTimeOptimizer:
    """Create demonstration real-time optimization system."""
    # Create hybrid quantum-photonic system
    hybrid_system = create_quantum_photonic_demo()
    
    # Create real-time optimizer
    optimizer = RealTimeOptimizer(
        photonic_system=hybrid_system,
        optimization_interval=0.5,  # 500ms optimization interval
        max_optimization_threads=2
    )
    
    return optimizer


async def run_realtime_optimization_demo():
    """Run real-time optimization demonstration."""
    print("⚡ Real-Time Adaptive Optimization Demo")
    print("=" * 45)
    
    # Create real-time optimizer
    optimizer = create_realtime_optimization_demo()
    
    try:
        # Start optimization
        optimizer.start_optimization()
        
        # Generate streaming data
        torch.manual_seed(42)
        np.random.seed(42)
        
        batch_size = 8
        input_dim = 20
        n_classes = 3
        n_batches = 20
        
        print(f"Processing {n_batches} batches with real-time optimization...")
        
        performance_history = []
        
        for batch_idx in range(n_batches):
            # Generate batch data
            test_data = torch.randn(batch_size, input_dim)
            test_labels = torch.randint(0, n_classes, (batch_size,))
            
            # Perform inference with optimization
            outputs, metrics = await optimizer.async_inference(test_data, test_labels)
            performance_history.append(metrics)
            
            # Print progress
            if batch_idx % 5 == 0:
                print(f"Batch {batch_idx:2d}: Accuracy={metrics.accuracy:.3f}, "
                      f"Latency={metrics.latency*1000:.1f}ms, "
                      f"Optical={metrics.optical_efficiency:.3f}")
            
            # Simulate realistic inference intervals
            await asyncio.sleep(0.1)
        
        # Final statistics
        print("\n📊 Optimization Results:")
        
        # Performance improvement analysis
        early_performance = np.mean([m.accuracy for m in performance_history[:5]])
        late_performance = np.mean([m.accuracy for m in performance_history[-5:]])
        improvement = late_performance - early_performance
        
        print(f"Initial accuracy: {early_performance:.4f}")
        print(f"Final accuracy: {late_performance:.4f}")
        print(f"Performance improvement: {improvement:.4f}")
        
        # Get optimization statistics
        opt_stats = optimizer.get_optimization_statistics()
        print(f"Total optimizations performed: {opt_stats.get('total_optimizations', 0)}")
        print(f"Optimization frequency: {opt_stats.get('optimization_frequency', 0):.2f} Hz")
        print(f"Average reward: {opt_stats.get('average_reward', 0):.4f}")
        
        # Performance trends
        trends = opt_stats.get('performance_trends', {})
        if 'accuracy' in trends and len(trends['accuracy']) > 1:
            accuracy_trend = np.polyfit(range(len(trends['accuracy'])), trends['accuracy'], 1)[0]
            print(f"Accuracy trend: {accuracy_trend:.6f} per sample {'↗' if accuracy_trend > 0 else '↘'}")
        
        return optimizer, performance_history
        
    finally:
        # Cleanup
        optimizer.stop_optimization()


def run_realtime_demo():
    """Run the real-time optimization demo (sync wrapper)."""
    return asyncio.run(run_realtime_optimization_demo())


if __name__ == "__main__":
    run_realtime_demo()