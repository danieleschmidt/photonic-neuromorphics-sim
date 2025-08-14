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


class QuantumInspiredOptimizer:
    """
    Quantum-inspired optimization for photonic neural networks.
    
    Implements quantum annealing algorithms and variational optimization
    techniques specifically designed for photonic computing systems.
    """
    
    def __init__(self, num_qubits: int = 16, temperature: float = 1.0):
        self.num_qubits = num_qubits
        self.temperature = temperature
        self.quantum_state = None
        self.optimization_history = []
        
        # Initialize quantum-inspired components
        self._initialize_quantum_state()
        
        self._logger = logging.getLogger(__name__)
    
    def _initialize_quantum_state(self):
        """Initialize quantum state representation."""
        # Simulate quantum superposition for optimization variables
        self.quantum_state = {
            'amplitudes': np.random.complex128((2**self.num_qubits,)),
            'phases': np.random.uniform(0, 2*np.pi, (2**self.num_qubits,)),
            'entanglement_matrix': np.random.random((self.num_qubits, self.num_qubits))
        }
        
        # Normalize amplitudes
        norm = np.linalg.norm(self.quantum_state['amplitudes'])
        self.quantum_state['amplitudes'] /= norm
    
    def optimize_wavelength_allocation(
        self,
        photonic_network,
        optimization_target: str = "energy_efficiency"
    ) -> Dict[str, Any]:
        """
        Optimize wavelength allocation using quantum-inspired algorithms.
        
        Args:
            photonic_network: The photonic neural network to optimize
            optimization_target: Target metric to optimize
            
        Returns:
            Optimization results with improved wavelength allocation
        """
        start_time = time.perf_counter()
        
        # Extract current network parameters
        current_params = self._extract_network_parameters(photonic_network)
        
        # Define optimization objective
        objective_func = self._create_objective_function(
            photonic_network, optimization_target
        )
        
        # Quantum annealing optimization
        best_params = self._quantum_annealing_optimization(
            objective_func, current_params
        )
        
        # Apply optimized parameters
        optimization_improvement = self._apply_optimized_parameters(
            photonic_network, best_params
        )
        
        optimization_time = time.perf_counter() - start_time
        
        results = {
            "optimization_target": optimization_target,
            "improvement_factor": optimization_improvement,
            "optimization_time": optimization_time,
            "optimized_parameters": best_params,
            "quantum_state_evolution": len(self.optimization_history),
            "convergence_achieved": optimization_improvement > 1.05  # 5% improvement
        }
        
        self.optimization_history.append(results)
        self._logger.info(f"Quantum optimization completed: {optimization_improvement:.2f}x improvement")
        
        return results
    
    def _extract_network_parameters(self, network) -> Dict[str, np.ndarray]:
        """Extract optimizable parameters from photonic network."""
        params = {}
        
        # Extract wavelength allocations
        if hasattr(network, 'wavelength_channels'):
            params['wavelength_spacing'] = np.array([0.8e-9] * network.wavelength_channels)
            params['power_distribution'] = np.ones(network.wavelength_channels)
            
        # Extract coupling efficiencies
        if hasattr(network, 'optical_params'):
            params['coupling_efficiency'] = np.array([network.optical_params.coupling_efficiency])
            params['propagation_loss'] = np.array([network.optical_params.loss])
        
        # Extract neural weights for photonic optimization
        if hasattr(network, 'layers'):
            for i, layer in enumerate(network.layers):
                if hasattr(layer, 'weight'):
                    params[f'layer_{i}_weights'] = layer.weight.detach().numpy().flatten()
        
        return params
    
    def _create_objective_function(self, network, target: str) -> Callable:
        """Create objective function for optimization."""
        def objective(params: Dict[str, np.ndarray]) -> float:
            # Simulate network performance with given parameters
            if target == "energy_efficiency":
                # Calculate energy efficiency metric
                total_power = sum(p.sum() for p in params.values() if 'power' in str(p))
                total_efficiency = 1.0 / (total_power + 1e-8)
                return total_efficiency
                
            elif target == "latency":
                # Calculate latency metric (lower is better, so negate for maximization)
                processing_delay = sum(len(p) for p in params.values()) * 1e-9
                return -processing_delay
                
            elif target == "accuracy":
                # Simulate accuracy based on parameter quality
                param_quality = sum(np.std(p) for p in params.values())
                return 1.0 / (param_quality + 1e-8)
            
            else:
                return 1.0  # Default objective
        
        return objective
    
    def _quantum_annealing_optimization(
        self,
        objective_func: Callable,
        initial_params: Dict[str, np.ndarray],
        max_iterations: int = 1000
    ) -> Dict[str, np.ndarray]:
        """Perform quantum annealing optimization."""
        current_params = {k: v.copy() for k, v in initial_params.items()}
        best_params = {k: v.copy() for k, v in initial_params.items()}
        
        current_energy = objective_func(current_params)
        best_energy = current_energy
        
        # Annealing schedule
        initial_temp = self.temperature
        
        for iteration in range(max_iterations):
            # Update temperature (cooling schedule)
            temperature = initial_temp * (1 - iteration / max_iterations)
            
            # Generate quantum-inspired perturbation
            perturbed_params = self._quantum_perturbation(current_params, temperature)
            
            # Evaluate new configuration
            new_energy = objective_func(perturbed_params)
            
            # Quantum acceptance criterion
            if self._quantum_acceptance(current_energy, new_energy, temperature):
                current_params = perturbed_params
                current_energy = new_energy
                
                # Update best solution
                if new_energy > best_energy:
                    best_params = {k: v.copy() for k, v in perturbed_params.items()}
                    best_energy = new_energy
            
            # Update quantum state
            self._evolve_quantum_state(iteration / max_iterations)
        
        return best_params
    
    def _quantum_perturbation(
        self,
        params: Dict[str, np.ndarray],
        temperature: float
    ) -> Dict[str, np.ndarray]:
        """Generate quantum-inspired parameter perturbations."""
        perturbed = {}
        
        for key, values in params.items():
            # Quantum amplitude-based perturbation
            quantum_noise = np.random.normal(0, temperature * 0.1, values.shape)
            
            # Apply quantum interference effects
            if len(values) > 1:
                interference = np.sin(np.arange(len(values)) * np.pi / len(values))
                quantum_noise *= interference
            
            perturbed[key] = values + quantum_noise
            
            # Ensure physical constraints
            if 'power' in key or 'efficiency' in key:
                perturbed[key] = np.clip(perturbed[key], 0.0, 1.0)
            elif 'wavelength' in key:
                perturbed[key] = np.clip(perturbed[key], 1e-9, 2e-6)  # Valid optical range
        
        return perturbed
    
    def _quantum_acceptance(self, current_energy: float, new_energy: float, temperature: float) -> bool:
        """Quantum-inspired acceptance criterion."""
        if new_energy > current_energy:
            return True
        
        # Quantum tunneling probability
        energy_diff = current_energy - new_energy
        tunneling_prob = np.exp(-energy_diff / (temperature + 1e-8))
        
        # Add quantum coherence effects
        coherence_factor = np.abs(np.sum(self.quantum_state['amplitudes'][:4]))
        enhanced_prob = tunneling_prob * (1 + 0.1 * coherence_factor)
        
        return np.random.random() < enhanced_prob
    
    def _evolve_quantum_state(self, progress: float):
        """Evolve quantum state during optimization."""
        # Simulate quantum state evolution
        evolution_operator = np.exp(1j * progress * np.pi)
        self.quantum_state['amplitudes'] *= evolution_operator
        
        # Add decoherence effects
        decoherence = 1 - 0.1 * progress
        self.quantum_state['amplitudes'] *= decoherence
        
        # Renormalize
        norm = np.linalg.norm(self.quantum_state['amplitudes'])
        if norm > 0:
            self.quantum_state['amplitudes'] /= norm
    
    def _apply_optimized_parameters(
        self,
        network,
        optimized_params: Dict[str, np.ndarray]
    ) -> float:
        """Apply optimized parameters to network and calculate improvement."""
        # Store original performance
        original_performance = self._measure_network_performance(network)
        
        # Apply optimized parameters
        if hasattr(network, 'wavelength_channels') and 'power_distribution' in optimized_params:
            # Update power distribution (simplified)
            power_dist = optimized_params['power_distribution']
            if hasattr(network, 'interference_weights'):
                network.interference_weights.data = torch.from_numpy(power_dist).float()
        
        # Measure improved performance
        improved_performance = self._measure_network_performance(network)
        
        improvement_factor = improved_performance / (original_performance + 1e-8)
        return improvement_factor
    
    def _measure_network_performance(self, network) -> float:
        """Measure current network performance."""
        # Simplified performance metric
        if hasattr(network, 'wavelength_channels'):
            return float(network.wavelength_channels) * 0.1
        return 1.0


class HyperParameterOptimizer:
    """
    Advanced hyperparameter optimization for photonic neural networks.
    
    Implements Bayesian optimization, genetic algorithms, and photonic-specific
    optimization strategies for automatic hyperparameter tuning.
    """
    
    def __init__(self, search_space: Dict[str, Any]):
        self.search_space = search_space
        self.evaluation_history = []
        self.best_parameters = None
        self.best_score = -float('inf')
        
        # Bayesian optimization components
        self.gaussian_process = None
        self.acquisition_function = "expected_improvement"
        
        # Genetic algorithm components
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
        self._logger = logging.getLogger(__name__)
    
    def optimize(
        self,
        objective_function: Callable,
        n_trials: int = 100,
        optimization_method: str = "bayesian"
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using specified method.
        
        Args:
            objective_function: Function to optimize (higher is better)
            n_trials: Number of optimization trials
            optimization_method: Method to use ('bayesian', 'genetic', 'random')
            
        Returns:
            Optimization results
        """
        start_time = time.perf_counter()
        
        if optimization_method == "bayesian":
            results = self._bayesian_optimization(objective_function, n_trials)
        elif optimization_method == "genetic":
            results = self._genetic_algorithm_optimization(objective_function, n_trials)
        elif optimization_method == "photonic_aware":
            results = self._photonic_aware_optimization(objective_function, n_trials)
        else:
            results = self._random_search(objective_function, n_trials)
        
        optimization_time = time.perf_counter() - start_time
        
        return {
            "best_parameters": self.best_parameters,
            "best_score": self.best_score,
            "optimization_method": optimization_method,
            "optimization_time": optimization_time,
            "total_evaluations": len(self.evaluation_history),
            "convergence_curve": [eval_data["score"] for eval_data in self.evaluation_history]
        }
    
    def _bayesian_optimization(self, objective_func: Callable, n_trials: int) -> Dict[str, Any]:
        """Bayesian optimization using Gaussian processes."""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
            from sklearn.preprocessing import StandardScaler
            from scipy.optimize import minimize
            
            # Initialize Gaussian Process
            kernel = Matern(length_scale=1.0, nu=2.5)
            self.gaussian_process = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42
            )
            
            # Random initialization
            for _ in range(min(10, n_trials)):
                params = self._sample_random_parameters()
                score = objective_func(params)
                self._update_history(params, score)
            
            # Bayesian optimization loop
            for trial in range(len(self.evaluation_history), n_trials):
                # Fit GP to current data
                X = np.array([self._encode_parameters(eval_data["parameters"]) 
                             for eval_data in self.evaluation_history])
                y = np.array([eval_data["score"] for eval_data in self.evaluation_history])
                
                self.gaussian_process.fit(X, y)
                
                # Optimize acquisition function
                next_params = self._optimize_acquisition_function()
                
                # Evaluate objective
                score = objective_func(next_params)
                self._update_history(next_params, score)
            
            return {"method": "bayesian", "gp_fitted": True}
            
        except ImportError:
            self._logger.warning("scikit-learn not available, falling back to random search")
            return self._random_search(objective_func, n_trials)
    
    def _genetic_algorithm_optimization(self, objective_func: Callable, n_trials: int) -> Dict[str, Any]:
        """Genetic algorithm optimization."""
        # Initialize population
        population = [self._sample_random_parameters() for _ in range(self.population_size)]
        
        generations = n_trials // self.population_size
        
        for generation in range(generations):
            # Evaluate population
            fitness_scores = []
            for individual in population:
                score = objective_func(individual)
                self._update_history(individual, score)
                fitness_scores.append(score)
            
            # Selection (tournament selection)
            selected = self._tournament_selection(population, fitness_scores)
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % len(selected)]
                
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if np.random.random() < self.mutation_rate:
                    child1 = self._mutate(child1)
                if np.random.random() < self.mutation_rate:
                    child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        return {"method": "genetic", "generations": generations}
    
    def _photonic_aware_optimization(self, objective_func: Callable, n_trials: int) -> Dict[str, Any]:
        """Photonic-aware optimization considering optical constraints."""
        # Start with physically meaningful parameters
        photonic_priors = {
            'wavelength': 1550e-9,  # Standard telecom wavelength
            'power': 1e-3,          # 1 mW
            'coupling_efficiency': 0.9,
            'loss': 0.1             # dB/cm
        }
        
        # Optimization considering optical physics
        for trial in range(n_trials):
            if trial < 10:
                # Start with physics-based initialization
                params = self._sample_physics_informed_parameters(photonic_priors)
            else:
                # Adaptive sampling based on optical performance
                params = self._adaptive_photonic_sampling()
            
            score = objective_func(params)
            self._update_history(params, score)
        
        return {"method": "photonic_aware", "physics_informed": True}
    
    def _sample_physics_informed_parameters(self, priors: Dict[str, float]) -> Dict[str, Any]:
        """Sample parameters with physics constraints."""
        params = {}
        
        for key, (param_type, bounds) in self.search_space.items():
            if key in priors:
                # Use physics-informed prior with small perturbation
                prior_value = priors[key]
                if param_type == "float":
                    # Gaussian perturbation around physical prior
                    noise_scale = (bounds[1] - bounds[0]) * 0.1
                    value = np.clip(
                        np.random.normal(prior_value, noise_scale),
                        bounds[0], bounds[1]
                    )
                    params[key] = value
                else:
                    params[key] = self._sample_parameter(param_type, bounds)
            else:
                params[key] = self._sample_parameter(param_type, bounds)
        
        return params
    
    def _adaptive_photonic_sampling(self) -> Dict[str, Any]:
        """Adaptive sampling based on photonic performance."""
        if len(self.evaluation_history) < 5:
            return self._sample_random_parameters()
        
        # Analyze best performing configurations
        top_configs = sorted(
            self.evaluation_history,
            key=lambda x: x["score"],
            reverse=True
        )[:5]
        
        # Sample around best configurations with adaptive noise
        base_config = np.random.choice(top_configs)["parameters"]
        
        params = {}
        for key, (param_type, bounds) in self.search_space.items():
            if key in base_config:
                base_value = base_config[key]
                
                # Adaptive noise based on parameter sensitivity
                noise_scale = self._estimate_parameter_sensitivity(key) * 0.1
                
                if param_type == "float":
                    value = np.clip(
                        np.random.normal(base_value, noise_scale),
                        bounds[0], bounds[1]
                    )
                    params[key] = value
                else:
                    params[key] = base_value
            else:
                params[key] = self._sample_parameter(param_type, bounds)
        
        return params
    
    def _estimate_parameter_sensitivity(self, param_name: str) -> float:
        """Estimate parameter sensitivity from evaluation history."""
        if len(self.evaluation_history) < 10:
            return 1.0
        
        # Simple sensitivity analysis
        param_values = []
        scores = []
        
        for eval_data in self.evaluation_history:
            if param_name in eval_data["parameters"]:
                param_values.append(eval_data["parameters"][param_name])
                scores.append(eval_data["score"])
        
        if len(param_values) < 5:
            return 1.0
        
        # Calculate correlation between parameter and score
        param_array = np.array(param_values)
        score_array = np.array(scores)
        
        correlation = np.corrcoef(param_array, score_array)[0, 1]
        sensitivity = abs(correlation) if not np.isnan(correlation) else 1.0
        
        return sensitivity
    
    def _sample_random_parameters(self) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        params = {}
        for key, (param_type, bounds) in self.search_space.items():
            params[key] = self._sample_parameter(param_type, bounds)
        return params
    
    def _sample_parameter(self, param_type: str, bounds: Any) -> Any:
        """Sample a single parameter."""
        if param_type == "float":
            return np.random.uniform(bounds[0], bounds[1])
        elif param_type == "int":
            return np.random.randint(bounds[0], bounds[1] + 1)
        elif param_type == "choice":
            return np.random.choice(bounds)
        elif param_type == "log_uniform":
            log_low, log_high = np.log10(bounds[0]), np.log10(bounds[1])
            return 10 ** np.random.uniform(log_low, log_high)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    def _update_history(self, params: Dict[str, Any], score: float):
        """Update evaluation history."""
        self.evaluation_history.append({
            "parameters": params.copy(),
            "score": score,
            "timestamp": time.time()
        })
        
        if score > self.best_score:
            self.best_score = score
            self.best_parameters = params.copy()
    
    def _encode_parameters(self, params: Dict[str, Any]) -> np.ndarray:
        """Encode parameters for GP input."""
        encoded = []
        for key in sorted(self.search_space.keys()):
            if key in params:
                value = params[key]
                if isinstance(value, (int, float)):
                    encoded.append(float(value))
                else:
                    # Handle categorical parameters
                    param_type, bounds = self.search_space[key]
                    if param_type == "choice":
                        encoded.append(float(bounds.index(value)))
                    else:
                        encoded.append(0.0)
            else:
                encoded.append(0.0)
        return np.array(encoded)
    
    def _optimize_acquisition_function(self) -> Dict[str, Any]:
        """Optimize acquisition function to find next point."""
        # Simplified acquisition optimization
        best_acquisition = -float('inf')
        best_params = None
        
        # Random sampling of acquisition function
        for _ in range(1000):
            candidate_params = self._sample_random_parameters()
            acquisition_value = self._expected_improvement(candidate_params)
            
            if acquisition_value > best_acquisition:
                best_acquisition = acquisition_value
                best_params = candidate_params
        
        return best_params
    
    def _expected_improvement(self, params: Dict[str, Any]) -> float:
        """Calculate expected improvement acquisition function."""
        if self.gaussian_process is None:
            return np.random.random()
        
        X_candidate = self._encode_parameters(params).reshape(1, -1)
        
        try:
            mu, sigma = self.gaussian_process.predict(X_candidate, return_std=True)
            
            # Expected improvement calculation
            current_best = self.best_score
            improvement = mu - current_best
            Z = improvement / (sigma + 1e-8)
            
            from scipy.stats import norm
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            return ei[0]
            
        except Exception:
            return np.random.random()
    
    def _tournament_selection(self, population: List, fitness_scores: List) -> List:
        """Tournament selection for genetic algorithm."""
        selected = []
        tournament_size = max(2, len(population) // 10)
        
        for _ in range(len(population)):
            # Random tournament
            tournament_indices = np.random.choice(
                len(population), tournament_size, replace=False
            )
            
            # Select best from tournament
            best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
            selected.append(population[best_index].copy())
        
        return selected
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Crossover operation for genetic algorithm."""
        child1, child2 = parent1.copy(), parent2.copy()
        
        # Uniform crossover
        for key in parent1.keys():
            if np.random.random() < 0.5:
                child1[key], child2[key] = child2[key], child1[key]
        
        return child1, child2
    
    def _mutate(self, individual: Dict) -> Dict:
        """Mutation operation for genetic algorithm."""
        mutated = individual.copy()
        
        # Gaussian mutation for numerical parameters
        for key, (param_type, bounds) in self.search_space.items():
            if key in mutated and np.random.random() < 0.3:  # 30% mutation rate per parameter
                if param_type == "float":
                    noise_scale = (bounds[1] - bounds[0]) * 0.1
                    mutated[key] = np.clip(
                        mutated[key] + np.random.normal(0, noise_scale),
                        bounds[0], bounds[1]
                    )
                elif param_type == "int":
                    mutated[key] = np.random.randint(bounds[0], bounds[1] + 1)
                elif param_type == "choice":
                    mutated[key] = np.random.choice(bounds)
        
        return mutated
    
    def _random_search(self, objective_func: Callable, n_trials: int) -> Dict[str, Any]:
        """Random search baseline."""
        for _ in range(n_trials):
            params = self._sample_random_parameters()
            score = objective_func(params)
            self._update_history(params, score)
        
        return {"method": "random"}


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