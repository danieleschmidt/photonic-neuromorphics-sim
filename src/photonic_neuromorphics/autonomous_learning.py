"""
Autonomous Learning Framework for Photonic Neuromorphics

Advanced self-improving algorithms that automatically optimize photonic neural network
performance through meta-learning, adaptive parameter tuning, and evolutionary strategies.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging
from collections import defaultdict, deque
import time
import copy
from pathlib import Path
import json

from .core import PhotonicSNN, OpticalParameters, encode_to_spikes
from .exceptions import ValidationError, OpticalModelError, PhotonicNeuromorphicsException
from .monitoring import MetricsCollector


@dataclass
class LearningMetrics:
    """Metrics for autonomous learning performance."""
    accuracy: float = 0.0
    loss: float = float('inf')
    convergence_rate: float = 0.0
    optical_efficiency: float = 0.0
    energy_per_inference: float = 0.0
    training_time: float = 0.0
    stability_score: float = 0.0
    adaptation_speed: float = 0.0
    
    def __post_init__(self):
        self.timestamp = time.time()
    
    def composite_score(self) -> float:
        """Calculate composite performance score."""
        # Weighted combination of key metrics
        weights = {
            'accuracy': 0.3,
            'optical_efficiency': 0.25,
            'energy_efficiency': 0.2,
            'stability': 0.15,
            'adaptation_speed': 0.1
        }
        
        # Normalize and weight metrics
        energy_efficiency = 1.0 / (1.0 + self.energy_per_inference * 1e6)  # Higher is better
        stability = min(self.stability_score, 1.0)
        adaptation = min(self.adaptation_speed, 1.0)
        
        score = (
            weights['accuracy'] * self.accuracy +
            weights['optical_efficiency'] * self.optical_efficiency +
            weights['energy_efficiency'] * energy_efficiency +
            weights['stability'] * stability +
            weights['adaptation_speed'] * adaptation
        )
        
        return max(0.0, min(1.0, score))


class MetaLearningOptimizer:
    """Meta-learning optimizer for photonic neural networks."""
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 meta_lr: float = 0.01,
                 adaptation_steps: int = 5):
        self.learning_rate = learning_rate
        self.meta_lr = meta_lr
        self.adaptation_steps = adaptation_steps
        self.meta_parameters = {}
        self.adaptation_history = deque(maxlen=100)
        self.logger = logging.getLogger(__name__)
    
    def meta_update(self, 
                   network: PhotonicSNN, 
                   support_data: torch.Tensor,
                   support_labels: torch.Tensor,
                   query_data: torch.Tensor,
                   query_labels: torch.Tensor) -> LearningMetrics:
        """
        Perform meta-learning update using Model-Agnostic Meta-Learning (MAML).
        
        Args:
            network: Photonic neural network
            support_data: Support set data for adaptation
            support_labels: Support set labels
            query_data: Query set data for evaluation
            query_labels: Query set labels
            
        Returns:
            LearningMetrics: Performance metrics after meta-update
        """
        start_time = time.time()
        
        # Save original parameters
        original_params = {name: param.clone() for name, param in network.named_parameters()}
        
        # Inner loop: Fast adaptation on support set
        optimizer = torch.optim.SGD(network.parameters(), lr=self.learning_rate)
        
        for step in range(self.adaptation_steps):
            optimizer.zero_grad()
            
            # Forward pass on support set
            support_output = network.forward(support_data)
            support_loss = nn.functional.cross_entropy(support_output, support_labels)
            
            # Backward pass
            support_loss.backward()
            optimizer.step()
            
            # Track adaptation progress
            with torch.no_grad():
                support_acc = (support_output.argmax(1) == support_labels).float().mean()
                self.logger.debug(f"Adaptation step {step}: loss={support_loss:.4f}, acc={support_acc:.4f}")
        
        # Evaluate on query set
        with torch.no_grad():
            query_output = network.forward(query_data)
            query_loss = nn.functional.cross_entropy(query_output, query_labels)
            query_acc = (query_output.argmax(1) == query_labels).float().mean()
        
        # Meta-gradient computation
        meta_gradients = {}
        for name, param in network.named_parameters():
            if param.grad is not None:
                meta_gradients[name] = param.grad.clone()
        
        # Restore original parameters and apply meta-update
        for name, param in network.named_parameters():
            param.data = original_params[name]
            if name in meta_gradients:
                param.data -= self.meta_lr * meta_gradients[name]
        
        # Calculate metrics
        training_time = time.time() - start_time
        optical_efficiency = self._calculate_optical_efficiency(network)
        energy_per_inference = self._estimate_energy_consumption(network)
        
        metrics = LearningMetrics(
            accuracy=query_acc.item(),
            loss=query_loss.item(),
            optical_efficiency=optical_efficiency,
            energy_per_inference=energy_per_inference,
            training_time=training_time,
            adaptation_speed=1.0 / (training_time + 1e-6)
        )
        
        self.adaptation_history.append(metrics)
        return metrics
    
    def _calculate_optical_efficiency(self, network: PhotonicSNN) -> float:
        """Calculate optical efficiency of the network."""
        # Simplified efficiency calculation based on wavelength optimization
        wavelength_nm = network.wavelength * 1e9
        
        # Optimal efficiency around 1550nm
        optimal_wavelength = 1550.0
        wavelength_deviation = abs(wavelength_nm - optimal_wavelength) / optimal_wavelength
        efficiency = max(0.1, 1.0 - wavelength_deviation)
        
        return efficiency
    
    def _estimate_energy_consumption(self, network: PhotonicSNN) -> float:
        """Estimate energy consumption per inference."""
        # Simplified energy model based on network size and optical parameters
        total_synapses = sum(a * b for a, b in zip(network.topology[:-1], network.topology[1:]))
        base_energy = total_synapses * 0.1e-12  # 0.1 pJ per synapse
        
        # Optical power contribution
        optical_energy = network.optical_params.power * 1e-9  # 1ns inference time
        
        return base_energy + optical_energy


class EvolutionaryOptimizer:
    """Evolutionary strategy for optimizing photonic network architectures."""
    
    def __init__(self, 
                 population_size: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elitism_ratio: float = 0.2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        self.generation = 0
        self.best_genome = None
        self.best_fitness = -float('inf')
        self.population_history = []
        self.logger = logging.getLogger(__name__)
    
    def evolve_architecture(self, 
                          base_topology: List[int],
                          fitness_function: Callable,
                          generations: int = 50) -> Tuple[List[int], float]:
        """
        Evolve optimal network architecture using genetic algorithms.
        
        Args:
            base_topology: Starting network topology
            fitness_function: Function to evaluate network fitness
            generations: Number of evolutionary generations
            
        Returns:
            Tuple of (best_topology, best_fitness)
        """
        # Initialize population
        population = self._initialize_population(base_topology)
        
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate fitness for each individual
            fitness_scores = []
            for genome in population:
                try:
                    fitness = fitness_function(genome)
                    fitness_scores.append(fitness)
                except Exception as e:
                    self.logger.warning(f"Fitness evaluation failed for genome {genome}: {e}")
                    fitness_scores.append(-float('inf'))
            
            # Track best individual
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_genome = population[best_idx].copy()
                self.logger.info(f"Generation {gen}: New best fitness {self.best_fitness:.4f}")
            
            # Selection and reproduction
            new_population = self._select_and_reproduce(population, fitness_scores)
            population = new_population
            
            # Record generation statistics
            gen_stats = {
                'generation': gen,
                'best_fitness': max(fitness_scores),
                'avg_fitness': np.mean(fitness_scores),
                'std_fitness': np.std(fitness_scores)
            }
            self.population_history.append(gen_stats)
            
            self.logger.debug(f"Generation {gen}: best={gen_stats['best_fitness']:.4f}, "
                            f"avg={gen_stats['avg_fitness']:.4f}")
        
        return self.best_genome, self.best_fitness
    
    def _initialize_population(self, base_topology: List[int]) -> List[List[int]]:
        """Initialize random population based on base topology."""
        population = []
        
        for _ in range(self.population_size):
            # Create variation of base topology
            genome = base_topology.copy()
            
            # Randomly modify hidden layers
            for i in range(1, len(genome) - 1):
                if np.random.random() < self.mutation_rate:
                    # Mutation: change layer size
                    variation = np.random.randint(-50, 51)
                    new_size = max(10, genome[i] + variation)
                    genome[i] = new_size
            
            # Occasionally add/remove layers
            if np.random.random() < 0.1 and len(genome) < 6:
                # Add layer
                insert_pos = np.random.randint(1, len(genome) - 1)
                new_layer_size = np.random.randint(50, 300)
                genome.insert(insert_pos, new_layer_size)
            elif np.random.random() < 0.05 and len(genome) > 3:
                # Remove layer
                remove_pos = np.random.randint(1, len(genome) - 2)
                genome.pop(remove_pos)
            
            population.append(genome)
        
        return population
    
    def _select_and_reproduce(self, 
                            population: List[List[int]], 
                            fitness_scores: List[float]) -> List[List[int]]:
        """Selection and reproduction with elitism."""
        new_population = []
        
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        # Elitism: keep best individuals
        elite_count = int(self.population_size * self.elitism_ratio)
        for i in range(elite_count):
            new_population.append(population[sorted_indices[i]].copy())
        
        # Reproduction to fill remaining slots
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
                new_population.extend([child1, child2])
            else:
                new_population.extend([parent1.copy(), parent2.copy()])
        
        # Trim to exact population size
        new_population = new_population[:self.population_size]
        
        # Mutation
        for genome in new_population[elite_count:]:  # Don't mutate elites
            self._mutate(genome)
        
        return new_population
    
    def _tournament_selection(self, 
                            population: List[List[int]], 
                            fitness_scores: List[float],
                            tournament_size: int = 3) -> List[int]:
        """Tournament selection for parent selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Single-point crossover for topology evolution."""
        # Ensure both parents have same input/output
        if parent1[0] != parent2[0] or parent1[-1] != parent2[-1]:
            return parent1.copy(), parent2.copy()
        
        # Single-point crossover in hidden layers
        if len(parent1) > 2 and len(parent2) > 2:
            crossover_point = np.random.randint(1, min(len(parent1), len(parent2)) - 1)
            
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            
            return child1, child2
        
        return parent1.copy(), parent2.copy()
    
    def _mutate(self, genome: List[int]) -> None:
        """Mutate genome in-place."""
        for i in range(1, len(genome) - 1):  # Skip input/output layers
            if np.random.random() < self.mutation_rate:
                # Gaussian mutation
                mutation = int(np.random.normal(0, genome[i] * 0.1))
                genome[i] = max(10, genome[i] + mutation)


class AdaptiveOpticalTuner:
    """Adaptive tuning of optical parameters for optimal performance."""
    
    def __init__(self, 
                 tuning_range: Dict[str, Tuple[float, float]] = None,
                 adaptation_rate: float = 0.01):
        self.tuning_range = tuning_range or {
            'wavelength': (1260e-9, 1675e-9),  # O, E, S, C, L bands
            'power': (1e-6, 10e-3),           # 1μW to 10mW
            'coupling_efficiency': (0.1, 0.99),
            'detector_efficiency': (0.1, 0.95)
        }
        self.adaptation_rate = adaptation_rate
        self.optimization_history = []
        self.logger = logging.getLogger(__name__)
    
    def tune_parameters(self, 
                       network: PhotonicSNN,
                       validation_data: torch.Tensor,
                       validation_labels: torch.Tensor,
                       iterations: int = 100) -> OpticalParameters:
        """
        Adaptively tune optical parameters using gradient-free optimization.
        
        Args:
            network: Photonic neural network
            validation_data: Validation dataset
            validation_labels: Validation labels
            iterations: Number of tuning iterations
            
        Returns:
            OpticalParameters: Optimized optical parameters
        """
        # Initialize with current parameters
        current_params = copy.deepcopy(network.optical_params)
        best_params = copy.deepcopy(current_params)
        best_score = self._evaluate_parameters(network, current_params, 
                                             validation_data, validation_labels)
        
        # Coordinate descent optimization
        for iteration in range(iterations):
            improved = False
            
            # Try to improve each parameter
            for param_name in self.tuning_range.keys():
                if not hasattr(current_params, param_name):
                    continue
                
                current_value = getattr(current_params, param_name)
                param_min, param_max = self.tuning_range[param_name]
                
                # Try increasing the parameter
                new_value_up = min(param_max, current_value * (1 + self.adaptation_rate))
                test_params = copy.deepcopy(current_params)
                setattr(test_params, param_name, new_value_up)
                
                score_up = self._evaluate_parameters(network, test_params, 
                                                   validation_data, validation_labels)
                
                # Try decreasing the parameter
                new_value_down = max(param_min, current_value * (1 - self.adaptation_rate))
                test_params = copy.deepcopy(current_params)
                setattr(test_params, param_name, new_value_down)
                
                score_down = self._evaluate_parameters(network, test_params, 
                                                     validation_data, validation_labels)
                
                # Choose best direction
                if score_up > best_score and score_up > score_down:
                    setattr(current_params, param_name, new_value_up)
                    best_score = score_up
                    best_params = copy.deepcopy(current_params)
                    improved = True
                    self.logger.debug(f"Improved {param_name}: {current_value:.6e} -> {new_value_up:.6e}")
                elif score_down > best_score:
                    setattr(current_params, param_name, new_value_down)
                    best_score = score_down
                    best_params = copy.deepcopy(current_params)
                    improved = True
                    self.logger.debug(f"Improved {param_name}: {current_value:.6e} -> {new_value_down:.6e}")
            
            # Record optimization progress
            self.optimization_history.append({
                'iteration': iteration,
                'score': best_score,
                'parameters': copy.deepcopy(best_params)
            })
            
            # Early stopping if no improvement
            if not improved:
                self.logger.info(f"Optical tuning converged after {iteration} iterations")
                break
        
        self.logger.info(f"Optical tuning completed: final score = {best_score:.4f}")
        return best_params
    
    def _evaluate_parameters(self, 
                           network: PhotonicSNN,
                           optical_params: OpticalParameters,
                           validation_data: torch.Tensor,
                           validation_labels: torch.Tensor) -> float:
        """Evaluate network performance with given optical parameters."""
        try:
            # Temporarily update network parameters
            original_params = network.optical_params
            network.optical_params = optical_params
            
            # Evaluate performance
            with torch.no_grad():
                outputs = network.forward(validation_data)
                accuracy = (outputs.argmax(1) == validation_labels).float().mean()
                
                # Calculate optical efficiency
                wavelength_nm = optical_params.wavelength * 1e9
                optical_efficiency = max(0.1, 1.0 - abs(wavelength_nm - 1550) / 1550)
                
                # Energy efficiency
                energy_per_inference = optical_params.power * 1e-9  # 1ns inference
                energy_efficiency = 1.0 / (1.0 + energy_per_inference * 1e6)
                
                # Composite score
                score = 0.5 * accuracy + 0.3 * optical_efficiency + 0.2 * energy_efficiency
            
            # Restore original parameters
            network.optical_params = original_params
            return score.item()
            
        except Exception as e:
            self.logger.warning(f"Parameter evaluation failed: {e}")
            network.optical_params = original_params
            return 0.0


class AutonomousLearningFramework:
    """Complete autonomous learning framework for photonic neuromorphics."""
    
    def __init__(self, 
                 meta_learning: bool = True,
                 evolutionary_optimization: bool = True,
                 adaptive_tuning: bool = True):
        self.meta_learning_enabled = meta_learning
        self.evolutionary_enabled = evolutionary_optimization
        self.adaptive_tuning_enabled = adaptive_tuning
        
        self.meta_optimizer = MetaLearningOptimizer() if meta_learning else None
        self.evolutionary_optimizer = EvolutionaryOptimizer() if evolutionary_optimization else None
        self.optical_tuner = AdaptiveOpticalTuner() if adaptive_tuning else None
        
        self.learning_history = []
        self.best_configurations = {}
        self.logger = logging.getLogger(__name__)
    
    def autonomous_optimize(self,
                          initial_network: PhotonicSNN,
                          train_data: torch.Tensor,
                          train_labels: torch.Tensor,
                          val_data: torch.Tensor,
                          val_labels: torch.Tensor,
                          optimization_budget: int = 1000) -> PhotonicSNN:
        """
        Perform complete autonomous optimization of photonic neural network.
        
        Args:
            initial_network: Starting network configuration
            train_data: Training dataset
            train_labels: Training labels
            val_data: Validation dataset
            val_labels: Validation labels
            optimization_budget: Total optimization steps
            
        Returns:
            PhotonicSNN: Optimized network
        """
        self.logger.info("Starting autonomous optimization of photonic neural network")
        start_time = time.time()
        
        current_network = copy.deepcopy(initial_network)
        best_network = copy.deepcopy(initial_network)
        best_metrics = self._evaluate_network(current_network, val_data, val_labels)
        
        # Phase 1: Evolutionary architecture optimization (if enabled)
        if self.evolutionary_enabled and optimization_budget > 100:
            self.logger.info("Phase 1: Evolutionary architecture optimization")
            
            def fitness_function(topology):
                try:
                    test_network = PhotonicSNN(
                        topology=topology,
                        neuron_type=current_network.neuron_type,
                        synapse_type=current_network.synapse_type,
                        wavelength=current_network.wavelength,
                        optical_params=current_network.optical_params
                    )
                    metrics = self._evaluate_network(test_network, val_data, val_labels)
                    return metrics.composite_score()
                except Exception as e:
                    self.logger.warning(f"Fitness evaluation failed: {e}")
                    return 0.0
            
            optimal_topology, best_fitness = self.evolutionary_optimizer.evolve_architecture(
                current_network.topology, fitness_function, 
                generations=min(20, optimization_budget // 20)
            )
            
            # Create network with optimal topology
            if best_fitness > best_metrics.composite_score():
                current_network = PhotonicSNN(
                    topology=optimal_topology,
                    neuron_type=current_network.neuron_type,
                    synapse_type=current_network.synapse_type,
                    wavelength=current_network.wavelength,
                    optical_params=current_network.optical_params
                )
                best_network = copy.deepcopy(current_network)
                best_metrics = self._evaluate_network(current_network, val_data, val_labels)
                self.logger.info(f"Improved topology: {optimal_topology}")
        
        # Phase 2: Adaptive optical parameter tuning (if enabled)
        if self.adaptive_tuning_enabled and optimization_budget > 50:
            self.logger.info("Phase 2: Adaptive optical parameter tuning")
            
            optimized_params = self.optical_tuner.tune_parameters(
                current_network, val_data, val_labels, 
                iterations=min(100, optimization_budget // 10)
            )
            
            # Update network with optimized parameters
            current_network.optical_params = optimized_params
            tuned_metrics = self._evaluate_network(current_network, val_data, val_labels)
            
            if tuned_metrics.composite_score() > best_metrics.composite_score():
                best_network = copy.deepcopy(current_network)
                best_metrics = tuned_metrics
                self.logger.info("Improved optical parameters")
        
        # Phase 3: Meta-learning fine-tuning (if enabled)
        if self.meta_learning_enabled and optimization_budget > 20:
            self.logger.info("Phase 3: Meta-learning fine-tuning")
            
            # Split data for meta-learning
            split_idx = len(train_data) // 2
            support_data, query_data = train_data[:split_idx], train_data[split_idx:]
            support_labels, query_labels = train_labels[:split_idx], train_labels[split_idx:]
            
            meta_iterations = min(50, optimization_budget // 20)
            for i in range(meta_iterations):
                meta_metrics = self.meta_optimizer.meta_update(
                    current_network, support_data, support_labels,
                    query_data, query_labels
                )
                
                # Evaluate on validation set
                val_metrics = self._evaluate_network(current_network, val_data, val_labels)
                
                if val_metrics.composite_score() > best_metrics.composite_score():
                    best_network = copy.deepcopy(current_network)
                    best_metrics = val_metrics
                    self.logger.debug(f"Meta-learning iteration {i}: improved performance")
        
        # Record final results
        optimization_time = time.time() - start_time
        final_results = {
            'optimization_time': optimization_time,
            'initial_score': self._evaluate_network(initial_network, val_data, val_labels).composite_score(),
            'final_score': best_metrics.composite_score(),
            'improvement': best_metrics.composite_score() - self._evaluate_network(initial_network, val_data, val_labels).composite_score(),
            'final_topology': best_network.topology,
            'final_optical_params': best_network.optical_params
        }
        
        self.learning_history.append(final_results)
        self.logger.info(f"Autonomous optimization completed in {optimization_time:.2f}s")
        self.logger.info(f"Performance improvement: {final_results['improvement']:.4f}")
        
        return best_network
    
    def _evaluate_network(self, 
                         network: PhotonicSNN,
                         data: torch.Tensor,
                         labels: torch.Tensor) -> LearningMetrics:
        """Evaluate network performance and return comprehensive metrics."""
        try:
            start_time = time.time()
            
            with torch.no_grad():
                outputs = network.forward(data)
                loss = nn.functional.cross_entropy(outputs, labels)
                accuracy = (outputs.argmax(1) == labels).float().mean()
            
            # Calculate additional metrics
            wavelength_nm = network.wavelength * 1e9
            optical_efficiency = max(0.1, 1.0 - abs(wavelength_nm - 1550) / 1550)
            energy_per_inference = network.optical_params.power * 1e-9
            
            # Estimate stability (simplified)
            weight_variance = torch.var(torch.cat([p.flatten() for p in network.parameters()]))
            stability_score = 1.0 / (1.0 + weight_variance.item())
            
            evaluation_time = time.time() - start_time
            
            return LearningMetrics(
                accuracy=accuracy.item(),
                loss=loss.item(),
                optical_efficiency=optical_efficiency,
                energy_per_inference=energy_per_inference,
                training_time=evaluation_time,
                stability_score=stability_score,
                adaptation_speed=1.0 / (evaluation_time + 1e-6)
            )
            
        except Exception as e:
            self.logger.error(f"Network evaluation failed: {e}")
            return LearningMetrics()  # Return default metrics
    
    def save_learning_state(self, filepath: str) -> None:
        """Save current learning state to file."""
        state = {
            'learning_history': self.learning_history,
            'best_configurations': self.best_configurations,
            'meta_optimizer_state': self.meta_optimizer.adaptation_history if self.meta_optimizer else None,
            'evolutionary_state': self.evolutionary_optimizer.population_history if self.evolutionary_optimizer else None,
            'optical_tuning_history': self.optical_tuner.optimization_history if self.optical_tuner else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"Learning state saved to {filepath}")
    
    def load_learning_state(self, filepath: str) -> None:
        """Load learning state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.learning_history = state.get('learning_history', [])
            self.best_configurations = state.get('best_configurations', {})
            
            self.logger.info(f"Learning state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load learning state: {e}")


def create_autonomous_learning_demo() -> AutonomousLearningFramework:
    """Create a demonstration autonomous learning framework."""
    return AutonomousLearningFramework(
        meta_learning=True,
        evolutionary_optimization=True,
        adaptive_tuning=True
    )


def run_autonomous_learning_demo():
    """Run autonomous learning demonstration."""
    import matplotlib.pyplot as plt
    
    # Create demonstration data
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Simple classification dataset
    n_samples = 1000
    n_features = 20
    n_classes = 3
    
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_X, val_X, test_X = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    train_y, val_y, test_y = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
    
    # Convert to spike trains
    train_spikes = encode_to_spikes(train_X.numpy())
    val_spikes = encode_to_spikes(val_X.numpy())
    
    # Create initial network
    from .core import PhotonicSNN, WaveguideNeuron
    initial_network = PhotonicSNN(
        topology=[n_features, 50, 25, n_classes],
        neuron_type=WaveguideNeuron,
        wavelength=1550e-9
    )
    
    # Create autonomous learning framework
    autonomous_learner = create_autonomous_learning_demo()
    
    print("Starting autonomous learning demonstration...")
    
    # Perform autonomous optimization
    optimized_network = autonomous_learner.autonomous_optimize(
        initial_network=initial_network,
        train_data=train_spikes,
        train_labels=train_y,
        val_data=val_spikes,
        val_labels=val_y,
        optimization_budget=200
    )
    
    # Evaluate final performance
    initial_metrics = autonomous_learner._evaluate_network(initial_network, val_spikes, val_y)
    final_metrics = autonomous_learner._evaluate_network(optimized_network, val_spikes, val_y)
    
    print(f"\n=== Autonomous Learning Results ===")
    print(f"Initial accuracy: {initial_metrics.accuracy:.4f}")
    print(f"Final accuracy: {final_metrics.accuracy:.4f}")
    print(f"Improvement: {final_metrics.accuracy - initial_metrics.accuracy:.4f}")
    print(f"Initial composite score: {initial_metrics.composite_score():.4f}")
    print(f"Final composite score: {final_metrics.composite_score():.4f}")
    print(f"Score improvement: {final_metrics.composite_score() - initial_metrics.composite_score():.4f}")
    print(f"Final topology: {optimized_network.topology}")
    print(f"Optical efficiency: {final_metrics.optical_efficiency:.4f}")
    print(f"Energy per inference: {final_metrics.energy_per_inference:.2e} J")
    
    return autonomous_learner, initial_network, optimized_network


if __name__ == "__main__":
    run_autonomous_learning_demo()