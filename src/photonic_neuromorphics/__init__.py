"""
Photonic Neuromorphics Simulation Framework

A comprehensive toolkit for designing and simulating silicon-photonic spiking neural networks,
with automatic RTL generation for MPW tape-outs.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.ai"

# Core imports for easy access
from .core import PhotonicSNN, WaveguideNeuron, encode_to_spikes, create_mnist_photonic_snn
from .simulator import PhotonicSimulator, SimulationMode, create_optimized_simulator
from .rtl import RTLGenerator, create_rtl_for_mnist, create_high_performance_rtl
from .components import (
    MachZehnderNeuron, MicroringResonator, PhotonicCrystalCavity,
    PhaseChangeMaterial, WaveguideCrossing, create_component_library
)
from .architectures import (
    PhotonicCrossbar, PhotonicReservoir, ConvolutionalPhotonicNetwork,
    create_mnist_photonic_crossbar, create_temporal_photonic_reservoir
)
from .benchmarks import (
    NeuronMorphicBenchmark, create_mnist_benchmark, create_temporal_benchmark,
    run_comprehensive_comparison
)
from .multiwavelength import (
    WDMMultiplexer, MultiWavelengthNeuron, WDMCrossbar, AttentionMechanism,
    create_multiwavelength_mnist_network, simulate_multiwavelength_network
)
from .physical_validation import (
    PhysicalValidationPipeline, FDTDSimulator, ThermalAnalyzer,
    ProcessVariationAnalyzer, create_validation_pipeline, validate_neuron_design
)
from .security import (
    SecurityManager, SecureSimulationSession, InputValidator, OutputSanitizer,
    create_secure_environment
)
from .enhanced_logging import (
    PhotonicLogger, CorrelationContext, PerformanceTracker, LogAnalyzer,
    setup_photonic_logging, logged_operation
)
from .robust_error_handling import (
    ErrorHandler, CircuitBreaker, robust_operation, error_recovery_context,
    create_robust_error_system
)
from .xr_agent_mesh import (
    XRAgentMesh, XRAgent, SpatialAnchorAgent, ObjectTrackerAgent,
    PhotonicXRProcessor, XRCoordinate, XRMessage, XRDataType, XRAgentType,
    create_xr_demo_mesh, run_xr_mesh_simulation
)
from .xr_spatial_computing import (
    PhotonicSpatialProcessor, SpatialMemoryManager, SpatialObject, SpatialRegion,
    create_spatial_computing_demo, run_spatial_computing_demo
)
from .xr_visualization import (
    PhotonicInteractionProcessor, XRVisualizationEngine, XRInteraction, HapticFeedback,
    InteractionType, RenderingMode, run_xr_visualization_demo, create_interaction_demo_sequence
)
from .autonomous_learning import (
    AutonomousLearningFramework, MetaLearningOptimizer, EvolutionaryOptimizer,
    AdaptiveOpticalTuner, LearningMetrics, create_autonomous_learning_demo,
    run_autonomous_learning_demo
)
from .quantum_photonic_interface import (
    QuantumPhotonicProcessor, PhotonicQubit, HybridQuantumPhotonic, QuantumState,
    create_quantum_photonic_demo, run_quantum_photonic_demo
)
from .realtime_adaptive_optimization import (
    RealTimeOptimizer, RealTimeProfiler, AdaptiveParameterTuner, PerformanceMetrics,
    create_realtime_optimization_demo, run_realtime_optimization_demo, run_realtime_demo
)
from .distributed_computing import (
    NodeManager, DistributedPhotonicSimulator, NodeInfo, ComputeTask,
    create_distributed_demo_cluster, run_distributed_computing_demo
)
from .advanced_analytics import (
    AdvancedAnalyticsFramework, PerformanceAnalyzer, OptimizationAnalyzer,
    SystemHealthAnalyzer, AnalyticsMetric, InsightReport,
    create_advanced_analytics_demo, run_advanced_analytics_demo
)

__all__ = [
    # Core functionality
    "PhotonicSNN",
    "WaveguideNeuron", 
    "encode_to_spikes",
    "create_mnist_photonic_snn",
    
    # Simulation
    "PhotonicSimulator",
    "SimulationMode",
    "create_optimized_simulator",
    
    # RTL Generation
    "RTLGenerator",
    "create_rtl_for_mnist",
    "create_high_performance_rtl",
    
    # Advanced Components
    "MachZehnderNeuron",
    "MicroringResonator", 
    "PhotonicCrystalCavity",
    "PhaseChangeMaterial",
    "WaveguideCrossing",
    "create_component_library",
    
    # Architectures
    "PhotonicCrossbar",
    "PhotonicReservoir",
    "ConvolutionalPhotonicNetwork",
    "create_mnist_photonic_crossbar",
    "create_temporal_photonic_reservoir",
    
    # Benchmarking
    "NeuronMorphicBenchmark",
    "create_mnist_benchmark",
    "create_temporal_benchmark",
    "run_comprehensive_comparison",
    
    # Multi-wavelength Computing
    "WDMMultiplexer",
    "MultiWavelengthNeuron", 
    "WDMCrossbar",
    "AttentionMechanism",
    "create_multiwavelength_mnist_network",
    "simulate_multiwavelength_network",
    
    # Physical Validation
    "PhysicalValidationPipeline",
    "FDTDSimulator",
    "ThermalAnalyzer", 
    "ProcessVariationAnalyzer",
    "create_validation_pipeline",
    "validate_neuron_design",
    
    # Security
    "SecurityManager",
    "SecureSimulationSession",
    "InputValidator",
    "OutputSanitizer",
    "create_secure_environment",
    
    # Enhanced Logging
    "PhotonicLogger",
    "CorrelationContext",
    "PerformanceTracker",
    "LogAnalyzer",
    "setup_photonic_logging",
    "logged_operation",
    
    # Robust Error Handling
    "ErrorHandler",
    "CircuitBreaker",
    "robust_operation",
    "error_recovery_context",
    "create_robust_error_system",
    
    # XR Agent Mesh
    "XRAgentMesh",
    "XRAgent", 
    "SpatialAnchorAgent",
    "ObjectTrackerAgent",
    "PhotonicXRProcessor",
    "XRCoordinate",
    "XRMessage",
    "XRDataType",
    "XRAgentType",
    "create_xr_demo_mesh",
    "run_xr_mesh_simulation",
    
    # XR Spatial Computing
    "PhotonicSpatialProcessor",
    "SpatialMemoryManager",
    "SpatialObject",
    "SpatialRegion",
    "create_spatial_computing_demo",
    "run_spatial_computing_demo",
    
    # XR Visualization
    "PhotonicInteractionProcessor",
    "XRVisualizationEngine",
    "XRInteraction",
    "HapticFeedback",
    "InteractionType",
    "RenderingMode",
    "run_xr_visualization_demo",
    "create_interaction_demo_sequence",
    
    # Autonomous Learning
    "AutonomousLearningFramework",
    "MetaLearningOptimizer", 
    "EvolutionaryOptimizer",
    "AdaptiveOpticalTuner",
    "create_autonomous_learning_demo",
    "run_autonomous_learning_demo",
    
    # Quantum-Photonic Interface
    "QuantumPhotonicProcessor",
    "PhotonicQubit",
    "HybridQuantumPhotonic",
    "QuantumState",
    "create_quantum_photonic_demo",
    "run_quantum_photonic_demo",
    
    # Real-Time Adaptive Optimization
    "RealTimeOptimizer",
    "RealTimeProfiler", 
    "AdaptiveParameterTuner",
    "PerformanceMetrics",
    "create_realtime_optimization_demo",
    "run_realtime_optimization_demo",
    "run_realtime_demo",
    
    # Distributed Computing
    "NodeManager",
    "DistributedPhotonicSimulator",
    "NodeInfo",
    "ComputeTask",
    "create_distributed_demo_cluster",
    "run_distributed_computing_demo",
    
    # Advanced Analytics
    "AdvancedAnalyticsFramework",
    "PerformanceAnalyzer",
    "OptimizationAnalyzer", 
    "SystemHealthAnalyzer",
    "AnalyticsMetric",
    "InsightReport",
    "create_advanced_analytics_demo",
    "run_advanced_analytics_demo",
]