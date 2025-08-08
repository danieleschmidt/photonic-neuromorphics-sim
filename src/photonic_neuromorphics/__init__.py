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
]