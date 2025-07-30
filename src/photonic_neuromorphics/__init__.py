"""
Photonic Neuromorphics Simulation Framework

A comprehensive toolkit for designing and simulating silicon-photonic spiking neural networks,
with automatic RTL generation for MPW tape-outs.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.ai"

# Core imports for easy access
from .core import PhotonicSNN, WaveguideNeuron
from .simulator import PhotonicSimulator
from .rtl import RTLGenerator

__all__ = [
    "PhotonicSNN",
    "WaveguideNeuron", 
    "PhotonicSimulator",
    "RTLGenerator",
]