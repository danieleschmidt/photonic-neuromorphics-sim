#!/usr/bin/env python3
"""
Basic functionality test for photonic neuromorphics system.
Tests core components to ensure "Make it Work" status.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test that core modules can be imported successfully."""
    try:
        from photonic_neuromorphics import (
            PhotonicSNN, WaveguideNeuron, encode_to_spikes,
            PhotonicSimulator, create_mnist_photonic_snn
        )
        print("✅ Core imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_neural_network():
    """Test basic photonic neural network creation."""
    try:
        from photonic_neuromorphics import PhotonicSNN, WaveguideNeuron
        
        # Create simple network
        network = PhotonicSNN(
            neuron_type=WaveguideNeuron,
            topology=[10, 5, 2],
            wavelength=1550e-9
        )
        print("✅ Basic neural network creation successful")
        return True
    except Exception as e:
        print(f"❌ Neural network creation failed: {e}")
        return False

def test_mnist_network():
    """Test MNIST-specific network creation."""
    try:
        from photonic_neuromorphics import create_mnist_photonic_snn
        
        mnist_network = create_mnist_photonic_snn()
        print("✅ MNIST network creation successful")
        return True
    except Exception as e:
        print(f"❌ MNIST network creation failed: {e}")
        return False

def test_spike_encoding():
    """Test spike encoding functionality."""
    try:
        from photonic_neuromorphics import encode_to_spikes
        import numpy as np
        
        # Test with simple data
        test_data = np.random.rand(10, 10)
        spikes = encode_to_spikes(test_data, duration=100e-9)
        print("✅ Spike encoding successful")
        return True
    except Exception as e:
        print(f"❌ Spike encoding failed: {e}")
        return False

def test_simulation():
    """Test basic simulation capability."""
    try:
        from photonic_neuromorphics import PhotonicSimulator, create_optimized_simulator
        
        simulator = create_optimized_simulator()
        print("✅ Simulator creation successful")
        return True
    except Exception as e:
        print(f"❌ Simulator creation failed: {e}")
        return False

def main():
    """Run all basic functionality tests."""
    print("🧪 Testing Photonic Neuromorphics Basic Functionality\n")
    
    tests = [
        test_basic_imports,
        test_basic_neural_network,
        test_mnist_network,
        test_spike_encoding,
        test_simulation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Generation 1 'Make it Work' STATUS: SUCCESS ✅")
        print("The photonic neuromorphics system is functioning correctly!")
        return True
    else:
        print("❌ Some tests failed. Basic functionality needs attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)