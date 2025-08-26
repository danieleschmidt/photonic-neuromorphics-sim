#!/usr/bin/env python3
"""
Simple test runner to validate our implementations work.
"""

import sys
import os
sys.path.insert(0, '/root/repo')
sys.path.insert(0, '/root/repo/test_mock_torch.py')

# Mock torch before importing our modules
import test_mock_torch
sys.modules['torch'] = test_mock_torch.torch

import numpy as np
import time

def test_quantum_temporal_entanglement():
    """Test quantum temporal entanglement basic functionality."""
    try:
        from src.photonic_neuromorphics.quantum_temporal_entanglement import (
            QuantumTemporalState, QuantumTemporalParameters, 
            QuantumTemporalEntanglementProcessor
        )
        
        print("✓ Successfully imported quantum temporal entanglement modules")
        
        # Test quantum state
        state = QuantumTemporalState(4)
        assert state.dimensions == 4
        assert len(state.amplitude) == 4
        print("✓ QuantumTemporalState initialization works")
        
        # Test parameters
        params = QuantumTemporalParameters()
        assert params.coherence_time == 100e-9
        print("✓ QuantumTemporalParameters initialization works")
        
        # Test processor
        processor = QuantumTemporalEntanglementProcessor(num_neurons=5, temporal_modes=4)
        assert processor.num_neurons == 5
        assert processor.temporal_modes == 4
        print("✓ QuantumTemporalEntanglementProcessor initialization works")
        
        # Test basic processing
        spike_train = test_mock_torch.torch.rand(10, 5)
        processed_spikes, metrics = processor.process_spike_train_quantum(spike_train)
        assert processed_spikes.shape == spike_train.shape
        assert "entanglement_fidelity" in metrics
        print("✓ Quantum spike processing works")
        
        print("✓ All quantum temporal entanglement tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Quantum temporal entanglement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ultra_high_performance_caching():
    """Test ultra high performance caching functionality."""
    try:
        from src.photonic_neuromorphics.ultra_high_performance_caching import (
            UltraHighPerformanceCache, CacheEntry, CacheLevel, ReplacementPolicy
        )
        
        print("✓ Successfully imported caching modules")
        
        # Test cache entry
        entry = CacheEntry(key="test", value="data", size_bytes=100)
        assert entry.key == "test"
        assert entry.value == "data"
        print("✓ CacheEntry initialization works")
        
        # Test cache
        cache = UltraHighPerformanceCache(
            l1_capacity=10,
            l2_capacity=20,
            enable_prefetching=False
        )
        print("✓ UltraHighPerformanceCache initialization works")
        
        # Test basic operations
        cache.put("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"
        print("✓ Basic cache put/get works")
        
        # Test with computation function
        def compute():
            return "computed_value"
        
        result = cache.get("missing_key", compute)
        assert result == "computed_value"
        print("✓ Cache miss with computation works")
        
        # Test statistics
        stats = cache.get_cache_statistics()
        assert "overall" in stats
        assert "hit_rate" in stats["overall"]
        print("✓ Cache statistics work")
        
        print("✓ All caching tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Caching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantum_accelerated_optimization():
    """Test quantum accelerated optimization functionality."""
    try:
        from src.photonic_neuromorphics.quantum_accelerated_optimization import (
            QuantumState, QuantumGates, QuantumAcceleratedOptimizer,
            QuantumOptimizationParameters, QuantumOptimizationMethod
        )
        
        print("✓ Successfully imported quantum optimization modules")
        
        # Test quantum state
        state = QuantumState(2)
        assert state.num_qubits == 2
        assert state.dimension == 4
        print("✓ QuantumState initialization works")
        
        # Test quantum gates
        pauli_x = QuantumGates.X
        assert pauli_x.shape == (2, 2)
        print("✓ QuantumGates work")
        
        # Test optimizer parameters
        params = QuantumOptimizationParameters(
            method=QuantumOptimizationMethod.QAOA,
            num_qubits=4
        )
        assert params.num_qubits == 4
        print("✓ QuantumOptimizationParameters initialization works")
        
        # Test optimizer
        optimizer = QuantumAcceleratedOptimizer(params)
        assert optimizer.params.num_qubits == 4
        print("✓ QuantumAcceleratedOptimizer initialization works")
        
        print("✓ All quantum optimization tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Quantum optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_neuromorphic_metamaterials():
    """Test neuromorphic metamaterials functionality.""" 
    try:
        from src.photonic_neuromorphics.neuromorphic_photonic_metamaterials import (
            NeuromorphicPhotonicMetamaterial, MetamaterialUnitCell, 
            MetamaterialParameters, MetamaterialType
        )
        
        print("✓ Successfully imported metamaterials modules")
        
        # Test parameters
        params = MetamaterialParameters()
        assert params.unit_cell_size == 200e-9
        print("✓ MetamaterialParameters initialization works")
        
        # Test unit cell
        unit_cell = MetamaterialUnitCell(
            position=(0, 0, 0),
            refractive_index=1.5 + 0.1j,
            structure_type=MetamaterialType.SPLIT_RING_RESONATOR
        )
        assert unit_cell.refractive_index == 1.5 + 0.1j
        print("✓ MetamaterialUnitCell initialization works")
        
        # Test metamaterial
        metamaterial = NeuromorphicPhotonicMetamaterial(
            grid_size=(10, 10, 5),
            metamaterial_params=params
        )
        assert metamaterial.grid_size == (10, 10, 5)
        print("✓ NeuromorphicPhotonicMetamaterial initialization works")
        
        # Test processing
        spike_train = test_mock_torch.torch.rand(20, 50)  # Match grid capacity
        processed_output, metrics = metamaterial.process_neural_activity(spike_train)
        assert processed_output.shape == spike_train.shape
        assert "reconfigurations" in metrics
        print("✓ Metamaterial processing works")
        
        print("✓ All metamaterials tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Metamaterials test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Running comprehensive functionality tests...\n")
    
    tests = [
        ("Quantum Temporal Entanglement", test_quantum_temporal_entanglement),
        ("Ultra High Performance Caching", test_ultra_high_performance_caching),
        ("Quantum Accelerated Optimization", test_quantum_accelerated_optimization),
        ("Neuromorphic Metamaterials", test_neuromorphic_metamaterials),
    ]
    
    results = []
    total_start = time.time()
    
    for test_name, test_func in tests:
        print(f"🔍 Testing {test_name}...")
        start_time = time.time()
        
        try:
            success = test_func()
            duration = time.time() - start_time
            results.append((test_name, success, duration))
            
            if success:
                print(f"✅ {test_name} passed in {duration:.2f}s\n")
            else:
                print(f"❌ {test_name} failed in {duration:.2f}s\n")
                
        except Exception as e:
            duration = time.time() - start_time
            results.append((test_name, False, duration))
            print(f"💥 {test_name} crashed in {duration:.2f}s: {e}\n")
    
    total_duration = time.time() - total_start
    
    # Summary
    print("=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, duration in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name:<35} ({duration:.2f}s)")
    
    print("-" * 60)
    print(f"📊 Results: {passed}/{total} tests passed")
    print(f"⏱️  Total time: {total_duration:.2f}s")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is working correctly.")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed. Review implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)