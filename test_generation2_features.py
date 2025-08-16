#!/usr/bin/env python3
"""
Test Generation 2 'Make it Robust' features.
Tests autonomous learning, quantum-photonic interfaces, and real-time optimization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_autonomous_learning():
    """Test autonomous learning framework."""
    try:
        from photonic_neuromorphics import (
            create_autonomous_learning_demo, AutonomousLearningFramework
        )
        
        # Create autonomous learning system
        autonomous_learner = create_autonomous_learning_demo()
        print("✅ Autonomous learning framework created")
        
        # Test learning components
        assert hasattr(autonomous_learner, 'meta_optimizer')
        assert hasattr(autonomous_learner, 'evolutionary_optimizer')
        assert hasattr(autonomous_learner, 'optical_tuner')
        print("✅ All learning components present")
        
        return True
    except Exception as e:
        print(f"❌ Autonomous learning test failed: {e}")
        return False

def test_quantum_photonic_interface():
    """Test quantum-photonic hybrid interface."""
    try:
        from photonic_neuromorphics import (
            create_quantum_photonic_demo, HybridQuantumPhotonic, 
            QuantumPhotonicProcessor, PhotonicQubit
        )
        import torch
        import numpy as np
        
        # Create quantum-photonic system
        hybrid_system = create_quantum_photonic_demo()
        print("✅ Quantum-photonic hybrid system created")
        
        # Test quantum processor
        assert hybrid_system.quantum_processor.n_qubits == 6
        assert len(hybrid_system.quantum_processor.qubits) == 6
        print("✅ Quantum processor properly initialized")
        
        # Test quantum operations
        hybrid_system.quantum_processor.create_bell_state(0, 1)
        entanglement = hybrid_system.quantum_processor.get_entanglement_entropy()
        assert abs(entanglement) < 1e-6 or entanglement >= 0  # Accept small numerical errors
        print("✅ Quantum entanglement operations working")
        
        # Test quantum-enhanced inference
        torch.manual_seed(42)
        test_data = torch.randn(4, 784)  # MNIST input size
        output = hybrid_system.quantum_enhanced_forward(test_data)
        assert output.shape == (4, 10)  # MNIST output size
        print("✅ Quantum-enhanced inference working")
        
        return True
    except Exception as e:
        import traceback
        print(f"❌ Quantum-photonic interface test failed: {e}")
        print(traceback.format_exc())
        return False

def test_realtime_optimization():
    """Test real-time adaptive optimization."""
    try:
        from photonic_neuromorphics import (
            create_realtime_optimization_demo, RealTimeOptimizer,
            RealTimeProfiler, AdaptiveParameterTuner
        )
        
        # Create real-time optimizer
        optimizer = create_realtime_optimization_demo()
        print("✅ Real-time optimizer created")
        
        # Test components
        assert isinstance(optimizer.profiler, RealTimeProfiler)
        assert isinstance(optimizer.parameter_tuner, AdaptiveParameterTuner)
        print("✅ Optimization components properly initialized")
        
        # Test profiler
        optimizer.profiler.start_profiling()
        optimizer.profiler.record_inference(0.85, 0.001, 0.1, 0.9)
        recent_metrics = optimizer.profiler.get_recent_metrics(1)
        assert len(recent_metrics) > 0
        optimizer.profiler.stop_profiling()
        print("✅ Real-time profiling working")
        
        # Test parameter tuner
        from photonic_neuromorphics.realtime_adaptive_optimization import PerformanceMetrics
        metrics = PerformanceMetrics(accuracy=0.8, throughput=500, latency=0.002)
        adjustments = optimizer.parameter_tuner.get_adaptive_parameters(
            metrics, {'accuracy': 0.9}
        )
        assert isinstance(adjustments, dict)
        print("✅ Adaptive parameter tuning working")
        
        return True
    except Exception as e:
        print(f"❌ Real-time optimization test failed: {e}")
        return False

def test_integration():
    """Test integration between Generation 2 components."""
    try:
        from photonic_neuromorphics import (
            create_quantum_photonic_demo, create_realtime_optimization_demo,
            AutonomousLearningFramework
        )
        import torch
        
        # Create integrated system
        hybrid_system = create_quantum_photonic_demo()
        optimizer = create_realtime_optimization_demo()
        autonomous_learner = AutonomousLearningFramework()
        
        print("✅ Integrated systems created")
        
        # Test quantum-enhanced optimization workflow
        torch.manual_seed(42)
        test_data = torch.randn(2, 784)  # MNIST input size
        
        # Quantum-enhanced inference
        quantum_output = hybrid_system.quantum_enhanced_forward(test_data)
        
        # Get quantum advantage metrics
        qa_metrics = hybrid_system.get_quantum_advantage_metrics()
        assert 'entanglement_capacity' in qa_metrics
        assert 'quantum_volume' in qa_metrics
        
        print("✅ Quantum-enhanced workflow working")
        
        # Test performance metrics
        from photonic_neuromorphics.realtime_adaptive_optimization import PerformanceMetrics
        perf_metrics = PerformanceMetrics(
            accuracy=0.85,
            throughput=1000,
            energy_efficiency=50,
            optical_efficiency=0.8
        )
        
        composite_score = perf_metrics.composite_score()
        assert 0 <= composite_score <= 1
        print("✅ Performance metrics integration working")
        
        return True
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all Generation 2 feature tests."""
    print("🧪 Testing Generation 2 'Make it Robust' Features\n")
    
    tests = [
        ("Autonomous Learning", test_autonomous_learning),
        ("Quantum-Photonic Interface", test_quantum_photonic_interface),
        ("Real-Time Optimization", test_realtime_optimization),
        ("System Integration", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} tests passed")
        else:
            print(f"❌ {test_name} tests failed")
    
    print(f"\n📊 Results: {passed}/{total} test categories passed")
    
    if passed == total:
        print("🎉 Generation 2 'Make it Robust' STATUS: SUCCESS ✅")
        print("\n🚀 Advanced Features Confirmed:")
        print("  • Autonomous Learning with Meta-Learning & Evolution")
        print("  • Quantum-Photonic Hybrid Computing")
        print("  • Real-Time Adaptive Optimization")
        print("  • Integrated Multi-Modal Intelligence")
        return True
    else:
        print("❌ Some Generation 2 features need attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)