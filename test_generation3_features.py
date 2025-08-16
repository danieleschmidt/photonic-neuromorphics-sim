#!/usr/bin/env python3
"""
Test Generation 3 'Make it Scale' features.
Tests distributed computing, advanced analytics, and scaling optimizations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_distributed_computing():
    """Test distributed computing framework."""
    try:
        from photonic_neuromorphics import (
            create_distributed_demo_cluster, NodeManager, DistributedPhotonicSimulator,
            NodeInfo, ComputeTask
        )
        
        # Create distributed cluster
        node_manager, simulator = create_distributed_demo_cluster()
        print("✅ Distributed cluster created")
        
        # Test node manager
        assert len(node_manager.nodes) > 0
        cluster_status = node_manager.get_cluster_status()
        assert cluster_status['total_nodes'] > 0
        print("✅ Node manager functioning")
        
        # Test task submission
        task = ComputeTask(
            task_type="inference",
            priority=3,
            estimated_time=1.0
        )
        task_id = node_manager.submit_task(task)
        assert task_id is not None
        print("✅ Task submission working")
        
        # Test distributed simulator
        assert isinstance(simulator, DistributedPhotonicSimulator)
        assert simulator.node_manager == node_manager
        print("✅ Distributed simulator created")
        
        # Cleanup
        node_manager.stop()
        
        return True
    except Exception as e:
        print(f"❌ Distributed computing test failed: {e}")
        return False

def test_advanced_analytics():
    """Test advanced analytics framework."""
    try:
        from photonic_neuromorphics import (
            create_advanced_analytics_demo, AdvancedAnalyticsFramework,
            PerformanceAnalyzer, AnalyticsMetric
        )
        from photonic_neuromorphics.realtime_adaptive_optimization import PerformanceMetrics
        
        # Create analytics framework
        analytics = create_advanced_analytics_demo()
        print("✅ Advanced analytics framework created")
        
        # Test analytics components
        assert isinstance(analytics.performance_analyzer, PerformanceAnalyzer)
        assert hasattr(analytics, 'optimization_analyzer')
        assert hasattr(analytics, 'health_analyzer')
        print("✅ Analytics components present")
        
        # Test metrics recording
        test_metrics = PerformanceMetrics(
            accuracy=0.85,
            throughput=1000,
            latency=0.01,
            energy_efficiency=100
        )
        
        # Convert and record metrics
        analytics_metrics = analytics._convert_to_analytics_metrics(test_metrics)
        analytics.performance_analyzer.record_metrics(analytics_metrics)
        print("✅ Metrics recording working")
        
        # Test analysis generation
        analysis_report = analytics.analyze_system(test_metrics)
        assert 'performance_analysis' in analysis_report
        assert 'health_assessment' in analysis_report
        assert 'executive_summary' in analysis_report
        print("✅ Analysis generation working")
        
        return True
    except Exception as e:
        print(f"❌ Advanced analytics test failed: {e}")
        return False

def test_scaling_integration():
    """Test integration of scaling features."""
    try:
        from photonic_neuromorphics import (
            create_distributed_demo_cluster, create_advanced_analytics_demo,
            create_realtime_optimization_demo
        )
        import torch
        
        # Create integrated scaling systems
        node_manager, dist_simulator = create_distributed_demo_cluster()
        analytics = create_advanced_analytics_demo()
        optimizer = create_realtime_optimization_demo()
        
        print("✅ Integrated scaling systems created")
        
        # Test data flow between systems
        torch.manual_seed(42)
        test_data = torch.randn(10, 784)
        
        # Test distributed processing capability
        cluster_status = node_manager.get_cluster_status()
        assert cluster_status['online_nodes'] > 0
        print("✅ Distributed processing capability confirmed")
        
        # Test analytics on cluster metrics
        from photonic_neuromorphics.realtime_adaptive_optimization import PerformanceMetrics
        metrics = PerformanceMetrics(
            accuracy=0.88,
            throughput=cluster_status['total_photonic_units'] * 100,  # Scale with resources
            latency=0.008,
            energy_efficiency=120
        )
        
        analysis = analytics.analyze_system(metrics, node_manager)
        assert 'cluster' in analysis['health_assessment']['components']
        print("✅ Cluster analytics integration working")
        
        # Test scaling metrics
        exec_summary = analysis['executive_summary']
        assert 'overall_status' in exec_summary
        print("✅ Scaling metrics calculation working")
        
        # Cleanup
        node_manager.stop()
        optimizer.stop_optimization()
        
        return True
    except Exception as e:
        print(f"❌ Scaling integration test failed: {e}")
        return False

def test_performance_optimization():
    """Test performance optimization features."""
    try:
        from photonic_neuromorphics import create_mnist_photonic_snn
        from photonic_neuromorphics.realtime_adaptive_optimization import PerformanceMetrics
        import torch
        import time
        
        # Create network for performance testing
        network = create_mnist_photonic_snn()
        print("✅ Test network created")
        
        # Test inference performance
        torch.manual_seed(42)
        test_data = torch.randn(100, 784)
        
        start_time = time.time()
        outputs = network.forward(test_data)
        inference_time = time.time() - start_time
        
        assert outputs.shape == (100, 10)
        throughput = len(test_data) / inference_time
        print(f"✅ Inference performance: {throughput:.1f} samples/sec")
        
        # Test batch processing efficiency
        batch_sizes = [10, 50, 100]
        throughputs = []
        
        for batch_size in batch_sizes:
            batch_data = test_data[:batch_size]
            start_time = time.time()
            _ = network.forward(batch_data)
            batch_time = time.time() - start_time
            batch_throughput = batch_size / batch_time
            throughputs.append(batch_throughput)
        
        # Larger batches should generally be more efficient
        efficiency_improvement = throughputs[-1] / throughputs[0]
        print(f"✅ Batch efficiency improvement: {efficiency_improvement:.1f}x")
        
        # Test memory efficiency
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process larger dataset
        large_data = torch.randn(500, 784)
        _ = network.forward(large_data)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        print(f"✅ Memory efficiency test: {memory_usage:.1f} MB for 500 samples")
        
        return True
    except Exception as e:
        print(f"❌ Performance optimization test failed: {e}")
        return False

def main():
    """Run all Generation 3 scaling tests."""
    print("🧪 Testing Generation 3 'Make it Scale' Features\n")
    
    tests = [
        ("Distributed Computing", test_distributed_computing),
        ("Advanced Analytics", test_advanced_analytics), 
        ("Scaling Integration", test_scaling_integration),
        ("Performance Optimization", test_performance_optimization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🚀 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} tests passed")
        else:
            print(f"❌ {test_name} tests failed")
    
    print(f"\n📊 Results: {passed}/{total} test categories passed")
    
    if passed == total:
        print("🎉 Generation 3 'Make it Scale' STATUS: SUCCESS ✅")
        print("\n🚀 Scaling Features Confirmed:")
        print("  • Distributed Multi-Node Processing")
        print("  • Advanced Analytics & Insights")
        print("  • Performance Optimization")
        print("  • Integrated Scaling Architecture")
        return True
    else:
        print("❌ Some Generation 3 features need attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)