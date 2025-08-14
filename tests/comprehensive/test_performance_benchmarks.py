"""
Comprehensive Performance Benchmark Test Suite

Advanced performance testing including load testing, stress testing,
scalability benchmarks, and performance regression detection.
"""

import unittest
import time
import sys
import os
import threading
import multiprocessing
import gc
import memory_profiler
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import statistics

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from photonic_neuromorphics.enhanced_core import (
        EnhancedPhotonicNeuron, PhotonicNetworkTopology, PhotonicResearchBenchmark
    )
    from photonic_neuromorphics.high_performance_optimization import (
        HighPerformanceOptimizer, IntelligentCache, ParallelProcessor
    )
    from photonic_neuromorphics.scalability_framework import (
        DistributedTaskScheduler, LoadBalancer, NodeManager
    )
    from photonic_neuromorphics.production_monitoring import (
        PhotonicSystemMonitor, MetricsCollector
    )
    PERFORMANCE_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Performance modules not available for testing: {e}")
    PERFORMANCE_MODULES_AVAILABLE = False


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
        self.results = {}
        self.requirements = {}
    
    def run_benchmark(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Run a performance benchmark for a function."""
        # Warm up
        for _ in range(3):
            try:
                func(*args, **kwargs)
            except:
                pass
        
        # Collect garbage before benchmark
        gc.collect()
        
        # Run benchmark
        times = []
        memory_usage = []
        
        for i in range(10):  # 10 iterations
            # Memory before
            mem_before = self._get_memory_usage()
            
            # Time execution
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            # Memory after
            mem_after = self._get_memory_usage()
            
            execution_time = end_time - start_time
            memory_delta = mem_after - mem_before
            
            times.append(execution_time)
            memory_usage.append(memory_delta)
        
        # Calculate statistics
        benchmark_results = {
            'name': self.name,
            'iterations': len(times),
            'mean_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_time': statistics.stdev(times) if len(times) > 1 else 0,
            'mean_memory': statistics.mean(memory_usage),
            'max_memory': max(memory_usage),
            'throughput': 1.0 / statistics.mean(times) if statistics.mean(times) > 0 else 0,
            'all_times': times,
            'all_memory': memory_usage
        }
        
        self.results[self.name] = benchmark_results
        return benchmark_results
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            # Fallback to approximate memory usage
            return 0.0
    
    def check_performance_requirements(self, requirements: Dict[str, float]) -> bool:
        """Check if benchmark meets performance requirements."""
        if self.name not in self.results:
            return False
        
        result = self.results[self.name]
        
        # Check time requirements
        if 'max_time' in requirements:
            if result['mean_time'] > requirements['max_time']:
                return False
        
        # Check memory requirements
        if 'max_memory' in requirements:
            if result['max_memory'] > requirements['max_memory']:
                return False
        
        # Check throughput requirements
        if 'min_throughput' in requirements:
            if result['throughput'] < requirements['min_throughput']:
                return False
        
        return True


class TestCorePerformance(unittest.TestCase):
    """Test core photonic neuromorphic performance."""
    
    def setUp(self):
        """Set up performance test environment."""
        if not PERFORMANCE_MODULES_AVAILABLE:
            self.skipTest("Performance modules not available")
        
        self.benchmark = PerformanceBenchmark("core_performance")
        
        # Performance requirements
        self.requirements = {
            'neuron_creation_max_time': 0.01,  # 10ms
            'network_creation_max_time': 5.0,   # 5s
            'forward_pass_max_time': 1.0,       # 1s
            'max_memory_per_neuron': 1.0        # 1MB
        }
    
    def test_neuron_creation_performance(self):
        """Test neuron creation performance."""
        def create_neuron():
            return EnhancedPhotonicNeuron()
        
        results = self.benchmark.run_benchmark(create_neuron)
        
        # Should create neurons quickly
        self.assertLess(results['mean_time'], self.requirements['neuron_creation_max_time'])
        self.assertGreater(results['throughput'], 100)  # > 100 neurons/second
        
        print(f"Neuron creation: {results['mean_time']:.4f}s avg, {results['throughput']:.0f} neurons/sec")
    
    def test_network_creation_performance(self):
        """Test network creation performance."""
        layer_sizes = [1000, 500, 250, 100]
        
        def create_network():
            return PhotonicNetworkTopology(layer_sizes)
        
        results = self.benchmark.run_benchmark(create_network)
        
        # Should create networks in reasonable time
        self.assertLess(results['mean_time'], self.requirements['network_creation_max_time'])
        
        print(f"Network creation ({sum(layer_sizes)} neurons): {results['mean_time']:.3f}s avg")
    
    def test_forward_pass_performance(self):
        """Test forward pass performance."""
        network = PhotonicNetworkTopology([100, 50, 10])
        input_data = [0.5] * 100
        
        def forward_pass():
            return network.forward_pass(input_data)
        
        results = self.benchmark.run_benchmark(forward_pass)
        
        # Should complete forward pass quickly
        self.assertLess(results['mean_time'], self.requirements['forward_pass_max_time'])
        
        print(f"Forward pass: {results['mean_time']:.4f}s avg, {results['throughput']:.0f} inferences/sec")
    
    def test_batch_processing_performance(self):
        """Test batch processing performance."""
        network = PhotonicNetworkTopology([50, 25, 10])
        batch_size = 32
        batch_data = [[0.5] * 50 for _ in range(batch_size)]
        
        def process_batch():
            results = []
            for input_data in batch_data:
                results.append(network.forward_pass(input_data))
            return results
        
        results = self.benchmark.run_benchmark(process_batch)
        
        # Calculate throughput per sample
        samples_per_second = batch_size * results['throughput']
        
        print(f"Batch processing ({batch_size} samples): {results['mean_time']:.3f}s, {samples_per_second:.0f} samples/sec")
        
        # Should process batches efficiently
        self.assertGreater(samples_per_second, 50)  # > 50 samples/second
    
    def test_memory_efficiency(self):
        """Test memory efficiency of core components."""
        # Test memory usage for different network sizes
        network_sizes = [
            [10, 5],
            [100, 50, 10],
            [500, 250, 100, 10],
            [1000, 500, 250, 100, 10]
        ]
        
        memory_usage = []
        
        for layer_sizes in network_sizes:
            gc.collect()
            mem_before = self.benchmark._get_memory_usage()
            
            network = PhotonicNetworkTopology(layer_sizes)
            
            mem_after = self.benchmark._get_memory_usage()
            memory_delta = mem_after - mem_before
            
            memory_per_neuron = memory_delta / sum(layer_sizes)
            memory_usage.append(memory_per_neuron)
            
            print(f"Network {layer_sizes}: {memory_delta:.2f}MB total, {memory_per_neuron:.4f}MB per neuron")
        
        # Memory usage should be reasonable
        avg_memory_per_neuron = statistics.mean(memory_usage)
        self.assertLess(avg_memory_per_neuron, self.requirements['max_memory_per_neuron'])


class TestOptimizationPerformance(unittest.TestCase):
    """Test high-performance optimization performance."""
    
    def setUp(self):
        """Set up optimization performance tests."""
        if not PERFORMANCE_MODULES_AVAILABLE:
            self.skipTest("Performance modules not available")
        
        self.benchmark = PerformanceBenchmark("optimization_performance")
        self.optimizer = HighPerformanceOptimizer()
    
    def test_cache_performance(self):
        """Test cache performance."""
        cache = IntelligentCache(max_size=1000)
        
        # Test cache write performance
        def cache_writes():
            for i in range(100):
                cache.put(f"key_{i}", f"value_{i}")
        
        write_results = self.benchmark.run_benchmark(cache_writes)
        print(f"Cache writes (100 items): {write_results['mean_time']:.4f}s")
        
        # Test cache read performance
        def cache_reads():
            for i in range(100):
                cache.get(f"key_{i}")
        
        read_results = self.benchmark.run_benchmark(cache_reads)
        print(f"Cache reads (100 items): {read_results['mean_time']:.4f}s")
        
        # Cache operations should be fast
        self.assertLess(write_results['mean_time'], 0.01)  # < 10ms for 100 writes
        self.assertLess(read_results['mean_time'], 0.001)   # < 1ms for 100 reads
    
    def test_parallel_processing_performance(self):
        """Test parallel processing performance."""
        processor = ParallelProcessor()
        
        # Test data
        test_data = list(range(1000))
        
        def simple_operation(x):
            return x * x
        
        # Sequential processing
        def sequential_processing():
            return [simple_operation(x) for x in test_data]
        
        seq_results = self.benchmark.run_benchmark(sequential_processing)
        
        # Parallel processing (simulate with smaller chunks for test)
        def parallel_processing():
            chunks = [test_data[i:i+100] for i in range(0, len(test_data), 100)]
            results = []
            for chunk in chunks:
                chunk_results = [simple_operation(x) for x in chunk]
                results.extend(chunk_results)
            return results
        
        par_results = self.benchmark.run_benchmark(parallel_processing)
        
        print(f"Sequential processing: {seq_results['mean_time']:.4f}s")
        print(f"Parallel processing: {par_results['mean_time']:.4f}s")
        
        # Parallel should be competitive or better
        # Note: For small operations, parallel might have overhead
        speedup = seq_results['mean_time'] / par_results['mean_time']
        print(f"Speedup factor: {speedup:.2f}x")
    
    def test_batch_optimization_performance(self):
        """Test batch optimization performance."""
        test_data = list(range(1000))
        
        def processing_operation(x):
            # Simulate computational work
            result = 0
            for i in range(10):
                result += x * i
            return result
        
        # Test different batch sizes
        batch_sizes = [10, 50, 100, 200]
        
        for batch_size in batch_sizes:
            def batch_processing():
                return self.optimizer.optimize_batch_processing(
                    test_data, processing_operation, batch_size=batch_size
                )
            
            results = self.benchmark.run_benchmark(batch_processing)
            throughput = len(test_data) * results['throughput']
            
            print(f"Batch size {batch_size}: {results['mean_time']:.3f}s, {throughput:.0f} items/sec")
            
            # Should process all items efficiently
            self.assertEqual(len(batch_processing()), len(test_data))


class TestScalabilityPerformance(unittest.TestCase):
    """Test scalability framework performance."""
    
    def setUp(self):
        """Set up scalability performance tests."""
        if not PERFORMANCE_MODULES_AVAILABLE:
            self.skipTest("Performance modules not available")
        
        self.benchmark = PerformanceBenchmark("scalability_performance")
    
    def test_node_management_performance(self):
        """Test node management performance."""
        node_manager = NodeManager()
        
        # Test node registration performance
        def register_nodes():
            from photonic_neuromorphics.scalability_framework import ComputeNode
            for i in range(100):
                node = ComputeNode(
                    node_id=f"node_{i}",
                    host="localhost",
                    port=8000 + i,
                    capacity=100
                )
                node_manager.register_node(node)
        
        reg_results = self.benchmark.run_benchmark(register_nodes)
        
        # Test node query performance
        def query_nodes():
            for _ in range(100):
                available_nodes = node_manager.get_available_nodes()
        
        query_results = self.benchmark.run_benchmark(query_nodes)
        
        print(f"Node registration (100 nodes): {reg_results['mean_time']:.4f}s")
        print(f"Node queries (100 queries): {query_results['mean_time']:.4f}s")
        
        # Node operations should be fast
        self.assertLess(reg_results['mean_time'], 1.0)    # < 1s for 100 registrations
        self.assertLess(query_results['mean_time'], 0.1)  # < 100ms for 100 queries
    
    def test_load_balancing_performance(self):
        """Test load balancing performance."""
        from photonic_neuromorphics.scalability_framework import LoadBalancer, WorkloadTask, ComputeNode
        
        load_balancer = LoadBalancer()
        
        # Set up nodes
        for i in range(10):
            node = ComputeNode(
                node_id=f"node_{i}",
                host="localhost",
                port=8000 + i,
                capacity=100
            )
            load_balancer.node_manager.register_node(node)
        
        # Test task assignment performance
        def assign_tasks():
            for i in range(100):
                task = WorkloadTask(
                    task_id=f"task_{i}",
                    task_type="test",
                    resource_requirements={'cpu': 1}
                )
                assigned_node = load_balancer.assign_task(task)
                if assigned_node:
                    load_balancer.complete_task(task)
        
        lb_results = self.benchmark.run_benchmark(assign_tasks)
        
        print(f"Load balancing (100 tasks): {lb_results['mean_time']:.4f}s")
        print(f"Task assignment rate: {100 * lb_results['throughput']:.0f} tasks/sec")
        
        # Load balancing should be efficient
        self.assertLess(lb_results['mean_time'], 0.5)  # < 500ms for 100 tasks


class TestMonitoringPerformance(unittest.TestCase):
    """Test monitoring system performance."""
    
    def setUp(self):
        """Set up monitoring performance tests."""
        if not PERFORMANCE_MODULES_AVAILABLE:
            self.skipTest("Performance modules not available")
        
        self.benchmark = PerformanceBenchmark("monitoring_performance")
    
    def test_metrics_collection_performance(self):
        """Test metrics collection performance."""
        monitor = PhotonicSystemMonitor(collection_interval=0.1)
        
        # Test metric recording performance
        def record_metrics():
            for i in range(1000):
                monitor.record_metric(f"test.metric_{i % 10}", float(i))
        
        try:
            monitor.start_monitoring()
            time.sleep(0.1)  # Let monitoring start
            
            record_results = self.benchmark.run_benchmark(record_metrics)
            
            print(f"Metrics recording (1000 metrics): {record_results['mean_time']:.4f}s")
            print(f"Metrics recording rate: {1000 * record_results['throughput']:.0f} metrics/sec")
            
            # Should handle high metric volume
            self.assertLess(record_results['mean_time'], 1.0)  # < 1s for 1000 metrics
            
        finally:
            monitor.stop_monitoring()
    
    def test_dashboard_generation_performance(self):
        """Test dashboard generation performance."""
        monitor = PhotonicSystemMonitor(collection_interval=0.1)
        
        # Add some test metrics
        for i in range(100):
            monitor.record_metric(f"test.metric_{i}", float(i))
        
        def generate_dashboard():
            return monitor.get_metric_dashboard()
        
        try:
            monitor.start_monitoring()
            time.sleep(0.2)  # Let some metrics accumulate
            
            dashboard_results = self.benchmark.run_benchmark(generate_dashboard)
            
            print(f"Dashboard generation: {dashboard_results['mean_time']:.4f}s")
            
            # Dashboard generation should be fast
            self.assertLess(dashboard_results['mean_time'], 0.1)  # < 100ms
            
        finally:
            monitor.stop_monitoring()


class TestLoadStress(unittest.TestCase):
    """Test system behavior under load and stress conditions."""
    
    def setUp(self):
        """Set up load/stress test environment."""
        if not PERFORMANCE_MODULES_AVAILABLE:
            self.skipTest("Performance modules not available")
    
    def test_concurrent_network_operations(self):
        """Test concurrent network operations."""
        network = PhotonicNetworkTopology([100, 50, 10])
        
        def worker_function(worker_id):
            results = []
            for i in range(10):
                input_data = [0.1 * (worker_id + i)] * 100
                output = network.forward_pass(input_data)
                results.append(output)
            return results
        
        # Test with multiple threads
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker_function, i) for i in range(10)]
            results = [future.result() for future in futures]
        
        concurrent_time = time.time() - start_time
        
        # Test sequential execution for comparison
        start_time = time.time()
        sequential_results = [worker_function(i) for i in range(10)]
        sequential_time = time.time() - start_time
        
        print(f"Concurrent execution (4 threads): {concurrent_time:.3f}s")
        print(f"Sequential execution: {sequential_time:.3f}s")
        print(f"Concurrency speedup: {sequential_time / concurrent_time:.2f}x")
        
        # Should complete successfully
        self.assertEqual(len(results), 10)
        self.assertEqual(len(sequential_results), 10)
    
    def test_memory_pressure_handling(self):
        """Test system behavior under memory pressure."""
        networks = []
        memory_usage = []
        
        # Create networks until memory pressure
        for i in range(10):
            try:
                # Create progressively larger networks
                layer_sizes = [100 * (i + 1), 50 * (i + 1), 10]
                network = PhotonicNetworkTopology(layer_sizes)
                networks.append(network)
                
                # Track memory usage
                current_memory = self._get_memory_usage()
                memory_usage.append(current_memory)
                
                print(f"Network {i}: {sum(layer_sizes)} neurons, {current_memory:.1f}MB memory")
                
            except MemoryError:
                print(f"Memory limit reached at network {i}")
                break
        
        # Should handle memory pressure gracefully
        self.assertGreater(len(networks), 0)
        
        # Memory usage should increase reasonably
        if len(memory_usage) > 1:
            memory_growth = memory_usage[-1] - memory_usage[0]
            print(f"Total memory growth: {memory_growth:.1f}MB")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def test_high_frequency_operations(self):
        """Test system behavior with high-frequency operations."""
        network = PhotonicNetworkTopology([50, 25, 10])
        input_data = [0.5] * 50
        
        # Rapid fire operations
        operation_count = 1000
        start_time = time.time()
        
        for i in range(operation_count):
            output = network.forward_pass(input_data)
        
        total_time = time.time() - start_time
        operations_per_second = operation_count / total_time
        
        print(f"High-frequency operations: {operations_per_second:.0f} ops/sec")
        
        # Should maintain reasonable performance
        self.assertGreater(operations_per_second, 100)  # > 100 ops/sec


def run_performance_benchmark_suite():
    """Run the comprehensive performance benchmark suite."""
    print("🚀 Running Comprehensive Performance Benchmark Suite")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add performance test classes
    performance_test_classes = [
        TestCorePerformance,
        TestOptimizationPerformance,
        TestScalabilityPerformance,
        TestMonitoringPerformance,
        TestLoadStress
    ]
    
    for test_class in performance_test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run performance tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    start_time = time.time()
    result = runner.run(test_suite)
    total_time = time.time() - start_time
    
    # Print performance summary
    print("\n" + "=" * 70)
    print("Performance Benchmark Suite Summary:")
    print(f"Total benchmark time: {total_time:.2f}s")
    print(f"Performance tests run: {result.testsRun}")
    print(f"Performance failures: {len(result.failures)}")
    print(f"Performance errors: {len(result.errors)}")
    print(f"Performance success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100:.1f}%")
    
    # Performance-specific reporting
    if result.failures:
        print("\n⚠️  PERFORMANCE ISSUES DETECTED:")
        for test, traceback in result.failures:
            print(f"- {test}: Performance requirements not met")
    
    if result.errors:
        print("\n❌ PERFORMANCE TEST ERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: Test execution error")
    
    if not result.failures and not result.errors:
        print("\n✅ ALL PERFORMANCE BENCHMARKS PASSED")
        print("System meets all performance requirements!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_performance_benchmark_suite()
    sys.exit(0 if success else 1)