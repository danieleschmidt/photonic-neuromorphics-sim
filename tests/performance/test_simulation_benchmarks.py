"""
Performance benchmarks for photonic neuromorphics simulation.

These tests measure performance characteristics and detect regressions.
"""

import pytest
import time
import psutil
import numpy as np
from memory_profiler import profile
from pathlib import Path

# These imports would be available when actual modules are implemented
# from photonic_neuromorphics import PhotonicSNN, PhotonicSimulator
# from photonic_neuromorphics.components import WaveguideNeuron

# Mock implementations for testing infrastructure
class MockPhotonicSNN:
    def __init__(self, topology, **kwargs):
        self.topology = topology
        self.weights = [np.random.randn(topology[i], topology[i+1]) 
                       for i in range(len(topology)-1)]
    
    def forward(self, input_data):
        # Simulate computation time
        time.sleep(0.01 * len(self.topology))
        return np.random.rand(self.topology[-1])

class MockPhotonicSimulator:
    def run(self, model, input_data, duration=1e-6):
        # Simulate optical simulation
        time.sleep(0.05)
        return np.random.poisson(0.1, size=(100, model.topology[-1]))


@pytest.fixture
def small_network():
    """Small network for quick tests."""
    return MockPhotonicSNN(topology=[10, 5, 2])

@pytest.fixture
def medium_network():
    """Medium network for standard benchmarks."""
    return MockPhotonicSNN(topology=[100, 50, 20, 5])

@pytest.fixture
def large_network():
    """Large network for stress testing."""
    return MockPhotonicSNN(topology=[1000, 500, 100, 10])

@pytest.fixture
def simulator():
    """Photonic simulator instance."""
    return MockPhotonicSimulator()


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.benchmark
    def test_small_network_inference_time(self, benchmark, small_network):
        """Benchmark inference time for small network."""
        input_data = np.random.rand(10)
        
        result = benchmark(small_network.forward, input_data)
        assert result is not None
        assert len(result) == 2
        
    @pytest.mark.benchmark  
    def test_medium_network_inference_time(self, benchmark, medium_network):
        """Benchmark inference time for medium network."""
        input_data = np.random.rand(100)
        
        result = benchmark(medium_network.forward, input_data)
        assert result is not None
        assert len(result) == 5
        
    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_large_network_inference_time(self, benchmark, large_network):
        """Benchmark inference time for large network."""
        input_data = np.random.rand(1000)
        
        result = benchmark(large_network.forward, input_data)
        assert result is not None
        assert len(result) == 10
        
    def test_simulation_memory_usage(self, medium_network, simulator):
        """Test memory usage during simulation."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        input_spikes = np.random.poisson(0.1, size=(1000, 100))
        result = simulator.run(medium_network, input_spikes)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for test)
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.2f}MB"
        assert result is not None
        
    @pytest.mark.parametrize("network_size", [10, 50, 100, 500])
    def test_scaling_performance(self, network_size, simulator):
        """Test performance scaling with network size."""
        network = MockPhotonicSNN(topology=[network_size, network_size//2, 10])
        input_data = np.random.rand(network_size)
        
        start_time = time.time()
        result = network.forward(input_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance should scale reasonably with network size
        expected_max_time = network_size * 0.001  # 1ms per neuron max
        assert execution_time < expected_max_time, \
            f"Execution time {execution_time:.4f}s exceeded expected {expected_max_time:.4f}s"
        assert result is not None
        
    @pytest.mark.slow
    def test_memory_leak_detection(self, small_network):
        """Test for memory leaks during repeated operations."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run many iterations
        for i in range(100):
            input_data = np.random.rand(10)
            result = small_network.forward(input_data)
            
            # Check memory every 20 iterations
            if i % 20 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be minimal (< 50MB after 100 iterations)
                if memory_growth > 50:
                    pytest.fail(f"Memory leak detected: {memory_growth:.2f}MB growth at iteration {i}")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        assert total_growth < 50, f"Total memory growth: {total_growth:.2f}MB"


class TestOpticalPerformance:
    """Optical simulation specific performance tests."""
    
    @pytest.mark.benchmark
    def test_waveguide_propagation_time(self, benchmark, simulator):
        """Benchmark waveguide propagation simulation."""
        # Mock waveguide parameters
        network = MockPhotonicSNN(topology=[50, 25, 10])
        input_spikes = np.random.poisson(0.1, size=(100, 50))
        
        result = benchmark(simulator.run, network, input_spikes, 1e-6)
        assert result is not None
        assert result.shape == (100, 10)
        
    @pytest.mark.parametrize("duration", [1e-9, 1e-8, 1e-7, 1e-6])
    def test_simulation_duration_scaling(self, duration, simulator):
        """Test how simulation time scales with duration."""
        network = MockPhotonicSNN(topology=[20, 10, 5])
        input_spikes = np.random.poisson(0.1, size=(50, 20))
        
        start_time = time.time()
        result = simulator.run(network, input_spikes, duration)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Execution time should not scale linearly with simulation duration
        # (this is a mock test - real implementation would have different scaling)
        assert execution_time < 1.0, f"Simulation took too long: {execution_time:.4f}s"
        assert result is not None


class TestConcurrencyPerformance:
    """Test performance under concurrent operations."""
    
    @pytest.mark.slow
    def test_parallel_simulation_performance(self, simulator):
        """Test performance when running multiple simulations in parallel."""
        import concurrent.futures
        import threading
        
        def run_simulation(network_id):
            network = MockPhotonicSNN(topology=[50, 25, 10])
            input_spikes = np.random.poisson(0.1, size=(100, 50))
            return simulator.run(network, input_spikes)
        
        start_time = time.time()
        
        # Run 4 simulations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_simulation, i) for i in range(4)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        parallel_time = end_time - start_time
        
        # All results should be valid
        assert all(result is not None for result in results)
        assert all(result.shape == (100, 10) for result in results)
        
        # Parallel execution should be faster than sequential
        # (This is a rough check - actual performance depends on implementation)
        assert parallel_time < 1.0, f"Parallel execution took too long: {parallel_time:.4f}s"


# Performance regression detection
@pytest.mark.performance_regression
class TestRegressionDetection:
    """Tests to detect performance regressions."""
    
    def test_inference_time_regression(self, medium_network):
        """Detect regressions in inference time."""
        input_data = np.random.rand(100)
        
        # Baseline performance expectation
        expected_max_time = 0.5  # 500ms max for medium network
        
        start_time = time.time()
        result = medium_network.forward(input_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert execution_time < expected_max_time, \
            f"Performance regression detected: {execution_time:.4f}s > {expected_max_time}s"
        assert result is not None
        
    def test_memory_usage_regression(self, large_network):
        """Detect regressions in memory usage."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        input_data = np.random.rand(1000)
        result = large_network.forward(input_data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should not exceed baseline
        max_expected_memory = 200  # 200MB max increase
        
        assert memory_increase < max_expected_memory, \
            f"Memory regression detected: {memory_increase:.2f}MB > {max_expected_memory}MB"
        assert result is not None


# Fixtures for performance test configuration
@pytest.fixture(scope="session")
def performance_config():
    """Configuration for performance tests."""
    return {
        "timeout": 300,  # 5 minutes max per test
        "memory_limit": 1024,  # 1GB memory limit
        "cpu_limit": 80,  # 80% CPU usage limit
    }

# Custom markers for pytest
pytestmark = [
    pytest.mark.performance,
]