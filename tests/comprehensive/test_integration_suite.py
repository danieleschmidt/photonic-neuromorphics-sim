"""
Comprehensive Integration Test Suite for Photonic Neuromorphic Systems

Complete integration testing covering all modules, performance validation,
security testing, and end-to-end workflow validation.
"""

import unittest
import time
import sys
import os
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import modules to test
try:
    from photonic_neuromorphics.enhanced_core import (
        EnhancedPhotonicNeuron, PhotonicNetworkTopology, PhotonicResearchBenchmark,
        PhotonicActivationFunction, PhotonicParameters
    )
    from photonic_neuromorphics.quantum_photonic_interface import (
        QuantumPhotonicNetwork, QuantumPhotonicResearchSuite, QuantumPhotonicMode
    )
    from photonic_neuromorphics.ml_assisted_optimization import (
        MLAssistedOptimizationSuite, OptimizationObjective, PhotonicDesignParameters
    )
    from photonic_neuromorphics.robust_validation_system import (
        PhotonicValidationFramework, ValidationLevel, ValidationCategory
    )
    from photonic_neuromorphics.production_monitoring import (
        PhotonicSystemMonitor, MetricType, AlertSeverity
    )
    from photonic_neuromorphics.enterprise_reliability import (
        EnterpriseReliabilityFramework, SystemState, ComponentHealth
    )
    from photonic_neuromorphics.high_performance_optimization import (
        HighPerformanceOptimizer, CacheStrategy, ProcessingMode
    )
    from photonic_neuromorphics.scalability_framework import (
        DistributedTaskScheduler, LoadBalancingStrategy, ScalingMode
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Some modules not available for testing: {e}")
    MODULES_AVAILABLE = False


class TestPhotonicCoreIntegration(unittest.TestCase):
    """Test integration of core photonic neuromorphic functionality."""
    
    def setUp(self):
        """Set up test environment."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.test_data = {
            'layer_sizes': [100, 50, 10],
            'wavelength': 1550e-9,
            'power': 1e-3,
            'efficiency': 0.8
        }
    
    def test_enhanced_neuron_creation(self):
        """Test enhanced photonic neuron creation and functionality."""
        neuron = EnhancedPhotonicNeuron(
            activation=PhotonicActivationFunction.MACH_ZEHNDER,
            params=PhotonicParameters()
        )
        
        self.assertIsNotNone(neuron)
        self.assertEqual(neuron.activation, PhotonicActivationFunction.MACH_ZEHNDER)
        self.assertGreaterEqual(neuron.spike_count, 0)
    
    def test_network_topology_creation(self):
        """Test photonic network topology creation."""
        network = PhotonicNetworkTopology(
            layer_sizes=self.test_data['layer_sizes'],
            wavelength_multiplexing=True
        )
        
        self.assertIsNotNone(network)
        self.assertEqual(len(network.neurons), len(self.test_data['layer_sizes']))
        self.assertTrue(network.wavelength_multiplexing)
    
    def test_network_forward_pass(self):
        """Test network forward pass functionality."""
        network = PhotonicNetworkTopology(layer_sizes=[10, 5, 2])
        input_data = [0.5] * 10
        
        output = network.forward_pass(input_data)
        
        self.assertIsNotNone(output)
        self.assertEqual(len(output), 2)  # Output layer size
        self.assertIsInstance(output, list)
    
    def test_research_benchmark(self):
        """Test research benchmarking functionality."""
        benchmark = PhotonicResearchBenchmark()
        network = PhotonicNetworkTopology([50, 25, 10])
        
        results = benchmark.run_mnist_benchmark(network, test_samples=10)
        
        self.assertIsNotNone(results)
        self.assertIn('accuracy', results)
        self.assertIn('avg_energy_per_inference', results)
        self.assertGreaterEqual(results['accuracy'], 0.0)
        self.assertLessEqual(results['accuracy'], 1.0)


class TestQuantumPhotonicIntegration(unittest.TestCase):
    """Test quantum-photonic interface integration."""
    
    def setUp(self):
        """Set up quantum test environment."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
    
    def test_quantum_network_creation(self):
        """Test quantum photonic network creation."""
        layer_sizes = [20, 10, 5]
        quantum_modes = [
            QuantumPhotonicMode.QUANTUM_INTERFERENCE,
            QuantumPhotonicMode.ENTANGLED_PHOTONS,
            QuantumPhotonicMode.SQUEEZED_LIGHT
        ]
        
        network = QuantumPhotonicNetwork(layer_sizes, quantum_modes)
        
        self.assertIsNotNone(network)
        self.assertEqual(len(network.quantum_neurons), len(layer_sizes))
    
    def test_quantum_forward_pass(self):
        """Test quantum-enhanced forward pass."""
        network = QuantumPhotonicNetwork([10, 5])
        input_data = [0.1 * i for i in range(10)]
        
        output = network.quantum_forward_pass(input_data)
        
        self.assertIsNotNone(output)
        self.assertEqual(len(output), 5)
        self.assertIsInstance(output, list)
    
    def test_quantum_research_suite(self):
        """Test quantum research suite functionality."""
        research_suite = QuantumPhotonicResearchSuite()
        
        results = research_suite.run_quantum_advantage_experiment([20, 10, 5])
        
        self.assertIsNotNone(results)
        self.assertIn('quantum_advantage_demonstrated', results)
        self.assertIn('advantage_ratio', results)


class TestMLOptimizationIntegration(unittest.TestCase):
    """Test ML-assisted optimization integration."""
    
    def setUp(self):
        """Set up ML optimization test environment."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
    
    def test_optimization_suite_creation(self):
        """Test ML optimization suite creation."""
        suite = MLAssistedOptimizationSuite()
        
        self.assertIsNotNone(suite)
        self.assertIsInstance(suite.optimization_results, dict)
    
    def test_design_parameters_validation(self):
        """Test photonic design parameters validation."""
        params = PhotonicDesignParameters(
            layer_sizes=[100, 50, 10],
            wavelengths=[1550e-9, 1551e-9],
            power_levels=[1e-3, 0.8e-3],
            coupling_efficiencies=[0.9, 0.85]
        )
        
        self.assertTrue(params.validate())
        self.assertEqual(len(params.wavelengths), len(params.power_levels))


class TestValidationSystemIntegration(unittest.TestCase):
    """Test robust validation system integration."""
    
    def setUp(self):
        """Set up validation test environment."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.validator = PhotonicValidationFramework(ValidationLevel.NORMAL)
    
    def test_validation_framework_creation(self):
        """Test validation framework creation."""
        self.assertIsNotNone(self.validator)
        self.assertEqual(self.validator.validation_level, ValidationLevel.NORMAL)
    
    def test_data_validation(self):
        """Test data validation functionality."""
        test_data = 1550e-9  # Valid wavelength
        
        result = self.validator.validate_data(test_data, [ValidationCategory.INPUT_VALIDATION])
        
        self.assertTrue(result.passed)
        self.assertEqual(len(result.errors), 0)
    
    def test_invalid_data_validation(self):
        """Test validation of invalid data."""
        test_data = -1  # Invalid wavelength
        
        result = self.validator.validate_data(test_data, [ValidationCategory.INPUT_VALIDATION])
        
        # Should either fail or auto-fix
        self.assertTrue(not result.passed or result.validated_data != test_data)
    
    def test_photonic_parameters_validation(self):
        """Test photonic parameters validation."""
        params = {
            'wavelength': 1550e-9,
            'power': 1e-3,
            'efficiency': 0.8,
            'temperature': 298.15
        }
        
        result = self.validator.validate_photonic_parameters(params)
        
        self.assertTrue(result.passed)


class TestMonitoringIntegration(unittest.TestCase):
    """Test production monitoring integration."""
    
    def setUp(self):
        """Set up monitoring test environment."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.monitor = PhotonicSystemMonitor(collection_interval=0.1)
    
    def tearDown(self):
        """Clean up monitoring resources."""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring()
    
    def test_monitor_creation(self):
        """Test monitoring system creation."""
        self.assertIsNotNone(self.monitor)
        self.assertEqual(self.monitor.collection_interval, 0.1)
    
    def test_metric_recording(self):
        """Test metric recording functionality."""
        self.monitor.record_metric("test.metric", 42.0, metric_type=MetricType.GAUGE)
        
        # Allow some time for processing
        time.sleep(0.2)
        
        latest_value = self.monitor.metrics_collector.get_latest_value("test.metric")
        self.assertEqual(latest_value, 42.0)
    
    def test_monitoring_start_stop(self):
        """Test monitoring start and stop functionality."""
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.monitoring_active)
        
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.monitoring_active)
    
    def test_dashboard_generation(self):
        """Test dashboard generation."""
        self.monitor.start_monitoring()
        time.sleep(0.2)
        
        dashboard = self.monitor.get_metric_dashboard()
        
        self.assertIsNotNone(dashboard)
        self.assertIn('timestamp', dashboard)
        self.assertIn('system_health', dashboard)


class TestReliabilityIntegration(unittest.TestCase):
    """Test enterprise reliability integration."""
    
    def setUp(self):
        """Set up reliability test environment."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.reliability = EnterpriseReliabilityFramework()
    
    def tearDown(self):
        """Clean up reliability resources."""
        if hasattr(self, 'reliability'):
            self.reliability.stop_reliability_monitoring()
    
    def test_reliability_framework_creation(self):
        """Test reliability framework creation."""
        self.assertIsNotNone(self.reliability)
        self.assertEqual(self.reliability.system_state, SystemState.HEALTHY)
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation and functionality."""
        cb = self.reliability.add_circuit_breaker("test_cb")
        
        self.assertIsNotNone(cb)
        self.assertEqual(cb.name, "test_cb")
    
    def test_reliable_operation_context(self):
        """Test reliable operation context manager."""
        def test_operation():
            return "success"
        
        with self.reliability.reliable_operation("test_op"):
            result = test_operation()
        
        self.assertEqual(result, "success")
    
    def test_reliability_dashboard(self):
        """Test reliability dashboard generation."""
        dashboard = self.reliability.get_reliability_dashboard()
        
        self.assertIsNotNone(dashboard)
        self.assertIn('system_state', dashboard)
        self.assertIn('reliability_metrics', dashboard)


class TestPerformanceIntegration(unittest.TestCase):
    """Test high-performance optimization integration."""
    
    def setUp(self):
        """Set up performance test environment."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        
        # Create a simpler optimizer to avoid pickling issues
        self.optimizer = HighPerformanceOptimizer()
        # Disable parallel processing for tests to avoid pickling issues
        self.optimizer.parallel_processor.mode = ProcessingMode.SEQUENTIAL
    
    def test_optimizer_creation(self):
        """Test performance optimizer creation."""
        self.assertIsNotNone(self.optimizer)
        self.assertTrue(self.optimizer.optimization_enabled)
    
    def test_cache_functionality(self):
        """Test caching functionality."""
        # Test cache put and get
        self.optimizer.cache.put("test_key", "test_value")
        cached_value = self.optimizer.cache.get("test_key")
        
        self.assertEqual(cached_value, "test_value")
    
    def test_batch_processing(self):
        """Test batch processing functionality."""
        test_data = list(range(20))
        
        def simple_operation(x):
            return x * 2
        
        results = self.optimizer.optimize_batch_processing(test_data, simple_operation, batch_size=5)
        
        self.assertEqual(len(results), len(test_data))
        self.assertEqual(results[0], 0)
        self.assertEqual(results[1], 2)
    
    def test_performance_dashboard(self):
        """Test performance dashboard generation."""
        dashboard = self.optimizer.get_performance_dashboard()
        
        self.assertIsNotNone(dashboard)
        self.assertIn('cache_performance', dashboard)
        self.assertIn('system_info', dashboard)


class TestScalabilityIntegration(unittest.TestCase):
    """Test scalability framework integration."""
    
    def setUp(self):
        """Set up scalability test environment."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
    
    def test_scheduler_creation(self):
        """Test distributed scheduler creation."""
        scheduler = DistributedTaskScheduler()
        
        self.assertIsNotNone(scheduler)
        self.assertFalse(scheduler.scheduler_active)
    
    def test_load_balancer_configuration(self):
        """Test load balancer configuration."""
        scheduler = DistributedTaskScheduler()
        
        # Test different load balancing strategies
        strategies = [
            LoadBalancingStrategy.ROUND_ROBIN,
            LoadBalancingStrategy.LEAST_CONNECTIONS,
            LoadBalancingStrategy.PERFORMANCE_BASED
        ]
        
        for strategy in strategies:
            scheduler.load_balancer.strategy = strategy
            self.assertEqual(scheduler.load_balancer.strategy, strategy)


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflow integration."""
    
    def setUp(self):
        """Set up end-to-end test environment."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
    
    def test_complete_photonic_simulation_workflow(self):
        """Test complete photonic simulation workflow."""
        # 1. Create and validate photonic network
        validator = PhotonicValidationFramework()
        
        network_params = {
            'layer_sizes': [50, 25, 10],
            'wavelength': 1550e-9,
            'power': 1e-3,
            'efficiency': 0.8
        }
        
        validation_result = validator.validate_photonic_parameters(network_params)
        self.assertTrue(validation_result.passed)
        
        # 2. Create photonic network
        layer_sizes = validation_result.validated_data['layer_sizes']
        network = PhotonicNetworkTopology(layer_sizes)
        
        # 3. Run simulation
        input_data = [0.5] * layer_sizes[0]
        output = network.forward_pass(input_data)
        
        self.assertIsNotNone(output)
        self.assertEqual(len(output), layer_sizes[-1])
        
        # 4. Benchmark performance
        benchmark = PhotonicResearchBenchmark()
        results = benchmark.run_mnist_benchmark(network, test_samples=5)
        
        self.assertIn('accuracy', results)
        self.assertGreaterEqual(results['accuracy'], 0.0)
    
    def test_quantum_enhanced_workflow(self):
        """Test quantum-enhanced photonic workflow."""
        # 1. Create quantum photonic network
        layer_sizes = [20, 10, 5]
        quantum_modes = [QuantumPhotonicMode.QUANTUM_INTERFERENCE] * len(layer_sizes)
        
        quantum_network = QuantumPhotonicNetwork(layer_sizes, quantum_modes)
        
        # 2. Run quantum simulation
        input_data = [0.1 * i for i in range(layer_sizes[0])]
        quantum_output = quantum_network.quantum_forward_pass(input_data)
        
        self.assertIsNotNone(quantum_output)
        self.assertEqual(len(quantum_output), layer_sizes[-1])
        
        # 3. Compare with classical network
        classical_network = PhotonicNetworkTopology(layer_sizes)
        classical_output = classical_network.forward_pass(input_data)
        
        # Should have same output dimensions
        self.assertEqual(len(quantum_output), len(classical_output))
    
    def test_optimized_production_workflow(self):
        """Test optimized production workflow with monitoring."""
        # 1. Create optimized system
        optimizer = HighPerformanceOptimizer()
        optimizer.parallel_processor.mode = ProcessingMode.SEQUENTIAL  # Avoid pickling issues
        
        monitor = PhotonicSystemMonitor(collection_interval=0.1)
        monitor.start_monitoring()
        
        try:
            # 2. Run optimized operations
            test_data = list(range(50))
            
            def processing_operation(x):
                monitor.record_metric("processing.input", x, metric_type=MetricType.GAUGE)
                result = x * x
                monitor.record_metric("processing.output", result, metric_type=MetricType.GAUGE)
                return result
            
            results = optimizer.optimize_batch_processing(test_data, processing_operation)
            
            # 3. Validate results
            self.assertEqual(len(results), len(test_data))
            self.assertEqual(results[5], 25)  # 5^2 = 25
            
            # 4. Check monitoring
            time.sleep(0.2)  # Allow metrics to be processed
            dashboard = monitor.get_metric_dashboard()
            
            self.assertIn('system_health', dashboard)
            
        finally:
            monitor.stop_monitoring()


class TestSecurityValidation(unittest.TestCase):
    """Test security aspects of the system."""
    
    def setUp(self):
        """Set up security test environment."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.validator = PhotonicValidationFramework()
    
    def test_input_sanitization(self):
        """Test input sanitization against malicious data."""
        malicious_inputs = [
            "exec('import os; os.system(\"rm -rf /\")')",
            "__import__('os').system('echo pwned')",
            "eval('1+1')",
            "' OR 1=1 --",
            "<script>alert('xss')</script>"
        ]
        
        for malicious_input in malicious_inputs:
            result = self.validator.validate_data(malicious_input, [ValidationCategory.SECURITY_CHECKS])
            
            # Should either fail validation or sanitize the input
            self.assertTrue(not result.passed or result.validated_data != malicious_input)
    
    def test_data_size_limits(self):
        """Test data size validation for DoS protection."""
        large_data = "x" * 20000  # 20KB data
        
        result = self.validator.validate_data(large_data, [ValidationCategory.SECURITY_CHECKS])
        
        # Should either fail or truncate the data
        self.assertTrue(not result.passed or len(str(result.validated_data)) < len(large_data))
    
    def test_parameter_bounds_enforcement(self):
        """Test that parameter bounds are enforced."""
        invalid_params = {
            'wavelength': -1,  # Negative wavelength
            'power': 100,      # Excessive power
            'efficiency': 2.0, # > 100% efficiency
            'temperature': 1000 # Extreme temperature
        }
        
        result = self.validator.validate_photonic_parameters(invalid_params)
        
        # Should auto-fix or report errors
        if result.passed:
            # Parameters should be fixed to valid ranges
            validated = result.validated_data
            self.assertGreater(validated['wavelength'], 0)
            self.assertLessEqual(validated['power'], 1.0)
            self.assertLessEqual(validated['efficiency'], 1.0)
            self.assertLess(validated['temperature'], 500)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test performance benchmarks and requirements."""
    
    def setUp(self):
        """Set up performance test environment."""
        if not MODULES_AVAILABLE:
            self.skipTest("Required modules not available")
    
    def test_network_creation_performance(self):
        """Test network creation performance."""
        layer_sizes = [1000, 500, 100]
        
        start_time = time.time()
        network = PhotonicNetworkTopology(layer_sizes)
        creation_time = time.time() - start_time
        
        # Should create network in reasonable time
        self.assertLess(creation_time, 5.0, "Network creation took too long")
        self.assertEqual(len(network.neurons), len(layer_sizes))
    
    def test_forward_pass_performance(self):
        """Test forward pass performance."""
        network = PhotonicNetworkTopology([100, 50, 10])
        input_data = [0.5] * 100
        
        # Warm up
        network.forward_pass(input_data)
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            output = network.forward_pass(input_data)
        execution_time = time.time() - start_time
        
        avg_time = execution_time / 10
        
        # Should complete forward pass quickly
        self.assertLess(avg_time, 1.0, "Forward pass too slow")
        self.assertEqual(len(output), 10)
    
    def test_validation_performance(self):
        """Test validation system performance."""
        validator = PhotonicValidationFramework()
        
        test_params = {
            'wavelength': 1550e-9,
            'power': 1e-3,
            'efficiency': 0.8
        }
        
        # Benchmark validation
        start_time = time.time()
        for _ in range(100):
            result = validator.validate_photonic_parameters(test_params)
        validation_time = time.time() - start_time
        
        avg_time = validation_time / 100
        
        # Validation should be fast
        self.assertLess(avg_time, 0.01, "Validation too slow")


def run_comprehensive_test_suite():
    """Run the comprehensive test suite."""
    print("🧪 Running Comprehensive Integration Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestPhotonicCoreIntegration,
        TestQuantumPhotonicIntegration,
        TestMLOptimizationIntegration,
        TestValidationSystemIntegration,
        TestMonitoringIntegration,
        TestReliabilityIntegration,
        TestPerformanceIntegration,
        TestScalabilityIntegration,
        TestEndToEndWorkflow,
        TestSecurityValidation,
        TestPerformanceBenchmarks
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Suite Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100:.1f}%")
    
    # Print detailed failure information
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.splitlines()[-1] if traceback.splitlines() else 'Unknown failure'}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.splitlines()[-1] if traceback.splitlines() else 'Unknown error'}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_test_suite()
    sys.exit(0 if success else 1)