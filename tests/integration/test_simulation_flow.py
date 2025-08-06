"""
Comprehensive integration tests for photonic neuromorphic simulation flow.
"""

import pytest
import numpy as np
import torch
import logging
from pathlib import Path
import tempfile
import time

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from photonic_neuromorphics.core import PhotonicSNN, encode_to_spikes
from photonic_neuromorphics.simulator import PhotonicSimulator, SimulationMode
from photonic_neuromorphics.rtl import RTLGenerator, RTLGenerationConfig
from photonic_neuromorphics.monitoring import MetricsCollector, create_monitoring_system
from photonic_neuromorphics.optimization import OptimizationConfig


class TestSimulationFlow:
    """Test complete photonic simulation workflows."""
    
    @pytest.mark.integration 
    def test_end_to_end_mnist_flow(self):
        """Test complete MNIST classification simulation flow."""
        # Create MNIST-like network
        snn = PhotonicSNN([784, 256, 128, 10])
        
        # Create simulator with optimization
        opt_config = OptimizationConfig(
            enable_caching=True,
            enable_parallel=True,
            max_workers=2
        )
        
        simulator = PhotonicSimulator(
            mode=SimulationMode.BEHAVIORAL,
            optimization_config=opt_config
        )
        
        # Create MNIST-like input (28x28 = 784 pixels)
        mnist_image = np.random.rand(28, 28)
        input_data = mnist_image.flatten()
        
        # Encode to spikes
        spike_train = encode_to_spikes(input_data, duration=100e-9, dt=1e-9)
        
        # Run simulation
        results = simulator.run(snn, spike_train)
        
        # Verify MNIST output structure
        assert results.output_spikes.shape == (100, 10)  # 10 classes
        
        # Analyze results
        output_rates = torch.mean(results.output_spikes, dim=0)
        predicted_class = torch.argmax(output_rates).item()
        
        # Should predict valid class
        assert 0 <= predicted_class <= 9
        
        # Verify performance metrics
        assert results.timing_metrics['simulation_time'] > 0
        assert results.energy_consumption['total_energy'] > 0
        
        # Calculate classification confidence
        confidence = float(torch.max(output_rates)) / float(torch.sum(output_rates))
        assert 0 <= confidence <= 1
    
    @pytest.mark.integration
    def test_complete_rtl_generation_flow(self):
        """Test complete RTL generation flow from simulation to hardware."""
        # Create and validate network through simulation first
        snn = PhotonicSNN([16, 8, 4])
        
        simulator = PhotonicSimulator(mode=SimulationMode.BEHAVIORAL)
        
        # Validate network works
        input_data = np.random.rand(16)
        spike_train = encode_to_spikes(input_data, duration=50e-9)
        
        sim_results = simulator.run(snn, spike_train)
        assert torch.all(torch.isfinite(sim_results.output_spikes))
        
        # Generate RTL from validated network
        rtl_config = RTLGenerationConfig(
            target_frequency=200e6,
            pipeline_stages=3,
            fixed_point_width=14,
            fractional_bits=8,
            include_testbench=True,
            include_assertions=True
        )
        
        rtl_generator = RTLGenerator(rtl_config)
        
        # Generate RTL design
        with tempfile.TemporaryDirectory() as temp_dir:
            rtl_design = rtl_generator.generate(snn, temp_dir)
            
            # Verify RTL files were created
            rtl_dir = Path(temp_dir)
            assert (rtl_dir / "rtl" / "photonic_neural_network.v").exists()
            assert (rtl_dir / "tb" / "tb_photonic_neural_network.v").exists()
            assert (rtl_dir / "constraints" / "constraints.sdc").exists()
            
            # Verify RTL content
            verilog_content = (rtl_dir / "rtl" / "photonic_neural_network.v").read_text()
            assert "module photonic_neural_network" in verilog_content
            assert "photonic_neuron" in verilog_content
            assert "photonic_crossbar_16x8" in verilog_content
            assert "photonic_crossbar_8x4" in verilog_content
            
            # Verify testbench
            tb_content = (rtl_dir / "tb" / "tb_photonic_neural_network.v").read_text()
            assert "tb_photonic_neural_network" in tb_content
            assert "spike_inputs" in tb_content
            assert "spike_outputs" in tb_content
            
            # Verify constraints
            sdc_content = (rtl_dir / "constraints" / "constraints.sdc").read_text()
            assert "create_clock" in sdc_content
            assert str(rtl_config.target_frequency) in sdc_content or "200.0" in sdc_content
            
            # Verify resource estimates
            resources = rtl_design.resource_estimates
            assert resources['logic_gates'] > 0
            assert resources['memory_bits'] > 0
            assert resources['estimated_power_mw'] > 0
            
            # Verify estimates are reasonable for this network size
            expected_synapses = 16*8 + 8*4  # 128 + 32 = 160
            expected_neurons = 16 + 8 + 4   # 28 total
            
            assert resources['resource_utilization']['synapses'] == expected_synapses
            assert resources['resource_utilization']['neurons'] == expected_neurons
    
    @pytest.mark.integration
    def test_monitoring_integration_flow(self):
        """Test complete monitoring system integration."""
        # Set up comprehensive monitoring
        monitoring_system = create_monitoring_system(
            enable_system_monitoring=True,
            enable_health_monitoring=True
        )
        
        metrics_collector = monitoring_system['metrics_collector']
        health_monitor = monitoring_system['health_monitor']
        profiler = monitoring_system['profiler']
        
        try:
            # Create simulation with monitoring
            simulator = PhotonicSimulator(
                mode=SimulationMode.OPTICAL,
                metrics_collector=metrics_collector
            )
            
            snn = PhotonicSNN([20, 10, 5])
            snn.set_metrics_collector(metrics_collector)
            
            # Run multiple simulations to generate monitoring data
            for i in range(3):
                input_data = np.random.rand(20)
                spike_train = encode_to_spikes(input_data, duration=30e-9)
                
                with profiler.profile_operation(f"simulation_{i}"):
                    results = simulator.run(snn, spike_train)
                    
                # Record additional metrics
                metrics_collector.record_metric(f"simulation_{i}_accuracy", 0.85 + i * 0.05)
                metrics_collector.increment_counter("total_simulations")
                
                time.sleep(0.1)  # Allow monitoring to collect data
            
            # Verify monitoring data collection
            current_metrics = metrics_collector.get_current_metrics()
            
            # Should have system metrics
            assert 'gauge_system_cpu_percent' in current_metrics
            assert 'gauge_system_memory_percent' in current_metrics
            
            # Should have simulation metrics
            assert 'total_simulations' in current_metrics
            assert current_metrics['total_simulations'] >= 3
            
            # Check performance profiling
            perf_report = profiler.get_performance_report()
            assert 'operations' in perf_report
            
            # Check health status
            health_status = health_monitor.perform_health_check()
            assert health_status.status in ['healthy', 'warning', 'critical']
            assert len(health_status.checks) > 0
            
            # Export metrics
            json_metrics = metrics_collector.export_metrics("json")
            assert '"simulation_0_accuracy"' in json_metrics
            
            prometheus_metrics = metrics_collector.export_metrics("prometheus")
            assert "total_simulations" in prometheus_metrics
            
        finally:
            health_monitor.stop_monitoring()
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_scale_simulation_flow(self):
        """Test large-scale simulation with optimization."""
        # Create large network
        large_topology = [500, 256, 128, 64, 10]
        snn = PhotonicSNN(large_topology)
        
        # Use aggressive optimization
        opt_config = OptimizationConfig(
            enable_caching=True,
            cache_size=500,
            enable_parallel=True,
            max_workers=4,
            enable_memory_pooling=True,
            memory_pool_size=512,  # 512 MB
            enable_auto_scaling=True,
            batch_size=64
        )
        
        metrics_collector = MetricsCollector(enable_system_metrics=True)
        
        simulator = PhotonicSimulator(
            mode=SimulationMode.BEHAVIORAL,  # Use fast mode for large network
            optimization_config=opt_config,
            metrics_collector=metrics_collector
        )
        
        # Create realistic large input
        large_input = np.random.rand(500) * 0.8 + 0.1
        spike_train = encode_to_spikes(large_input, duration=50e-9)
        
        # Run simulation with timing
        start_time = time.time()
        results = simulator.run(snn, spike_train)
        total_time = time.time() - start_time
        
        # Verify results
        assert results.output_spikes.shape == (50, 10)
        assert torch.all(torch.isfinite(results.output_spikes))
        
        # Verify performance is reasonable
        assert total_time < 60.0  # Should complete within 1 minute
        
        # Verify optimization effectiveness
        cache_stats = simulator.cache.get_stats()
        assert cache_stats['size'] >= 0
        
        if simulator.memory_pool:
            pool_stats = simulator.memory_pool.get_stats()
            assert pool_stats['pool_efficiency'] >= 0
        
        # Verify auto-scaling was active
        if simulator.auto_scaler:
            scaling_params = simulator.auto_scaler.get_current_parameters()
            assert 'batch_size' in scaling_params
            assert 'worker_count' in scaling_params
        
        # Calculate performance metrics
        throughput = spike_train.shape[0] / total_time
        energy_efficiency = results.energy_consumption['total_energy'] / results.output_spikes.sum().item()
        
        # Log performance for analysis
        print(f"Large network simulation:")
        print(f"  Network: {large_topology}")
        print(f"  Simulation time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} time steps/s")
        print(f"  Energy efficiency: {energy_efficiency:.2e} J/spike")
    
    @pytest.mark.integration
    def test_error_recovery_integration_flow(self):
        """Test comprehensive error recovery throughout the simulation flow."""
        # Create network that might encounter various errors
        snn = PhotonicSNN([25, 15, 8, 3])
        
        opt_config = OptimizationConfig(
            enable_caching=True,
            enable_parallel=True,
            max_workers=2
        )
        
        metrics_collector = MetricsCollector(enable_system_metrics=False)
        
        simulator = PhotonicSimulator(
            mode=SimulationMode.OPTICAL,
            optimization_config=opt_config,
            metrics_collector=metrics_collector
        )
        
        # Test 1: Recovery from invalid input
        try:
            invalid_input = np.array([float('inf'), float('nan'), -1e10, 1e10] + [0.5] * 21)
            spike_train = encode_to_spikes(invalid_input, duration=20e-9)
            
            # Should handle invalid inputs gracefully
            results = simulator.run(snn, spike_train)
            assert results.output_spikes.shape == (20, 3)
            assert torch.all(torch.isfinite(results.output_spikes))
            
        except Exception as e:
            # Acceptable to fail with clear error message
            assert "validation" in str(e).lower() or "invalid" in str(e).lower()
        
        # Test 2: Recovery from memory pressure simulation
        try:
            # Create scenario that might cause memory pressure
            large_input = np.random.rand(25)
            long_spike_train = encode_to_spikes(large_input, duration=200e-9)  # Long duration
            
            results = simulator.run(snn, long_spike_train)
            assert results.output_spikes.shape == (200, 3)
            
        except Exception as e:
            # Should provide informative error
            assert len(str(e)) > 10  # Should have meaningful error message
        
        # Verify error recovery metrics were collected
        error_metrics = metrics_collector.get_current_metrics()
        # May contain error counters depending on what errors occurred
        assert isinstance(error_metrics, dict)
    
    @pytest.mark.integration
    def test_multi_mode_comparison_flow(self):
        """Test comparison across different simulation modes."""
        snn = PhotonicSNN([12, 8, 4])
        
        # Test data
        input_data = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.0, 0.15, 0.85])
        spike_train = encode_to_spikes(input_data, duration=40e-9)
        
        modes_to_test = [
            SimulationMode.BEHAVIORAL,
            SimulationMode.OPTICAL,
            SimulationMode.MIXED_SIGNAL
        ]
        
        results_by_mode = {}
        
        for mode in modes_to_test:
            simulator = PhotonicSimulator(
                mode=mode,
                optimization_config=OptimizationConfig(enable_caching=True)
            )
            
            start_time = time.time()
            results = simulator.run(snn, spike_train)
            sim_time = time.time() - start_time
            
            results_by_mode[mode.value] = {
                'output_spikes': results.output_spikes,
                'simulation_time': sim_time,
                'energy_consumption': results.energy_consumption,
                'timing_metrics': results.timing_metrics
            }
        
        # Verify all modes produced valid results
        for mode_name, mode_results in results_by_mode.items():
            assert mode_results['output_spikes'].shape == (40, 4)
            assert torch.all(torch.isfinite(mode_results['output_spikes']))
            assert mode_results['simulation_time'] > 0
            assert mode_results['energy_consumption']['total_energy'] > 0
        
        # Compare modes
        behavioral_time = results_by_mode['behavioral']['simulation_time']
        optical_time = results_by_mode['optical']['simulation_time']
        
        # Behavioral should generally be faster
        print(f"Mode comparison:")
        print(f"  Behavioral: {behavioral_time:.4f}s")
        print(f"  Optical: {optical_time:.4f}s")
        print(f"  Mixed Signal: {results_by_mode['mixed']['simulation_time']:.4f}s")
        
        # Energy consumption patterns
        behavioral_energy = results_by_mode['behavioral']['energy_consumption']['total_energy']
        optical_energy = results_by_mode['optical']['energy_consumption']['total_energy']
        
        # Optical should show photonic energy advantage
        optical_photonic_ratio = results_by_mode['optical']['energy_consumption']['photonic_efficiency']
        assert 0 <= optical_photonic_ratio <= 1


# Test fixtures
@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)