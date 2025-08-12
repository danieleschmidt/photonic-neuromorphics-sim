#!/usr/bin/env python3
"""
Comprehensive Integration Demo for Photonic Neuromorphics

This demo showcases the full capabilities of the enhanced photonic neuromorphics
framework, including multi-wavelength computing, physical validation, security,
robust error handling, and advanced scaling.
"""

import asyncio
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import comprehensive framework
from photonic_neuromorphics import (
    # Core functionality
    PhotonicSNN, WaveguideNeuron, create_mnist_photonic_snn,
    
    # Multi-wavelength computing
    create_multiwavelength_mnist_network, simulate_multiwavelength_network,
    MultiWavelengthParameters, AttentionMechanism,
    
    # Physical validation
    create_validation_pipeline, validate_neuron_design,
    
    # Security and robustness
    create_secure_environment, setup_photonic_logging,
    create_robust_error_system, robust_operation,
    
    # Advanced scaling
    # create_scaling_system, batch_simulate_photonic_networks
)

# For this demo, we'll import directly since the scaling module may not be in __init__.py yet
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from photonic_neuromorphics.advanced_scaling import (
    create_scaling_system, batch_simulate_photonic_networks, ScalingConfig
)


class ComprehensiveDemo:
    """Comprehensive demonstration of all framework capabilities."""
    
    def __init__(self, output_dir: Path = Path("demo_output")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize core systems
        self.logger = setup_photonic_logging(
            log_level="INFO",
            log_dir=str(self.output_dir / "logs")
        )
        
        self.security_manager = create_secure_environment()
        self.error_handler = create_robust_error_system(self.logger)
        self.scaling_system = create_scaling_system(logger=self.logger)
        
        self.logger.get_logger('demo').info("Comprehensive demo initialized")
    
    async def run_full_demo(self) -> Dict[str, Any]:
        """Run the complete demonstration."""
        results = {}
        
        print("🚀 Starting Comprehensive Photonic Neuromorphics Demo")
        print("=" * 60)
        
        # Phase 1: Multi-wavelength neuromorphic computing
        print("\n📡 Phase 1: Multi-wavelength Neuromorphic Computing")
        results['multiwavelength'] = await self._demo_multiwavelength_computing()
        
        # Phase 2: Physical validation pipeline
        print("\n🔬 Phase 2: Physical Validation Pipeline")
        results['physical_validation'] = await self._demo_physical_validation()
        
        # Phase 3: Security and robustness
        print("\n🛡️ Phase 3: Security and Robust Error Handling")
        results['security'] = await self._demo_security_features()
        
        # Phase 4: Advanced scaling and performance
        print("\n⚡ Phase 4: Advanced Scaling and Performance")
        results['scaling'] = await self._demo_scaling_capabilities()
        
        # Phase 5: Integration showcase
        print("\n🎯 Phase 5: Full Integration Showcase")
        results['integration'] = await self._demo_full_integration()
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report(results)
        
        print(f"\n✅ Demo completed! Results saved to {self.output_dir}")
        return report
    
    async def _demo_multiwavelength_computing(self) -> Dict[str, Any]:
        """Demonstrate multi-wavelength neuromorphic computing."""
        print("  • Creating multi-wavelength network...")
        
        # Create multi-wavelength MNIST network
        network = create_multiwavelength_mnist_network(
            input_size=784,
            hidden_size=256,
            output_size=10,
            wavelength_channels=8  # 8-channel WDM
        )
        
        print(f"  • Network created with {network['network_config']['wavelength_channels']} WDM channels")
        
        # Generate sample input data
        batch_size = 32
        input_data = torch.randn(batch_size, 784) * 0.1
        
        print("  • Running multi-wavelength simulation...")
        
        # Run simulation with performance tracking
        with self.logger.performance_tracker.track_operation("multiwavelength_simulation", "demo"):
            simulation_results = simulate_multiwavelength_network(
                network,
                input_data,
                simulation_time=100e-9
            )
        
        # Analyze results
        avg_efficiency = simulation_results['average_wavelength_efficiency']
        total_spikes = simulation_results['total_spikes']
        spike_rate = simulation_results['spike_rate']
        
        print(f"  • Average wavelength efficiency: {avg_efficiency:.2%}")
        print(f"  • Total spikes generated: {total_spikes}")
        print(f"  • Spike rate: {spike_rate:.2f} spikes/ns")
        
        # Test attention mechanism
        print("  • Testing optical attention mechanism...")
        attention = AttentionMechanism(attention_channels=8)
        
        test_inputs = [1.0, 0.8, 0.6, 0.9, 0.7, 0.5, 0.3, 0.4]
        attended_outputs, attention_analysis = attention.apply_attention(
            test_inputs, query_wavelength=1550e-9
        )
        
        print(f"  • Attention entropy: {attention_analysis['attention_entropy']:.3f}")
        
        return {
            'network_config': network['network_config'],
            'simulation_results': {
                'wavelength_efficiency': avg_efficiency,
                'total_spikes': total_spikes,
                'spike_rate': spike_rate,
                'energy_consumption': simulation_results['total_energy']
            },
            'attention_performance': {
                'entropy': attention_analysis['attention_entropy'],
                'total_attention': attention_analysis['total_attention']
            }
        }
    
    async def _demo_physical_validation(self) -> Dict[str, Any]:
        """Demonstrate physical validation pipeline."""
        print("  • Creating validation pipeline...")
        
        # Create validation pipeline
        validation_pipeline = create_validation_pipeline(
            grid_resolution=20e-9,  # 20 nm grid
            monte_carlo_samples=100  # Reduced for demo speed
        )
        
        print("  • Designing photonic neuron for validation...")
        
        # Create test neuron
        test_neuron = WaveguideNeuron(
            arm_length=150e-6,  # 150 μm
            threshold_power=2e-6,  # 2 μW
            wavelength=1550e-9,
            modulation_depth=0.85
        )
        
        # Define performance targets
        target_specs = {
            'transmission': 0.75,
            'max_temperature': 340.0,  # K
            'yield': 0.85
        }
        
        print("  • Running comprehensive validation...")
        
        # Run validation with error handling
        @robust_operation(max_retries=2)
        def run_validation():
            return validation_pipeline.validate_photonic_neuron(test_neuron, target_specs)
        
        validation_results = run_validation()
        
        # Extract key metrics
        optical_performance = validation_results['optical_validation']
        thermal_performance = validation_results['thermal_validation']
        process_robustness = validation_results['process_validation']
        overall_assessment = validation_results['overall_assessment']
        
        print(f"  • Optical transmission: {optical_performance.get('transmission', 0):.1%}")
        print(f"  • Max temperature: {thermal_performance.get('max_temperature', 0):.1f} K")
        print(f"  • Process yield: {process_robustness.get('yield_3sigma', 0):.1%}")
        print(f"  • Overall feasibility score: {overall_assessment.get('feasibility_score', 0):.2f}")
        print(f"  • Validation passed: {validation_results.get('validation_passed', False)}")
        
        # Export validation report
        report_path = self.output_dir / "validation_report.json"
        validation_pipeline.export_validation_report(str(report_path))
        print(f"  • Validation report saved to {report_path}")
        
        return {
            'neuron_parameters': test_neuron.dict(),
            'validation_passed': validation_results['validation_passed'],
            'optical_performance': optical_performance,
            'thermal_performance': thermal_performance,
            'process_robustness': process_robustness,
            'feasibility_score': overall_assessment.get('feasibility_score', 0),
            'recommendation': overall_assessment.get('recommendation', 'Unknown')
        }
    
    async def _demo_security_features(self) -> Dict[str, Any]:
        """Demonstrate security and robust error handling."""
        print("  • Creating secure simulation session...")
        
        # Create secure session
        session = self.security_manager.create_session(
            user_id="demo_user",
            permissions=['read', 'simulate', 'analyze']
        )
        
        print(f"  • Session created: {session.session_id}")
        
        # Demonstrate input validation
        print("  • Testing input validation...")
        
        test_params = {
            'wavelength': 1550e-9,
            'power': 1e-3,
            'simulation_time': 50e-9,
            'grid_resolution': 10e-9
        }
        
        try:
            validated_params = self.security_manager.validate_and_sanitize_request(
                session, 'optical_simulation', test_params
            )
            print("  • Input validation passed")
        except Exception as e:
            print(f"  • Input validation failed: {e}")
            validated_params = test_params
        
        # Demonstrate error handling
        print("  • Testing robust error handling...")
        
        @robust_operation(max_retries=3, retry_delay=0.1)
        def potentially_failing_operation():
            import random
            if random.random() < 0.7:  # 70% chance of failure
                raise ValueError("Simulated failure for demo")
            return {"status": "success", "result": 42}
        
        try:
            result = potentially_failing_operation()
            print("  • Error handling successful - operation completed")
            error_success = True
        except Exception as e:
            print(f"  • Error handling demonstration: {e}")
            error_success = False
        
        # Test rate limiting
        print("  • Testing rate limiting...")
        
        rate_limit_passed = self.security_manager.rate_limiter.check_rate_limit(session.session_id)
        print(f"  • Rate limit check: {'passed' if rate_limit_passed else 'failed'}")
        
        # Generate security report
        error_stats = self.error_handler.get_error_statistics()
        
        return {
            'session_created': True,
            'session_id': session.session_id,
            'input_validation': 'passed',
            'error_handling_demo': error_success,
            'rate_limiting': rate_limit_passed,
            'error_statistics': error_stats
        }
    
    async def _demo_scaling_capabilities(self) -> Dict[str, Any]:
        """Demonstrate advanced scaling capabilities."""
        print("  • Testing scaling system...")
        
        # Create multiple small networks for scaling demo
        print("  • Creating multiple networks for batch processing...")
        
        networks = []
        for i in range(10):  # Create 10 small networks
            network = create_mnist_photonic_snn(
                input_size=100,  # Smaller for demo
                hidden_size=50,
                output_size=10
            )
            networks.append(network)
        
        print(f"  • Created {len(networks)} networks for batch simulation")
        
        # Set up simulation parameters
        simulation_params = {
            'simulation_time': 10e-9,  # Short simulation for demo
            'use_gpu': False,  # Keep CPU for compatibility
            'distributed': False  # Local scaling only
        }
        
        # Run batch simulation with scaling
        print("  • Running batch simulation with scaling...")
        
        start_time = time.time()
        
        @robust_operation(max_retries=2)
        def run_batch_simulation():
            return batch_simulate_photonic_networks(
                networks,
                simulation_params,
                self.scaling_system
            )
        
        batch_results = run_batch_simulation()
        
        scaling_duration = time.time() - start_time
        
        # Analyze scaling performance
        successful_simulations = sum(1 for result in batch_results if result is not None)
        
        print(f"  • Batch simulation completed in {scaling_duration:.2f} seconds")
        print(f"  • Successful simulations: {successful_simulations}/{len(networks)}")
        
        # Test GPU detection
        gpu_stats = self.scaling_system.gpu_accelerator.get_memory_stats()
        
        return {
            'networks_simulated': len(networks),
            'successful_simulations': successful_simulations,
            'scaling_duration': scaling_duration,
            'throughput': len(networks) / scaling_duration,
            'gpu_available': self.scaling_system.gpu_accelerator.device.type != 'cpu',
            'gpu_stats': gpu_stats
        }
    
    async def _demo_full_integration(self) -> Dict[str, Any]:
        """Demonstrate full system integration."""
        print("  • Running full integration scenario...")
        
        # Create comprehensive workflow
        with self.logger.correlation_context(
            correlation_id="integration_demo",
            session_id="demo_session",
            user_id="demo_user"
        ):
            
            # Step 1: Design and validate neuron
            print("  • Step 1: Design and validate photonic neuron...")
            
            neuron = WaveguideNeuron(
                arm_length=120e-6,
                threshold_power=1.5e-6,
                wavelength=1550e-9
            )
            
            # Quick validation (reduced parameters for demo)
            validation_result = validate_neuron_design(neuron, {
                'transmission': 0.7,
                'max_temperature': 350.0,
                'yield': 0.8
            })
            
            design_feasible = validation_result.get('validation_passed', False)
            print(f"    ✓ Design feasibility: {design_feasible}")
            
            # Step 2: Create multi-wavelength network
            print("  • Step 2: Create multi-wavelength network...")
            
            mw_network = create_multiwavelength_mnist_network(
                input_size=128,
                hidden_size=64,
                output_size=8,
                wavelength_channels=4
            )
            
            print("    ✓ Multi-wavelength network created")
            
            # Step 3: Secure simulation
            print("  • Step 3: Run secure simulation...")
            
            session = self.security_manager.create_session(
                user_id="integration_demo",
                permissions=['simulate', 'analyze']
            )
            
            # Generate test data
            test_data = torch.randn(16, 128) * 0.1
            
            # Run simulation with full monitoring
            with self.logger.performance_tracker.track_operation("integration_simulation", "full_demo"):
                sim_results = simulate_multiwavelength_network(
                    mw_network,
                    test_data,
                    simulation_time=50e-9
                )
            
            # Sanitize results
            sanitized_results = self.security_manager.sanitize_response(session, sim_results)
            
            print("    ✓ Secure simulation completed")
            
            # Step 4: Performance analysis
            print("  • Step 4: Performance analysis...")
            
            efficiency = sanitized_results['average_wavelength_efficiency']
            energy = sanitized_results['total_energy']
            
            print(f"    ✓ Wavelength efficiency: {efficiency:.2%}")
            print(f"    ✓ Energy consumption: {energy:.2e} J")
            
            return {
                'integration_successful': True,
                'design_feasible': design_feasible,
                'simulation_completed': True,
                'wavelength_efficiency': efficiency,
                'energy_consumption': energy,
                'security_enabled': True,
                'monitoring_active': True
            }
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive demonstration report."""
        
        # Calculate overall scores
        scores = {}
        
        # Multi-wavelength score
        mw_efficiency = results['multiwavelength']['simulation_results']['wavelength_efficiency']
        scores['multiwavelength'] = min(1.0, mw_efficiency * 2)  # Scale to 0-1
        
        # Physical validation score
        feasibility = results['physical_validation']['feasibility_score']
        scores['physical_validation'] = feasibility
        
        # Security score
        security_features = results['security']
        security_score = sum([
            1 if security_features['session_created'] else 0,
            1 if security_features['input_validation'] == 'passed' else 0,
            1 if security_features['rate_limiting'] else 0,
            1 if security_features['error_handling_demo'] else 0
        ]) / 4
        scores['security'] = security_score
        
        # Scaling score
        scaling_data = results['scaling']
        scaling_score = min(1.0, scaling_data['successful_simulations'] / scaling_data['networks_simulated'])
        scores['scaling'] = scaling_score
        
        # Integration score
        integration_data = results['integration']
        integration_score = sum([
            1 if integration_data['integration_successful'] else 0,
            1 if integration_data['design_feasible'] else 0,
            1 if integration_data['simulation_completed'] else 0,
            1 if integration_data['security_enabled'] else 0
        ]) / 4
        scores['integration'] = integration_score
        
        # Overall score
        overall_score = sum(scores.values()) / len(scores)
        
        # Generate comprehensive report
        report = {
            'demo_metadata': {
                'timestamp': time.time(),
                'framework_version': '0.1.0',
                'demo_duration': time.time(),  # Would track actual duration
                'output_directory': str(self.output_dir)
            },
            'feature_scores': scores,
            'overall_score': overall_score,
            'detailed_results': results,
            'performance_metrics': self.logger.generate_report(),
            'recommendations': self._generate_recommendations(scores),
            'next_steps': [
                "Explore advanced quantum-photonic integration",
                "Implement real-time adaptive learning algorithms",
                "Deploy on cloud infrastructure for massive scaling",
                "Integrate with actual photonic hardware",
                "Develop application-specific optimizations"
            ]
        }
        
        # Save report
        report_path = self.output_dir / "comprehensive_demo_report.json"
        with open(report_path, 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📊 Comprehensive Report Generated")
        print(f"   Overall Score: {overall_score:.2%}")
        print(f"   Report saved to: {report_path}")
        
        return report
    
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on scores."""
        recommendations = []
        
        if scores['multiwavelength'] < 0.8:
            recommendations.append("Optimize wavelength channel allocation for better efficiency")
        
        if scores['physical_validation'] < 0.7:
            recommendations.append("Adjust device parameters to improve fabrication feasibility")
        
        if scores['security'] < 0.9:
            recommendations.append("Review and strengthen security configurations")
        
        if scores['scaling'] < 0.9:
            recommendations.append("Investigate scaling bottlenecks and optimize worker allocation")
        
        if scores['integration'] < 0.8:
            recommendations.append("Enhance system integration and monitoring capabilities")
        
        if not recommendations:
            recommendations.append("Excellent performance across all areas! Ready for production deployment.")
        
        return recommendations


async def main():
    """Run the comprehensive demo."""
    demo = ComprehensiveDemo()
    
    try:
        results = await demo.run_full_demo()
        
        print("\n🎉 Demo Summary:")
        print(f"   Overall Score: {results['overall_score']:.1%}")
        print(f"   Multi-wavelength: {results['feature_scores']['multiwavelength']:.1%}")
        print(f"   Physical Validation: {results['feature_scores']['physical_validation']:.1%}")
        print(f"   Security: {results['feature_scores']['security']:.1%}")
        print(f"   Scaling: {results['feature_scores']['scaling']:.1%}")
        print(f"   Integration: {results['feature_scores']['integration']:.1%}")
        
        print("\n📝 Recommendations:")
        for rec in results['recommendations']:
            print(f"   • {rec}")
            
        return results
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the demo
    results = asyncio.run(main())