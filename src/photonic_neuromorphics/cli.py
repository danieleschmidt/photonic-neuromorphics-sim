"""
Command-line interface for photonic neuromorphics simulation.

Provides comprehensive CLI tools for creating, simulating, and analyzing
photonic spiking neural networks with RTL generation capabilities.
"""

import click
import json
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch

from .core import PhotonicSNN, WaveguideNeuron, create_mnist_photonic_snn, encode_to_spikes
from .simulator import PhotonicSimulator, SimulationMode, NoiseParameters
from .rtl import RTLGenerator, RTLGenerationConfig, ConstraintsConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(verbose: bool):
    """Photonic Neuromorphics Simulation Framework CLI."""
    setup_logging(verbose)


@main.command()
@click.option('--topology', '-t', default='784,256,128,10', 
              help='Network topology (comma-separated layer sizes)')
@click.option('--neuron-type', default='waveguide', 
              help='Neuron type (waveguide, microring)')
@click.option('--wavelength', '-w', default=1550e-9, type=float,
              help='Operating wavelength in meters')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file for model')
def create_network(topology: str, neuron_type: str, wavelength: float, output: Optional[str]):
    """Create a new photonic spiking neural network."""
    topology_list = [int(x.strip()) for x in topology.split(',')]
    
    click.echo(f"Creating photonic SNN with topology: {topology_list}")
    click.echo(f"Neuron type: {neuron_type}")
    click.echo(f"Wavelength: {wavelength*1e9:.0f} nm")
    
    # Create network
    model = PhotonicSNN(
        topology=topology_list,
        neuron_type=WaveguideNeuron,
        wavelength=wavelength
    )
    
    # Display network info
    info = model.get_network_info()
    click.echo("\nNetwork Information:")
    for key, value in info.items():
        click.echo(f"  {key}: {value}")
    
    if output:
        # Save model (simplified serialization)
        model_data = {
            'topology': topology_list,
            'neuron_type': neuron_type,
            'wavelength': wavelength,
            'network_info': info
        }
        
        with open(output, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        click.echo(f"\nModel saved to: {output}")
    
    click.echo(click.style("✓ Network created successfully!", fg='green'))


@main.command()
@click.option('--input-data', '-i', type=click.Path(exists=True),
              help='Input spike data file (numpy format)')
@click.option('--model', '-m', type=click.Path(exists=True),
              help='Model configuration file')
@click.option('--duration', '-d', default=100e-9, type=float,
              help='Simulation duration in seconds')
@click.option('--mode', type=click.Choice(['behavioral', 'optical', 'mixed', 'spice']),
              default='optical', help='Simulation mode')
@click.option('--noise/--no-noise', default=True, help='Enable optical noise')
@click.option('--output', '-o', type=click.Path(), help='Output results file')
def simulate(input_data: Optional[str], model: Optional[str], duration: float, 
             mode: str, noise: bool, output: Optional[str]):
    """Simulate photonic neural network."""
    click.echo(f"Running {mode} simulation for {duration*1e9:.1f} ns")
    
    # Create or load model
    if model:
        with open(model, 'r') as f:
            model_config = json.load(f)
        snn = PhotonicSNN(
            topology=model_config['topology'],
            wavelength=model_config.get('wavelength', 1550e-9)
        )
    else:
        # Default MNIST model
        snn = create_mnist_photonic_snn()
    
    # Load or generate input data
    if input_data:
        spike_data = torch.from_numpy(np.load(input_data)).float()
    else:
        # Generate random test data
        click.echo("Generating random test spike train...")
        test_data = np.random.rand(784) 
        spike_data = encode_to_spikes(test_data, duration)
    
    click.echo(f"Input data shape: {spike_data.shape}")
    
    # Configure simulator
    sim_mode = SimulationMode(mode.upper())
    noise_params = NoiseParameters() if noise else NoiseParameters(
        shot_noise_enabled=False,
        thermal_noise_enabled=False,
        phase_noise_enabled=False
    )
    
    simulator = PhotonicSimulator(
        mode=sim_mode,
        noise_params=noise_params
    )
    
    # Run simulation
    with click.progressbar(length=100, label='Simulating') as bar:
        results = simulator.run(snn, spike_data, duration, detailed_logging=True)
        bar.update(100)
    
    # Display results
    click.echo("\nSimulation Results:")
    click.echo(f"  Output spikes: {torch.sum(results.output_spikes).item():.0f}")
    click.echo(f"  Simulation time: {results.timing_metrics['simulation_time']:.3f}s")
    click.echo(f"  Energy consumption: {results.energy_consumption['total_energy']*1e12:.2f} pJ")
    click.echo(f"  Power consumption: {results.energy_consumption['power_consumption']*1e3:.2f} mW")
    
    if results.noise_analysis:
        click.echo(f"  Input SNR: {results.noise_analysis['input_snr_db']:.1f} dB")
        click.echo(f"  Output SNR: {results.noise_analysis['output_snr_db']:.1f} dB")
    
    # Save results
    if output:
        results_data = {
            'output_spikes': results.output_spikes.numpy().tolist(),
            'energy_consumption': results.energy_consumption,
            'timing_metrics': results.timing_metrics,
            'noise_analysis': results.noise_analysis,
            'simulation_parameters': {
                'duration': duration,
                'mode': mode,
                'noise_enabled': noise
            }
        }
        
        with open(output, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        click.echo(f"Results saved to: {output}")
    
    click.echo(click.style("✓ Simulation completed successfully!", fg='green'))


@main.command()
@click.option('--model', '-m', type=click.Path(exists=True),
              help='Model configuration file')
@click.option('--target-freq', default=1e9, type=float,
              help='Target frequency in Hz')
@click.option('--technology', default='skywater130',
              help='Target technology (skywater130, tsmc28)')
@click.option('--fixed-point', default=16, type=int,
              help='Fixed-point width in bits')
@click.option('--optimization', type=click.Choice(['0', '1', '2']),
              default='2', help='Optimization level')
@click.option('--output-dir', '-o', type=click.Path(),
              help='Output directory for RTL files')
def generate_rtl(model: Optional[str], target_freq: float, technology: str,
                 fixed_point: int, optimization: str, output_dir: Optional[str]):
    """Generate RTL for photonic neural network."""
    click.echo(f"Generating RTL for {technology} at {target_freq/1e6:.0f} MHz")
    
    # Create or load model
    if model:
        with open(model, 'r') as f:
            model_config = json.load(f)
        snn = PhotonicSNN(
            topology=model_config['topology'],
            wavelength=model_config.get('wavelength', 1550e-9)
        )
    else:
        # Default MNIST model  
        snn = create_mnist_photonic_snn()
    
    # Configure RTL generation
    config = RTLGenerationConfig(
        target_frequency=target_freq,
        fixed_point_width=fixed_point,
        optimization_level=int(optimization),
        target_technology=technology
    )
    
    constraints = ConstraintsConfig(
        target_frequency=target_freq
    )
    
    generator = RTLGenerator(config, constraints, technology)
    
    # Generate RTL
    with click.progressbar(length=100, label='Generating RTL') as bar:
        rtl_design = generator.generate(snn, output_dir)
        bar.update(100)
    
    # Display resource estimates
    click.echo("\nResource Estimates:")
    resources = rtl_design.resource_estimates
    click.echo(f"  Logic gates: {resources['logic_gates']:,}")
    click.echo(f"  Memory bits: {resources['memory_bits']:,}")
    click.echo(f"  Total area: {resources['total_area_um2']:.0f} μm²")
    click.echo(f"  Estimated max freq: {resources['estimated_max_frequency_hz']/1e6:.0f} MHz")
    click.echo(f"  Estimated power: {resources['estimated_power_mw']:.2f} mW")
    
    if output_dir:
        click.echo(f"\nRTL files saved to: {output_dir}")
        click.echo("  Generated files:")
        click.echo("    rtl/photonic_neural_network.v")
        click.echo("    tb/tb_photonic_neural_network.v")
        click.echo("    constraints/constraints.sdc")
        click.echo("    scripts/dc_synthesis.tcl")
        click.echo("    reports/resource_estimates.json")
    
    click.echo(click.style("✓ RTL generation completed successfully!", fg='green'))


@main.command()
@click.option('--input-file', '-i', type=click.Path(exists=True), required=True,
              help='Input data file (numpy format)')
@click.option('--model', '-m', type=click.Path(exists=True),
              help='Model configuration file')
@click.option('--modes', default='behavioral,optical',
              help='Simulation modes to benchmark (comma-separated)')
@click.option('--output', '-o', type=click.Path(),
              help='Output benchmark results file')
def benchmark(input_file: str, model: Optional[str], modes: str, output: Optional[str]):
    """Benchmark performance across different simulation modes."""
    click.echo("Running benchmark across simulation modes...")
    
    # Load model
    if model:
        with open(model, 'r') as f:
            model_config = json.load(f)
        snn = PhotonicSNN(
            topology=model_config['topology'],
            wavelength=model_config.get('wavelength', 1550e-9)
        )
    else:
        snn = create_mnist_photonic_snn()
    
    # Load test data
    input_data = np.load(input_file)
    test_cases = [encode_to_spikes(data) for data in input_data[:5]]  # First 5 samples
    
    # Parse modes
    mode_list = [SimulationMode(mode.strip().upper()) for mode in modes.split(',')]
    
    # Create simulator and run benchmark
    simulator = PhotonicSimulator()
    
    with click.progressbar(length=len(mode_list), label='Benchmarking') as bar:
        results = simulator.benchmark_performance(snn, test_cases, mode_list)
        bar.update(len(mode_list))
    
    # Display results
    click.echo("\nBenchmark Results:")
    click.echo(f"{'Mode':<15} {'Avg Time (s)':<15} {'Throughput (Hz)':<15} {'Avg Energy (pJ)':<15}")
    click.echo("-" * 60)
    
    for mode, metrics in results.items():
        click.echo(f"{mode:<15} {metrics['avg_time_per_case']:<15.4f} "
                  f"{metrics['throughput']:<15.2f} {metrics['avg_energy_per_case']*1e12:<15.2f}")
    
    # Save results
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nBenchmark results saved to: {output}")
    
    click.echo(click.style("✓ Benchmark completed successfully!", fg='green'))


@main.command()
@click.option('--model-file', '-m', type=click.Path(exists=True), required=True,
              help='Model configuration file')
def analyze(model_file: str):
    """Analyze photonic neural network characteristics."""
    click.echo(f"Analyzing model: {model_file}")
    
    # Load model
    with open(model_file, 'r') as f:
        model_config = json.load(f)
    
    snn = PhotonicSNN(
        topology=model_config['topology'],
        wavelength=model_config.get('wavelength', 1550e-9)
    )
    
    # Get comprehensive analysis
    info = snn.get_network_info()
    
    # Generate test data for analysis
    test_data = np.random.rand(info['topology'][0])
    spike_train = encode_to_spikes(test_data)
    
    # Estimate performance characteristics
    energy_estimate = snn.estimate_energy_consumption(spike_train)
    
    click.echo("\n" + "="*50)
    click.echo("PHOTONIC NEURAL NETWORK ANALYSIS")
    click.echo("="*50)
    
    click.echo(f"\nNetwork Architecture:")
    click.echo(f"  Topology: {info['topology']}")
    click.echo(f"  Total neurons: {info['total_neurons']:,}")
    click.echo(f"  Total synapses: {info['total_synapses']:,}")
    click.echo(f"  Neuron type: {info['neuron_type']}")
    
    click.echo(f"\nOptical Parameters:")
    opt_params = info['optical_parameters']
    click.echo(f"  Wavelength: {opt_params['wavelength']*1e9:.0f} nm")
    click.echo(f"  Input power: {opt_params['power']*1e3:.1f} mW")
    click.echo(f"  Propagation loss: {opt_params['loss']} dB/cm")
    click.echo(f"  Coupling efficiency: {opt_params['coupling_efficiency']*100:.1f}%")
    
    click.echo(f"\nPerformance Estimates:")
    click.echo(f"  Energy per spike: {energy_estimate['energy_per_spike']*1e15:.2f} fJ")
    click.echo(f"  Total energy (test): {energy_estimate['total_energy']*1e12:.2f} pJ")
    click.echo(f"  Power consumption: {energy_estimate['power_consumption']*1e3:.2f} mW")
    
    # Comparison with electronic equivalent
    electronic_energy_per_op = 1e-12  # 1 pJ per operation
    electronic_total = info['total_synapses'] * electronic_energy_per_op
    
    improvement = electronic_total / energy_estimate['total_energy'] if energy_estimate['total_energy'] > 0 else float('inf')
    
    click.echo(f"\nPhotonic vs Electronic Comparison:")
    click.echo(f"  Electronic energy estimate: {electronic_total*1e12:.2f} pJ")
    click.echo(f"  Energy improvement factor: {improvement:.0f}x")
    
    click.echo(f"\nOptical Design Characteristics:")
    click.echo(f"  Wavelength band: {'C-band' if 1530e-9 <= info['optical_parameters']['wavelength'] <= 1565e-9 else 'Other'}")
    click.echo(f"  Estimated propagation delay: {info['total_synapses'] * 1e-12 * 3.3:.2f} ps")  # Speed of light in silicon
    
    click.echo(click.style("\n✓ Analysis completed!", fg='green'))


if __name__ == '__main__':
    main()