"""
Unit tests for physical validation pipeline functionality.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

from photonic_neuromorphics.physical_validation import (
    FDTDSimulator, ThermalAnalyzer, ProcessVariationAnalyzer,
    PhysicalValidationPipeline, FabricationConstraints, ProcessVariationModel,
    create_validation_pipeline, validate_neuron_design
)
from photonic_neuromorphics.core import WaveguideNeuron


class TestFDTDSimulator:
    """Test FDTD simulation functionality."""
    
    def test_fdtd_initialization(self):
        """Test FDTD simulator initialization."""
        fdtd = FDTDSimulator(
            grid_resolution=10e-9,
            simulation_time=100e-15,
            boundary_conditions="PML"
        )
        
        assert fdtd.grid_resolution == 10e-9
        assert fdtd.simulation_time == 100e-15
        assert fdtd.boundary_conditions == "PML"
        assert fdtd._field_components == {}
        assert fdtd._material_grid is None
    
    def test_setup_simulation_domain(self):
        """Test simulation domain setup."""
        fdtd = FDTDSimulator(grid_resolution=10e-9)
        
        device_geometry = {
            'domain_size': (20e-6, 20e-6, 2e-6),
            'waveguides': [
                {
                    'x_bounds': (100, 1900),
                    'y_bounds': (990, 1010),
                    'z_bounds': (0, 200)
                }
            ]
        }
        
        material_properties = {
            'core_index': 3.4,
            'substrate_index': 1.44
        }
        
        fdtd.setup_simulation_domain(device_geometry, material_properties)
        
        assert fdtd._field_components is not None
        assert 'Ex' in fdtd._field_components
        assert 'Ey' in fdtd._field_components
        assert 'Ez' in fdtd._field_components
        assert fdtd._material_grid is not None
    
    def test_run_simulation(self):
        """Test FDTD simulation execution."""
        fdtd = FDTDSimulator(
            grid_resolution=50e-9,  # Larger for faster test
            simulation_time=10e-15   # Shorter for faster test
        )
        
        # Setup minimal simulation domain
        device_geometry = {
            'domain_size': (1e-6, 1e-6, 0.5e-6),
            'waveguides': []
        }
        material_properties = {'core_index': 3.4, 'substrate_index': 1.44}
        
        fdtd.setup_simulation_domain(device_geometry, material_properties)
        
        source_params = {
            'wavelength': 1550e-9,
            'amplitude': 1.0,
            'position': (10, 10, 1)
        }
        
        results = fdtd.run_simulation(source_params)
        
        assert isinstance(results, dict)
        assert 'field_evolution' in results
        assert 'transmission' in results
        assert 'reflection' in results
        assert isinstance(results['transmission'], float)
        assert isinstance(results['reflection'], float)
    
    def test_add_waveguide_to_grid(self):
        """Test waveguide addition to material grid."""
        fdtd = FDTDSimulator()
        grid = np.ones((100, 100, 20)) * 1.44  # Substrate
        
        waveguide = {
            'x_bounds': (10, 90),
            'y_bounds': (45, 55),
            'z_bounds': (0, 20)
        }
        materials = {'core_index': 3.4}
        
        fdtd._add_waveguide_to_grid(grid, waveguide, materials)
        
        # Check that core region has higher index
        assert grid[50, 50, 10] == 3.4  # Core region
        assert grid[50, 30, 10] == 1.44  # Substrate region


class TestThermalAnalyzer:
    """Test thermal analysis functionality."""
    
    def test_thermal_analyzer_initialization(self):
        """Test thermal analyzer initialization."""
        analyzer = ThermalAnalyzer(
            ambient_temperature=300.0,
            thermal_conductivity=150.0,
            power_dissipation=1e-3
        )
        
        assert analyzer.ambient_temperature == 300.0
        assert analyzer.thermal_conductivity == 150.0
        assert analyzer.power_dissipation == 1e-3
    
    def test_analyze_thermal_effects(self):
        """Test thermal effects analysis."""
        analyzer = ThermalAnalyzer()
        
        device_geometry = {
            'size': (100e-6, 50e-6),
            'volume': 1e-15
        }
        
        operating_conditions = {
            'heat_sources': [
                {
                    'position': (25, 12),
                    'power': 1e-3
                }
            ]
        }
        
        results = analyzer.analyze_thermal_effects(device_geometry, operating_conditions)
        
        assert isinstance(results, dict)
        assert 'temperature_distribution' in results
        assert 'refractive_index_shift' in results
        assert 'max_temperature' in results
        assert 'thermal_stability' in results
        assert 'thermal_time_constant' in results
        
        assert results['max_temperature'] > analyzer.ambient_temperature
        assert isinstance(results['thermal_time_constant'], float)
    
    def test_thermal_index_shift_calculation(self):
        """Test thermal refractive index shift calculation."""
        analyzer = ThermalAnalyzer(ambient_temperature=300.0)
        
        temperature_map = np.array([[310.0, 305.0], [308.0, 302.0]])
        index_shift = analyzer._calculate_thermal_index_shift(temperature_map)
        
        assert index_shift.shape == temperature_map.shape
        assert np.all(index_shift >= 0)  # Positive temperature increases
        assert index_shift[0, 0] > index_shift[1, 1]  # Higher temp -> larger shift
    
    def test_thermal_stability_assessment(self):
        """Test thermal stability assessment."""
        analyzer = ThermalAnalyzer()
        
        # Uniform temperature (high stability)
        uniform_temp = np.ones((10, 10)) * 305.0
        uniform_stability = analyzer._assess_thermal_stability(uniform_temp)
        
        # Non-uniform temperature (lower stability)
        gradient_temp = np.linspace(300, 350, 100).reshape(10, 10)
        gradient_stability = analyzer._assess_thermal_stability(gradient_temp)
        
        assert uniform_stability['stability_score'] > gradient_stability['stability_score']
        assert uniform_stability['temperature_uniformity'] < gradient_stability['temperature_uniformity']


class TestProcessVariationAnalyzer:
    """Test process variation analysis functionality."""
    
    def test_process_analyzer_initialization(self):
        """Test process variation analyzer initialization."""
        variation_model = ProcessVariationModel(
            width_variation=0.02,
            thickness_variation=0.03
        )
        
        analyzer = ProcessVariationAnalyzer(
            variation_model=variation_model,
            monte_carlo_samples=100
        )
        
        assert analyzer.variation_model.width_variation == 0.02
        assert analyzer.monte_carlo_samples == 100
    
    def test_generate_process_variations(self):
        """Test process variation generation."""
        variation_model = ProcessVariationModel(width_variation=0.1)
        analyzer = ProcessVariationAnalyzer(
            variation_model=variation_model,
            monte_carlo_samples=10
        )
        
        nominal_device = {
            'width': 450e-9,
            'thickness': 220e-9,
            'refractive_index': 3.4
        }
        
        variations = analyzer._generate_process_variations(nominal_device)
        
        assert len(variations) == 10
        assert all('width' in device for device in variations)
        
        # Check that variations are different from nominal
        widths = [device['width'] for device in variations]
        assert len(set(widths)) > 1  # Should have variety
        assert all(w > 0 for w in widths)  # All positive
    
    @patch('photonic_neuromorphics.physical_validation.ProcessPoolExecutor')
    def test_analyze_process_sensitivity(self, mock_executor):
        """Test process sensitivity analysis."""
        # Mock the executor to avoid actual parallel processing
        mock_future = Mock()
        mock_future.result.return_value = 0.8
        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
        
        variation_model = ProcessVariationModel()
        analyzer = ProcessVariationAnalyzer(
            variation_model=variation_model,
            monte_carlo_samples=10
        )
        
        nominal_device = {'width': 450e-9, 'thickness': 220e-9}
        
        def simple_performance_metric(device):
            return 0.8  # Constant for testing
        
        results = analyzer.analyze_process_sensitivity(nominal_device, simple_performance_metric)
        
        assert isinstance(results, dict)
        assert 'nominal_performance' in results
        assert 'mean_performance' in results
        assert 'yield_3sigma' in results
        assert 'sensitivity_coefficient' in results
        assert 'process_corners' in results
    
    def test_process_corners_analysis(self):
        """Test process corners analysis."""
        variation_model = ProcessVariationModel()
        analyzer = ProcessVariationAnalyzer(variation_model=variation_model)
        
        # Create test performance samples
        performance_samples = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.75, 0.65, 0.55, 0.95])
        
        corners = analyzer._analyze_process_corners(performance_samples)
        
        assert isinstance(corners, dict)
        assert 'slow_corner' in corners
        assert 'fast_corner' in corners
        assert 'nominal' in corners
        assert corners['slow_corner'] <= corners['nominal'] <= corners['fast_corner']


class TestPhysicalValidationPipeline:
    """Test complete physical validation pipeline."""
    
    def test_pipeline_initialization(self):
        """Test validation pipeline initialization."""
        pipeline = create_validation_pipeline(
            grid_resolution=20e-9,
            monte_carlo_samples=10
        )
        
        assert isinstance(pipeline, PhysicalValidationPipeline)
        assert pipeline.fdtd_simulator.grid_resolution == 20e-9
        assert pipeline.process_analyzer.monte_carlo_samples == 10
    
    def test_neuron_to_geometry_conversion(self):
        """Test neuron parameter to geometry conversion."""
        pipeline = create_validation_pipeline()
        
        neuron = WaveguideNeuron(
            arm_length=100e-6,
            wavelength=1550e-9,
            threshold_power=1e-6
        )
        
        geometry = pipeline._neuron_to_geometry(neuron)
        
        assert isinstance(geometry, dict)
        assert geometry['arm_length'] == 100e-6
        assert geometry['wavelength'] == 1550e-9
        assert 'waveguide_width' in geometry
        assert 'domain_size' in geometry
        assert 'refractive_index' in geometry
    
    @patch.object(PhysicalValidationPipeline, '_validate_optical_performance')
    @patch.object(PhysicalValidationPipeline, '_validate_thermal_performance')
    @patch.object(PhysicalValidationPipeline, '_validate_process_robustness')
    @patch.object(PhysicalValidationPipeline, '_validate_fabrication_feasibility')
    def test_validate_photonic_neuron(self, mock_fab, mock_process, mock_thermal, mock_optical):
        """Test complete neuron validation."""
        # Mock the validation methods
        mock_optical.return_value = {'transmission': 0.8, 'insertion_loss_db': 1.0}
        mock_thermal.return_value = {'max_temperature': 310.0}
        mock_process.return_value = {'yield_3sigma': 0.9}
        mock_fab.return_value = {'fabrication_feasible': True}
        
        pipeline = create_validation_pipeline()
        neuron = WaveguideNeuron()
        target_performance = {
            'transmission': 0.7,
            'max_temperature': 350.0,
            'yield': 0.9
        }
        
        results = pipeline.validate_photonic_neuron(neuron, target_performance)
        
        assert isinstance(results, dict)
        assert 'optical_validation' in results
        assert 'thermal_validation' in results
        assert 'process_validation' in results
        assert 'fabrication_validation' in results
        assert 'overall_assessment' in results
        assert 'validation_passed' in results
        
        # Check that all validation methods were called
        mock_optical.assert_called_once()
        mock_thermal.assert_called_once()
        mock_process.assert_called_once()
        mock_fab.assert_called_once()
    
    def test_fabrication_feasibility_validation(self):
        """Test fabrication feasibility validation."""
        pipeline = create_validation_pipeline()
        
        # Valid geometry
        valid_geometry = {
            'waveguide_width': 450e-9,
            'gap': 200e-9,
            'waveguide_height': 220e-9,
            'arm_length': 100e-6
        }
        
        results = pipeline._validate_fabrication_feasibility(valid_geometry)
        
        assert isinstance(results, dict)
        assert 'fabrication_feasible' in results
        assert 'constraint_violations' in results
        assert 'min_feature_size' in results
        assert 'aspect_ratio' in results
        assert 'estimated_yield' in results
        
        # Invalid geometry (too small features)
        invalid_geometry = {
            'waveguide_width': 50e-9,  # Too small
            'gap': 30e-9,              # Too small
            'waveguide_height': 220e-9,
            'arm_length': 100e-6
        }
        
        invalid_results = pipeline._validate_fabrication_feasibility(invalid_geometry)
        assert not invalid_results['fabrication_feasible']
        assert len(invalid_results['constraint_violations']) > 0
    
    def test_overall_assessment(self):
        """Test overall feasibility assessment."""
        pipeline = create_validation_pipeline()
        
        # High performance results
        optical = {'transmission': 0.9}
        thermal = {'max_temperature': 310.0}
        process = {'yield_3sigma': 0.95}
        fabrication = {'fabrication_feasible': True}
        targets = {'transmission': 0.7, 'max_temperature': 350.0, 'yield': 0.9}
        
        assessment = pipeline._assess_overall_feasibility(
            optical, thermal, process, fabrication, targets
        )
        
        assert isinstance(assessment, dict)
        assert 'individual_scores' in assessment
        assert 'feasibility_score' in assessment
        assert 'recommendation' in assessment
        assert 'critical_issues' in assessment
        assert 'optimization_suggestions' in assessment
        
        assert assessment['feasibility_score'] > 0.8  # Should be high
        assert "highly feasible" in assessment['recommendation'].lower()
    
    def test_export_validation_report(self):
        """Test validation report export."""
        pipeline = create_validation_pipeline()
        
        # Set up a mock validation report
        pipeline._validation_report = {
            'test_data': 'test_value',
            'timestamp': 1234567890
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name
        
        try:
            pipeline.export_validation_report(filename)
            
            # Verify file was created and contains expected data
            with open(filename, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data['test_data'] == 'test_value'
            assert 'timestamp' in loaded_data
        finally:
            import os
            if os.path.exists(filename):
                os.unlink(filename)


class TestValidationFunctions:
    """Test standalone validation functions."""
    
    def test_validate_neuron_design(self):
        """Test convenient neuron design validation function."""
        neuron = WaveguideNeuron(
            arm_length=100e-6,
            threshold_power=1e-6
        )
        
        with patch.object(PhysicalValidationPipeline, 'validate_photonic_neuron') as mock_validate:
            mock_validate.return_value = {'validation_passed': True}
            
            results = validate_neuron_design(neuron)
            
            assert isinstance(results, dict)
            mock_validate.assert_called_once()
    
    def test_validate_neuron_design_with_custom_specs(self):
        """Test neuron validation with custom specifications."""
        neuron = WaveguideNeuron()
        custom_specs = {
            'transmission': 0.9,
            'max_temperature': 300.0,
            'yield': 0.95
        }
        
        with patch.object(PhysicalValidationPipeline, 'validate_photonic_neuron') as mock_validate:
            mock_validate.return_value = {'validation_passed': True}
            
            results = validate_neuron_design(neuron, custom_specs)
            
            # Check that custom specs were passed
            call_args = mock_validate.call_args
            assert call_args[0][1] == custom_specs  # Second argument should be specs


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_validation_report_export(self):
        """Test error when exporting empty validation report."""
        pipeline = create_validation_pipeline()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name
        
        with pytest.raises(Exception):  # Should raise PhysicalValidationError
            pipeline.export_validation_report(filename)
    
    def test_unsupported_export_format(self):
        """Test error for unsupported export format."""
        pipeline = create_validation_pipeline()
        pipeline._validation_report = {'test': 'data'}
        
        with pytest.raises(ValueError):
            pipeline.export_validation_report('test.txt')  # Unsupported format
    
    def test_invalid_geometry_thermal_analysis(self):
        """Test thermal analysis with invalid geometry."""
        analyzer = ThermalAnalyzer()
        
        # Missing required geometry fields
        invalid_geometry = {}
        operating_conditions = {'heat_sources': []}
        
        # Should handle gracefully or provide defaults
        results = analyzer.analyze_thermal_effects(invalid_geometry, operating_conditions)
        
        assert isinstance(results, dict)
        assert 'temperature_distribution' in results


if __name__ == '__main__':
    pytest.main([__file__])