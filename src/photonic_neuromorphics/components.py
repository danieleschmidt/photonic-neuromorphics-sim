"""
Advanced Photonic Components for Neuromorphic Computing.

This module provides sophisticated photonic components including Mach-Zehnder
interferometers, microring resonators, photodetectors, and phase-change materials
for building complex neuromorphic systems.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

from .core import OpticalParameters, WaveguideNeuron
from .exceptions import OpticalModelError, ValidationError


@dataclass
class ComponentParameters:
    """Base parameters for photonic components."""
    wavelength: float = 1550e-9  # Operating wavelength
    temperature: float = 300.0   # Operating temperature (K)
    power: float = 1e-3         # Input power (W)
    loss: float = 0.1           # Insertion loss (dB)


class PhotonicComponent(ABC):
    """Abstract base class for all photonic components."""
    
    def __init__(self, params: ComponentParameters):
        self.params = params
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def transfer_function(self, wavelength: float, power: float) -> Tuple[float, float]:
        """Calculate transfer function (transmission, phase)."""
        pass
    
    @abstractmethod
    def get_spice_model(self) -> str:
        """Generate SPICE model for the component."""
        pass
    
    def get_insertion_loss(self, wavelength: float = None) -> float:
        """Calculate insertion loss in dB."""
        if wavelength is None:
            wavelength = self.params.wavelength
        return self.params.loss


class MachZehnderNeuron(PhotonicComponent):
    """
    Advanced Mach-Zehnder interferometer-based photonic neuron.
    
    Implements a sophisticated neuron model with thermal phase shifters,
    realistic propagation delays, and nonlinear activation functions.
    """
    
    def __init__(
        self,
        arm_length: float = 100e-6,  # 100 μm
        phase_shifter_type: str = "thermal",
        modulation_depth: float = 0.9,
        threshold_power: float = 1e-6,  # 1 μW
        coupling_ratio: float = 0.5,
        extinction_ratio: float = 20,  # dB
        params: Optional[ComponentParameters] = None
    ):
        super().__init__(params or ComponentParameters())
        self.arm_length = arm_length
        self.phase_shifter_type = phase_shifter_type
        self.modulation_depth = modulation_depth
        self.threshold_power = threshold_power
        self.coupling_ratio = coupling_ratio
        self.extinction_ratio = extinction_ratio
        
        # Derived parameters
        self.free_spectral_range = self._calculate_fsr()
        self.thermal_efficiency = 2e-4 if phase_shifter_type == "thermal" else 0  # rad/mW
        
        # Internal state
        self.membrane_potential = 0.0
        self.last_spike_time = 0.0
        self.refractory_period = 5e-9  # 5 ns
    
    def _calculate_fsr(self) -> float:
        """Calculate free spectral range."""
        effective_index = 2.44  # Silicon effective index
        return self.params.wavelength**2 / (2 * effective_index * self.arm_length)
    
    def transfer_function(self, wavelength: float, power: float) -> Tuple[float, float]:
        """Calculate MZI transfer function with realistic effects."""
        # Wavelength-dependent phase
        k = 2 * np.pi / wavelength
        effective_index = 2.44
        intrinsic_phase = k * effective_index * self.arm_length
        
        # Thermal phase shift (power-dependent)
        thermal_phase = power * self.thermal_efficiency * 1e3  # Convert W to mW
        
        # Total phase difference
        total_phase = intrinsic_phase + thermal_phase
        
        # MZI transmission with coupling ratio effects
        t1 = np.sqrt(self.coupling_ratio)
        t2 = np.sqrt(1 - self.coupling_ratio)
        
        transmission = (
            t1**2 * np.cos(total_phase/2)**2 + 
            t2**2 * np.sin(total_phase/2)**2
        )
        
        # Apply extinction ratio limit
        min_transmission = 10**(-self.extinction_ratio/10)
        transmission = max(transmission, min_transmission)
        
        return transmission, total_phase
    
    def forward(self, optical_input: float, time: float, phase_control: float = 0.0) -> bool:
        """Advanced neuron processing with realistic dynamics."""
        try:
            # Check refractory period
            if time - self.last_spike_time < self.refractory_period:
                return False
            
            # Apply transfer function
            transmission, phase = self.transfer_function(
                self.params.wavelength, 
                optical_input
            )
            
            # Integrate membrane potential with leak
            leak_rate = 0.95  # 5% leak per time step
            self.membrane_potential = self.membrane_potential * leak_rate + transmission
            
            # Nonlinear activation with phase control
            activation_threshold = self.threshold_power * (1 + 0.1 * np.sin(phase_control))
            
            if self.membrane_potential > activation_threshold:
                self.membrane_potential = 0.0
                self.last_spike_time = time
                
                self.logger.debug(f"Spike at time {time:.2e}s, threshold {activation_threshold:.2e}")
                return True
            
            return False
            
        except Exception as e:
            raise OpticalModelError("mach_zehnder_neuron", "forward", optical_input, str(e))
    
    def get_spice_model(self) -> str:
        """Generate comprehensive SPICE model."""
        return f"""
* Mach-Zehnder Photonic Neuron SPICE Model
.subckt mz_neuron optical_in phase_ctrl spike_out
* Parameters
.param arm_length={self.arm_length*1e6}u
.param coupling_ratio={self.coupling_ratio}
.param extinction_ratio={self.extinction_ratio}

* Optical splitter
E_split1 arm1_in 0 optical_in 0 {np.sqrt(self.coupling_ratio)}
E_split2 arm2_in 0 optical_in 0 {np.sqrt(1-self.coupling_ratio)}

* Phase shifters
G_phase1 0 phase1 phase_ctrl 0 {self.thermal_efficiency*1e3}
G_phase2 0 phase2 CCCS=0 0 0

* Optical delay lines
R_delay1 arm1_in arm1_out {50 * self.arm_length * 1e6}
C_delay1 arm1_out 0 {10e-15 * self.arm_length * 1e6}p

R_delay2 arm2_in arm2_out {50 * self.arm_length * 1e6}
C_delay2 arm2_out 0 {10e-15 * self.arm_length * 1e6}p

* Optical combiner with phase
E_combine spike_out 0 POLY(2) arm1_out 0 arm2_out phase1 0 0 1 0 1 0 0 0

.ends mz_neuron
"""


class MicroringResonator(PhotonicComponent):
    """
    Microring resonator for wavelength filtering and memory applications.
    
    Provides high-Q optical filtering with tunable resonance frequency
    for implementing synaptic plasticity and memory functions.
    """
    
    def __init__(
        self,
        radius: float = 10e-6,  # 10 μm radius
        coupling_gap: float = 200e-9,  # 200 nm gap
        quality_factor: float = 10000,
        group_index: float = 4.2,
        params: Optional[ComponentParameters] = None
    ):
        super().__init__(params or ComponentParameters())
        self.radius = radius
        self.coupling_gap = coupling_gap
        self.quality_factor = quality_factor
        self.group_index = group_index
        
        # Calculate derived parameters
        self.circumference = 2 * np.pi * radius
        self.free_spectral_range = self._calculate_fsr()
        self.coupling_coefficient = self._calculate_coupling()
        self.round_trip_loss = self._calculate_loss()
        
        # Memory state for plasticity
        self.resonance_shift = 0.0  # For tunable resonance
        self.stored_energy = 0.0
    
    def _calculate_fsr(self) -> float:
        """Calculate free spectral range."""
        effective_index = 2.44
        return self.params.wavelength**2 / (effective_index * self.circumference)
    
    def _calculate_coupling(self) -> float:
        """Calculate coupling coefficient based on gap."""
        # Empirical model for coupling vs gap
        return 0.5 * np.exp(-self.coupling_gap / 100e-9)  # Exponential decay
    
    def _calculate_loss(self) -> float:
        """Calculate round-trip loss."""
        # Loss from material absorption and scattering
        material_loss = 0.1  # dB/cm
        scattering_loss = 0.05  # dB/cm
        return (material_loss + scattering_loss) * self.circumference * 100  # Convert to dB
    
    def transfer_function(self, wavelength: float, power: float) -> Tuple[float, float]:
        """Calculate microring transfer function."""
        # Detuning from resonance
        resonance_wavelength = self.params.wavelength + self.resonance_shift
        detuning = 2 * np.pi * (wavelength - resonance_wavelength) / self.free_spectral_range
        
        # Coupling and loss parameters
        kappa = self.coupling_coefficient
        alpha = 10**(-self.round_trip_loss / 20)  # Convert dB to linear
        
        # Transfer function (all-pass filter response)
        denominator = 1 - alpha * np.exp(1j * detuning)
        h = (alpha * np.exp(1j * detuning) - 1) / denominator
        
        transmission = abs(h)**2
        phase = np.angle(h)
        
        # Power-dependent nonlinearity (Kerr effect)
        if power > 1e-3:  # Above 1 mW
            kerr_shift = 2e-6 * power  # Nonlinear phase shift
            phase += kerr_shift
        
        # Energy storage for plasticity
        if transmission > 0.5:  # On resonance
            self.stored_energy += power * transmission * 1e-9  # Accumulate energy
        
        return transmission, phase
    
    def update_plasticity(self, learning_rate: float = 1e-6) -> None:
        """Update synaptic strength based on stored energy."""
        # Hebbian-like plasticity rule
        if self.stored_energy > 1e-12:  # Above threshold
            self.resonance_shift += learning_rate * np.sign(self.stored_energy)
            self.stored_energy *= 0.99  # Decay stored energy
        
        # Bounds on resonance shift
        self.resonance_shift = np.clip(self.resonance_shift, -10e-9, 10e-9)
    
    def get_spice_model(self) -> str:
        """Generate SPICE model for microring."""
        return f"""
* Microring Resonator SPICE Model
.subckt microring optical_in optical_out drop_out
* Parameters
.param radius={self.radius*1e6}u
.param Q={self.quality_factor}
.param kappa={self.coupling_coefficient}

* Resonator model (simplified RLC circuit)
L_ring ring_node 0 {1e-9 * self.radius * 1e6}n
C_ring ring_node 0 {1e-15 / (self.quality_factor * self.radius * 1e6)}p
R_ring ring_node 0 {50 * self.quality_factor}

* Coupling elements
K_couple optical_in ring_node {self.coupling_coefficient}
K_drop ring_node drop_out {self.coupling_coefficient * 0.5}

* Through port
E_through optical_out 0 optical_in ring_node 1 -{self.coupling_coefficient}

.ends microring
"""


class PhotonicCrystalCavity(PhotonicComponent):
    """
    Photonic crystal cavity for ultra-high Q resonances and strong light confinement.
    
    Enables sophisticated optical memory and wavelength-selective processing
    with extremely high quality factors.
    """
    
    def __init__(
        self,
        lattice_constant: float = 420e-9,  # 420 nm
        hole_radius: float = 120e-9,  # 120 nm
        num_periods: int = 15,
        defect_shift: float = 50e-9,  # 50 nm shift
        quality_factor: float = 1e6,  # Ultra-high Q
        mode_volume: float = 1e-18,  # 1 cubic wavelength
        params: Optional[ComponentParameters] = None
    ):
        super().__init__(params or ComponentParameters())
        self.lattice_constant = lattice_constant
        self.hole_radius = hole_radius
        self.num_periods = num_periods
        self.defect_shift = defect_shift
        self.quality_factor = quality_factor
        self.mode_volume = mode_volume
        
        # Calculate cavity properties
        self.finesse = quality_factor * np.pi / 2
        self.photon_lifetime = quality_factor / (2 * np.pi * 3e8 / self.params.wavelength)
        self.cavity_enhancement = quality_factor / (2 * np.pi)
    
    def transfer_function(self, wavelength: float, power: float) -> Tuple[float, float]:
        """Calculate photonic crystal cavity response."""
        # Resonance condition
        resonance_wavelength = 2 * self.lattice_constant  # Simplified
        detuning = (wavelength - resonance_wavelength) / resonance_wavelength
        
        # Lorentzian response
        gamma = 1 / (2 * self.quality_factor)
        response = 1 / (1 + (detuning / gamma)**2)
        
        # Phase response
        phase = -np.arctan(detuning / gamma)
        
        # Nonlinear effects at high power
        if power > 1e-6:  # Above 1 μW
            # Kerr nonlinearity enhanced by high Q
            nonlinear_shift = 2e-3 * power * self.cavity_enhancement
            phase += nonlinear_shift
        
        # Transmission includes cavity loss
        transmission = response * (1 - 1/self.quality_factor)
        
        return transmission, phase
    
    def get_spice_model(self) -> str:
        """Generate SPICE model for photonic crystal cavity."""
        return f"""
* Photonic Crystal Cavity SPICE Model
.subckt pc_cavity optical_in optical_out
* Ultra-high Q cavity parameters
.param Q={self.quality_factor}
.param mode_volume={self.mode_volume*1e18}e-18
.param finesse={self.finesse}

* High-Q resonator equivalent circuit
L_cavity cavity_node 0 {1e-6 * self.quality_factor}u
C_cavity cavity_node 0 {1e-18 / self.quality_factor}f
R_cavity cavity_node 0 {1e6 * self.quality_factor}

* Input coupling
K_in optical_in cavity_node {1e-3}

* Output coupling with high finesse
E_out optical_out 0 cavity_node 0 {1/self.finesse}

.ends pc_cavity
"""


class PhaseChangeMaterial(PhotonicComponent):
    """
    Phase-change material (PCM) component for non-volatile optical memory.
    
    Implements GST (Ge2Sb2Te5) or similar materials for creating
    non-volatile synaptic weights and optical memory elements.
    """
    
    def __init__(
        self,
        material: str = "GST",  # Ge2Sb2Te5
        length: float = 1e-6,  # 1 μm
        width: float = 500e-9,  # 500 nm
        thickness: float = 20e-9,  # 20 nm
        switching_energy: float = 100e-15,  # 100 fJ
        switching_time: float = 10e-9,  # 10 ns
        params: Optional[ComponentParameters] = None
    ):
        super().__init__(params or ComponentParameters())
        self.material = material
        self.length = length
        self.width = width
        self.thickness = thickness
        self.switching_energy = switching_energy
        self.switching_time = switching_time
        
        # Material properties database
        self.material_db = {
            "GST": {
                "n_amorphous": 4.0 + 0.5j,
                "n_crystalline": 6.0 + 2.0j,
                "switch_temp": 423,  # Kelvin
                "melt_temp": 873    # Kelvin
            },
            "GSST": {
                "n_amorphous": 3.8 + 0.3j,
                "n_crystalline": 5.5 + 1.5j,
                "switch_temp": 373,
                "melt_temp": 823
            }
        }
        
        # Current state
        self.crystalline_fraction = 0.0  # 0=amorphous, 1=crystalline
        self.accumulated_energy = 0.0
        self.last_switch_time = 0.0
    
    def transfer_function(self, wavelength: float, power: float) -> Tuple[float, float]:
        """Calculate PCM transmission based on current state."""
        material_props = self.material_db.get(self.material, self.material_db["GST"])
        
        # Interpolate optical constants based on crystalline fraction
        n_amorphous = material_props["n_amorphous"]
        n_crystalline = material_props["n_crystalline"]
        
        n_eff = (
            (1 - self.crystalline_fraction) * n_amorphous + 
            self.crystalline_fraction * n_crystalline
        )
        
        # Calculate transmission through thin film
        k = 2 * np.pi / wavelength
        phase_thickness = k * n_eff.real * self.thickness
        absorption = k * n_eff.imag * self.thickness
        
        transmission = np.exp(-absorption)
        phase = phase_thickness
        
        return transmission, phase
    
    def apply_pulse(self, pulse_energy: float, pulse_duration: float, current_time: float) -> bool:
        """Apply optical pulse to potentially switch the material state."""
        self.accumulated_energy += pulse_energy
        
        # Check if switching threshold is reached
        if self.accumulated_energy >= self.switching_energy:
            # Check if enough time has passed since last switch
            if current_time - self.last_switch_time >= self.switching_time:
                # Determine switching direction based on pulse characteristics
                if pulse_duration < 1e-9:  # Short pulse -> amorphization
                    self.crystalline_fraction = 0.0
                else:  # Long pulse -> crystallization
                    self.crystalline_fraction = 1.0
                
                self.accumulated_energy = 0.0
                self.last_switch_time = current_time
                
                self.logger.info(f"PCM switched to {'crystalline' if self.crystalline_fraction > 0.5 else 'amorphous'} state")
                return True
        
        # Energy decay
        self.accumulated_energy *= 0.95  # 5% decay per time step
        return False
    
    def get_weight_value(self) -> float:
        """Get current synaptic weight value (0-1 based on crystalline fraction)."""
        return self.crystalline_fraction
    
    def set_weight_value(self, weight: float, programming_energy: float = None) -> None:
        """Set synaptic weight by programming the PCM state."""
        if programming_energy is None:
            programming_energy = self.switching_energy * 2  # Ensure switching
        
        target_fraction = np.clip(weight, 0.0, 1.0)
        
        # Simple programming model
        if abs(target_fraction - self.crystalline_fraction) > 0.1:
            self.crystalline_fraction = target_fraction
            self.accumulated_energy = 0.0
            self.logger.debug(f"PCM programmed to weight {weight:.2f}")
    
    def get_spice_model(self) -> str:
        """Generate SPICE model for phase-change material."""
        return f"""
* Phase-Change Material SPICE Model
.subckt pcm optical_in optical_out control_pulse
* PCM parameters
.param length={self.length*1e6}u
.param switching_energy={self.switching_energy*1e15}f
.param crystalline_fraction={self.crystalline_fraction}

* Variable resistor model for PCM
R_pcm_amorphous optical_in n1 {{1e6*(1-crystalline_fraction)}}
R_pcm_crystalline n1 optical_out {{100*crystalline_fraction}}

* Switching control
B_switch crystalline_fraction 0 V=if(V(control_pulse)>{self.switching_energy*1e15}, 1-V(crystalline_fraction), V(crystalline_fraction))

* Capacitive coupling for optical signal
C_optical optical_in optical_out {1e-15 * self.length * 1e6}f

.ends pcm
"""


class WaveguideCrossing(PhotonicComponent):
    """
    Low-loss waveguide crossing for complex photonic routing.
    
    Implements optimized crossings with minimal crosstalk and loss
    for building large-scale photonic networks.
    """
    
    def __init__(
        self,
        crossing_angle: float = 90,  # degrees
        width1: float = 450e-9,      # 450 nm
        width2: float = 450e-9,      # 450 nm
        crossing_loss: float = 0.1,  # dB
        crosstalk: float = -30,      # dB
        params: Optional[ComponentParameters] = None
    ):
        super().__init__(params or ComponentParameters())
        self.crossing_angle = crossing_angle
        self.width1 = width1
        self.width2 = width2
        self.crossing_loss = crossing_loss
        self.crosstalk = crosstalk
        
        # Calculate scattering parameters
        self.through_transmission = 10**(-crossing_loss/10)
        self.crosstalk_coefficient = 10**(crosstalk/10)
    
    def transfer_function(self, wavelength: float, power: float) -> Tuple[float, float]:
        """Calculate crossing transfer function."""
        # Wavelength-dependent effects
        wavelength_detuning = (wavelength - self.params.wavelength) / self.params.wavelength
        wavelength_penalty = 0.1 * wavelength_detuning**2  # Quadratic penalty
        
        # Through transmission
        transmission = self.through_transmission * (1 - wavelength_penalty)
        
        # Phase shift (minimal for good crossing)
        phase = 0.1 * np.sin(2 * np.pi * wavelength / self.params.wavelength)
        
        return transmission, phase
    
    def calculate_crosstalk(self, input_power: float) -> float:
        """Calculate crosstalk power to orthogonal port."""
        return input_power * self.crosstalk_coefficient
    
    def get_spice_model(self) -> str:
        """Generate SPICE model for waveguide crossing."""
        return f"""
* Waveguide Crossing SPICE Model
.subckt wg_crossing in1 out1 in2 out2
* Crossing parameters
.param crossing_loss={self.crossing_loss}
.param crosstalk={self.crosstalk}

* Through paths
E_through1 out1 0 in1 0 {self.through_transmission}
E_through2 out2 0 in2 0 {self.through_transmission}

* Crosstalk paths
E_crosstalk12 out2 0 in1 0 {self.crosstalk_coefficient}
E_crosstalk21 out1 0 in2 0 {self.crosstalk_coefficient}

* Parasitic elements
C_parasitic1 in1 out1 {1e-18}f
C_parasitic2 in2 out2 {1e-18}f

.ends wg_crossing
"""


# Component factory functions
def create_high_performance_neuron(
    wavelength: float = 1550e-9,
    target_frequency: float = 1e9  # 1 GHz operation
) -> MachZehnderNeuron:
    """Create optimized Mach-Zehnder neuron for high-speed operation."""
    params = ComponentParameters(
        wavelength=wavelength,
        temperature=300.0,
        power=1e-3,  # 1 mW
        loss=0.05    # Low loss
    )
    
    return MachZehnderNeuron(
        arm_length=50e-6,  # Shorter for high speed
        phase_shifter_type="thermal",
        modulation_depth=0.95,
        threshold_power=1e-7,  # 100 nW threshold
        coupling_ratio=0.5,
        extinction_ratio=25,  # High extinction
        params=params
    )


def create_memory_synapse(
    wavelength: float = 1550e-9,
    memory_time: float = 1e-3  # 1 ms memory
) -> MicroringResonator:
    """Create microring resonator optimized for synaptic memory."""
    params = ComponentParameters(
        wavelength=wavelength,
        temperature=300.0,
        power=100e-6,  # 100 μW
        loss=0.2
    )
    
    # Calculate radius for desired memory time
    quality_factor = memory_time * 3e8 / wavelength / np.pi
    radius = wavelength / (2 * np.pi * 2.44)  # Single-mode condition
    
    return MicroringResonator(
        radius=radius,
        coupling_gap=150e-9,  # Tight coupling
        quality_factor=min(quality_factor, 50000),  # Practical limit
        group_index=4.5,
        params=params
    )


def create_nonvolatile_weight(material: str = "GST") -> PhaseChangeMaterial:
    """Create phase-change material for non-volatile synaptic weights."""
    params = ComponentParameters(
        wavelength=1550e-9,
        temperature=300.0,
        power=1e-6,  # 1 μW programming
        loss=1.0     # Higher loss acceptable for memory
    )
    
    return PhaseChangeMaterial(
        material=material,
        length=2e-6,      # 2 μm length
        width=500e-9,     # 500 nm width
        thickness=50e-9,   # 50 nm thickness
        switching_energy=50e-15,  # 50 fJ switching
        switching_time=5e-9,      # 5 ns switching
        params=params
    )


def create_component_library(
    wavelength: float = 1550e-9
) -> Dict[str, PhotonicComponent]:
    """Create a comprehensive library of photonic components."""
    library = {
        "mz_neuron": create_high_performance_neuron(wavelength),
        "memory_synapse": create_memory_synapse(wavelength),
        "nonvolatile_weight": create_nonvolatile_weight(),
        "pc_cavity": PhotonicCrystalCavity(params=ComponentParameters(wavelength=wavelength)),
        "crossing": WaveguideCrossing(params=ComponentParameters(wavelength=wavelength))
    }
    
    return library
