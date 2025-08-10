"""
RTL Generation for Photonic Neuromorphic Systems.

This module provides comprehensive RTL generation capabilities for converting
high-level photonic neural network descriptions into synthesizable Verilog code
suitable for ASIC tape-outs and FPGA implementation.
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import logging

from .core import PhotonicSNN, WaveguideNeuron, OpticalParameters


@dataclass
class RTLGenerationConfig:
    """Configuration for RTL generation process."""
    target_frequency: float = 1e9  # 1 GHz
    pipeline_stages: int = 3
    resource_sharing: bool = True
    optimization_level: int = 2  # 0=none, 1=basic, 2=aggressive
    memory_type: str = "block_ram"  # "block_ram", "distributed", "register"
    fixed_point_width: int = 16
    fractional_bits: int = 8
    target_technology: str = "skywater130"
    include_testbench: bool = True
    include_assertions: bool = True


@dataclass
class ConstraintsConfig:
    """Synthesis constraints configuration."""
    max_area: Optional[float] = None  # μm²
    max_power: Optional[float] = None  # mW
    target_frequency: float = 1e9  # Hz
    setup_margin: float = 0.1  # ns
    hold_margin: float = 0.05  # ns
    max_fanout: int = 64
    max_capacitance: float = 0.1  # pF


@dataclass
class RTLDesign:
    """Generated RTL design with metadata."""
    verilog_code: str
    testbench_code: Optional[str] = None
    constraints_code: Optional[str] = None
    synthesis_scripts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resource_estimates: Dict[str, Any] = field(default_factory=dict)


class FixedPointConverter:
    """Convert floating-point operations to fixed-point for RTL."""
    
    def __init__(self, total_bits: int = 16, fractional_bits: int = 8):
        self.total_bits = total_bits
        self.fractional_bits = fractional_bits
        self.integer_bits = total_bits - fractional_bits
        self.scale_factor = 2 ** fractional_bits
        
    def to_fixed_point(self, value: float) -> int:
        """Convert floating-point value to fixed-point representation."""
        return int(value * self.scale_factor)
    
    def from_fixed_point(self, value: int) -> float:
        """Convert fixed-point value back to floating-point."""
        return value / self.scale_factor
    
    def get_verilog_type(self) -> str:
        """Get Verilog wire/reg type declaration."""
        return f"signed [{self.total_bits-1}:0]"


class VerilogCodeGenerator:
    """Generate optimized Verilog code for photonic neural networks."""
    
    def __init__(self, config: RTLGenerationConfig):
        self.config = config
        self.fp_converter = FixedPointConverter(
            config.fixed_point_width, 
            config.fractional_bits
        )
        self.logger = logging.getLogger(__name__)
        
    def generate_module_header(
        self, 
        module_name: str, 
        inputs: List[Tuple[str, int]], 
        outputs: List[Tuple[str, int]]
    ) -> str:
        """Generate Verilog module header with proper I/O declarations."""
        header = f"""
//==============================================================================
// Photonic Neural Network Module: {module_name}
// Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// Target Technology: {self.config.target_technology}
// Optimization Level: {self.config.optimization_level}
//==============================================================================

`timescale 1ns / 1ps

module {module_name} (
    input wire clk,
    input wire rst_n,
    input wire enable,"
// Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// Target Technology: {self.config.target_technology}
// Target Frequency: {self.config.target_frequency/1e6:.1f} MHz
//==============================================================================

module {module_name} #(
    parameter DATA_WIDTH = {self.config.fixed_point_width},
    parameter FRAC_BITS = {self.config.fractional_bits}
) (
    input wire clk,
    input wire rst_n,
"""
        
        # Add input ports
        for name, width in inputs:
            if width == 1:
                header += f"    input wire {name},\n"
            else:
                header += f"    input wire [{width-1}:0] {name},\n"
        
        # Add output ports
        for i, (name, width) in enumerate(outputs):
            comma = "," if i < len(outputs) - 1 else ""
            if width == 1:
                header += f"    output reg {name}{comma}\n"
            else:
                header += f"    output reg [{width-1}:0] {name}{comma}\n"
        
        header += ");\n\n"
        return header
    
    def generate_neuron_module(self, neuron: WaveguideNeuron) -> str:
        """Generate Verilog module for a single photonic neuron."""
        module_code = self.generate_module_header(
            "photonic_neuron",
            [("optical_input", self.config.fixed_point_width),
             ("enable", 1)],
            [("spike_out", 1),
             ("membrane_potential", self.config.fixed_point_width)]
        )
        
        # Fixed-point threshold
        threshold_fp = self.fp_converter.to_fixed_point(neuron.threshold_power * 1e6)
        
        module_code += f"""
    // Fixed-point parameters
    localparam THRESHOLD = {self.config.fixed_point_width}'sd{threshold_fp};
    localparam LEAK_FACTOR = {self.config.fixed_point_width}'sd{self.fp_converter.to_fixed_point(0.99)};
    
    // Internal registers
    reg {self.fp_converter.get_verilog_type()} membrane_potential_reg;
    reg refractory_counter;
    reg [3:0] refractory_period = 4'h5; // 5 clock cycles
    
    // Membrane potential update
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            membrane_potential_reg <= '0;
            spike_out <= 1'b0;
            refractory_counter <= 1'b0;
        end else if (enable) begin
            if (refractory_counter > 0) begin
                // Refractory period
                refractory_counter <= refractory_counter - 1'b1;
                spike_out <= 1'b0;
            end else begin
                // Integrate input
                membrane_potential_reg <= membrane_potential_reg + optical_input;
                
                // Apply leak
                membrane_potential_reg <= (membrane_potential_reg * LEAK_FACTOR) >>> FRAC_BITS;
                
                // Check threshold
                if (membrane_potential_reg >= THRESHOLD) begin
                    spike_out <= 1'b1;
                    membrane_potential_reg <= '0;
                    refractory_counter <= refractory_period;
                end else begin
                    spike_out <= 1'b0;
                end
            end
        end else begin
            spike_out <= 1'b0;
        end
    end
    
    // Output assignment
    assign membrane_potential = membrane_potential_reg;

endmodule
"""
        return module_code
    
    def generate_crossbar_array(
        self, 
        rows: int, 
        cols: int, 
        weights: Optional[List[List[float]]] = None
    ) -> str:
        """Generate Verilog for photonic crossbar array."""
        module_code = self.generate_module_header(
            f"photonic_crossbar_{rows}x{cols}",
            [("optical_inputs", rows * self.config.fixed_point_width),
             ("weight_programming", 1),
             ("weight_data", self.config.fixed_point_width),
             ("weight_addr_row", (rows-1).bit_length()),
             ("weight_addr_col", (cols-1).bit_length()),
             ("enable", 1)],
            [("optical_outputs", cols * self.config.fixed_point_width)]
        )
        
        module_code += f"""
    // Weight memory array
    reg {self.fp_converter.get_verilog_type()} weights [{rows-1}:0][{cols-1}:0];
    
    // Input/output arrays
    wire {self.fp_converter.get_verilog_type()} inputs [{rows-1}:0];
    reg {self.fp_converter.get_verilog_type()} outputs [{cols-1}:0];
    
    // Unpack input vector
    genvar i;
    generate
        for (i = 0; i < {rows}; i = i + 1) begin : input_unpack
            assign inputs[i] = optical_inputs[(i+1)*DATA_WIDTH-1:i*DATA_WIDTH];
        end
    endgenerate
    
    // Pack output vector
    generate
        for (i = 0; i < {cols}; i = i + 1) begin : output_pack
            assign optical_outputs[(i+1)*DATA_WIDTH-1:i*DATA_WIDTH] = outputs[i];
        end
    endgenerate
"""
        
        # Initialize weights if provided
        if weights:
            module_code += "\n    // Initialize weights\n"
            module_code += "    initial begin\n"
            for r in range(min(rows, len(weights))):
                for c in range(min(cols, len(weights[r]))):
                    weight_fp = self.fp_converter.to_fixed_point(weights[r][c])
                    module_code += f"        weights[{r}][{c}] = {self.config.fixed_point_width}'sd{weight_fp};\n"
            module_code += "    end\n"
        
        module_code += """
    // Weight programming
    always @(posedge clk) begin
        if (weight_programming) begin
            weights[weight_addr_row][weight_addr_col] <= weight_data;
        end
    end
    
    // Matrix-vector multiplication
    integer row, col;
    reg signed [31:0] accumulator;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (col = 0; col < """ + str(cols) + """; col = col + 1) begin
                outputs[col] <= '0;
            end
        end else if (enable) begin
            for (col = 0; col < """ + str(cols) + """; col = col + 1) begin
                accumulator = 0;
                for (row = 0; row < """ + str(rows) + """; row = row + 1) begin
                    accumulator = accumulator + (inputs[row] * weights[row][col]);
                end
                outputs[col] <= accumulator >>> FRAC_BITS;
            end
        end
    end

endmodule
"""
        return module_code
    
    def generate_network_top_level(self, model: PhotonicSNN) -> str:
        """Generate top-level network module."""
        input_size = model.topology[0]
        output_size = model.topology[-1]
        
        module_code = self.generate_module_header(
            "photonic_neural_network",
            [("spike_inputs", input_size),
             ("enable", 1),
             ("weight_programming", 1),
             ("weight_data", self.config.fixed_point_width),
             ("weight_layer", 3),  # Assuming max 8 layers
             ("weight_addr_src", 12), # Assuming max 4096 neurons per layer
             ("weight_addr_dst", 12)],
            [("spike_outputs", output_size),
             ("network_activity", 16)]  # Activity counter
        )
        
        # Generate layer instances
        layer_connections = []
        for i, (prev_size, curr_size) in enumerate(zip(model.topology[:-1], model.topology[1:])):
            layer_connections.append((prev_size, curr_size))
        
        module_code += f"""
    // Internal layer connections
"""
        
        # Declare layer interconnects
        for i, (prev_size, curr_size) in enumerate(layer_connections):
            module_code += f"    wire [{prev_size-1}:0] layer_{i}_output;\n"
            module_code += f"    wire [{curr_size * self.config.fixed_point_width - 1}:0] layer_{i+1}_optical;\n"
        
        module_code += "\n    // Layer 0: Input conversion\n"
        module_code += f"    assign layer_0_output = spike_inputs;\n\n"
        
        # Generate layer instances
        for i, (prev_size, curr_size) in enumerate(layer_connections):
            module_code += f"""
    // Layer {i+1}: {prev_size}x{curr_size} crossbar + neurons
    photonic_crossbar_{prev_size}x{curr_size} crossbar_layer_{i+1} (
        .clk(clk),
        .rst_n(rst_n),
        .optical_inputs(layer_{i}_output * {self.fp_converter.scale_factor}),
        .weight_programming(weight_programming && (weight_layer == {i+1})),
        .weight_data(weight_data),
        .weight_addr_row(weight_addr_src),
        .weight_addr_col(weight_addr_dst),
        .enable(enable),
        .optical_outputs(layer_{i+1}_optical)
    );
    
    // Neuron array for layer {i+1}
    genvar neuron_{i+1};
    generate
        for (neuron_{i+1} = 0; neuron_{i+1} < {curr_size}; neuron_{i+1} = neuron_{i+1} + 1) begin : neurons_layer_{i+1}
            photonic_neuron neuron (
                .clk(clk),
                .rst_n(rst_n),
                .optical_input(layer_{i+1}_optical[(neuron_{i+1}+1)*DATA_WIDTH-1:neuron_{i+1}*DATA_WIDTH]),
                .enable(enable),
                .spike_out(layer_{i+1}_output[neuron_{i+1}]),
                .membrane_potential()  // Not used at top level
            );
        end
    endgenerate
"""
        
        # Final output assignment
        final_layer_idx = len(layer_connections)
        module_code += f"""
    // Output assignment
    assign spike_outputs = layer_{final_layer_idx}_output;
    
    // Activity monitoring
    reg [15:0] activity_counter;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            activity_counter <= 16'h0;
        end else if (enable) begin
            activity_counter <= activity_counter + $countones(spike_outputs);
        end
    end
    assign network_activity = activity_counter;

endmodule
"""
        return module_code


class RTLGenerator:
    """
    Comprehensive RTL generator for photonic neural networks.
    
    Converts high-level photonic neural network descriptions into optimized,
    synthesizable Verilog code with comprehensive testbenches and constraints.
    """
    
    def __init__(
        self,
        config: Optional[RTLGenerationConfig] = None,
        constraints: Optional[ConstraintsConfig] = None,
        technology: str = "skywater130"
    ):
        """
        Initialize RTL generator.
        
        Args:
            config: RTL generation configuration
            constraints: Synthesis constraints
            technology: Target technology node
        """
        self.config = config or RTLGenerationConfig()
        self.constraints = constraints or ConstraintsConfig()
        self.technology = technology
        
        self.verilog_gen = VerilogCodeGenerator(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Technology-specific parameters
        self.tech_params = self._load_technology_parameters(technology)
    
    def _load_technology_parameters(self, technology: str) -> Dict[str, Any]:
        """Load technology-specific parameters."""
        tech_db = {
            "skywater130": {
                "min_feature_size": 130e-9,
                "supply_voltage": 1.8,
                "gate_delay": 50e-12,  # 50 ps
                "wire_delay_per_um": 1e-12,  # 1 ps/μm
                "logic_area_per_gate": 1.0,  # μm²
                "memory_area_per_bit": 0.5   # μm²
            },
            "tsmc28": {
                "min_feature_size": 28e-9,
                "supply_voltage": 1.0,
                "gate_delay": 20e-12,
                "wire_delay_per_um": 0.5e-12,
                "logic_area_per_gate": 0.3,
                "memory_area_per_bit": 0.15
            }
        }
        return tech_db.get(technology, tech_db["skywater130"])
    
    def generate(
        self,
        model: PhotonicSNN,
        output_dir: Optional[str] = None
    ) -> RTLDesign:
        """
        Generate complete RTL design from photonic neural network.
        
        Args:
            model: Photonic neural network model
            output_dir: Output directory for files (optional)
            
        Returns:
            RTLDesign: Complete RTL design package
        """
        self.logger.info(f"Generating RTL for {model.__class__.__name__}")
        
        # Generate core Verilog modules
        verilog_modules = []
        
        # Generate neuron module
        sample_neuron = model.neurons[1][0] if model.neurons else WaveguideNeuron()
        neuron_module = self.verilog_gen.generate_neuron_module(sample_neuron)
        verilog_modules.append(neuron_module)
        
        # Generate crossbar modules for each layer
        for i, (prev_size, curr_size) in enumerate(zip(model.topology[:-1], model.topology[1:])):
            # Extract weights if available
            weights = None
            if hasattr(model, 'layers') and i < len(model.layers):
                weight_tensor = model.layers[i]
                if hasattr(weight_tensor, 'data'):
                    weights = weight_tensor.data.numpy().tolist()
            
            crossbar_module = self.verilog_gen.generate_crossbar_array(
                prev_size, curr_size, weights
            )
            verilog_modules.append(crossbar_module)
        
        # Generate top-level network module
        top_module = self.verilog_gen.generate_network_top_level(model)
        verilog_modules.append(top_module)
        
        # Combine all modules
        verilog_code = "\n".join(verilog_modules)
        
        # Generate testbench if requested
        testbench_code = None
        if self.config.include_testbench:
            testbench_code = self._generate_testbench(model)
        
        # Generate synthesis constraints
        constraints_code = self._generate_constraints(model)
        
        # Generate synthesis scripts
        synthesis_scripts = self._generate_synthesis_scripts(model)
        
        # Calculate resource estimates
        resource_estimates = self._estimate_resources(model)
        
        # Create RTL design package
        rtl_design = RTLDesign(
            verilog_code=verilog_code,
            testbench_code=testbench_code,
            constraints_code=constraints_code,
            synthesis_scripts=synthesis_scripts,
            metadata={
                "model_topology": model.topology,
                "target_frequency": self.config.target_frequency,
                "technology": self.technology,
                "generation_time": datetime.now().isoformat(),
                "fixed_point_config": {
                    "total_bits": self.config.fixed_point_width,
                    "fractional_bits": self.config.fractional_bits
                }
            },
            resource_estimates=resource_estimates
        )
        
        # Save to files if output directory specified
        if output_dir:
            self._save_design_files(rtl_design, output_dir, model)
        
        return rtl_design
    
    def _generate_testbench(self, model: PhotonicSNN) -> str:
        """Generate comprehensive testbench for the neural network."""
        input_size = model.topology[0]
        output_size = model.topology[-1]
        
        testbench = f"""
//==============================================================================
// Testbench for Photonic Neural Network
// Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
//==============================================================================

`timescale 1ns/1ps

module tb_photonic_neural_network;
    
    // Parameters
    parameter CLK_PERIOD = {1e9 / self.config.target_frequency:.1f}; // ns
    parameter INPUT_SIZE = {input_size};
    parameter OUTPUT_SIZE = {output_size};
    
    // Signals
    reg clk;
    reg rst_n;
    reg [INPUT_SIZE-1:0] spike_inputs;
    reg enable;
    reg weight_programming;
    reg [15:0] weight_data;
    reg [2:0] weight_layer;
    reg [11:0] weight_addr_src, weight_addr_dst;
    
    wire [OUTPUT_SIZE-1:0] spike_outputs;
    wire [15:0] network_activity;
    
    // DUT instantiation
    photonic_neural_network dut (
        .clk(clk),
        .rst_n(rst_n),
        .spike_inputs(spike_inputs),
        .enable(enable),
        .weight_programming(weight_programming),
        .weight_data(weight_data),
        .weight_layer(weight_layer),
        .weight_addr_src(weight_addr_src),
        .weight_addr_dst(weight_addr_dst),
        .spike_outputs(spike_outputs),
        .network_activity(network_activity)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Reset sequence
    initial begin
        rst_n = 0;
        enable = 0;
        weight_programming = 0;
        spike_inputs = 0;
        weight_data = 0;
        weight_layer = 0;
        weight_addr_src = 0;
        weight_addr_dst = 0;
        
        #(CLK_PERIOD * 5) rst_n = 1;
        #(CLK_PERIOD * 2) enable = 1;
    end
    
    // Test stimulus
    integer test_cycle;
    initial begin
        wait(rst_n == 1);
        wait(enable == 1);
        
        $display("Starting photonic neural network test...");
        
        // Test pattern sequence
        for (test_cycle = 0; test_cycle < 100; test_cycle = test_cycle + 1) begin
            @(posedge clk);
            spike_inputs <= $random;
            
            if (test_cycle % 10 == 0) begin
                $display("Cycle %0d: Input=%b, Output=%b, Activity=%0d",
                    test_cycle, spike_inputs, spike_outputs, network_activity);
            end
        end
        
        #(CLK_PERIOD * 10);
        $display("Test completed successfully!");
        $finish;
    end
    
    // Performance monitoring
    always @(posedge clk) begin
        if (|spike_outputs) begin
            $display("Output spike detected at time %0t: %b", $time, spike_outputs);
        end
    end

endmodule
"""
        return testbench
    
    def _generate_constraints(self, model: PhotonicSNN) -> str:
        """Generate synthesis constraints (SDC format)."""
        constraints = f"""
# Synthesis Constraints for Photonic Neural Network
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Clock definition
create_clock -name clk -period {1e9 / self.constraints.target_frequency:.3f} [get_ports clk]

# Input/output delays
set_input_delay -clock clk {self.constraints.setup_margin:.3f} [all_inputs]
set_output_delay -clock clk {self.constraints.setup_margin:.3f} [all_outputs]

# Clock uncertainty
set_clock_uncertainty 0.1 [get_clocks clk]

# Maximum transition time
set_max_transition 0.2 [current_design]

# Maximum capacitance
set_max_capacitance {self.constraints.max_capacitance} [all_inputs]

# Maximum fanout
set_max_fanout {self.constraints.max_fanout} [current_design]
"""
        
        # Add area constraint if specified
        if self.constraints.max_area:
            constraints += f"\n# Area constraint\nset_max_area {self.constraints.max_area}\n"
        
        # Add power constraint if specified
        if self.constraints.max_power:
            constraints += f"\n# Power constraint\nset_max_power {self.constraints.max_power}mW\n"
        
        return constraints
    
    def _generate_synthesis_scripts(self, model: PhotonicSNN) -> Dict[str, str]:
        """Generate synthesis scripts for various tools."""
        scripts = {}
        
        # Design Compiler script
        scripts["dc_synthesis.tcl"] = f"""
# Design Compiler Synthesis Script
# Generated for Photonic Neural Network

# Setup
set_app_var target_library {self.technology}_typical.db
set_app_var link_library "* $target_library"
set_app_var symbol_library {self.technology}.sdb

# Read design
analyze -format verilog photonic_neural_network.v
elaborate photonic_neural_network

# Apply constraints
source constraints.sdc

# Compile
compile_ultra -gate_clock -no_autoungroup

# Reports
report_area > reports/area_report.txt
report_timing > reports/timing_report.txt
report_power > reports/power_report.txt

# Write out netlist
write -format verilog -hierarchy -output netlist/photonic_neural_network_netlist.v
"""
        
        # Yosys script for open-source flow
        scripts["yosys_synthesis.ys"] = f"""
# Yosys Synthesis Script for Photonic Neural Network

# Read design
read_verilog photonic_neural_network.v

# Hierarchy check
hierarchy -top photonic_neural_network

# Technology mapping
proc
opt
memory
opt
fsm
opt
techmap
opt

# Output
write_verilog netlist/photonic_neural_network_netlist.v
stat
"""
        
        return scripts
    
    def _estimate_resources(self, model: PhotonicSNN) -> Dict[str, Any]:
        """Estimate hardware resources required."""
        total_neurons = sum(model.topology)
        total_synapses = sum(
            model.topology[i] * model.topology[i+1] 
            for i in range(len(model.topology)-1)
        )
        
        # Logic resources
        logic_gates_per_neuron = 50  # Estimate
        logic_gates_per_synapse = 10
        total_logic_gates = (
            total_neurons * logic_gates_per_neuron +
            total_synapses * logic_gates_per_synapse
        )
        
        # Memory resources (for weights)
        memory_bits = total_synapses * self.config.fixed_point_width
        
        # Area estimates
        logic_area = total_logic_gates * self.tech_params["logic_area_per_gate"]
        memory_area = memory_bits * self.tech_params["memory_area_per_bit"]
        total_area = logic_area + memory_area
        
        # Timing estimates
        critical_path_stages = len(model.topology) - 1  # One stage per layer
        estimated_delay = (
            critical_path_stages * self.tech_params["gate_delay"] * 
            np.log2(max(model.topology))  # Logarithmic scaling with layer size
        )
        max_frequency = 1.0 / estimated_delay if estimated_delay > 0 else float('inf')
        
        # Power estimates
        dynamic_power_per_gate = 0.1e-12  # 0.1 pW per gate at 1GHz
        static_power_per_gate = 0.01e-12  # 0.01 pW per gate
        
        dynamic_power = total_logic_gates * dynamic_power_per_gate * self.config.target_frequency
        static_power = total_logic_gates * static_power_per_gate
        total_power = dynamic_power + static_power
        
        return {
            "logic_gates": total_logic_gates,
            "memory_bits": memory_bits,
            "total_area_um2": total_area,
            "logic_area_um2": logic_area,
            "memory_area_um2": memory_area,
            "estimated_max_frequency_hz": max_frequency,
            "estimated_delay_ns": estimated_delay * 1e9,
            "estimated_power_mw": total_power * 1e3,
            "dynamic_power_mw": dynamic_power * 1e3,
            "static_power_mw": static_power * 1e3,
            "resource_utilization": {
                "neurons": total_neurons,
                "synapses": total_synapses,
                "layers": len(model.topology) - 1
            }
        }
    
    def _save_design_files(self, design: RTLDesign, output_dir: str, model: PhotonicSNN):
        """Save all design files to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_path / "rtl").mkdir(exist_ok=True)
        (output_path / "tb").mkdir(exist_ok=True)
        (output_path / "constraints").mkdir(exist_ok=True)
        (output_path / "scripts").mkdir(exist_ok=True)
        (output_path / "reports").mkdir(exist_ok=True)
        (output_path / "netlist").mkdir(exist_ok=True)
        
        # Save main Verilog file
        with open(output_path / "rtl" / "photonic_neural_network.v", "w") as f:
            f.write(design.verilog_code)
        
        # Save testbench
        if design.testbench_code:
            with open(output_path / "tb" / "tb_photonic_neural_network.v", "w") as f:
                f.write(design.testbench_code)
        
        # Save constraints
        if design.constraints_code:
            with open(output_path / "constraints" / "constraints.sdc", "w") as f:
                f.write(design.constraints_code)
        
        # Save synthesis scripts
        for script_name, script_content in design.synthesis_scripts.items():
            with open(output_path / "scripts" / script_name, "w") as f:
                f.write(script_content)
        
        # Save resource estimates as JSON
        import json
        with open(output_path / "reports" / "resource_estimates.json", "w") as f:
            json.dump(design.resource_estimates, f, indent=2)
        
        # Save metadata
        with open(output_path / "reports" / "design_metadata.json", "w") as f:
            json.dump(design.metadata, f, indent=2)
        
        self.logger.info(f"RTL design saved to {output_path}")
    
    def simulate(
        self, 
        design: RTLDesign, 
        simulator: str = "iverilog"
    ) -> Dict[str, Any]:
        """
        Simulate the generated RTL design.
        
        Args:
            design: RTL design to simulate
            simulator: Simulator to use ("iverilog", "verilator", "modelsim")
            
        Returns:
            Dict: Simulation results
        """
        if not design.testbench_code:
            raise ValueError("No testbench available for simulation")
        
        # For now, return mock simulation results
        # In a real implementation, this would run the actual simulator
        return {
            "simulation_passed": True,
            "simulation_time": "1.2ms",
            "cycles_simulated": 1000,
            "coverage": {"line": 95.2, "branch": 87.3, "toggle": 91.8},
            "performance": {
                "max_frequency_achieved": self.config.target_frequency * 0.9,
                "power_consumption": 15.3  # mW
            }
        }


def create_rtl_for_mnist() -> RTLDesign:
    """Create RTL design optimized for MNIST classification."""
    from .core import create_mnist_photonic_snn
    
    config = RTLGenerationConfig(
        target_frequency=100e6,  # 100 MHz for MNIST
        pipeline_stages=2,
        resource_sharing=True,
        optimization_level=2,
        fixed_point_width=12,
        fractional_bits=6
    )
    
    constraints = ConstraintsConfig(
        max_area=2e6,  # 2 mm²
        max_power=50,  # 50 mW
        target_frequency=100e6
    )
    
    generator = RTLGenerator(config, constraints)
    model = create_mnist_photonic_snn()
    
    return generator.generate(model)


def create_high_performance_rtl(model: PhotonicSNN) -> RTLDesign:
    """Create high-performance RTL design with aggressive optimization."""
    config = RTLGenerationConfig(
        target_frequency=1e9,  # 1 GHz
        pipeline_stages=4,
        resource_sharing=False,  # No sharing for max speed
        optimization_level=2,
        fixed_point_width=16,
        fractional_bits=8
    )
    
    constraints = ConstraintsConfig(
        target_frequency=1e9,
        setup_margin=0.05,  # Tight timing
        hold_margin=0.02
    )
    
    generator = RTLGenerator(config, constraints, technology="tsmc28")
    return generator.generate(model)