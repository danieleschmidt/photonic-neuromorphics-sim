# ADR-0001: Photonic Simulation Framework Architecture

## Status
Accepted

## Context
The photonic neuromorphics simulation requires a unified framework that can handle:
- High-level neural network models (PyTorch/TensorFlow)
- Photonic component modeling and simulation
- Optical-electrical co-simulation
- RTL generation for tape-out

Traditional electronic EDA tools lack photonic simulation capabilities, while photonic simulators don't integrate well with neuromorphic algorithms.

## Decision
Implement a Python-based simulation framework with the following architecture:
1. **Frontend**: PyTorch integration for neural network definition
2. **Transpiler**: Automatic conversion to photonic implementations
3. **Component Library**: Validated photonic building blocks
4. **Co-Simulation**: Unified optical-electrical simulation
5. **Backend**: RTL generation and layout export

## Consequences

### Positive
- Unified design flow from algorithm to silicon
- Reusable component library for rapid prototyping
- Automatic optimization and layout generation
- Integration with existing ML frameworks
- Support for multiple PDKs (SiEPIC, SkyWater)

### Negative
- Complex simulation engine requiring multiple domains
- Performance overhead from Python-based implementation
- Learning curve for users familiar with traditional EDA tools
- Dependency on external simulation engines (SPICE, optical solvers)

### Neutral
- Framework requires significant initial development effort
- Need for comprehensive validation against experimental data

## Implementation
1. Core framework in Python with NumPy/SciPy for numerical operations
2. Component library using object-oriented design patterns
3. SPICE integration through PySpice for electrical simulation
4. Optical simulation through custom solvers and third-party tools
5. RTL generation using Mako templates and Verilog AST manipulation

## Alternatives Considered
- **MATLAB-based framework**: Rejected due to licensing costs and limited extensibility
- **C++ implementation**: Rejected due to development complexity and Python ecosystem integration
- **Existing photonic simulators (Lumerical, Ansys)**: Rejected due to cost and limited neuromorphic support
- **Pure HDL approach**: Rejected as it lacks high-level algorithm support

## References
- [Photonic Neural Networks: A Survey](https://example.com/photonic-survey)
- [Silicon Photonic Integration for Neuromorphics](https://example.com/silicon-photonic-neuro)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [SiEPIC PDK](https://github.com/SiEPIC/SiEPIC_EBeam_PDK)