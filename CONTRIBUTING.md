# Contributing to Photonic Neuromorphics Sim

We welcome contributions to the photonic neuromorphics simulation framework! This guide will help you get started.

## 🚀 Quick Start

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/photonic-neuromorphics-sim.git
   cd photonic-neuromorphics-sim
   ```

2. **Set up development environment**
   ```bash
   make dev-setup
   # or manually:
   pip install -e ".[dev,docs,test]"
   pre-commit install
   ```

3. **Run tests to verify setup**
   ```bash
   make test
   ```

## 📋 Development Workflow

### Creating a Branch
```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b fix/issue-description
```

### Making Changes
1. Write code following our style guidelines
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass: `make test`
5. Run linting: `make lint`

### Submitting Changes
1. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add photonic reservoir computing support"
   ```

2. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request**
   - Use our PR template
   - Include comprehensive description
   - Link to relevant issues

## 🎯 Priority Areas

We especially welcome contributions in:

- **Photonic neuron models**: New architectures and transfer functions
- **Routing algorithms**: Optimized optical interconnect design
- **Noise modeling**: Realistic photonic device behavior
- **PDK support**: Additional process design kits
- **Benchmarks**: Neuromorphic task implementations
- **Documentation**: Tutorials and examples

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Cross-component tests
└── fixtures/       # Test data and mocks
```

### Writing Tests
- Use descriptive test names: `test_mach_zehnder_transfer_function_linearity`
- Include docstrings explaining test purpose
- Use fixtures for common test data
- Mark slow tests: `@pytest.mark.slow`

### Running Tests
```bash
make test              # All tests
make test-cov         # With coverage
pytest -m "not slow"  # Skip slow tests
pytest tests/unit/    # Unit tests only
```

## 📝 Code Style

### Python Style
- **Line length**: 88 characters (Black default)
- **Import order**: `isort` compatible
- **Type hints**: Required for public APIs
- **Docstrings**: Google style

### Example
```python
def create_waveguide_neuron(
    arm_length: float,
    wavelength: float = 1550e-9,
    threshold_power: float = 1e-6,
) -> WaveguideNeuron:
    """Create a Mach-Zehnder interferometer neuron.
    
    Args:
        arm_length: Length of MZI arms in meters
        wavelength: Operating wavelength in meters
        threshold_power: Spike threshold in watts
        
    Returns:
        Configured waveguide neuron instance
        
    Raises:
        ValueError: If parameters are outside valid ranges
    """
    if arm_length <= 0:
        raise ValueError("Arm length must be positive")
    
    return WaveguideNeuron(
        arm_length=arm_length,
        wavelength=wavelength,
        threshold_power=threshold_power,
    )
```

### Verilog Style
- **Indentation**: 2 spaces
- **Naming**: `snake_case` for signals, `CamelCase` for modules
- **Comments**: Explain optical behavior and timing

## 🏗️ Architecture Guidelines

### Module Organization
```python
photonic_neuromorphics/
├── core/           # Basic neuron and network classes
├── components/     # Optical component library
├── simulator/      # SPICE and optical simulation
├── rtl/           # RTL generation and verification
├── benchmarks/    # Standard neuromorphic tasks
└── utils/         # Helper functions and tools
```

### Adding New Components
1. **Create component class**
   ```python
   class MyPhotonicComponent(OpticalComponent):
       def __init__(self, **params):
           super().__init__()
           self.validate_parameters(params)
           
       def get_transfer_function(self) -> callable:
           # Return optical transfer function
           pass
           
       def to_spice(self) -> str:
           # Generate SPICE model
           pass
   ```

2. **Add comprehensive tests**
3. **Update documentation**
4. **Add example usage**

## 📚 Documentation

### Building Docs
```bash
make docs          # Build static docs
make docs-live     # Live reload server
```

### Documentation Types
- **API Reference**: Auto-generated from docstrings
- **Tutorials**: Step-by-step guides in `docs/tutorials/`
- **Examples**: Jupyter notebooks in `examples/`
- **Architecture**: Design decisions in `docs/architecture/`

## 🐛 Bug Reports

### Before Reporting
1. Search existing issues
2. Try latest version
3. Reproduce with minimal example

### Bug Report Template
```markdown
## Bug Description
Brief description of the issue

## Reproduction Steps
1. Step 1
2. Step 2
3. Expected vs actual behavior

## Environment
- OS: 
- Python version:
- Package version:
- Dependencies:

## Minimal Example
```python
# Code that reproduces the issue
```

## 💡 Feature Requests

### Feature Request Template
```markdown
## Feature Description
What functionality would you like to see?

## Use Case
Why is this feature needed?

## Proposed API
How would users interact with this feature?

## Implementation Notes
Any technical considerations or constraints?
```

## 🔍 Code Review Process

### For Contributors
- Keep PRs focused and reasonably sized
- Respond to feedback promptly
- Update documentation and tests

### Review Criteria
- **Functionality**: Does it work correctly?
- **Tests**: Adequate test coverage?
- **Documentation**: Clear and up-to-date?
- **Style**: Follows project conventions?
- **Performance**: No significant regressions?

## 📄 Commit Guidelines

### Commit Message Format
```
type(scope): description

[optional body]

[optional footer]
```

### Types
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `test`: Test additions/modifications
- `refactor`: Code restructuring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### Examples
```bash
feat(components): add microring resonator model
fix(simulator): correct phase noise calculation
docs(tutorials): add RTL generation guide
test(integration): add SPICE co-simulation tests
```

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Invited to join the development team (active contributors)

## 📞 Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Report bugs and request features
- **Email**: Contact maintainers directly for sensitive issues

Thank you for contributing to photonic neuromorphics research! 🧠✨