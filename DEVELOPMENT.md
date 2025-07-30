# Development Guide

This guide covers setting up your development environment and contributing to the photonic neuromorphics simulation framework.

## 🚀 Quick Setup

### Prerequisites
- Python 3.9+ (3.11 recommended)
- Git
- Make (optional, for convenience commands)

### 1. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/photonic-neuromorphics-sim.git
cd photonic-neuromorphics-sim

# Set up development environment
make dev-setup
# or manually:
pip install -e ".[dev,docs,test]"
pre-commit install
```

### 2. Verify Installation
```bash
# Run tests to ensure everything works
make test

# Check code quality
make lint

# Build documentation
make docs
```

## 🛠️ Development Environment

### Recommended IDE Setup

#### VS Code
Install these extensions:
- Python
- Pylance
- Black Formatter
- Ruff
- Test Explorer UI

#### PyCharm
Configure:
- Python interpreter: Use the virtual environment
- Code style: Black (88 characters)
- Type checker: MyPy
- Test runner: pytest

### Environment Variables
Create a `.env` file for development:
```bash
# Simulation settings
PHOTONIC_SIMULATOR_DEBUG=true
SPICE_SIMULATOR_PATH=/usr/local/bin/ngspice
OPENLANE_ROOT=/opt/openlane

# Testing
PYTEST_ADDOPTS="--cov=photonic_neuromorphics --cov-report=term-missing"

# Documentation
DOCS_PORT=8000
```

## 📁 Project Structure

```
photonic-neuromorphics-sim/
├── src/
│   └── photonic_neuromorphics/    # Main package
│       ├── core/                  # Core classes and algorithms
│       ├── components/            # Optical component library
│       ├── simulator/             # Simulation engines
│       ├── rtl/                   # RTL generation
│       ├── benchmarks/            # Standard tasks
│       └── utils/                 # Helper functions
├── tests/
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── fixtures/                  # Test data
├── docs/                          # Documentation
│   ├── api/                       # API reference
│   ├── tutorials/                 # Step-by-step guides
│   └── examples/                  # Jupyter notebooks
├── scripts/                       # Development utilities
└── examples/                      # Usage examples
```

## 🧪 Testing Strategy

### Test Categories
- **Unit Tests**: Fast, isolated component testing
- **Integration Tests**: Cross-component functionality
- **Slow Tests**: Computationally intensive simulations
- **Property Tests**: Hypothesis-based testing

### Running Tests
```bash
# All tests
make test

# Unit tests only
pytest tests/unit/

# Skip slow tests
pytest -m "not slow"

# With coverage
make test-cov

# Specific test file
pytest tests/unit/test_core.py -v

# Run tests in parallel
pytest -n auto
```

### Writing Tests
```python
import pytest
from photonic_neuromorphics import WaveguideNeuron

class TestWaveguideNeuron:
    def test_initialization(self):
        """Test basic neuron initialization."""
        neuron = WaveguideNeuron(
            arm_length=100e-6,
            wavelength=1550e-9
        )
        assert neuron.arm_length == 100e-6
    
    @pytest.mark.slow
    def test_spice_simulation(self):
        """Test SPICE co-simulation (slow)."""
        # Implementation
        pass
    
    @pytest.mark.parametrize("wavelength,expected", [
        (1550e-9, True),
        (1310e-9, True),
        (800e-9, False),  # Outside C-band
    ])
    def test_wavelength_validation(self, wavelength, expected):
        """Test wavelength parameter validation."""
        # Implementation
        pass
```

## 🎨 Code Style Guidelines

### Python Style
Follow PEP 8 with these specifics:
- **Line length**: 88 characters (Black default)
- **Quotes**: Double quotes preferred
- **Imports**: Absolute imports, grouped by standard/third-party/local
- **Type hints**: Required for public APIs

### Example Code Style
```python
from typing import Optional, Tuple
import numpy as np
from .base import OpticalComponent

class MachZehnderNeuron(OpticalComponent):
    """Mach-Zehnder interferometer-based photonic neuron.
    
    This neuron uses optical interference to implement
    nonlinear activation functions suitable for spiking
    neural networks.
    
    Args:
        arm_length: Length of MZI arms in meters
        wavelength: Operating wavelength in meters
        threshold_power: Spike threshold in watts
        
    Example:
        >>> neuron = MachZehnderNeuron(
        ...     arm_length=100e-6,
        ...     wavelength=1550e-9,
        ...     threshold_power=1e-6
        ... )
        >>> transfer_func = neuron.get_transfer_function()
    """
    
    def __init__(
        self,
        arm_length: float,
        wavelength: float = 1550e-9,
        threshold_power: float = 1e-6,
    ) -> None:
        super().__init__()
        self._validate_parameters(arm_length, wavelength, threshold_power)
        
        self.arm_length = arm_length
        self.wavelength = wavelength
        self.threshold_power = threshold_power
    
    def get_transfer_function(self) -> callable:
        """Get the optical transfer function for this neuron."""
        def transfer(input_power: float, phase_shift: float) -> float:
            # Physics-based transfer function
            interference = np.cos(phase_shift) ** 2
            return input_power * interference
        
        return transfer
    
    def _validate_parameters(
        self, 
        arm_length: float, 
        wavelength: float, 
        threshold_power: float
    ) -> None:
        """Validate physical parameters."""
        if arm_length <= 0:
            raise ValueError("Arm length must be positive")
        
        if not (1300e-9 <= wavelength <= 1600e-9):
            raise ValueError("Wavelength must be in telecom bands")
        
        if threshold_power <= 0:
            raise ValueError("Threshold power must be positive")
```

### Documentation Style
Use Google-style docstrings:

```python
def create_crossbar_network(
    rows: int,
    cols: int,
    weight_matrix: np.ndarray,
    wavelength: float = 1550e-9,
) -> PhotonicCrossbar:
    """Create a photonic crossbar network.
    
    Args:
        rows: Number of input waveguides
        cols: Number of output waveguides  
        weight_matrix: Synaptic weight matrix (rows x cols)
        wavelength: Operating wavelength in meters
        
    Returns:
        Configured photonic crossbar instance
        
    Raises:
        ValueError: If weight matrix dimensions don't match rows/cols
        
    Example:
        >>> weights = np.random.randn(64, 64)
        >>> crossbar = create_crossbar_network(64, 64, weights)
        >>> layout = crossbar.generate_layout()
    """
    # Implementation
    pass
```

## 🔧 Development Tools

### Code Formatting
```bash
# Format code with Black
black src tests

# Sort imports
ruff check --fix src tests

# Check formatting without changes
black --check src tests
```

### Type Checking
```bash
# Run MyPy type checker
mypy src

# Check specific file
mypy src/photonic_neuromorphics/core.py
```

### Linting
```bash
# Run all linting checks
make lint

# Ruff linting only
ruff check src tests

# Check specific rules
ruff check --select E,W,F src
```

## 📚 Documentation

### Building Documentation
```bash
# Build HTML documentation
make docs

# Serve docs with live reload
make docs-live

# Clean documentation build
rm -rf docs/_build/
```

### Adding Documentation
1. **API Documentation**: Add comprehensive docstrings
2. **Tutorials**: Create `.md` files in `docs/tutorials/`
3. **Examples**: Add Jupyter notebooks in `examples/`
4. **Architecture**: Document design decisions in `docs/architecture/`

### Documentation Structure
```markdown
# Tutorial Title

Brief description of what this tutorial covers.

## Prerequisites
- List of required knowledge
- Installation requirements

## Step 1: Setup
```python
# Code examples with explanations
```

## Step 2: Implementation
Detailed explanation with code.

## Conclusion
Summary and next steps.
```

## 🚀 Release Process

### Version Management
We use semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Creating a Release
1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with changes
3. **Create pull request** for review
4. **Merge to main** after approval
5. **Create git tag**: `git tag v0.1.0`
6. **Push tag**: `git push origin v0.1.0`
7. **GitHub Actions** will automatically:
   - Build packages
   - Run tests
   - Upload to PyPI
   - Create GitHub release

## 🐛 Debugging Tips

### Common Issues
1. **Import errors**: Check Python path and virtual environment
2. **Test failures**: Run with `-v` flag for detailed output
3. **Type errors**: Use `reveal_type()` for MyPy debugging
4. **Performance issues**: Use `pytest-benchmark` for timing

### Debugging Tools
```python
# Use built-in debugger
import pdb; pdb.set_trace()

# Better debugging with ipdb
import ipdb; ipdb.set_trace()

# Logging for complex issues
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

### Profiling Performance
```bash
# Profile specific test
pytest tests/test_slow.py --profile

# Memory profiling
pip install memory-profiler
python -m memory_profiler script.py

# Line profiling
pip install line-profiler
kernprof -l -v script.py
```

## 🤝 Collaboration Workflow

### Git Workflow
1. **Create branch**: `git checkout -b feature/your-feature`
2. **Make changes**: Follow coding standards
3. **Commit changes**: Use conventional commit format
4. **Push branch**: `git push origin feature/your-feature`
5. **Create PR**: Use provided template
6. **Code review**: Address feedback
7. **Merge**: Squash and merge to main

### Commit Message Format
```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

Examples:
- `feat(components): add microring resonator model`
- `fix(simulator): correct phase noise calculation`
- `docs(tutorials): add RTL generation guide`

## 📞 Getting Help

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests  
- **Code Review**: Request review from maintainers
- **Documentation**: Check existing docs and tutorials

Happy developing! 🧠✨