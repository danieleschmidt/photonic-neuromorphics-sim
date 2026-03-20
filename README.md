# photonic-neuromorphics-sim

Python/NumPy simulation of photonic neuromorphic computing elements including Mach-Zehnder interferometers, ring resonator synapses, and optical spiking neurons.

## Install

```bash
pip install numpy
```

## Usage

```python
from photonic_neuromorphics.xor_demo import simulate_xor

results = simulate_xor()
print(results)
# {(0, 0): [0], (0, 1): [1], (1, 0): [1], (1, 1): [0]}
```

## Components

- **MachZehnderInterferometer**: transfer matrix model (2x2 unitary)
- **PhotonicSynapse**: ring resonator weight encoding via Lorentzian transmission
- **PhotonicNeuron**: optical integrate-and-fire spiking neuron
- **OpticalNetwork**: connect neurons into layers with photonic synapses

## Architecture

```
MachZehnderInterferometer  — beam routing / linear transformation
PhotonicSynapse            — weight encoding via ring resonator detuning
PhotonicNeuron             — integrate-and-fire with optical threshold
OpticalNetwork             — layered composition of neurons + synapses
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## License

MIT
