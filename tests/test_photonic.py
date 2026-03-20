"""Tests for photonic neuromorphic simulation components."""

import numpy as np
import pytest

from photonic_neuromorphics.mzi import MachZehnderInterferometer
from photonic_neuromorphics.synapse import PhotonicSynapse
from photonic_neuromorphics.neuron import PhotonicNeuron
from photonic_neuromorphics.network import OpticalNetwork
from photonic_neuromorphics.xor_demo import simulate_xor


# ─── MZI Tests ───────────────────────────────────────────────────────────────

def test_mzi_transfer_matrix_shape():
    """Transfer matrix must be 2x2."""
    mzi = MachZehnderInterferometer(theta=0.3, phi=0.1)
    T = mzi.transfer_matrix()
    assert T.shape == (2, 2), f"Expected shape (2,2), got {T.shape}"


def test_mzi_unitary():
    """Transfer matrix determinant magnitude must be 1 (unitary)."""
    mzi = MachZehnderInterferometer(theta=np.pi / 4, phi=np.pi / 6)
    T = mzi.transfer_matrix()
    det = np.linalg.det(T)
    assert abs(abs(det) - 1.0) < 1e-10, f"|det| = {abs(det)}, expected 1"


def test_mzi_forward_shape():
    """forward() output must be a complex 1D array of length 2."""
    mzi = MachZehnderInterferometer(theta=0.5, phi=0.2)
    field = np.array([1.0 + 0j, 0.0 + 0j])
    out = mzi.forward(field)
    assert out.shape == (2,), f"Expected shape (2,), got {out.shape}"
    assert np.iscomplexobj(out), "Output should be complex"


def test_mzi_energy_conservation():
    """Total optical power in == total optical power out (lossless)."""
    mzi = MachZehnderInterferometer(theta=np.pi / 3, phi=np.pi / 5)
    field = np.array([1.0 + 0j, 0.5 + 0.5j])
    out = mzi.forward(field)
    power_in = np.sum(np.abs(field) ** 2)
    power_out = np.sum(np.abs(out) ** 2)
    assert abs(power_in - power_out) < 1e-10, (
        f"Power not conserved: in={power_in}, out={power_out}"
    )


# ─── Synapse Tests ────────────────────────────────────────────────────────────

def test_synapse_zero_weight():
    """At weight=0, detuning=0, transmission should be exactly 1."""
    syn = PhotonicSynapse(weight=0.0)
    result = syn.transmit(1.0)
    # T = 1/(1+0) = 1, phase = exp(0) = 1
    assert abs(abs(result) - 1.0) < 1e-10, f"|transmit| = {abs(result)}, expected 1.0"


def test_synapse_weight_range():
    """transmit() should return a complex value with |output| <= |input| for all weights."""
    signal = 2.0
    for w in np.linspace(-1, 1, 21):
        syn = PhotonicSynapse(weight=float(w))
        out = syn.transmit(signal)
        assert abs(out) <= signal + 1e-10, (
            f"Output amplitude {abs(out)} exceeds input {signal} at weight={w}"
        )


def test_synapse_invalid_weight():
    """Weights outside [-1, 1] should raise ValueError."""
    with pytest.raises(ValueError):
        PhotonicSynapse(weight=1.5)
    with pytest.raises(ValueError):
        PhotonicSynapse(weight=-2.0)


def test_synapse_returns_complex():
    """transmit() should return a complex number."""
    syn = PhotonicSynapse(weight=0.5)
    result = syn.transmit(1.0)
    assert isinstance(result, complex), f"Expected complex, got {type(result)}"


# ─── Neuron Tests ─────────────────────────────────────────────────────────────

def test_neuron_fires_above_threshold():
    """Neuron should fire (return 1) when total input power exceeds threshold."""
    neuron = PhotonicNeuron(threshold=0.5)
    assert neuron.fire([0.6]) == 1


def test_neuron_silent_below_threshold():
    """Neuron should not fire (return 0) when total input power is below threshold."""
    neuron = PhotonicNeuron(threshold=0.5)
    assert neuron.fire([0.3]) == 0


def test_neuron_integrate_sum():
    """integrate() should return sum of absolute values of all inputs."""
    neuron = PhotonicNeuron(threshold=0.5)
    inputs = [0.2, 0.3, 0.1]
    result = neuron.integrate(inputs)
    assert abs(result - 0.6) < 1e-10, f"Expected 0.6, got {result}"


def test_neuron_integrate_complex_inputs():
    """integrate() should handle complex inputs and use |x| for each."""
    neuron = PhotonicNeuron(threshold=0.5)
    inputs = [1.0 + 0j, 0.0 + 1.0j]
    result = neuron.integrate(inputs)
    # |1+0j| + |0+1j| = 1 + 1 = 2
    assert abs(result - 2.0) < 1e-10


# ─── Network Tests ────────────────────────────────────────────────────────────

def test_network_add_layer():
    """add_layer() should append a layer with the correct number of neurons."""
    net = OpticalNetwork()
    net.add_layer(3)
    assert len(net.layers) == 1
    assert len(net.layers[0]) == 3
    net.add_layer(2)
    assert len(net.layers) == 2
    assert len(net.layers[1]) == 2


def test_network_forward_shape():
    """forward() output length should match number of neurons in final layer."""
    net = OpticalNetwork()
    net.add_layer(2)
    net.add_layer(3)
    net.connect(0, 1, np.ones((3, 2)) * 0.5)
    out = net.forward([1.0, 1.0])
    assert len(out) == 3, f"Expected output length 3, got {len(out)}"


def test_network_binary_output():
    """forward() output values should all be 0 or 1."""
    net = OpticalNetwork()
    net.add_layer(2)
    net.add_layer(1)
    net.connect(0, 1, np.array([[0.8, 0.8]]))
    out = net.forward([1.0, 1.0])
    for val in out:
        assert val in (0, 1), f"Output {val} is not binary"


# ─── XOR Demo Tests ───────────────────────────────────────────────────────────

def test_xor_demo_runs():
    """simulate_xor() should run without error and return a dict."""
    results = simulate_xor()
    assert isinstance(results, dict), "simulate_xor() should return a dict"
    assert len(results) == 4, f"Expected 4 entries, got {len(results)}"


def test_xor_outputs_binary():
    """All XOR demo outputs should be lists of 0 or 1."""
    results = simulate_xor()
    for inp, out in results.items():
        assert isinstance(out, list), f"Output for {inp} should be a list"
        assert len(out) == 1, f"Output for {inp} should have 1 element"
        assert out[0] in (0, 1), f"Output value {out[0]} for {inp} is not binary"
