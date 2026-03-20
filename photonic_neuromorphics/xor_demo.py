"""XOR demonstration using a 2-layer optical network."""

import numpy as np
from .network import OpticalNetwork


def simulate_xor() -> dict:
    """
    Build a 2-layer optical network that approximates XOR behaviour.

    Architecture:
        - Input layer:  2 neurons
        - Hidden layer: 2 neurons
        - Output layer: 1 neuron

    XOR truth table:
        (0, 0) -> 0
        (0, 1) -> 1
        (1, 0) -> 1
        (1, 1) -> 0

    The weights are hand-crafted to approximate XOR using optical synapses.
    Because the synapse transmission is a non-linear Lorentzian function of
    weight, we tune weights empirically so that:
        - one hidden neuron fires on OR  (fires when >= 1 input is 1)
        - one hidden neuron fires on AND (fires when both inputs are 1)
        - output neuron fires on OR-but-NOT-AND (XOR = OR NAND AND)

    Returns
    -------
    dict
        Keys are input tuples, values are output lists (single element).
        Example: {(0, 0): [0], (0, 1): [1], (1, 0): [1], (1, 1): [0]}
    """
    net = OpticalNetwork()

    # Layer 0: 2 input neurons (threshold low so they pass through any signal)
    net.add_layer(2, threshold=0.3)
    # Layer 1: 2 hidden neurons
    #   neuron 0 (OR-like):  fires if at least one input neuron fires
    #   neuron 1 (AND-like): fires only if both input neurons fire
    net.add_layer(2, threshold=0.3)
    # Layer 2: 1 output neuron
    net.add_layer(1, threshold=0.3)

    # Weights: input -> hidden
    # Hidden neuron 0 (OR): strong positive weights from both inputs
    # Hidden neuron 1 (AND): strong positive weights from both inputs but
    #   higher threshold needed (we rely on additive effect)
    # Weight matrix shape: (n_dst=2, n_src=2)
    w_input_hidden = np.array([
        [0.9, 0.9],   # OR neuron: fires if either input fires (sums to >0.3)
        [0.9, 0.9],   # AND neuron: same weights, but higher threshold set below
    ])
    net.connect(0, 1, w_input_hidden)

    # Override AND neuron threshold to require both inputs
    net.layers[1][1].threshold = 0.7  # needs both inputs to clear threshold

    # Weights: hidden -> output
    # Output fires if OR neuron fires (w=+) but NOT if AND neuron fires (w=-)
    # We use negative weight for AND contribution to suppress double-fire case.
    # weight must be in [-1, 1]; use -0.95 for strong inhibition
    w_hidden_output = np.array([
        [0.9, -0.95],  # +OR - AND
    ])
    net.connect(1, 2, w_hidden_output)

    xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    results = {}
    for a, b in xor_inputs:
        out = net.forward([float(a), float(b)])
        results[(a, b)] = out

    return results


if __name__ == "__main__":
    results = simulate_xor()
    print("XOR simulation results:")
    print(f"{'Input':>10}  {'Output':>8}  {'Expected':>10}")
    expected = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    for inp, out in sorted(results.items()):
        print(f"{str(inp):>10}  {str(out):>8}  {expected[inp]:>10}")
