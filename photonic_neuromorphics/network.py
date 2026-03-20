"""OpticalNetwork — connects PhotonicNeurons in layers via PhotonicSynapses."""

from typing import Dict, List, Tuple
import numpy as np

from .neuron import PhotonicNeuron
from .synapse import PhotonicSynapse


class OpticalNetwork:
    """
    Layered optical neural network.

    Neurons are organised in layers; adjacent layers are connected through
    PhotonicSynapse objects whose weights are stored in a weight matrix.

    Parameters
    ----------
    None — layers and synapses are added via `add_layer` and `connect`.
    """

    def __init__(self):
        self.layers: List[List[PhotonicNeuron]] = []
        # Key: (layer_i, neuron_j_in_layer_i, neuron_k_in_layer_i+1)
        self.synapses: Dict[Tuple[int, int, int], PhotonicSynapse] = {}

    def add_layer(self, n_neurons: int, threshold: float = 0.5) -> None:
        """
        Append a new layer with `n_neurons` PhotonicNeurons.

        Parameters
        ----------
        n_neurons : int
            Number of neurons in the new layer.
        threshold : float
            Firing threshold for each neuron in this layer (default 0.5).
        """
        self.layers.append([PhotonicNeuron(threshold=threshold) for _ in range(n_neurons)])

    def connect(self, layer_i: int, layer_j: int, weight_matrix: np.ndarray) -> None:
        """
        Connect layer `layer_i` to layer `layer_j` using a weight matrix.

        Each entry weight_matrix[k, j] is the synaptic weight from neuron j
        in layer_i to neuron k in layer_j.

        Parameters
        ----------
        layer_i : int
            Source layer index.
        layer_j : int
            Target layer index.
        weight_matrix : np.ndarray
            Shape (len(layers[layer_j]), len(layers[layer_i])).
            Weights must be in [-1, 1].
        """
        weight_matrix = np.asarray(weight_matrix, dtype=float)
        n_src = len(self.layers[layer_i])
        n_dst = len(self.layers[layer_j])
        assert weight_matrix.shape == (n_dst, n_src), (
            f"weight_matrix shape {weight_matrix.shape} does not match "
            f"({n_dst}, {n_src})"
        )
        # Clip weights to valid synapse range
        weight_matrix = np.clip(weight_matrix, -1.0, 1.0)
        for k in range(n_dst):
            for j in range(n_src):
                self.synapses[(layer_i, j, k)] = PhotonicSynapse(weight=weight_matrix[k, j])

    def forward(self, inputs: List[float]) -> List[int]:
        """
        Propagate an input vector through all layers of the network.

        Parameters
        ----------
        inputs : list of float
            Input signals for the first layer (one per neuron).

        Returns
        -------
        list of int
            Binary output spikes from the final layer.
        """
        if not self.layers:
            return []

        # First layer fires based on raw inputs
        current_outputs: List[float] = []
        for j, neuron in enumerate(self.layers[0]):
            inp = [inputs[j]] if j < len(inputs) else [0.0]
            current_outputs.append(float(neuron.fire(inp)))

        # Propagate through subsequent layers
        for layer_idx in range(1, len(self.layers)):
            prev_layer_idx = layer_idx - 1
            next_outputs: List[float] = []
            for k, neuron in enumerate(self.layers[layer_idx]):
                synapse_inputs = []
                for j in range(len(self.layers[prev_layer_idx])):
                    key = (prev_layer_idx, j, k)
                    if key in self.synapses:
                        transmitted = self.synapses[key].transmit(current_outputs[j])
                        synapse_inputs.append(abs(transmitted))
                    else:
                        synapse_inputs.append(0.0)
                next_outputs.append(float(neuron.fire(synapse_inputs)))
            current_outputs = next_outputs

        return [int(x) for x in current_outputs]
