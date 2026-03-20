"""Photonic Neuron — optical integrate-and-fire model."""

from typing import List


class PhotonicNeuron:
    """
    Optical integrate-and-fire neuron.

    Integrates optical power from multiple inputs and fires a binary spike
    when the integrated power exceeds the threshold.

    Parameters
    ----------
    threshold : float
        Firing threshold (default 0.5).
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.state: float = 0.0

    def integrate(self, inputs: List[float]) -> float:
        """
        Integrate optical input signals (sum of optical power).

        Parameters
        ----------
        inputs : list of float
            Optical signal amplitudes (real or complex magnitudes).

        Returns
        -------
        float
            Total integrated optical power.
        """
        self.state = sum(abs(x) for x in inputs)
        return self.state

    def fire(self, inputs: List[float]) -> int:
        """
        Integrate inputs and fire if total power exceeds threshold.

        Parameters
        ----------
        inputs : list of float
            Optical signal amplitudes.

        Returns
        -------
        int
            1 if neuron fires, 0 otherwise.
        """
        power = self.integrate(inputs)
        return 1 if power > self.threshold else 0
