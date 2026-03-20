"""Photonic Synapse modelled as a ring resonator with detuning-based weight encoding."""

import numpy as np


class PhotonicSynapse:
    """
    Photonic Synapse using ring resonator tuning to encode synaptic weight.

    The weight maps to a ring detuning:
        delta = weight * pi

    Lorentzian transmission:
        T = 1 / (1 + (2*delta/kappa)^2)

    The output signal includes a phase factor derived from the detuning.

    Parameters
    ----------
    weight : float
        Synaptic weight in the range [-1, 1].
    kappa : float
        Ring resonator linewidth (default 1.0).
    """

    def __init__(self, weight: float = 0.0, kappa: float = 1.0):
        if not -1.0 <= weight <= 1.0:
            raise ValueError(f"weight must be in [-1, 1], got {weight}")
        self.weight = weight
        self.kappa = kappa

    def transmit(self, signal: float) -> complex:
        """
        Transmit an optical signal through the ring resonator synapse.

        Transmission amplitude:
            T = 1 / (1 + (2*delta/kappa)^2)

        with a complex phase factor exp(i * delta):
            output = signal * T * exp(i * delta)

        Parameters
        ----------
        signal : float
            Input optical signal amplitude.

        Returns
        -------
        complex
            Transmitted complex signal.
        """
        delta = self.weight * np.pi
        T = 1.0 / (1.0 + (2.0 * delta / self.kappa) ** 2)
        phase_factor = np.exp(1j * delta)
        return signal * T * phase_factor
