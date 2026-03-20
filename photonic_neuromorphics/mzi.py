"""Mach-Zehnder Interferometer (MZI) using the 2x2 transfer matrix method."""

import numpy as np


class MachZehnderInterferometer:
    """
    Mach-Zehnder Interferometer modeled via the 2x2 transfer matrix method.

    Parameters
    ----------
    theta : float
        Phase shift applied in one arm (radians).
    phi : float
        Phase imbalance between the two arms (radians).
    """

    def __init__(self, theta: float = 0.0, phi: float = 0.0):
        self.theta = theta
        self.phi = phi

    def transfer_matrix(self) -> np.ndarray:
        """
        Compute the 2x2 complex transfer matrix of the MZI.

        The MZI is modelled as two 50/50 beam splitters with a phase shift
        in the upper arm.  The standard unitary representation is:

            T = (i/2) * [[exp(i*theta) + exp(i*phi),  exp(i*theta) - exp(i*phi)],
                         [exp(i*theta) - exp(i*phi), -(exp(i*theta) + exp(i*phi))]]

        This reduces to the well-known result for phi=0:
            T = i * [[cos(theta/2), sin(theta/2)],
                     [sin(theta/2), -cos(theta/2)]]

        Returns
        -------
        np.ndarray
            2x2 complex numpy array representing the transfer matrix.
        """
        t = self.theta
        p = self.phi
        # Standard MZI transfer matrix built from two 50/50 beam splitters
        # BS matrix: (1/sqrt(2)) * [[1, i],[i, 1]]
        # Phase matrix: [[exp(i*t), 0],[0, exp(i*p)]]
        # T = BS @ Phase @ BS
        bs = (1.0 / np.sqrt(2)) * np.array([[1, 1j], [1j, 1]], dtype=complex)
        phase = np.diag([np.exp(1j * t), np.exp(1j * p)])
        return bs @ phase @ bs

    def forward(self, input_field: np.ndarray) -> np.ndarray:
        """
        Apply the MZI transfer matrix to an input optical field.

        Parameters
        ----------
        input_field : np.ndarray
            1D complex array of length 2 representing input field amplitudes.

        Returns
        -------
        np.ndarray
            1D complex array of length 2 representing output field amplitudes.
        """
        input_field = np.asarray(input_field, dtype=complex)
        return self.transfer_matrix() @ input_field
