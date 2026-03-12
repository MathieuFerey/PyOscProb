import numpy as np
import astropy.units as u


class OscillationParameters:

    def __init__(
        self,
        dm21=7.49e-5 * u.eV**2,
        dm32=2.513e-3 * u.eV**2,
        sin212=0.308,
        sin223=0.470,
        sin213=2.215e-2,
        deltaCP=3.7,
        hierarchy=1,
    ):

        self.dm21 = dm21
        self.dm32 = dm32
        self.deltaCP = deltaCP
        self.hierarchy = hierarchy

        # internally store angles
        self._theta12 = np.arcsin(np.sqrt(sin212))
        self._theta23 = np.arcsin(np.sqrt(sin223))
        self._theta13 = np.arcsin(np.sqrt(sin213))

        

    # --------------------------------------------------
    # theta properties
    # --------------------------------------------------

    @property
    def theta12(self):
        return self._theta12

    @theta12.setter
    def theta12(self, val):
        self._theta12 = val

    @property
    def theta23(self):
        return self._theta23

    @theta23.setter
    def theta23(self, val):
        self._theta23 = val

    @property
    def theta13(self):
        return self._theta13

    @theta13.setter
    def theta13(self, val):
        self._theta13 = val

    # --------------------------------------------------
    # sin²(theta)
    # --------------------------------------------------

    @property
    def sin212(self):
        return np.sin(self._theta12) ** 2

    @sin212.setter
    def sin212(self, val):
        self._theta12 = np.arcsin(np.sqrt(val))

    @property
    def sin223(self):
        return np.sin(self._theta23) ** 2

    @sin223.setter
    def sin223(self, val):
        self._theta23 = np.arcsin(np.sqrt(val))

    @property
    def sin213(self):
        return np.sin(self._theta13) ** 2

    @sin213.setter
    def sin213(self, val):
        self._theta13 = np.arcsin(np.sqrt(val))

    # --------------------------------------------------
    # sin²(2theta)
    # --------------------------------------------------

    @property
    def sin22_12(self):
        return np.sin(2 * self._theta12) ** 2

    @sin22_12.setter
    def sin22_12(self, val):
        self._theta12 = 0.5 * np.arcsin(np.sqrt(val))

    @property
    def sin22_23(self):
        return np.sin(2 * self._theta23) ** 2

    @sin22_23.setter
    def sin22_23(self, val):
        self._theta23 = 0.5 * np.arcsin(np.sqrt(val))

    @property
    def sin22_13(self):
        return np.sin(2 * self._theta13) ** 2

    @sin22_13.setter
    def sin22_13(self, val):
        self._theta13 = 0.5 * np.arcsin(np.sqrt(val))

    # --------------------------------------------------
    # PMNS matrix
    # --------------------------------------------------

    @property
    def PMNS(self):

        c12 = np.cos(self._theta12)
        s12 = np.sin(self._theta12)

        c13 = np.cos(self._theta13)
        s13 = np.sin(self._theta13)

        c23 = np.cos(self._theta23)
        s23 = np.sin(self._theta23)

        d = self.deltaCP

        U = np.array(
            [
                [c12 * c13, s12 * c13, s13 * np.exp(-1j * d)],
                [
                    -s12 * c23 - c12 * s23 * s13 * np.exp(1j * d),
                    c12 * c23 - s12 * s23 * s13 * np.exp(1j * d),
                    s23 * c13,
                ],
                [
                    s12 * s23 - c12 * c23 * s13 * np.exp(1j * d),
                    -c12 * s23 - s12 * c23 * s13 * np.exp(1j * d),
                    c23 * c13,
                ],
            ]
        )

        return U

    # --------------------------------------------------
    # mass squared matrix
    # --------------------------------------------------

    @property
    def M2(self):

        if self.hierarchy == 1:  # NH

            return (
                np.diag(
                    [
                        0,
                        self.dm21.value,
                        self.dm32.value + self.dm21.value,
                    ]
                )
                * self.dm21.unit
            )

        else:  # IH

            return (
                np.diag(
                    [
                        0,
                        self.dm21.value,
                        -self.dm32.value - self.dm21.value,
                    ]
                )
                * self.dm21.unit
            )