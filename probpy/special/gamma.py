import numpy as np


# Taken from https://sv.wikipedia.org/wiki/Gammafunktionen
class Gamma:
    g = 7
    C = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]

    @staticmethod
    def _gamma(z):
        z = np.cast[np.complex64](z)
        if z.real < 0.5:
            return np.pi / (np.sin(np.pi * z) * Gamma._gamma(1 - z))
        else:
            z -= 1
            x = Gamma.C[0]
            for i in range(1, Gamma.g + 2):
                x += Gamma.C[i] / (z + i)
            t = z + Gamma.g + 0.5
            return np.sqrt(2 * np.pi) * t ** (z + 0.5) * np.exp(-t) * x

    @staticmethod
    def gamma(z):
        if z.shape != ():
            return np.array([Gamma._gamma(z_).real for z_ in z])
        return Gamma._gamma(z).real


