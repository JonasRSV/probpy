from probpy.core import Density
import numpy as np
from probpy.algorithms import mode_from_points
import numba


class URBK(Density):
    epsilon = 1e-2
    """Un-normalised Radial-Basis Kernel"""

    def __init__(self, variance: float = 1.0, **_):
        """

        :param variance: variance in rbf kernel
        :param _:
        """
        self.variance = variance
        self.bases = None
        self.particles = None
        self.lsq_coeff = None

    @staticmethod
    def kernel(x, y, variance):
        return np.exp(-(1 / variance) * np.square(x[:, None] - y).sum(axis=2))

    @staticmethod
    def distance(i, j):
        return np.square(i - j).sum(axis=1)

    def _place_bases(self, particles: np.ndarray, densities: np.ndarray):
        self.bases, self.lsq_coeff = mode_from_points(particles, densities, n=densities.size / 4)

    def fit(self, particles: np.ndarray, densities: np.ndarray):
        """

        :param particles: particles to estimate density
        :param densities: unnormalized density of particles
        :return:
        """
        if particles.ndim == 1: self.particles = self.particles.reshape(-1, 1)
        densities = densities / densities.sum()
        self.particles = particles
        self._place_bases(particles, densities)

    def p(self, particles: np.ndarray):
        """

        :param particles: particles to estimate
        :return: densities
        """
        particles = np.array(particles)
        if particles.ndim == 0: particles = particles.reshape(1, 1)
        if particles.ndim == 1: particles = particles.reshape(-1, 1)
        return self.kernel(particles, self.bases, self.variance) @ self.lsq_coeff




