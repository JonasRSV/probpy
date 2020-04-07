from probpy.core import Density
import numpy as np


class URBK(Density):
    epsilon = 1e-2
    """Un-normalised Radial-Basis Kernel"""

    def __init__(self, variance: float = 0.5,
                 bases: int = 5,
                 cl_iterations: int = 10,
                 use_cl: bool = False,
                 learning_rate: float = 1.0,
                 verbose: bool = False, **_):
        """

        :param variance: variance in rbf kernel
        :param bases: number of rbf bases
        :param cl_iterations: competitive learning iterations
        :param use_cl: use competitive learning to place bases
        :param learning_rate: learning rate of cl
        :param verbose: print cl error
        :param _:
        """
        self.variance = variance
        self.n_bases = bases
        self.bases = None
        self.cl_iterations = cl_iterations
        self.use_cl = use_cl
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.particles = None
        self.lsq_coeff = None

    @staticmethod
    def kernel(x, y, variance):
        return np.exp(-(1 / variance) * np.square(x[:, None] - y).sum(axis=2))

    @staticmethod
    def distance(i, j):
        return np.square(i - j).sum(axis=1)

    def _place_bases(self, particles: np.ndarray, densities: np.ndarray):
        densities = densities / densities.sum()
        indexes = np.arange(densities.size)

        self.bases = particles[np.random.choice(indexes, size=self.n_bases, p=densities)]

        if self.use_cl:
            prev_error = 0.0
            for i in range(self.cl_iterations):
                error = 0.0
                for j, particle in enumerate(particles):
                    closest_base = np.argmin(URBK.distance(self.bases, particle))

                    shift = self.learning_rate * densities[j] * (particle - self.bases[closest_base])
                    self.bases[closest_base] += shift

                    error += np.abs(shift).sum()

                if self.verbose: print(f"error: {error}")
                if error == prev_error:
                    break

                prev_error = error

    def fit(self, particles: np.ndarray, densities: np.ndarray):
        """

        :param particles: particles to estimate density
        :param densities: unnormalized density of particles
        :return:
        """
        if particles.ndim == 1: self.particles = self.particles.reshape(-1, 1)
        self.particles = particles
        self._place_bases(particles, densities)

        A = URBK.kernel(particles, self.bases, self.variance)

        self.lsq_coeff = np.linalg.inv(A.T @ A + URBK.epsilon) @ (A.T @ densities)

    def p(self, particles: np.ndarray):
        """

        :param particles: particles to estimate
        :return: densities
        """
        if particles.ndim == 1: particles = particles.reshape(-1, 1)
        return self.kernel(particles, self.bases, self.variance) @ self.lsq_coeff
