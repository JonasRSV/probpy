from probpy.core import Density
import numpy as np
import random


class URBK(Density):
    epsilon = 1e-2
    """Un-normalised Radial-Basis Kernel"""

    def __init__(self, variance: float,
                 bases: int = 5,
                 cl_iterations: int = 100,
                 use_cl: bool = False,
                 learning_rate: float = 1.0,
                 verbose: bool = False, **_):
        """Kernel is a function [x] -> [0, 1]"""
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
        return np.exp(-variance * np.square(x[:, None] - y).sum(axis=2))

    @staticmethod
    def distance(i, j):
        return np.square(i - j).sum()

    def _place_bases(self, particles: np.ndarray, densities: np.ndarray):
        self.bases = np.array(random.choices(particles, k=self.n_bases))

        if self.use_cl:
            prev_error = 0.0
            for i in range(self.cl_iterations):
                error = 0.0
                for j, particle in enumerate(particles):
                    closest_base = np.argmin(np.abs(self.bases - particle))

                    shift = self.learning_rate * densities[j] * (particle - self.bases[closest_base])
                    self.bases[closest_base] += shift

                    error += np.abs(shift).sum()

                if self.verbose: print(f"error: {error}")
                if error == prev_error:
                    break

                prev_error = error

    def fit(self, particles: np.ndarray, densities: np.ndarray):
        if particles.ndim == 1: self.particles = self.particles.reshape(-1, 1)
        self.particles = particles
        self._place_bases(particles, densities)

        A = URBK.kernel(particles, self.bases, self.variance)

        self.lsq_coeff = np.linalg.inv(A.T @ A + URBK.epsilon) @ (A.T @ densities)

    def p(self, particles: np.ndarray):
        if particles.ndim == 1: particles = particles.reshape(-1, 1)
        return self.kernel(particles, self.bases, self.variance) @ self.lsq_coeff
