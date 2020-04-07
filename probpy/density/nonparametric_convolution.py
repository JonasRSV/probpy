from probpy.core import Density
import numpy as np
from probpy.integration import uniform_importance_sampling
from probpy.distributions import multivariate_uniform


class UCKD(Density):
    """Un-normalised Convolution Kernel Density"""

    def __init__(self, variance: float = 2.0, **_):
        """

        :param variance: variance of kernel
        :param _:
        """
        self.variance = variance
        self.particles = None

    @staticmethod
    def kernel(x, y, variance):
        return np.exp(-(1 / variance) * np.square(x[:, None] - y).sum(axis=2)).sum(axis=1)

    def fit(self, particles: np.ndarray):
        """

        :param particles: particles to use for estimate
        :return:
        """
        self.particles = particles
        if particles.ndim == 1: self.particles = self.particles.reshape(-1, 1)

    def p(self, particles: np.ndarray):
        """

        :param particles: densities to estimtae
        :return: densities
        """
        if particles.ndim == 1: particles = particles.reshape(-1, 1)
        return self.kernel(particles, self.particles, self.variance)


class RCKD(Density):
    """Renormalized Convolution Kernel Density"""

    def __init__(self,
                 variance: float = 2.0,
                 sampling_sz: int = 100,
                 error: float = 1e-1,
                 verbose: bool = False):
        """

        :param variance: variance of kernel
        :param sampling_sz: sampling size in normalization integral
        :param error: error metric for normalization constant
        :param verbose: print progress of estimating normalization constant
        """
        self.variance = variance
        self.sampling_sz = sampling_sz
        self.error = error
        self.verbose = verbose
        self.particles = None
        self.partition = None

    @staticmethod
    def kernel(particles: np.ndarray, evidence: np.ndarray, variance: float):
        return np.exp(-(1 / variance) * np.square(particles[:, None] - evidence).sum(axis=2)).sum(axis=1)

    def _importance_sampling(self, particles: np.ndarray):
        def function(x: np.ndarray):
            return RCKD.kernel(x, particles, self.variance)

        dim = particles.shape[1]
        lb = np.array([particles[:, i].min() for i in range(dim)])
        ub = np.array([particles[:, i].max() for i in range(dim)])

        partition, previous, samples = 1.0, -9999, 1
        while np.abs(1 - partition / previous) > self.error:
            integral = uniform_importance_sampling(self.sampling_sz, function, (lb, ub),
                                                   multivariate_uniform.med(a=lb, b=ub))
            partition, samples, previous = (partition * samples + integral) / (samples + 1), samples + 1, partition

            if self.verbose:
                print(f"{samples} - {np.abs(1 - partition / previous)}")

        return partition, np.abs(1 - partition / previous)

    def fit(self, particles: np.ndarray):
        """

        :param particles: particles to use for estimate
        :return:
        """
        if particles.ndim == 1: particles = particles.reshape(-1, 1)
        self.particles = particles
        self.partition, error = self._importance_sampling(particles)
        return error

    def p(self, particles: np.ndarray):
        """

        :param particles: estimate density of particles
        :return: densities
        """
        if particles.ndim == 1: particles = particles.reshape(-1, 1)
        if particles[1].size != self.particles[1].size: raise Exception("Dimension mismatch in p RCKD")
        return RCKD.kernel(particles, self.particles, self.variance) / self.partition
