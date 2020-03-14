import numpy as np

from probpy.core import Distribution, FrozenDistribution
from probpy.special import gamma


class Dirichlet(Distribution):

    @classmethod
    def freeze(cls, alpha: np.ndarray) -> FrozenDistribution:
        return FrozenDistribution(cls, alpha)

    @staticmethod
    def sample(alpha: np.ndarray, shape=()) -> np.ndarray:
        return np.random.dirichlet(alpha, size=shape)

    @staticmethod
    def p(alpha: np.ndarray, x: np.ndarray) -> np.ndarray:
        # TODO: find out if there is a more numerically stable implementation
        normalizing_constant = np.prod(gamma(alpha)) / gamma(alpha.sum())
        return np.float_power(x, alpha - 1).prod(axis=1) / normalizing_constant

