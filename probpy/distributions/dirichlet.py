import numpy as np

from probpy.core import Distribution
from probpy.special import gamma


class Dirichlet(Distribution):

    @staticmethod
    def sample(alpha: np.ndarray, shape=()) -> np.ndarray:
        return np.random.dirichlet(alpha, size=shape)

    @staticmethod
    def p(alpha: np.ndarray, x: np.ndarray) -> np.ndarray:
        # TODO: find out if there is a more numerically stable implementation
        normalizing_constant = np.prod(gamma(alpha)) / gamma(alpha.sum())
        return np.float_power(x, alpha - 1).prod(axis=1) / normalizing_constant

