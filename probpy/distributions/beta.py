import numpy as np

from probpy.core import Distribution
from probpy.special import gamma


class Beta(Distribution):

    @staticmethod
    def sample(alpha: np.float32, beta: np.float32, shape=()) -> np.ndarray:
        return np.random.beta(alpha, beta, size=shape)

    @staticmethod
    def p(alpha: np.float32, beta: np.float32, x: np.ndarray) -> np.ndarray:
        # TODO: find out if there is a more numerically stable implementation
        normalizing_constant = np.prod(gamma(alpha)) / gamma(alpha.sum())
        return np.float_power(x, alpha - 1).prod(axis=1) / normalizing_constant

