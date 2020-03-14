import numpy as np

from probpy.core import Distribution, FrozenDistribution
from probpy.special import beta


class Beta(Distribution):

    @classmethod
    def freeze(cls, a: np.float32, b: np.float32) -> FrozenDistribution:
        return FrozenDistribution(cls, a, b)

    @staticmethod
    def sample(a: np.float32, b: np.float32, shape=()) -> np.ndarray:
        return np.random.beta(a, b, size=shape)

    @staticmethod
    def p(a: np.float32, b: np.float32, x: np.ndarray) -> np.ndarray:
        # TODO: find out if there is a more numerically stable implementation
        return np.float_power(x, a - 1) * np.float_power(1 - x, b - 1) / beta(a, b)

