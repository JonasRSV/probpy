import numpy as np

from probpy.core import Distribution, FrozenDistribution


class Binomial(Distribution):

    @classmethod
    def freeze(cls, n: int, p: np.float32) -> FrozenDistribution:
        return FrozenDistribution(cls, n, p)

    @staticmethod
    def _combinations_high_n(n, x):
        # Not broadcastable but deals well with large n
        # Might suffer some precision problems though
        terms = n - x
        res = 1
        for t in range(terms):
            res *= ((n - t) / (terms - t))
        return res

    @staticmethod
    def sample(n: int, p: np.float32, shape=()) -> np.ndarray:
        return np.random.binomial(n, p, size=shape)

    @staticmethod
    def p(n: int, p: np.float32, x: np.ndarray) -> np.ndarray:
        constants = np.array([Binomial._combinations_high_n(n, _x) for _x in x])
        return constants * np.float_power(p, x) * np.float_power(1 - p, n - x)
