import numpy as np

from probpy.core import Distribution, FrozenDistribution


class Multinomial(Distribution):

    @classmethod
    def freeze(cls, n: int, p: np.ndarray) -> FrozenDistribution:
        return FrozenDistribution(cls, n, p)

    @staticmethod
    def _combinations_high_n(_, x):
        # Not broadcastable but deals well with large n
        # Might suffer some precision problems though

        res = 1
        j = 1
        for i in x:
            for k in range(1, i + 1):
                res *= (j / k)
                j += 1

        return res

    @staticmethod
    def sample(n: int, p: np.ndarray, shape=()) -> np.ndarray:
        return np.random.multinomial(n, p, size=shape)

    @staticmethod
    def p(n: int, p: np.float32, x: np.ndarray) -> np.ndarray:
        constants = np.array([Multinomial._combinations_high_n(n, _x) for _x in x])
        return constants * np.prod(np.float_power(p, x), axis=1)

