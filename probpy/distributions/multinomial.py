import numpy as np

from probpy.core import Distribution, RandomVariable


class Multinomial(Distribution):

    @classmethod
    def freeze(cls, n: int = None, p: np.ndarray = None) -> RandomVariable:
        if n is None and p is None:
            _sample = Multinomial.sample
            _p = Multinomial.p
            shape = None
        elif n is None:
            def _sample(n: np.ndarray, shape: np.ndarray = ()): return Multinomial.sample(n, p, shape)
            def _p(x: np.ndarray, n: np.ndarray): return Multinomial.p(x, n, p)
            shape = p.size
        elif p is None:
            def _sample(p: np.ndarray, shape: np.ndarray = ()): return Multinomial.sample(n, p, shape)
            def _p(x: np.ndarray, p: np.ndarray): return Multinomial.p(x, n, p)
            shape = None
        else:
            def _sample(shape: np.ndarray = ()): return Multinomial.sample(n, p, shape)
            def _p(x: np.ndarray): return Multinomial.p(x, n, p)
            shape = p.size

        return RandomVariable(_sample, _p, shape=shape)

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
    def p(x: np.ndarray, n: int, p: np.float32) -> np.ndarray:
        constants = np.array([Multinomial._combinations_high_n(n, _x) for _x in x])
        return constants * np.prod(np.float_power(p, x), axis=1)

