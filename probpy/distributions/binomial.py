import numpy as np
import numba

from probpy.core import Distribution, RandomVariable


class Binomial(Distribution):

    @classmethod
    def freeze(cls, n: int = None, p: np.float32 = None) -> RandomVariable:
        if n is None and p is None:
            _sample = Binomial.sample
            _p = Binomial.p
        elif n is None:
            def _sample(n: np.ndarray, shape: np.ndarray = ()): return Binomial.sample(n, p, shape)
            def _p(x: np.ndarray, n: np.ndarray): return Binomial.p(x, n, p)
        elif p is None:
            def _sample(p: np.ndarray, shape: np.ndarray = ()): return Binomial.sample(n, p, shape)
            def _p(x: np.ndarray, p: np.ndarray): return Binomial.p(x, n, p)
        else:
            def _sample(shape: np.ndarray = ()): return Binomial.sample(n, p, shape)
            def _p(x: np.ndarray): return Binomial.p(x, n, p)

        return RandomVariable(_sample, _p, shape=())

    @staticmethod
    @numba.jit(nopython=True, forceobj=False)
    def _combinations_high_n(n, x):
        # Not broadcastable but deals well with large n
        # Might suffer some precision problems though
        terms = n - x
        res = 1
        for t in range(terms):
            res *= ((n - t) / (terms - t))
        return res

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(n: int, p: np.float32, shape=()) -> np.ndarray:
        return np.random.binomial(n, p, size=shape)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def p(x: np.ndarray, n: int, p: np.float32) -> np.ndarray:
        constants = np.array([Binomial._combinations_high_n(n, _x) for _x in x])
        return constants * np.float_power(p, x) * np.float_power(1 - p, n - x)
