import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class Multinomial(Distribution):
    n = "n"
    probabilities = "p"

    @classmethod
    def freeze(cls, n: int = None, p: np.ndarray = None, dim: int = None) -> RandomVariable:
        if n is None and p is None:
            _sample = Multinomial.sample
            _p = Multinomial.p
            shape = dim
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

        parameters = {
            Multinomial.n: Parameter(shape=(), value=n),
            Multinomial.probabilities: Parameter(shape=shape, value=p)
        }

        return RandomVariable(_sample, _p, shape=shape, parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=True, forceobj=False)
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
    @numba.jit(nopython=False, forceobj=True)
    def sample(n: int, p: np.ndarray, shape=()) -> np.ndarray:
        return np.random.multinomial(n, p, size=shape)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def p(x: np.ndarray, n: int, p: np.float32) -> np.ndarray:
        constants = np.array([Multinomial._combinations_high_n(n, _x) for _x in x])
        return constants * np.prod(np.float_power(p, x), axis=1)

