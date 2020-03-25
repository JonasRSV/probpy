import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class Multinomial(Distribution):
    n = "n"
    probabilities = "probabilities"

    @classmethod
    def freeze(cls, n: int = None, probabilities: np.ndarray = None, dim: int = None) -> RandomVariable:
        if n is None and probabilities is None:
            _sample = Multinomial.sample
            _p = Multinomial.p
            shape = dim
        elif n is None:
            def _sample(n: np.ndarray, shape: np.ndarray = ()): return Multinomial.sample(n, probabilities, shape)
            def _p(x: np.ndarray, n: np.ndarray): return Multinomial.p(x, n, probabilities)
            shape = probabilities.size
        elif probabilities is None:
            def _sample(probabilities: np.ndarray, shape: np.ndarray = ()): return Multinomial.sample(n, probabilities, shape)
            def _p(x: np.ndarray, probabilities: np.ndarray): return Multinomial.p(x, n, probabilities)
            shape = None
        else:
            def _sample(shape: np.ndarray = ()): return Multinomial.sample(n, probabilities, shape)
            def _p(x: np.ndarray): return Multinomial.p(x, n, probabilities)
            shape = probabilities.size

        parameters = {
            Multinomial.n: Parameter(shape=(), value=n),
            Multinomial.probabilities: Parameter(shape=shape, value=probabilities)
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
    def sample(n: int, probabilities: np.ndarray, shape=()) -> np.ndarray:
        return np.random.multinomial(n, probabilities, size=shape)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def p(x: np.ndarray, n: int, probabilities: np.ndarray) -> np.ndarray:
        constants = np.array([Multinomial._combinations_high_n(n, _x) for _x in x])
        return constants * np.prod(np.float_power(probabilities, x), axis=1)

