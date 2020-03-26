import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class Poisson(Distribution):
    lam = "lam"

    @classmethod
    def freeze(cls, lam: np.float32 = None) -> RandomVariable:
        if lam is None:
            _sample = Poisson.sample
            _p = Poisson.p
        else:
            def _sample(shape=()): return Poisson.sample(lam, shape)
            def _p(x): return Poisson.p(x, lam)

        parameters = { Poisson.lam: Parameter(shape=(), value=lam) }
        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(lam: np.float32, shape=()) -> np.ndarray:
        return np.random.poisson(lam, size=shape)

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def _factorial(x: np.ndarray):
        result = np.zeros(x.size)
        for i in range(x.size):
            result[i] = np.arange(1, x[i] + 1).prod()
        return result

    @staticmethod
    def p(x: np.ndarray, lam: np.float32) -> np.ndarray:
        return (np.float_power(lam, x) * np.exp(-lam)) / Poisson._factorial(x)

