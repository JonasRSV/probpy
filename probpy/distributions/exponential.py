import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class Exponential(Distribution):
    lam = "lam"

    @classmethod
    def freeze(cls, lam: np.float32 = None) -> RandomVariable:
        if lam is None:
            _sample = Exponential.sample
            _p = Exponential.p
        else:
            def _sample(shape=()): return Exponential.sample(lam, shape)
            def _p(x): return Exponential.p(x, lam)

        parameters = { Exponential.lam: Parameter(shape=(), value=lam) }
        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(lam: np.float32, shape=()) -> np.ndarray:
        return np.random.exponential(1 / lam, size=shape)

    @staticmethod
    @numba.jit(nopython=True, forceobj=False)
    def p(x: np.ndarray, lam: np.float32) -> np.ndarray:
        return lam * np.exp(-lam * x)

