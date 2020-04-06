import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class Exponential(Distribution):
    lam = "lam"

    @classmethod
    def med(cls, lam: np.float32 = None) -> RandomVariable:
        if lam is None:
            _sample = Exponential.sample
            _p = Exponential.p
        else:
            def _sample(size=()): return Exponential.sample(lam, size)
            def _p(x): return Exponential.p(x, lam)

        parameters = {Exponential.lam: Parameter(shape=(), value=lam)}
        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(lam: np.float32, size=()) -> np.ndarray:
        return np.random.exponential(1 / lam, size=size)

    @staticmethod
    def p(x: np.ndarray, lam: np.float32) -> np.ndarray:
        if type(x) != np.ndarray: x = np.array(x)
        result = np.zeros_like(x)
        lg_0 = x >= 0
        result[lg_0] = lam * np.exp(-lam * x[lg_0])
        return result

