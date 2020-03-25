import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class Bernoulli(Distribution):
    probability = "p"

    @classmethod
    def freeze(cls, p: np.float32 = None) -> RandomVariable:
        if p is None:
            _sample = Bernoulli.sample
            _p = Bernoulli.p
        else:
            def _sample(shape=()): return Bernoulli.sample(p, shape)
            def _p(x): return Bernoulli.p(x, p)

        parameters = { Bernoulli.probability: Parameter(shape=(), value=p) }
        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(p: np.float32, shape = ()) -> np.ndarray:
        if type(shape) == int:
            return (np.random.rand(shape) < p).astype(np.float32)
        return (np.random.rand(*shape) < p).astype(np.float32)

    @staticmethod
    @numba.jit(nopython=True, forceobj=False)
    def p(x: np.ndarray, p: np.float32) -> np.ndarray:
        res = np.zeros_like(x)
        res[x != 1.0] = 1 - p
        res[x == 1.0] = p
        return res

