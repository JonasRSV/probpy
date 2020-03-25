import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class Categorical(Distribution):
    probabilities = "p"

    @classmethod
    def freeze(cls, p: np.ndarray = None, dim: int = None) -> RandomVariable:
        if p is None:
            _sample = Categorical.sample
            _p = Categorical.p
            shape = dim
        else:
            def _sample(shape=()): return Categorical.sample(p, shape)
            def _p(x): return Categorical.p(x, p)
            shape = p.size

        parameters = { Categorical.probabilities: Parameter(shape=shape, value=p) }
        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(p: np.ndarray, shape=()) -> np.ndarray:
        return np.random.choice(np.arange(p.size), p=p, size=shape)

    @staticmethod
    @numba.jit(nopython=True, forceobj=False)
    def p(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[x]

    @staticmethod
    @numba.jit(nopython=True, forceobj=False)
    def one_hot(samples: np.ndarray, size: int) -> np.ndarray:
        return np.eye(size)[samples]
