import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class Categorical(Distribution):
    probabilities = "probabilities"

    @classmethod
    def med(cls, probabilities: np.ndarray = None, dim: int = None) -> RandomVariable:
        if probabilities is None:
            _sample = Categorical.sample
            _p = Categorical.p
            shape = dim
        else:
            def _sample(shape=()): return Categorical.sample(probabilities, shape)
            def _p(x): return Categorical.p(x, probabilities)
            shape = probabilities.size

        parameters = { Categorical.probabilities: Parameter(shape=shape, value=probabilities) }
        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(probabilities: np.ndarray, shape=()) -> np.ndarray:
        return np.random.choice(np.arange(probabilities.size), p=probabilities, size=shape)

    @staticmethod
    @numba.jit(nopython=True, forceobj=False)
    def p(x: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        return probabilities[x]

    @staticmethod
    @numba.jit(nopython=True, forceobj=False)
    def one_hot(samples: np.ndarray, size: int) -> np.ndarray:
        return np.eye(size)[samples]
