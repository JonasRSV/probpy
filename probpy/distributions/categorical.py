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
            def _sample(size: int = 1): return Categorical.sample(probabilities, size)
            def _p(x): return Categorical.p(x, probabilities)
            shape = probabilities.size

        parameters = {Categorical.probabilities: Parameter(shape=shape, value=probabilities)}
        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(probabilities: np.ndarray, size: int = 1) -> np.ndarray:
        return Categorical.one_hot(
            np.random.choice(np.arange(probabilities.size), p=probabilities, size=size),
            size=probabilities.size)

    @staticmethod
    def p(x: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        if type(x) != np.ndarray: x = np.array(x)
        if x.ndim == 2: x = np.argmax(x, axis=1)
        return probabilities[x]

    @staticmethod
    @numba.jit(nopython=True, forceobj=False)
    def one_hot(samples: np.ndarray, size: int) -> np.ndarray:
        return np.eye(size)[samples]
