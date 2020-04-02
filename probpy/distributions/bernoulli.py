import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class Bernoulli(Distribution):
    probability = "probability"

    @classmethod
    def med(cls, probability: np.float32 = None) -> RandomVariable:
        if probability is None:
            _sample = Bernoulli.sample
            _p = Bernoulli.p
        else:
            def _sample(size=()): return Bernoulli.sample(probability, size)
            def _p(x): return Bernoulli.p(x, probability)

        parameters = {Bernoulli.probability: Parameter(shape=(), value=probability)}
        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(probability: np.float32, size = ()) -> np.ndarray:
        if type(size) == int:
            return (np.random.rand(size) < probability).astype(np.float32)
        return (np.random.rand(*size) < probability).astype(np.float32)

    @staticmethod
    def p(x: np.ndarray, probability: np.float32) -> np.ndarray:
        res = np.zeros_like(x)
        res[x != 1.0] = 1 - probability
        res[x == 1.0] = probability
        return res

