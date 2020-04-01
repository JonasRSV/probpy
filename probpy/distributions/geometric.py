import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class Geometric(Distribution):
    probability = "probability"

    @classmethod
    def med(cls, probability: np.float32 = None) -> RandomVariable:
        if probability is None:
            _sample = Geometric.sample
            _p = Geometric.p
        else:
            def _sample(size=()): return Geometric.sample(probability, size)
            def _p(x: np.ndarray): return Geometric.p(x, probability)

        parameters = {
            Geometric.probability: Parameter(shape=(), value=probability)
        }
        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(probability: np.float32, size = ()) -> np.ndarray:
        return np.random.geometric(probability, size=size)

    @staticmethod
    def p(x: np.ndarray, probability: np.float32) -> np.ndarray:
        return np.float_power(1 - probability, x - 1) * probability