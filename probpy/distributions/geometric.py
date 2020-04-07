import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class Geometric(Distribution):
    """Geometric distribution"""
    probability = "probability"

    @classmethod
    def med(cls, probability: np.float = None) -> RandomVariable:
        """

        :param probability: probability of success
        :return: RandomVariable
        """
        if probability is None:
            _sample = Geometric.sample
            _p = Geometric.p
        else:
            def _sample(size: int = 1): return Geometric.sample(probability, size)
            def _p(x: np.ndarray): return Geometric.p(x, probability)

        parameters = {
            Geometric.probability: Parameter(shape=(), value=probability)
        }
        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(probability: np.float, size: int = 1) -> np.ndarray:
        return np.random.geometric(probability, size=size)

    @staticmethod
    def p(x: np.ndarray, probability: np.float) -> np.ndarray:
        if type(x) != np.ndarray: x = np.array(x)
        return np.float_power(1 - probability, x - 1) * probability
