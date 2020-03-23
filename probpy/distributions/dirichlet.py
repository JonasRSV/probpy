import numpy as np

from probpy.core import Distribution, RandomVariable
from probpy.special import gamma


class Dirichlet(Distribution):

    @classmethod
    def freeze(cls, alpha: np.ndarray = None) -> RandomVariable:
        if alpha is None:
            _sample = Dirichlet.sample
            _p = Dirichlet.p
            shape = None
        else:
            def _sample(shape=()): return Dirichlet.sample(alpha, shape)
            def _p(x): return Dirichlet.p(x, alpha)
            shape = (alpha.size, )

        return RandomVariable(_sample, _p, shape)

    @staticmethod
    def sample(alpha: np.ndarray, shape=()) -> np.ndarray:
        return np.random.dirichlet(alpha, size=shape)

    @staticmethod
    def p(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        # TODO: find out if there is a more numerically stable implementation
        normalizing_constant = np.prod(gamma(alpha)) / gamma(alpha.sum())
        return np.float_power(x, alpha - 1).prod(axis=1) / normalizing_constant

