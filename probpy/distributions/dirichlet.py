import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter
from probpy.special import gamma


class Dirichlet(Distribution):
    alpha = "alpha"

    @classmethod
    def med(cls, alpha: np.ndarray = None, dim: int = None) -> RandomVariable:
        if alpha is None:
            _sample = Dirichlet.sample
            _p = Dirichlet.p
            shape = dim
        else:
            def _sample(size=()): return Dirichlet.sample(alpha, size)
            def _p(x): return Dirichlet.p(x, alpha)
            shape = alpha.size

        parameters = {Dirichlet.alpha: Parameter(shape=shape, value=alpha)}
        return RandomVariable(_sample, _p, shape, parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(alpha: np.ndarray, size=()) -> np.ndarray:
        return np.random.dirichlet(alpha, size=size)

    @staticmethod
    def p(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        # TODO: find out if there is a more numerically stable implementation
        if type(x) != np.ndarray: x = np.array(x)
        if x.ndim == 1: x = x.reshape(1, -1)

        normalizing_constant = np.prod(gamma(alpha)) / gamma(alpha.sum())
        return np.float_power(x, alpha - 1).prod(axis=1) / normalizing_constant

