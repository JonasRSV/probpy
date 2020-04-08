import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter
from probpy.special import gamma


class Dirichlet(Distribution):
    """Dirichlet Distribution"""
    alpha = "alpha"

    @classmethod
    def med(cls, alpha: np.ndarray = None, categories: int = None) -> RandomVariable:
        """

        :param alpha: probability weights
        :param categories: number of categories
        :return:
        """
        if alpha is None:
            _sample = Dirichlet.sample
            _p = Dirichlet.p
            shape = categories
        else:
            def _sample(size: int = 1): return Dirichlet.sample(alpha, size)
            def _p(x): return Dirichlet.p(x, alpha)
            shape = alpha.size

        parameters = {Dirichlet.alpha: Parameter(shape=shape, value=alpha)}
        return RandomVariable(_sample, _p, shape, parameters=parameters, cls=cls)

    @staticmethod
    def sample(alpha: np.ndarray, size: int = 1) -> np.ndarray:
        """

        :param alpha: probability weights
        :param size: number of samples
        :return: array of samples
        """
        return np.random.dirichlet(alpha, size=size)

    @staticmethod
    def p(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """

        :param x: samples
        :param alpha: probability weights
        :return: densities
        """
        # TODO: find out if there is a more numerically stable implementation
        if type(x) != np.ndarray: x = np.array(x)
        if type(alpha) != np.ndarray: alpha = np.array(x)
        if x.ndim == 1: x = x.reshape(1, -1)

        if alpha.ndim == 2: raise Exception("Broadcasting on dirichlet not supported at the moment")

        normalizing_constant = np.prod(gamma(alpha)) / gamma(alpha.sum())
        return np.float_power(x, alpha - 1).prod(axis=1) / normalizing_constant

