import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class Exponential(Distribution):
    """Exponential distribution"""
    lam = "lam"

    @classmethod
    def med(cls, lam: np.float = None) -> RandomVariable:
        """

        :param lam: lambda, rate parameter
        :return: RandomVariable
        """

        if lam is None:
            _sample = Exponential.sample
            _p = Exponential.p
        else:
            def _sample(size: int = 1):
                return Exponential.sample(lam, size)

            def _p(x):
                return Exponential.p(x, lam)

        parameters = {Exponential.lam: Parameter(shape=(), value=lam)}
        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    def sample(lam: np.float, size: int = 1) -> np.ndarray:
        """

        :param lam: lambda, rate parameter
        :param size: number of samples
        :return: array of samples
        """
        return np.random.exponential(1 / lam, size=size)

    @staticmethod
    @numba.jit(fastmath=True, forceobj=False, nopython=True)
    def fast_p(x: np.ndarray, lam: np.float):
        return lam * np.exp(-lam * np.maximum(x, 0.0))

    @staticmethod
    def p(x: np.ndarray, lam: np.float) -> np.ndarray:
        """

        :param x: samples
        :param lam: lambda, rate parameter
        :return: densities
        """
        if type(x) != np.ndarray: x = np.array(x)
        if type(lam) != np.ndarray: lam = np.array(lam)
        lg_0 = x >= 0
        result = np.zeros_like(x)
        result[lg_0] = lam * np.exp(-lam * x[lg_0])

        return result

    @staticmethod
    def jit_probability(rv: RandomVariable):
        lam = rv.parameters[Exponential.lam].value

        _fast_p = Exponential.fast_p
        if lam is None:
            return _fast_p
        else:
            def fast_p(x: np.ndarray):
                return _fast_p(x, lam)

        fast_p = numba.jit(nopython=True, forceobj=False, fastmath=True)(fast_p)

        return fast_p
