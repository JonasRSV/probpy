import numpy as np

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
            def _sample(size: int = 1): return Exponential.sample(lam, size)
            def _p(x): return Exponential.p(x, lam)

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
    def p(x: np.ndarray, lam: np.float) -> np.ndarray:
        """

        :param x: samples
        :param lam: lambda, rate parameter
        :return: densities
        """
        if type(x) != np.ndarray: x = np.array(x)
        result = np.zeros_like(x)
        lg_0 = x >= 0
        result[lg_0] = lam * np.exp(-lam * x[lg_0])
        return result

