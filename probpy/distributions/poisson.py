import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class Poisson(Distribution):
    """Poisson distribution"""
    lam = "lam"

    @classmethod
    def med(cls, lam: np.float32 = None) -> RandomVariable:
        """

        :param lam: rate
        :return: RandomVariable
        """
        if lam is None:
            _sample = Poisson.sample
            _p = Poisson.p
        else:
            def _sample(size: int = 1): return Poisson.sample(lam, size)
            def _p(x): return Poisson.p(x, lam)

        parameters = {Poisson.lam: Parameter(shape=(), value=lam)}
        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(lam: np.float32, size: int = 1) -> np.ndarray:
        return np.random.poisson(lam, size=size)

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def _factorial(x: np.ndarray):
        result = np.zeros(x.size)
        for i in range(x.size):
            result[i] = np.arange(1, x[i] + 1).prod()
        return result

    @staticmethod
    def p(x: np.ndarray, lam: np.float32) -> np.ndarray:
        if type(x) != np.ndarray: x = np.array(x)
        if type(lam) != np.ndarray: lam = np.array(lam)

        if lam.ndim != 0:
            raise Exception("Broadcasting on poisson not supported at the moment")

        if x.ndim == 0: x = x.reshape(-1)
        return (np.float_power(lam, x) * np.exp(-lam)) / Poisson._factorial(x)

