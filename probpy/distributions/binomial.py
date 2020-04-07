import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class Binomial(Distribution):
    """Binomial distribution"""
    n = "n"
    probability = "probability"

    @classmethod
    def med(cls, n: int = None, probability: np.float32 = None) -> RandomVariable:
        """

        :param n: number of observations
        :param probability: probability of positive observation
        :return: RandomVariable
        """
        if n is None and probability is None:
            _sample = Binomial.sample
            _p = Binomial.p
        elif n is None:
            def _sample(n: np.int, size: int = 1): return Binomial.sample(n, probability, size)
            def _p(x: np.ndarray, n: np.int): return Binomial.p(x, n, probability)
        elif probability is None:
            def _sample(probability: np.float, size: int = 1): return Binomial.sample(n, probability, size)
            def _p(x: np.ndarray, probability: np.float): return Binomial.p(x, n, probability)
        else:
            def _sample(size: int = 1): return Binomial.sample(n, probability, size)
            def _p(x: np.ndarray): return Binomial.p(x, n, probability)

        parameters = {
            Binomial.n: Parameter(shape=(), value=n),
            Binomial.probability: Parameter(shape=(), value=probability)
        }

        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=True, forceobj=False)
    def _combinations_high_n(n, x):
        # Not broadcastable but deals well with large n
        # Might suffer some precision problems though
        terms = n - x
        res = 1
        for t in range(terms):
            res *= ((n - t) / (terms - t))
        return res

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(n: int, probability: np.float, size: int = 1) -> np.ndarray:
        """

        :param n: number of observations
        :param probability: probability of positive observations
        :param size: number of samples
        :return: array of samples
        """
        return np.random.binomial(n, probability, size=size)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def p(x: np.ndarray, n: int, probability: np.float) -> np.ndarray:
        """

        :param x: samples
        :param n: number of observations
        :param probability: probability of positive observatiosn
        :return: array of samples
        """
        if type(x) != np.ndarray: x = np.array(x)
        if x.ndim == 0: x = x.reshape(-1)
        constants = np.array([Binomial._combinations_high_n(n, _x) for _x in x])
        return constants * np.float_power(probability, x) * np.float_power(1 - probability, n - x)
