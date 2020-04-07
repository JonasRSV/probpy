import numpy as np

from probpy.core import Distribution, RandomVariable, Parameter
from probpy.special import beta


class Beta(Distribution):
    """Beta distribution"""
    a = "a"
    b = "b"

    @classmethod
    def med(cls, a: np.float = None, b: np.float = None) -> RandomVariable:
        """

        :param a: alpha shape
        :param b: beta shape
        :return: RandomVariable
        """
        if a is None and b is None:
            _sample = Beta.sample
            _p = Beta.p
        elif a is None:
            def _sample(a: np.float, size: int = 1): return Beta.sample(a, b, size)
            def _p(x: np.ndarray, a: np.float): return Beta.p(x, a, b)
        elif b is None:
            def _sample(b: np.float, size: int = 1): return Beta.sample(a, b, size)
            def _p(x: np.ndarray, b: np.float): return Beta.p(x, a, b)
        else:
            def _sample(size: int = 1): return Beta.sample(a, b, size)
            def _p(x: np.ndarray): return Beta.p(x, a, b)

        parameters = {
            Beta.a: Parameter(shape=(), value=a),
            Beta.b: Parameter(shape=(), value=b)
        }

        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    def sample(a: np.float, b: np.float, size: int = 1) -> np.ndarray:
        """

        :param a: alpha shape
        :param b: beta shape
        :param size: number of samples
        :return: array of samples
        """
        return np.random.beta(a, b, size=size)

    @staticmethod
    def p(x: np.ndarray, a: np.float, b: np.float) -> np.ndarray:
        """

        :param x: samples
        :param a: alpha shape
        :param b: beta shape
        :return: densities
        """
        if type(x) != np.ndarray: x = np.array(x)
        # TODO: find out if there is a more numerically stable implementation
        return np.float_power(x, a - 1) * np.float_power(1 - x, b - 1) / beta(a, b)
