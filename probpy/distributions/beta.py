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
            def _sample(a: np.float, size: int = 1):
                return Beta.sample(a, b, size)

            def _p(x: np.ndarray, a: np.float):
                return Beta.p(x, a, b)
        elif b is None:
            def _sample(b: np.float, size: int = 1):
                return Beta.sample(a, b, size)

            def _p(x: np.ndarray, b: np.float):
                return Beta.p(x, a, b)
        else:
            def _sample(size: int = 1):
                return Beta.sample(a, b, size)

            def _p(x: np.ndarray):
                return Beta.p(x, a, b)

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
    def _p(x: np.ndarray, a: np.float, b: np.float):
        return np.float_power(x, a - 1) * np.float_power(1 - x, b - 1) / beta(a, b)

    @staticmethod
    def p(x: np.ndarray, a: np.float, b: np.float) -> np.ndarray:
        """

        :param x: samples
        :param a: alpha shape
        :param b: beta shape
        :return: densities
        """
        # TODO: find out if there is a more numerically stable implementation

        if type(x) != np.ndarray: x = np.array(x)
        if type(a) != np.ndarray: a = np.array(a)
        if type(b) != np.ndarray: b = np.array(b)

        if a.ndim == 1 or b.ndim == 1:  # broadcast
            a = a.reshape(-1)
            b = b.reshape(-1)

            if a.size == 1:
                a = np.repeat(a, b.size)
            elif b.size == 1:
                b = np.repeat(b, a.size)
            elif a.size == b.size:
                pass
            else:
                raise Exception(f"Broadcasting beta with shapes {a.shape} {b.shape} does not work")

            result = []
            for i in range(a.size):
                result.append(
                    Beta._p(x, a[i], b[i])
                )
        else:
            result = Beta._p(x, a, b)

        return np.array(result)
