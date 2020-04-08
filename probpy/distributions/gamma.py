import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter
from probpy.special import gamma


class Gamma(Distribution):
    """Gamma distribution"""
    a = "a"
    b = "b"

    @classmethod
    def med(cls, a: np.float = None, b: np.float = None) -> RandomVariable:
        """

        :param a: shape
        :param b: rate
        :return: RandomVariable
        """
        if a is None and b is None:
            _sample = Gamma.sample
            _p = Gamma.p
        elif a is None:
            def _sample(a: np.float, size: int = 1): return Gamma.sample(a, b, size)
            def _p(x: np.ndarray, a: np.float): return Gamma.p(x, a, b)
        elif b is None:
            def _sample(b: np.float, size: int = 1): return Gamma.sample(a, b, size)
            def _p(x: np.ndarray, b: np.float): return Gamma.p(x, a, b)
        else:
            def _sample(size: int = 1): return Gamma.sample(a, b, size)
            def _p(x): return Gamma.p(x, a, b)

        parameters = {
            Gamma.a: Parameter(shape=(), value=a),
            Gamma.b: Parameter(shape=(), value=b)
        }

        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    def sample(a: np.float, b: np.float, size: int = 1) -> np.ndarray:
        """

        :param a: shape
        :param b: rate
        :param size: number of samples
        :return: array of samples
        """
        return np.random.gamma(a, 1 / b, size=size)

    @staticmethod
    def _p(x: np.ndarray, a: np.float, b: np.float):
        normalizing_constant = np.float_power(b, a) / gamma(a)
        return np.float_power(x, a - 1) * np.exp(-b * x) * normalizing_constant

    @staticmethod
    def p(x: np.ndarray, a: np.float, b: np.float) -> np.ndarray:
        """

        :param x: samples
        :param a: shape
        :param b: rate
        :return: densities
        """
        if type(x) != np.ndarray: x = np.array(x)
        if type(a) != np.ndarray: a = np.array(a)
        if type(b) != np.ndarray: b = np.array(b)

        #print(x.shape, a.shape, b.shape)

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
                    Gamma._p(x, a[i], b[i])
                )
        else:
            result = Gamma._p(x, a, b)

        return np.array(result)


