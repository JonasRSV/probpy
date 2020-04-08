import numpy as np
import numba
from typing import Tuple

from probpy.core import Distribution, RandomVariable, Parameter


class Uniform(Distribution):
    """Uniform Distribution"""
    a = "a"
    b = "b"

    @classmethod
    def med(cls, a: np.float = None, b: np.float = None) -> RandomVariable:
        """

        :param a: lower bound
        :param b: upper bound
        :return: RandomVariable
        """
        if a is None and b is None:
            _sample = Uniform.sample
            _p = Uniform.p
        elif a is None:
            def _sample(a: np.ndarray, size: int = 1): return Uniform.sample(a, b, size)
            def _p(x: np.ndarray, a: np.ndarray): return Uniform.p(x, a, b)
        elif b is None:
            def _sample(b: np.ndarray, size: int = 1): return Uniform.sample(a, b, size)
            def _p(x: np.ndarray, b: np.ndarray): return Uniform.p(x, a, b)
        else:
            def _sample(size: int = 1): return Uniform.sample(a, b, size)
            def _p(x: np.ndarray): return Uniform.p(x, a, b)

        parameters = {
            Uniform.a: Parameter((), a),
            Uniform.b: Parameter((), b)
        }

        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(a: np.float, b: np.float, size: int = 1) -> np.ndarray:
        return np.array(a + np.random.rand(size) * (b - a))

    @staticmethod
    def p(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if type(x) != np.ndarray: x = np.array(x)
        if type(a) != np.ndarray: a = np.array(a)
        if type(b) != np.ndarray: b = np.array(b)

        if a.ndim != 0 or b.ndim != 0:
            raise Exception("Broadcasting on uniform not supported at the moment")
        return ((a < x) & (x < b)).astype(np.float) / (b - a)


class MultiVariateUniform(Distribution):
    """Multivariate Uniform distribution"""
    a = "a"
    b = "b"

    @classmethod
    def med(cls, a: np.ndarray = None, b: np.ndarray = None, dimension: Tuple = None) -> RandomVariable:
        """

        :param a: lower bound
        :param b: upper bound
        :param dimension: dimension of r.v
        :return: RandomVariable
        """
        if a is None and b is None:
            _sample = MultiVariateUniform.sample
            _p = MultiVariateUniform.p
            shape = dimension
        elif a is None:
            def _sample(a: np.ndarray, size: int = 1): return MultiVariateUniform.sample(a, b, size)
            def _p(x: np.ndarray, a: np.ndarray): return MultiVariateUniform.p(x, a, b)
            shape = b.size
        elif b is None:
            def _sample(b: np.ndarray, size: int = 1): return MultiVariateUniform.sample(a, b, size)
            def _p(x: np.ndarray, b: np.ndarray): return MultiVariateUniform.p(x, a, b)
            shape = a.size
        else:
            def _sample(size: int = 1): return MultiVariateUniform.sample(a, b, size)
            def _p(x: np.ndarray): return MultiVariateUniform.p(x, a, b)
            shape = a.size

        parameters = {
            MultiVariateUniform.a: Parameter(shape, a),
            MultiVariateUniform.b: Parameter(shape, b)
        }

        return RandomVariable(_sample, _p, shape=shape, parameters=parameters, cls=cls)

    @staticmethod
    def sample(a: np.ndarray, b: np.ndarray, size: int = 1) -> np.ndarray:
        return a + np.random.rand(size, a.size) * (b - a)

    @staticmethod
    def p(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if type(x) != np.ndarray: x = np.array(x)
        if type(a) != np.ndarray: a = np.array(a)
        if type(b) != np.ndarray: b = np.array(b)

        if a.ndim != 1 or b.ndim != 1:
            raise Exception("Broadcasting on uniform not supported at the moment")

        if x.ndim == 1: x = x.reshape(-1, a.size)
        return ((a < x) & (x < b)).all(axis=1).astype(np.float) / np.product(b - a)
