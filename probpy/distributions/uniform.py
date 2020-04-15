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
    @numba.jit(nopython=True, forceobj=False)
    def fast_p(x: np.ndarray, a: np.float, b: np.float):
        return ((a < x) & (x < b)) / (b - a)

    @staticmethod
    def p(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if type(x) != np.ndarray: x = np.array(x)
        if type(a) != np.ndarray: a = np.array(a)
        if type(b) != np.ndarray: b = np.array(b)

        return Uniform.fast_p(x, a, b)

    @staticmethod
    def jit_probability(rv: RandomVariable):
        a = rv.parameters[Uniform.a].value
        b = rv.parameters[Uniform.b].value

        _fast_p = Uniform.fast_p
        if a is None and b is None:
            return _fast_p
        elif a is None:
            def fast_p(x: np.ndarray, a: np.float):
                return _fast_p(x, a, b)
        elif b is None:
            def fast_p(x: np.ndarray, b: np.float):
                return _fast_p(x, a, b)
        else:
            def fast_p(x: np.ndarray):
                return _fast_p(x, a, b)

        fast_p = numba.jit(nopython=True, forceobj=False, fastmath=True)(fast_p)

        return fast_p



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
    @numba.jit(nopython=True, fastmath=True, forceobj=False)
    def fast_p(x: np.ndarray, a: np.ndarray, b: np.ndarray):

        indicator_matrix = ((a < x) & (x < b))
        indicator_vector = np.array([np.all(indicator_matrix[i]) for i in range(len(x))])
        probability = 1 / np.prod(b - a)

        return indicator_vector * probability

    @staticmethod
    def p(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if type(x) != np.ndarray: x = np.array(x)
        if type(a) != np.ndarray: a = np.array(a)
        if type(b) != np.ndarray: b = np.array(b)
        if x.ndim == 1: x = x.reshape(-1, a.size)

        return MultiVariateUniform.fast_p(x, a, b)

    @staticmethod
    def jit_probability(rv: RandomVariable):
        a = rv.parameters[Uniform.a].value
        b = rv.parameters[Uniform.b].value

        _fast_p = MultiVariateUniform.fast_p
        if a is None and b is None:
            return _fast_p
        elif a is None:
            def fast_p(x: np.ndarray, a: np.float):
                return _fast_p(x, a, b)
        elif b is None:
            def fast_p(x: np.ndarray, b: np.float):
                return _fast_p(x, a, b)
        else:
            def fast_p(x: np.ndarray):
                return _fast_p(x, a, b)

        fast_p = numba.jit(nopython=True, forceobj=False, fastmath=True)(fast_p)

        return fast_p
