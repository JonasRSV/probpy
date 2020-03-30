import numpy as np
import numba
from typing import Tuple

from probpy.core import Distribution, RandomVariable, Parameter


class Uniform(Distribution):
    a = "a"
    b = "b"

    @classmethod
    def med(cls, a: np.float32 = None, b: np.float32 = None) -> RandomVariable:
        if a is None and b is None:
            _sample = Uniform.sample
            _p = Uniform.p
        elif a is None:
            def _sample(a: np.ndarray, shape: np.ndarray = ()): return Uniform.sample(a, b, shape)
            def _p(x: np.ndarray, a: np.ndarray): return Uniform.p(x, a, b)
        elif b is None:
            def _sample(b: np.ndarray, shape: np.ndarray = ()): return Uniform.sample(a, b, shape)
            def _p(x: np.ndarray, b: np.ndarray): return Uniform.p(x, a, b)
        else:
            def _sample(shape: np.ndarray = ()): return Uniform.sample(a, b, shape)
            def _p(x: np.ndarray): return Uniform.p(x, a, b)

        parameters = {
            Uniform.a: Parameter((), a),
            Uniform.b: Parameter((), b)
        }

        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(a: np.ndarray, b: np.ndarray, shape: np.ndarray = ()) -> np.ndarray:
        if type(shape) == int:
            return a + np.random.rand(shape) * (b - a)

        return a + np.random.rand(*shape) * (b - a)

    @staticmethod
    @numba.jit(nopython=True, forceobj=False)
    def p(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return ((a < x) & (x < b)).astype(np.float32) / (b - a)


class MultiVariateUniform(Distribution):
    a = "a"
    b = "b"

    @classmethod
    def med(cls, a: np.ndarray = None, b: np.ndarray = None, parameter_shape: Tuple = None) -> RandomVariable:
        if a is None and b is None:
            _sample = MultiVariateUniform.sample
            _p = MultiVariateUniform.p
            shape = parameter_shape
        elif a is None:
            def _sample(a: np.ndarray, shape: np.ndarray = ()): return MultiVariateUniform.sample(a, b, shape)
            def _p(x: np.ndarray, a: np.ndarray): return MultiVariateUniform.p(x, a, b)
            shape = b.size
        elif b is None:
            def _sample(b: np.ndarray, shape: np.ndarray = ()): return MultiVariateUniform.sample(a, b, shape)
            def _p(x: np.ndarray, b: np.ndarray): return MultiVariateUniform.p(x, a, b)
            shape = a.size
        else:
            def _sample(shape: np.ndarray = ()): return MultiVariateUniform.sample(a, b, shape)
            def _p(x: np.ndarray): return MultiVariateUniform.p(x, a, b)
            shape = a.size

        parameters = {
            MultiVariateUniform.a: Parameter(shape, a),
            MultiVariateUniform.b: Parameter(shape, b)
        }

        return RandomVariable(_sample, _p, shape=shape, parameters=parameters, cls=cls)

    @staticmethod
    def sample(a: np.ndarray, b: np.ndarray, shape: np.ndarray = ()) -> np.ndarray:
        if type(shape) == int:
            return a + np.random.rand(shape, a.shape[0]) * (b - a)

        if shape != ():
            if a.shape != shape[1:]:
                raise Exception(
                    f"shape of a needs to match provided shape a: {a.shape} -- provided: {shape} -- {a.shape} != {shape[1:]}")

            return a + np.random.rand(*shape) * (b - a)

        return a + np.random.rand(a.size) * (b - a)

    @staticmethod
    def p(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if x.ndim == 1: x = x.reshape(-1, a.size)
        return ((a < x) & (x < b)).all(axis=1).astype(np.float32) / np.product(b - a)
