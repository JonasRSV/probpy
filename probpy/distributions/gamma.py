import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter
from probpy.special import gamma


class Gamma(Distribution):
    a = "a"
    b = "b"

    @classmethod
    def freeze(cls, a: np.float32 = None, b: np.float32 = None) -> RandomVariable:
        if a is None and b is None:
            _sample = Gamma.sample
            _p = Gamma.p
        elif a is None:
            def _sample(a: np.ndarray, shape: np.ndarray = ()): return Gamma.sample(a, b, shape)
            def _p(x: np.ndarray, a: np.ndarray): return Gamma.p(x, a, b)
        elif b is None:
            def _sample(b: np.ndarray, shape: np.ndarray = ()): return Gamma.sample(a, b, shape)
            def _p(x: np.ndarray, b: np.ndarray): return Gamma.p(x, a, b)
        else:
            def _sample(shape=()): return Gamma.sample(a, b, shape)
            def _p(x): return Gamma.p(x, a, b)

        parameters = {
            Gamma.a: Parameter(shape=(), value=a),
            Gamma.b: Parameter(shape=(), value=b)
        }

        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(a: np.float32, b: np.float32, shape=()) -> np.ndarray:
        return np.random.gamma(a, 1 / b, size=shape)

    @staticmethod
    def p(x: np.ndarray, a: np.float32, b: np.float32) -> np.ndarray:
        normalizing_constant = np.float_power(b, a) / gamma(a)
        return np.float_power(x, a - 1) * np.exp(-b * x) * normalizing_constant

