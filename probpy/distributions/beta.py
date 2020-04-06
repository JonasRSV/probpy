import numpy as np

from probpy.core import Distribution, RandomVariable, Parameter
from probpy.special import beta


class Beta(Distribution):
    a = "a"
    b = "b"

    @classmethod
    def med(cls, a: np.float32 = None, b: np.float32 = None) -> RandomVariable:
        if a is None and b is None:
            _sample = Beta.sample
            _p = Beta.p
        elif a is None:
            def _sample(a: np.ndarray, size: np.ndarray = ()): return Beta.sample(a, b, size)
            def _p(x: np.ndarray, a: np.ndarray): return Beta.p(x, a, b)
        elif b is None:
            def _sample(b: np.ndarray, size: np.ndarray = ()): return Beta.sample(a, b, size)
            def _p(x: np.ndarray, b: np.ndarray): return Beta.p(x, a, b)
        else:
            def _sample(size: np.ndarray = ()): return Beta.sample(a, b, size)
            def _p(x: np.ndarray): return Beta.p(x, a, b)

        parameters = {
            Beta.a: Parameter(shape=(), value=a),
            Beta.b: Parameter(shape=(), value=b)
        }

        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    def sample(a: np.float32, b: np.float32, size=()) -> np.ndarray:
        return np.random.beta(a, b, size=size)

    @staticmethod
    def p(x: np.ndarray, a: np.float32, b: np.float32) -> np.ndarray:
        if type(x) != np.ndarray: x = np.array(x)
        # TODO: find out if there is a more numerically stable implementation
        return np.float_power(x, a - 1) * np.float_power(1 - x, b - 1) / beta(a, b)
