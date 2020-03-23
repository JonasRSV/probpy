import numpy as np

from probpy.core import Distribution, RandomVariable


class Bernoulli(Distribution):

    @classmethod
    def freeze(cls, p: np.float32 = None) -> RandomVariable:
        if p is None:
            _sample = Bernoulli.sample
            _p = Bernoulli.p
        else:
            def _sample(shape=()): return Bernoulli.sample(p, shape)
            def _p(x): return Bernoulli.p(x, p)

        return RandomVariable(_sample, _p, shape=())

    @staticmethod
    def sample(p: np.float32, shape = ()) -> np.ndarray:
        if type(shape) == int:
            return (np.random.rand(shape) < p).astype(np.float32)
        return (np.random.rand(*shape) < p).astype(np.float32)

    @staticmethod
    def p(x: np.ndarray, p: np.float32) -> np.ndarray:
        res = np.zeros_like(x)
        res[x != 1.0] = 1 - p
        res[x == 1.0] = p
        return res

