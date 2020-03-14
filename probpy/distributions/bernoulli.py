import numpy as np

from probpy.core import Distribution, FrozenDistribution


class Bernoulli(Distribution):

    @classmethod
    def freeze(cls, p: np.float32) -> FrozenDistribution:
        return FrozenDistribution(cls, p)

    @staticmethod
    def sample(p: np.float32, shape = ()) -> np.ndarray:
        if type(shape) == int:
            return (np.random.rand(shape) < p).astype(np.float32)
        return (np.random.rand(*shape) < p).astype(np.float32)

    @staticmethod
    def p(p: np.float32, x: np.ndarray) -> np.ndarray:
        res = np.zeros_like(x)
        res[x != 1.0] = 1 - p
        res[x == 1.0] = p
        return res

