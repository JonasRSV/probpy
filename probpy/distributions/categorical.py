import numpy as np

from probpy.core import Distribution, RandomVariable


class Categorical(Distribution):

    @classmethod
    def freeze(cls, p: np.ndarray = None) -> RandomVariable:
        if p is None:
            _sample = Categorical.sample
            _p = Categorical.p
        else:
            def _sample(shape=()): return Categorical.sample(p, shape)
            def _p(x): return Categorical.p(x, p)

        return RandomVariable(_sample, _p, shape=())

    @staticmethod
    def sample(p: np.ndarray, shape=()) -> np.ndarray:
        return np.random.choice(np.arange(p.size), p=p, size=shape)

    @staticmethod
    def p(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[x]

    @staticmethod
    def one_hot(samples: np.ndarray, size: int) -> np.ndarray:
        return np.eye(size)[samples]
