import numpy as np

from probpy.core import Distribution, FrozenDistribution


class Categorical(Distribution):

    @classmethod
    def freeze(cls, p: np.ndarray) -> FrozenDistribution:
        return FrozenDistribution(cls, p)

    @staticmethod
    def sample(p: np.ndarray, shape=()) -> np.ndarray:
        return np.random.choice(np.arange(p.size), p=p, size=shape)

    @staticmethod
    def p(p: np.ndarray, x: np.ndarray) -> np.ndarray:
        return p[x]

    @staticmethod
    def one_hot(samples: np.ndarray, size: int) -> np.ndarray:
        return np.eye(size)[samples]
