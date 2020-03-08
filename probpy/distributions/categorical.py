import numpy as np

from probpy.core import Distribution


class Categorical(Distribution):

    @staticmethod
    def sample(p: np.float32, shape=()) -> np.ndarray:
        return np.random.choice(np.arange(p.size), p=p, size=shape)

    @staticmethod
    def p(p: np.float32, x: np.ndarray) -> np.ndarray:
        return p[x]

    @staticmethod
    def one_hot(samples: np.ndarray, size: int) -> np.ndarray:
        return np.eye(size)[samples]
