import numpy as np

from probpy.core import Distribution


class Exponential(Distribution):

    @staticmethod
    def sample(lam: np.float32, shape=()) -> np.ndarray:
        return np.random.exponential(1 / lam, size=shape)

    @staticmethod
    def p(lam: np.float32, x: np.ndarray) -> np.ndarray:
        return lam * np.exp(-lam * x)

