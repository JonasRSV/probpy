import numpy as np

from probpy.core import Distribution, RandomVariable


class Unknown(Distribution):

    @classmethod
    def med(cls) -> RandomVariable:
        return RandomVariable(Unknown.sample, Unknown.p, shape=None, parameters={}, cls=cls)

    @staticmethod
    def sample(*args, shape: np.ndarray = ()) -> np.ndarray:
        raise Exception("Cannot sample from an unknown distribution")

    @staticmethod
    def p(x: np.ndarray, *args) -> np.ndarray:
        raise Exception("Cannot get probability from an unknown distribution")

