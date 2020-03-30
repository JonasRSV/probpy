import numpy as np

from probpy.core import Distribution, RandomVariable


class Generic(Distribution):
    @classmethod
    def med(cls, sampling=None, probability=None) -> RandomVariable:
        if sampling is None:
            def sampling(*args, **kwargs): raise NotImplementedError("Sampling not implemented in this Generic")

        if probability is None:
            def probability(*args, **kwargs): raise NotImplementedError("Probability not implemented in this Generic")

        return RandomVariable(sampling, probability, shape=None, parameters={}, cls=cls)

    @staticmethod
    def sample(*arg, **kwargs) -> np.ndarray:
        raise NotImplementedError("Static call of Points is not supported")

    @staticmethod
    def p(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Static call of Points is not supported")
