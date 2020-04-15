import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class Generic(Distribution):
    sampling_function="sampling function"
    probability_function="density function"
    fast_probability_function="fast probability function"
    """Generic distribution"""
    @classmethod
    def med(cls, sampling=None, probability=None, fast_p=None) -> RandomVariable:
        """

        :param sampling: sampling function
        :param probability: probability function
        :param fast_p: numba jitted probability function
        :return:
        """
        if sampling is None:
            def sampling(*args, **kwargs): raise NotImplementedError("Sampling not implemented in this Generic")

        if probability is None:
            def probability(*args, **kwargs): raise NotImplementedError("Probability not implemented in this Generic")

        parameters = {
            Generic.sampling_function: Parameter(None, sampling),
            Generic.probability_function: Parameter(None, probability),
            Generic.fast_probability_function: Parameter(None, fast_p)
        }

        return RandomVariable(sampling, probability, shape=None, parameters=parameters, cls=cls)

    @staticmethod
    def sample(*arg, **kwargs) -> np.ndarray:
        raise NotImplementedError("Static call of Points is not supported")

    @staticmethod
    def p(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Static call of Points is not supported")

    @staticmethod
    def jit_probability(rv: RandomVariable):
        fast_p = rv.parameters[Generic.fast_probability_function].value
        if fast_p is not None:
            return fast_p
        else:
            raise Exception("generic function did not have a numba probability")

