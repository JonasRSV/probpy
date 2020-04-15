import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class Bernoulli(Distribution):
    """Bernoulli Distribution"""
    probability = "probability"

    @classmethod
    def med(cls, probability: np.float32 = None) -> RandomVariable:
        """

        :param probability: probability of positive outcome
        :return: RandomVariable
        """
        if probability is None:
            _sample = Bernoulli.sample
            _p = Bernoulli.p
        else:
            def _sample(size: int = 1):
                return Bernoulli.sample(probability, size)

            def _p(x):
                return Bernoulli.p(x, probability)

        parameters = {Bernoulli.probability: Parameter(shape=(), value=probability)}
        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    def sample(probability: np.float32, size: int = 1) -> np.ndarray:
        """

        :param probability: probability of positive outcome
        :param size: number of samples
        :return: array of samples
        """
        return (np.random.rand(size) < probability).astype(np.float32)

    @staticmethod
    def p(x: np.ndarray, probability: np.float32) -> np.ndarray:
        """

        :param x:
        :param probability: probability of positive outcome
        :return: array of samples
        """
        if type(x) != np.ndarray: x = np.array(x)
        if type(probability) != np.ndarray: probability = np.array(probability)
        result = np.zeros_like(x)
        result[x != 1.0] = 1 - probability
        result[x == 1.0] = probability

        return result
