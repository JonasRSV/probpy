import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class Categorical(Distribution):
    """Categorical Distribution"""
    probabilities = "probabilities"

    @classmethod
    def med(cls, probabilities: np.ndarray = None, categories: int = None) -> RandomVariable:
        """

        :param probabilities: probability of categories
        :param categories: number of categories
        :return: RandomVariable
        """
        if probabilities is None:
            _sample = Categorical.sample
            _p = Categorical.p
            shape = categories
        else:
            def _sample(size: int = 1): return Categorical.sample(probabilities, size)
            def _p(x): return Categorical.p(x, probabilities)
            shape = probabilities.size

        parameters = {Categorical.probabilities: Parameter(shape=shape, value=probabilities)}
        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(probabilities: np.ndarray, size: int = 1) -> np.ndarray:
        """

        :param probabilities: probability of categories
        :param size: number of samples
        :return: array of samples
        """
        return Categorical.one_hot(
            np.random.choice(np.arange(probabilities.size), p=probabilities, size=size),
            size=probabilities.size)

    @staticmethod
    def p(x: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """

        :param x: samples
        :param probabilities: probability of categories
        :return: densities
        """
        if type(x) != np.ndarray: x = np.array(x)
        if type(probabilities) != np.ndarray: probabilities = np.array(probabilities)
        if x.ndim == 2: x = np.argmax(x, axis=1)

        if probabilities.ndim == 2: result = probabilities[:, x]
        else: result = probabilities[x]

        return result

    @staticmethod
    @numba.jit(nopython=True, forceobj=False)
    def one_hot(samples: np.ndarray, size: int) -> np.ndarray:
        return np.eye(size)[samples]
