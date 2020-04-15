from probpy.core import RandomVariable
from probpy.distributions import *
import numpy as np
import heapq
from probpy.algorithms import mode_from_points


def normal_mode(rv: RandomVariable, **_): return rv.mu


def multivariate_normal_mode(rv: RandomVariable, **_): return rv.mu


def gamma_mode(rv: RandomVariable, **_): return (rv.a - 1) / rv.b


def beta_mode(rv: RandomVariable, **_): return (rv.a - 1) / (rv.a + rv.b - 2)


def dirichlet_mode(rv: RandomVariable, **_): return (rv.alpha - 1) / (rv.alpha.sum() - rv.alpha.size)


def categorical_mode(rv: RandomVariable, **_): return np.eye(rv.probabilities.size)[np.argmax(rv.probabilities)]


def normal_inverse_gamma_mode(rv: RandomVariable, **_): return np.array([rv.mu, rv.b / (rv.a + 3 / 2)])


def points_mode(rv: RandomVariable, samples=100, n=10):
    samples = rv.sample(size=samples)
    probabilities = rv.p(samples)

    if samples.ndim == 1: samples = samples.reshape(-1, 1)

    return mode_from_points(samples, probabilities, n=n)[0]


implemented = {
    normal: normal_mode,
    multivariate_normal: multivariate_normal_mode,
    gamma: gamma_mode,
    beta: beta_mode,
    points: points_mode,
    dirichlet: dirichlet_mode,
    categorical: categorical_mode,
    normal_inverse_gamma: normal_inverse_gamma_mode,
    generic: points_mode
}


def mode(rv: RandomVariable, **kwargs):
    """

    :param rv: random variable to find mode for
    :param kwargs: arguments to find method
    :return:
    """
    if rv.cls in implemented:
        return implemented[rv.cls](rv, **kwargs)

    raise NotImplementedError(f"Mode not implemented for {rv.cls.__class__}")
