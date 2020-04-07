from typing import Callable, Tuple
from .core import RandomVariable
import numpy as np


def uniform_importance_sampling(size: int,
                                function: Callable[[np.ndarray], np.ndarray],
                                domain: Tuple[np.ndarray, np.ndarray],
                                proposal: RandomVariable):
    """

    :param size: samples to use in integral
    :param function: function to integrate
    :param domain: domain to integrate over
    :param proposal: proposal distribution
    :return:
    """
    lower_bounds, upper_bounds = domain

    samples = proposal.sample(size=size)

    accepted = (lower_bounds < samples) & (samples < upper_bounds)

    if accepted.ndim == 2:
        accepted = accepted.all(axis=1)

    return (function(samples[accepted]) / proposal.p(samples[accepted])).sum(axis=0) / size


def expected_value(size: int,
                   function: Callable[[np.ndarray], np.ndarray],
                   distribution: RandomVariable):
    """

    :param size: samples to estimate expectation
    :param function: function to estimate it with
    :param distribution: expectation with respect to this distribution
    :return:
    """
    return function(distribution.sample(size=size)).mean(axis=0)


def posterior_predictive_integration(size: int,
                                     likelihood: Callable,
                                     priors: Tuple[RandomVariable]):
    samples = [prior.sample(size=size) for prior in priors]
    return likelihood(*samples).mean(axis=0)
