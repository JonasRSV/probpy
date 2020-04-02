from typing import Callable, Tuple
from .core import RandomVariable
import numpy as np
import numba


@numba.jit(nopython=False, forceobj=True)
def uniform_importance_sampling(size: int,
                                function: Callable[[np.ndarray], np.ndarray],
                                domain: Tuple[np.ndarray, np.ndarray],
                                proposal: RandomVariable):
    lower_bounds, upper_bounds = domain

    samples = proposal.sample(size=size)

    accepted = (lower_bounds < samples) & (samples < upper_bounds)

    if accepted.ndim == 2:
        accepted = accepted.all(axis=1)

    return (function(samples[accepted]) / proposal.p(samples[accepted])).sum(axis=0) / size


@numba.jit(nopython=False, forceobj=True)
def expected_value(size: int,
                   function: Callable[[np.ndarray], np.ndarray],
                   distribution: RandomVariable):

    return function(distribution.sample(size=size)).mean(axis=0)

