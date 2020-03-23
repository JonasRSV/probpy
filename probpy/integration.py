from typing import Callable, Tuple
from .core import RandomVariable
import numpy as np


def uniform_importance_sampling(size: int,
                                function: Callable[[np.ndarray], np.ndarray],
                                domain: Tuple[np.ndarray, np.ndarray],
                                proposal: RandomVariable):
    lower_bounds, upper_bounds = domain

    samples = proposal.sample(shape=size)

    accepted = (lower_bounds < samples) & (samples < upper_bounds)

    if accepted.ndim == 2:
        accepted = accepted.all(axis=1)

    return (function(samples[accepted]) / proposal.p(samples[accepted])).sum(axis=0) / size
