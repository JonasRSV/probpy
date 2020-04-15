from probpy.core import Distribution
from typing import List, Tuple
import numpy as np
from .normal import (multivariate_normal_matcher, univariate_normal_matcher)
from .exponential import exponential_matcher

from probpy.distributions import (normal,
                                  multivariate_normal,
                                  bernoulli,
                                  categorical,
                                  exponential,
                                  binomial,
                                  multinomial,
                                  poisson,
                                  geometric,
                                  unilinear)

moment_matchers = {
    normal: univariate_normal_matcher,
    multivariate_normal: multivariate_normal_matcher,
    exponential: exponential_matcher
}


def attempt(samples: np.ndarray, match_for: Distribution):
    if match_for in moment_matchers:
        return moment_matchers[match_for](samples)
    return None

