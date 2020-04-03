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


def attempt(samples: List[np.ndarray], match_for: Tuple[Distribution]):
    if match_for is not None:
        if len(match_for) == 1:
            matcher = match_for[0]

            if matcher in moment_matchers:
                samples = np.concatenate(samples, axis=1)
                return moment_matchers[matcher](samples)

            raise Exception(f"no matcher implemented for {matcher.__name__}")

        if len(samples) == len(match_for):
            matches = []
            for sample, match in zip(samples, match_for):
                if match in moment_matchers:
                    matches.append(moment_matchers[match](sample))
                else:
                    raise Exception(f"no matcher implemented for {match.__name__}")

            return matches

    return None

