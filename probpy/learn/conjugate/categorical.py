from probpy.core import RandomVariable
from typing import Tuple
from probpy.distributions import categorical, dirichlet
from .identification import _check_no_none_parameters, _check_only_none_is
import numpy as np
import numba


class CategoricalDirichlet_PPrior:
    """Conjugate prior for categorical likelihood with unknown probability"""

    @staticmethod
    def is_conjugate(likelihood: RandomVariable, prior: RandomVariable):
        if prior.cls is dirichlet \
                and _check_no_none_parameters(prior) \
                and _check_only_none_is(likelihood, [categorical.probabilities]):
            return True
        return False

    @staticmethod
    def fast_loop(data: np.ndarray, categories: int):
        result = np.zeros(categories)
        for d in data:
            result[d] += 1

        return result

    @staticmethod
    def posterior(data: np.ndarray, _: RandomVariable, prior: RandomVariable) -> RandomVariable:
        data = np.array(data[0])
        if data.ndim == 0: data = data.reshape(-1)

        prior_alpha = prior.parameters[dirichlet.alpha].value

        if data.ndim == 1:
            posterior_alpha = prior_alpha + CategoricalDirichlet_PPrior.fast_loop(data, prior_alpha.size)
        else:
            posterior_alpha = prior_alpha + data.sum(axis=0)

        return dirichlet.med(alpha=posterior_alpha)
