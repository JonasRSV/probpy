from probpy.core import RandomVariable
from typing import Tuple
from probpy.distributions import multinomial, dirichlet
from .identification import _check_no_none_parameters, _check_only_none_is
import numpy as np


class MultinomialDirichlet_PPrior:
    """Conjugate prior for multinomial likelihood with unknown probability"""

    @staticmethod
    def is_conjugate(likelihood: RandomVariable, priors: Tuple[RandomVariable]):
        if priors[0].cls is dirichlet \
                and _check_no_none_parameters(priors[0]) \
                and _check_only_none_is(likelihood, [multinomial.probabilities]):
            return True
        return False

    @staticmethod
    def posterior(data: np.ndarray, _: RandomVariable, priors: Tuple[RandomVariable]) -> RandomVariable:
        data = np.array(data)
        prior = priors[0]

        prior_alpha = prior.parameters[dirichlet.alpha].value

        posterior_alpha = prior_alpha + data.sum(axis=0)

        return dirichlet.med(alpha=posterior_alpha)
