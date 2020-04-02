from probpy.core import RandomVariable
from typing import Tuple
from probpy.distributions import exponential, gamma
from .identification import _check_no_none_parameters, _check_only_none_is
import numpy as np


class ExponentialGamma_LambdaPrior:
    """Conjugate prior for exponential likelihood with unknown rate"""

    @staticmethod
    def is_conjugate(likelihood: RandomVariable, priors: Tuple[RandomVariable]):
        if priors[0].cls is gamma \
                and _check_no_none_parameters(priors[0]) \
                and _check_only_none_is(likelihood, [exponential.lam]):
            return True
        return False

    @staticmethod
    def posterior(data: np.ndarray, _: RandomVariable, priors: Tuple[RandomVariable]) -> RandomVariable:
        data = data[0]
        prior = priors[0]

        n = data.size

        prior_alpha = prior.parameters[gamma.a].value
        prior_beta = prior.parameters[gamma.b].value

        posterior_alpha = prior_alpha + n
        posterior_beta = prior_beta + data.sum()

        return gamma.med(a=posterior_alpha, b=posterior_beta)
