from probpy.core import RandomVariable
from typing import Tuple
from probpy.distributions import poisson, gamma
from .identification import _check_no_none_parameters, _check_only_none_is
import numpy as np


class PoissonGamma_LambdaPrior:
    """Conjugate prior for poisson likelihood with unknown rate"""

    @staticmethod
    def is_conjugate(likelihood: RandomVariable, prior: RandomVariable):
        if prior.cls is gamma \
                and _check_no_none_parameters(prior) \
                and _check_only_none_is(likelihood, [poisson.lam]):
            return True
        return False

    @staticmethod
    def posterior(data: np.ndarray, _: RandomVariable, prior: RandomVariable) -> RandomVariable:
        data = np.array(data[0])
        n = data.size

        prior_alpha = prior.parameters[gamma.a].value
        prior_beta = prior.parameters[gamma.b].value

        posterior_alpha = prior_alpha + data.sum()
        posterior_beta = prior_beta + n

        return gamma.med(a=posterior_alpha, b=posterior_beta)
