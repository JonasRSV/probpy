from probpy.core import RandomVariable
from typing import Tuple
from probpy.distributions import gaussian_process, unknown
from .identification import _check_no_none_parameters
import numpy as np


class UnknownGaussianProcess_FunctionPrior:
    """Conjugate prior for points likelihood with unknown function"""

    @staticmethod
    def is_conjugate(_: RandomVariable, priors: Tuple[RandomVariable]):
        if priors[0].cls is gaussian_process \
                and _check_no_none_parameters(priors[0]):
            return True
        return False

    @staticmethod
    def posterior(data: np.ndarray, _: RandomVariable, priors: Tuple[RandomVariable]) -> RandomVariable:
        prior = priors[0]

        prior_mu = prior.parameters[gaussian_process.mu].value
        prior_sigma = prior.parameters[gaussian_process.sigma].value
        prior_domain = prior.parameters[gaussian_process.domain].value
        prior_codomain = prior.parameters[gaussian_process.codomain].value

        domain, codomain = data

        posterior_domain = np.concatenate([prior_domain, domain])
        posterior_codomain = np.concatenate([prior_codomain, codomain])

        return gaussian_process.med(mu=prior_mu,
                                    sigma=prior_sigma,
                                    domain=posterior_domain,
                                    codomain=posterior_codomain)
