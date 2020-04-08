from probpy.core import RandomVariable
from typing import Tuple
from probpy.distributions import normal, multivariate_normal
from probpy.inference.conjugate.identification import _check_no_none_parameters, _check_only_none_is


class NormalNormal_MuPrior1D:
    """Conjugate prior for univariate normal likelihood with unknown mean"""

    @staticmethod
    def is_conjugate(likelihood: RandomVariable, priors: Tuple[RandomVariable]):
        if priors[0].cls is normal \
                and _check_no_none_parameters(priors[0]) \
                and _check_only_none_is(likelihood, [normal.mu]):
            return True
        return False

    @staticmethod
    def posterior(likelihood: RandomVariable, priors: Tuple[RandomVariable]) -> RandomVariable:
        prior = priors[0]

        prior_mu = prior.parameters[normal.mu].value
        prior_sigma = prior.parameters[normal.sigma].value

        likelihood_sigma = likelihood.parameters[normal.sigma].value

        return normal.med(mu=prior_mu, sigma=prior_sigma + likelihood_sigma)


class MultivariateNormalNormal_MuPrior:
    """Conjugate prior for multivariate normal likelihood with unknown mean"""

    @staticmethod
    def is_conjugate(likelihood: RandomVariable, priors: Tuple[RandomVariable]):
        if priors[0].cls is multivariate_normal \
                and _check_no_none_parameters(priors[0]) \
                and _check_only_none_is(likelihood, [multivariate_normal.mu]):
            return True
        return False

    @staticmethod
    def posterior(likelihood: RandomVariable, priors: Tuple[RandomVariable]) -> RandomVariable:
        prior = priors[0]

        prior_mu = prior.parameters[normal.mu].value
        prior_sigma = prior.parameters[normal.sigma].value

        likelihood_sigma = likelihood.parameters[normal.sigma].value

        return multivariate_normal.med(mu=prior_mu, sigma=prior_sigma + likelihood_sigma)

