from probpy.core import RandomVariable
from typing import Tuple
from probpy.distributions import multivariate_normal, unilinear
from .identification import _check_no_none_parameters, _check_only_none_is
import numpy as np


class UniLinearMultivariateNormal_VariablePrior:
    """Conjugate prior for univariate normal likelihood with unknown mean"""

    @staticmethod
    def is_conjugate(likelihood: RandomVariable, priors: Tuple[RandomVariable]):
        if priors[0].cls is multivariate_normal \
                and _check_no_none_parameters(priors[0]) \
                and _check_only_none_is(likelihood, [unilinear.variables]):
            return True
        return False

    @staticmethod
    def posterior(data: np.ndarray, likelihood: RandomVariable, priors: Tuple[RandomVariable]) -> RandomVariable:
        prior = priors[0]
        x, y = data
        x_dim = x.shape[0]

        x = np.concatenate([x, np.ones((x_dim, 1))], axis=1)  # Add bias term

        likelihood_sigma = np.eye(x_dim) / likelihood.parameters[unilinear.sigma].value

        prior_mu = prior.parameters[multivariate_normal.mu].value
        prior_sigma_inv = np.linalg.inv(prior.parameters[multivariate_normal.sigma].value)

        x_squiggle = x
        sigma_posterior = np.linalg.inv(x_squiggle.T @ likelihood_sigma @ x_squiggle + prior_sigma_inv)
        mu_posterior = (sigma_posterior @ x_squiggle.T) @ likelihood_sigma @ y \
                       + sigma_posterior @ prior_sigma_inv @ prior_mu

        return multivariate_normal.med(mu=mu_posterior, sigma=sigma_posterior)
