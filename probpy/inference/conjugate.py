from probpy.core import RandomVariable
from typing import Tuple
from probpy.distributions import normal, multivariate_normal
import numpy as np


def _check_no_none_parameters(rv: RandomVariable):
    for parameter in rv.parameters.values():
        if parameter.value is None:
            return False
    return True


def _check_only_none_is(rv: RandomVariable, none_name: str):
    for name, parameter in rv.parameters.items():
        if name == none_name:
            if parameter.value is not None:
                return False
        else:
            if parameter.value is None:
                return False
    return True


class NormalNormal_MuPrior1D:

    @staticmethod
    def check(likelihood: RandomVariable, priors: Tuple[RandomVariable]):
        if _check_no_none_parameters(priors[0]) and _check_only_none_is(likelihood, normal.mu):
            return True
        return False

    @staticmethod
    def posterior(data: np.ndarray, likelihood: RandomVariable, priors: Tuple[RandomVariable]) -> RandomVariable:
        prior = priors[0]

        n = data.size

        prior_sigma = prior.parameters[normal.sigma].value
        prior_mu = prior.parameters[normal.mu].value

        sigma = likelihood.parameters[normal.sigma].value

        n_sigma = n / sigma
        _inv_prior_sigma = 1 / prior_sigma

        posterior_sigma = 1 / (n_sigma + _inv_prior_sigma)
        posterior_mu = posterior_sigma * ((prior_mu / prior_sigma) + (data.sum() / sigma))

        return normal.freeze(mu=posterior_mu, sigma=posterior_sigma)


class MultivariateNormalNormal_MuPrior:

    @staticmethod
    def check(likelihood: RandomVariable, priors: Tuple[RandomVariable]):
        if _check_no_none_parameters(priors[0]) and _check_only_none_is(likelihood, multivariate_normal.mu):
            return True
        return False

    @staticmethod
    def posterior(data: np.ndarray, likelihood: RandomVariable, priors: Tuple[RandomVariable]) -> RandomVariable:
        prior = priors[0]

        n = data.shape[0]

        inv_prior_sigma = np.linalg.inv(prior.parameters[multivariate_normal.sigma].value)
        prior_mu = prior.parameters[normal.mu].value

        inv_sigma = np.linalg.inv(likelihood.parameters[multivariate_normal.sigma].value)

        posterior_sigma = np.linalg.inv(inv_prior_sigma + n * inv_sigma)
        posterior_mu = posterior_sigma @ (inv_prior_sigma @ prior_mu + n * inv_sigma @ data.mean(axis=0))

        return multivariate_normal.freeze(mu=posterior_mu, sigma=posterior_sigma)
