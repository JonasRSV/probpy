from probpy.core import RandomVariable
from typing import Tuple
from probpy.distributions import normal, multivariate_normal, normal_inverse_gamma
from .identification import _check_no_none_parameters, _check_only_none_is
import numpy as np


class NormalNormal_MuPrior1D:
    """Conjugate prior for univariate normal likelihood with unknown mean"""

    @staticmethod
    def is_conjugate(likelihood: RandomVariable, prior: RandomVariable):
        if prior.cls is normal \
                and _check_no_none_parameters(prior) \
                and _check_only_none_is(likelihood, [normal.mu]):
            return True
        return False

    @staticmethod
    def posterior(data: np.ndarray, likelihood: RandomVariable, prior: RandomVariable) -> RandomVariable:
        data = np.array(data[0])

        n = data.size

        prior_sigma = prior.parameters[normal.sigma].value
        prior_mu = prior.parameters[normal.mu].value

        sigma = likelihood.parameters[normal.sigma].value

        n_sigma = n / sigma
        _inv_prior_sigma = 1 / prior_sigma

        posterior_sigma = 1 / (n_sigma + _inv_prior_sigma)
        posterior_mu = posterior_sigma * ((prior_mu / prior_sigma) + (data.sum() / sigma))

        return normal.med(mu=posterior_mu, sigma=posterior_sigma)


class NormalNormal_NormalInverseGammaPrior1D:
    """Conjugate prior for univariate normal likelihood with unknown mean and variance"""

    @staticmethod
    def is_conjugate(likelihood: RandomVariable, prior: RandomVariable):
        if prior.cls is normal_inverse_gamma \
                and _check_no_none_parameters(prior) \
                and _check_only_none_is(likelihood, [normal.mu, normal.sigma]):
            return True
        return False

    @staticmethod
    def posterior(data: np.ndarray, _: RandomVariable, prior: RandomVariable) -> RandomVariable:
        data = np.array(data[0])
        if data.ndim == 0: data = data.reshape(-1)

        n = data.shape[0]

        prior_mu = prior.parameters[normal_inverse_gamma.mu].value
        prior_lam = prior.parameters[normal_inverse_gamma.lam].value
        prior_a = prior.parameters[normal_inverse_gamma.a].value
        prior_b = prior.parameters[normal_inverse_gamma.b].value

        posterior_mu = (prior_lam * prior_mu + n * data.mean()) / (prior_lam + n)
        posterior_lam = prior_lam + n
        posterior_a = prior_a + n / 2
        posterior_b = prior_b + 1 / 2 * np.square(data - data.mean()).sum() \
                      + (n * prior_lam) / (prior_lam + n) \
                      * np.square(data.mean() - prior_mu) / 2

        return normal_inverse_gamma.med(mu=posterior_mu, lam=posterior_lam, a=posterior_a, b=posterior_b)


class MultivariateNormalNormal_MuPrior:
    """Conjugate prior for multivariate normal likelihood with unknown mean"""

    @staticmethod
    def is_conjugate(likelihood: RandomVariable, prior: RandomVariable):
        if prior.cls is multivariate_normal \
                and _check_no_none_parameters(prior) \
                and _check_only_none_is(likelihood, [multivariate_normal.mu]):
            return True
        return False

    @staticmethod
    def posterior(data: np.ndarray, likelihood: RandomVariable, prior: RandomVariable) -> RandomVariable:
        data = np.array(data[0])
        if data.ndim == 1: data = data.reshape(1, -1)

        n = data.shape[0]

        inv_prior_sigma = np.linalg.inv(prior.parameters[multivariate_normal.sigma].value)
        prior_mu = prior.parameters[normal.mu].value

        inv_sigma = np.linalg.inv(likelihood.parameters[multivariate_normal.sigma].value)

        posterior_sigma = np.linalg.inv(inv_prior_sigma + n * inv_sigma)
        posterior_mu = posterior_sigma @ (inv_prior_sigma @ prior_mu + n * inv_sigma @ data.mean(axis=0))

        return multivariate_normal.med(mu=posterior_mu, sigma=posterior_sigma)
