import numpy as np
import numba
from typing import Callable

from probpy.core import Distribution, RandomVariable, Parameter
from probpy.distributions import multivariate_normal


class GaussianProcess(Distribution):
    mu = "mu"
    sigma = "sigma"
    domain = "domain"
    codomain = "codomain"

    probability_epsilon = 1e-1

    @classmethod
    def med(cls,
            mu: Callable[[np.ndarray], np.float] = None,
            sigma: Callable[[np.ndarray, np.ndarray], np.float] = None,
            domain: np.ndarray = None,
            codomain: np.ndarray = None) -> RandomVariable:

        params = [mu, sigma, domain, codomain]
        none = [i for i, param in enumerate(params) if param is None]
        not_none = [i for i, param in enumerate(params) if param is not None]

        def _p(x, *args):
            call_args = [None] * 4
            for i, arg in enumerate(args): call_args[none[i]] = arg
            for i in not_none: call_args[i] = params[i]

            return GaussianProcess.p(x, *call_args)

        def _sample(*args, shape=()):
            call_args = [None] * 4
            for i, arg in enumerate(args): call_args[none[i]] = arg
            for i in not_none: call_args[i] = params[i]

            return GaussianProcess.sample(*call_args, shape=shape)

        parameters = {
            GaussianProcess.mu: Parameter(None, mu),
            GaussianProcess.sigma: Parameter(None, sigma),
            GaussianProcess.domain: Parameter(None, domain),
            GaussianProcess.codomain: Parameter(None, codomain)
        }

        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def _build_parameters(
            mu: Callable[[np.ndarray], np.float],
            sigma: Callable[[np.ndarray, np.ndarray], np.float],
            domain: np.ndarray,
            codomain: np.ndarray):

        covariance_mat = np.zeros((codomain.size, codomain.size))
        mu_vec = np.zeros(codomain.size)

        for i in range(codomain.size):
            for j in range(codomain.size):
                covariance_mat[i, j] = sigma(domain[i], domain[j])
            mu_vec[i] = mu(domain[i])

        return mu_vec, covariance_mat

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(mu: Callable[[np.ndarray], np.float],
               sigma: Callable[[np.ndarray, np.ndarray], np.float],
               domain: np.ndarray,
               codomain: np.ndarray,
               shape: np.ndarray = ()) -> np.ndarray:

        mu_vec, covariance_mat = GaussianProcess._build_parameters(mu, sigma, domain, codomain)
        return multivariate_normal.sample(mu_vec, covariance_mat, shape=shape)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def p(X: np.ndarray,
          mu: Callable[[np.ndarray], np.float],
          sigma: Callable[[np.ndarray, np.ndarray], np.float],
          domain: np.ndarray,
          codomain: np.ndarray) -> np.ndarray:

        mu_vec, covariance_mat = GaussianProcess._build_parameters(mu, sigma, domain, codomain)
        covariance_mat = covariance_mat + np.eye(codomain.size) * GaussianProcess.probability_epsilon # for stability

        return multivariate_normal.p(X, mu_vec, covariance_mat)
