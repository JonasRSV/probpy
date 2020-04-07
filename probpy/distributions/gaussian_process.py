import numpy as np
import numba
from typing import Callable

from probpy.core import Distribution, RandomVariable, Parameter
from probpy.distributions import multivariate_normal


class GaussianProcess(Distribution):
    """Gaussian Process"""
    x = "x"
    mu = "mu"
    sigma = "sigma"
    X = "X"
    Y = "Y"

    probability_epsilon = 1e-1

    @classmethod
    def med(cls,
            x: np.ndarray = None,
            mu: Callable[[np.ndarray], np.float] = None,
            sigma: Callable[[np.ndarray, np.ndarray], np.float] = None,
            X: np.ndarray = None,
            Y: np.ndarray = None) -> RandomVariable:
        """

        :param x: non-observed samples
        :param mu: mean function
        :param sigma: variance function
        :param X: observed samples
        :param Y: observed values
        :return: RandomVariable
        """

        params = [x, mu, sigma, X, Y]
        none = [i for i, param in enumerate(params) if param is None]
        not_none = [i for i, param in enumerate(params) if param is not None]

        def _p(x, *args):
            call_args = [None] * 5
            for i, arg in enumerate(args): call_args[none[i]] = arg
            for i in not_none: call_args[i] = params[i]

            return GaussianProcess.p(x, *call_args)

        def _sample(*args, size: int = 1):
            call_args = [None] * 5
            for i, arg in enumerate(args): call_args[none[i]] = arg
            for i in not_none: call_args[i] = params[i]

            return GaussianProcess.sample(*call_args, size=size)

        parameters = {
            GaussianProcess.x: Parameter(None, x),
            GaussianProcess.mu: Parameter(None, mu),
            GaussianProcess.sigma: Parameter(None, sigma),
            GaussianProcess.X: Parameter(None, X),
            GaussianProcess.Y: Parameter(None, Y)
        }

        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def _build_parameters(
            mu: Callable[[np.ndarray], np.float],
            sigma: Callable[[np.ndarray, np.ndarray], np.float],
            X: np.ndarray):

        covariance_mat = np.zeros((X.shape[0], X.shape[0]))
        mu_vec = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                covariance_mat[i, j] = sigma(X[i], X[j])
            mu_vec[i] = mu(X[i])

        return mu_vec, covariance_mat

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def _build_off_diagnoal(
            mu: Callable[[np.ndarray], np.float],
            sigma: Callable[[np.ndarray, np.ndarray], np.float],
            x: np.ndarray, X: np.ndarray):

        x_dim, X_dim = x.shape[0], X.shape[0]
        off_diagonal = np.zeros((x_dim, X_dim))
        for i in range(x_dim):
            for j in range(X_dim):
                off_diagonal[i, j] = sigma(x[i], X[j])

        return off_diagonal

    @staticmethod
    def sample(x: np.ndarray,
               mu: Callable[[np.ndarray], np.float],
               sigma: Callable[[np.ndarray, np.ndarray], np.float],
               X: np.ndarray,
               Y: np.ndarray,
               size: int = 1) -> np.ndarray:
        """

        :param x: non-observed samples
        :param mu: mean function
        :param sigma: variance function
        :param X: observed samples
        :param Y: observed values
        :param size: number of samples
        :return: array of samples
        """

        x_mu_vec, x_covariance_mat = GaussianProcess._build_parameters(mu, sigma, x)
        X_mu_vec, X_covariance_mat = GaussianProcess._build_parameters(mu, sigma, X)

        off_diagonal = GaussianProcess._build_off_diagnoal(mu, sigma, x, X)

        inv_X_covariance_mat = np.linalg.inv(X_covariance_mat)

        posterior_mu = x_mu_vec + off_diagonal @ inv_X_covariance_mat @ (Y - X_mu_vec)
        posterior_sigma = x_covariance_mat - off_diagonal @ inv_X_covariance_mat @ off_diagonal.T

        return multivariate_normal.sample(posterior_mu, posterior_sigma, size=size)

    @staticmethod
    def p(y: np.ndarray,
          x: np.ndarray,
          mu: Callable[[np.ndarray], np.float],
          sigma: Callable[[np.ndarray, np.ndarray], np.float],
          X: np.ndarray,
          Y: np.ndarray) -> np.ndarray:
        """

        :param y: non-observed values
        :param x: non-observed samples
        :param mu: mean function
        :param sigma: variance function
        :param X: observed samples
        :param Y: observed values
        :return: densities
        """
        if y.ndim == 1: y = y.reshape(-1, 1)

        x_mu_vec, x_covariance_mat = GaussianProcess._build_parameters(mu, sigma, x)
        X_mu_vec, X_covariance_mat = GaussianProcess._build_parameters(mu, sigma, X)

        off_diagonal = GaussianProcess._build_off_diagnoal(mu, sigma, x, X)

        inv_X_covariance_mat = np.linalg.inv(X_covariance_mat)

        posterior_mu = x_mu_vec + off_diagonal @ inv_X_covariance_mat @ (Y - X_mu_vec)
        posterior_sigma = x_covariance_mat - off_diagonal @ inv_X_covariance_mat @ off_diagonal.T
        posterior_sigma = posterior_sigma + np.eye(posterior_sigma.shape[0]) * GaussianProcess.probability_epsilon

        return multivariate_normal.p(y, posterior_mu, posterior_sigma)
