import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class MultiVariateNormal(Distribution):
    mu = "mu"
    sigma = "sigma"

    @classmethod
    def med(cls, mu: np.ndarray = None, sigma: np.ndarray = None, k=None) -> RandomVariable:
        if mu is None and sigma is None:
            _sample = Normal.sample
            _p = Normal.p
            shape = k
        elif mu is None:
            def _sample(mu: np.ndarray, shape: np.ndarray = ()): return MultiVariateNormal.sample(mu, sigma, shape)
            def _p(x: np.ndarray, mu: np.ndarray): return MultiVariateNormal.p(x, mu, sigma)
            shape = sigma.shape[0]
        elif sigma is None:
            def _sample(sigma: np.ndarray, shape: np.ndarray = ()): return MultiVariateNormal.sample(mu, sigma, shape)
            def _p(x: np.ndarray, sigma: np.ndarray): return MultiVariateNormal.p(x, mu, sigma)
            shape = mu.size
        else:
            def _sample(shape: np.ndarray = ()): return MultiVariateNormal.sample(mu, sigma, shape)
            def _p(x: np.ndarray): return MultiVariateNormal.p(x, mu, sigma)
            shape = mu.size

        parameters = {
            MultiVariateNormal.mu: Parameter(shape=shape, value=mu),
            MultiVariateNormal.sigma: Parameter(shape=(shape, shape), value=sigma)
        }

        return RandomVariable(_sample, _p, shape=shape, parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(mu: np.ndarray, sigma: np.ndarray, shape: np.ndarray = ()) -> np.ndarray:
        return np.random.multivariate_normal(mu, sigma, size=shape)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def p(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        if X.ndim == 1: X = X.reshape(1, -1)
        normalizing_constant = np.float_power(2 * np.pi, -mu.size / 2) * np.float_power(np.linalg.det(sigma), -0.5)
        return np.array(
            [np.exp((-1 / 2) * (x - mu).T @ np.linalg.inv(sigma) @ (x - mu)) * normalizing_constant for x in X])


class Normal(Distribution):
    mu = "mu"
    sigma = "sigma"

    @classmethod
    def med(cls, mu: np.float = None, sigma: np.float = None) -> RandomVariable:
        if mu is None and sigma is None:
            _sample = Normal.sample
            _p = Normal.p
        elif mu is None:
            def _sample(mu: np.float, shape: np.ndarray = ()): return Normal.sample(mu, sigma, shape)
            def _p(x: np.ndarray, mu: np.float): return Normal.p(x, mu, sigma)
        elif sigma is None:
            def _sample(sigma: np.float, shape: np.ndarray = ()): return Normal.sample(mu, sigma, shape)
            def _p(x: np.ndarray, sigma: np.float): return Normal.p(x, mu, sigma)
        else:
            def _sample(shape: np.ndarray = ()): return Normal.sample(mu, sigma, shape)
            def _p(x: np.ndarray): return Normal.p(x, mu, sigma)

        parameters = {
            Normal.mu: Parameter((), mu),
            Normal.sigma: Parameter((), sigma)
        }

        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(mu: np.float, sigma: np.float, shape: np.ndarray = ()) -> np.ndarray:
        return np.random.normal(mu, np.sqrt(sigma), size=shape)

    @staticmethod
    @numba.jit(nopython=True, forceobj=False)
    def p(x: np.ndarray, mu: np.float, sigma: np.float) -> np.ndarray:
        normalizing_constant = np.sqrt(2 * np.pi * sigma)
        return np.exp((-1 / 2) * np.square(x - mu) / sigma) / normalizing_constant
