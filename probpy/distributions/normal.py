import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter


class MultiVariateNormal(Distribution):
    mu = "mu"
    sigma = "sigma"

    @classmethod
    def med(cls, mu: np.ndarray = None, sigma: np.ndarray = None, k=None) -> RandomVariable:
        if mu is None and sigma is None:
            _sample = MultiVariateNormal.sample
            _p = MultiVariateNormal.p
            shape = k
        elif mu is None:
            def _sample(mu: np.ndarray, size: np.ndarray = ()): return MultiVariateNormal.sample(mu, sigma, size)
            def _p(x: np.ndarray, mu: np.ndarray): return MultiVariateNormal.p(x, mu, sigma)
            shape = sigma.shape[0]
        elif sigma is None:
            def _sample(sigma: np.ndarray, size: np.ndarray = ()): return MultiVariateNormal.sample(mu, sigma, size)
            def _p(x: np.ndarray, sigma: np.ndarray): return MultiVariateNormal.p(x, mu, sigma)
            shape = mu.size
        else:
            def _sample(size: np.ndarray = ()): return MultiVariateNormal.sample(mu, sigma, size)
            def _p(x: np.ndarray): return MultiVariateNormal.p(x, mu, sigma)
            shape = mu.size

        parameters = {
            MultiVariateNormal.mu: Parameter(shape=shape, value=mu),
            MultiVariateNormal.sigma: Parameter(shape=(shape, shape), value=sigma)
        }

        return RandomVariable(_sample, _p, shape=shape, parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(mu: np.ndarray, sigma: np.ndarray, size: np.ndarray = ()) -> np.ndarray:
        return np.random.multivariate_normal(mu, sigma, size=size)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def p(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        if x.ndim == 1: x = x.reshape(1, -1)
        # Broadcasting over parameters
        if mu.ndim == 2: mu = mu[:, None]

        normalizing_constant = np.float_power(2 * np.pi, -mu.shape[-1] / 2) * np.float_power(np.linalg.det(sigma), -0.5)
        # broadcasting parameters + data
        if mu.ndim == 3:
            quadratic_form = ((x - mu) @ np.linalg.inv(sigma) * (x - mu)).sum(axis=2)
            return np.exp(-1 / 2 * quadratic_form) * normalizing_constant
        else: # broadcasting data
            quadratic_form = (((x - mu) @ np.linalg.inv(sigma)) * (x - mu)).sum(axis=1)
            return np.exp(-1 / 2 * quadratic_form) * normalizing_constant


class Normal(Distribution):
    mu = "mu"
    sigma = "sigma"

    @classmethod
    def med(cls, mu: np.float = None, sigma: np.float = None) -> RandomVariable:
        if mu is not None: mu = np.array(mu)
        if sigma is not None: sigma = np.array(sigma)

        if mu is None and sigma is None:
            _sample = Normal.sample
            _p = Normal.p
        elif mu is None:
            def _sample(mu: np.float, size: np.ndarray = ()): return Normal.sample(mu, sigma, size)
            def _p(x: np.ndarray, mu: np.float): return Normal.p(x, mu, sigma)
        elif sigma is None:
            def _sample(sigma: np.float, size: np.ndarray = ()): return Normal.sample(mu, sigma, size)
            def _p(x: np.ndarray, sigma: np.float): return Normal.p(x, mu, sigma)
        else:
            def _sample(size: np.ndarray = ()): return Normal.sample(mu, sigma, size)
            def _p(x: np.ndarray): return Normal.p(x, mu, sigma)

        parameters = {
            Normal.mu: Parameter((), mu),
            Normal.sigma: Parameter((), sigma)
        }

        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(mu: np.float, sigma: np.float, size: np.ndarray = ()) -> np.ndarray:
        return np.random.normal(mu, np.sqrt(sigma), size=size)

    @staticmethod
    def p(x: np.ndarray, mu: np.float, sigma: np.float) -> np.ndarray:
        if type(mu) == float: mu = np.array(mu)
        if type(sigma) == float: sigma = np.array(sigma)
        if mu.ndim == 1: mu = mu[:, None]
        if sigma.ndim == 1: sigma = sigma[:, None]

        normalizing_constant = np.sqrt(2 * np.pi * sigma)
        return np.exp((-1 / 2) * np.square(x - mu) / sigma) / normalizing_constant


