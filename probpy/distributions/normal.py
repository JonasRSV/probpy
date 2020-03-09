import numpy as np

from probpy.core import Distribution


class MultiVariateNormal(Distribution):

    @staticmethod
    def sample(mu: np.ndarray, sigma: np.ndarray, shape: np.ndarray = ()) -> np.ndarray:
        return np.random.multivariate_normal(mu, sigma, size=shape)

    @staticmethod
    def p(mu: np.ndarray, sigma: np.ndarray, X: np.ndarray) -> np.ndarray:
        normalizing_constant = np.float_power(np.sqrt(2 * np.pi * np.linalg.det(sigma)), -mu.ndim)
        return np.array([np.exp(-1/2 * (x - mu).T @ np.linalg.inv(sigma) @ (x - mu)) / normalizing_constant for x in X])


class Normal(Distribution):

    @staticmethod
    def sample(mu: np.ndarray, sigma: np.ndarray, shape: np.ndarray = ()) -> np.ndarray:
        return np.random.normal(mu, np.sqrt(sigma), size=shape)

    @staticmethod
    def p(mu: np.ndarray, sigma: np.ndarray, x: np.ndarray) -> np.ndarray:
        normalizing_constant = np.sqrt(2 * np.pi * sigma)
        return np.exp((-1/2) * np.square(x - mu) / sigma) / normalizing_constant

