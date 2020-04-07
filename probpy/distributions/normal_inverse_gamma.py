import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter
from probpy.special import gamma as _gamma
from probpy.distributions import normal, gamma


class NormalInverseGamma(Distribution):
    """Normal Inverse Gamma distribution"""
    mu = "mu"
    lam = "lam"
    a = "a"
    b = "b"

    @classmethod
    def med(cls,
            mu: np.float = None,
            lam: np.float = None,
            a: np.float = None,
            b: np.float = None) -> RandomVariable:
        """

        :param mu: mean
        :param lam: precision
        :param a: shape
        :param b: rate
        :return: RandomVariable
        """
        params = [mu, lam, a, b]
        none = [i for i, param in enumerate(params) if param is None]
        not_none = [i for i, param in enumerate(params) if param is not None]

        def _p(x, *args):
            call_args = [None] * 4
            for i, arg in enumerate(args): call_args[none[i]] = arg
            for i in not_none: call_args[i] = params[i]

            return NormalInverseGamma.p(x, *call_args)

        def _sample(*args, size: int = 1):
            call_args = [None] * 4
            for i, arg in enumerate(args[:len(none)]): call_args[none[i]] = arg
            for i in not_none: call_args[i] = params[i]

            if len(args) > len(none): size = args[-1]

            return NormalInverseGamma.sample(*call_args, size=size)

        parameters = {
            NormalInverseGamma.mu: Parameter((), mu),
            NormalInverseGamma.lam: Parameter((), lam),
            NormalInverseGamma.a: Parameter((), a),
            NormalInverseGamma.b: Parameter((), b)
        }

        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(mu: np.float, lam: np.float, a: np.float, b: np.float, size: int = 1) -> np.ndarray:
        sigma = 1 / gamma.sample(a, b, size=size)
        x = normal.sample(mu=mu, sigma=sigma / lam, size=size)
        return np.concatenate([x[:, None], sigma[:, None]], axis=1)

    @staticmethod
    def p(x: np.ndarray, mu: np.float, lam: np.float, a: np.float, b: np.float) -> np.ndarray:
        if type(x) != np.ndarray: x = np.array(x)
        if x.ndim == 1: x = x.reshape(1, 2)
        lam_norm = np.sqrt(lam) / np.sqrt(2 * np.pi * x[:, 1])
        beta_norm = np.float_power(b, a) / _gamma(a)
        sigma_norm = np.float_power(x[:, 1], - (a + 1))
        exp_term = np.exp(-(2 * b + lam * np.square(x[:, 0] - mu)) / (2 * x[:, 1]))
        return lam_norm * beta_norm * sigma_norm * exp_term
