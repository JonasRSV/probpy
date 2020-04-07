import numpy as np
import numba
from typing import Tuple

from probpy.core import Distribution, RandomVariable, Parameter
from probpy.distributions import normal


class UniLinear(Distribution):
    """Linear with one target distribution"""
    x = "x"
    variables = "variables"
    sigma = "sigma"

    @classmethod
    def med(cls, x: np.ndarray = None, variables: np.ndarray = None, sigma: np.float = None) -> RandomVariable:
        """

        :param x: input
        :param variables: weights
        :param sigma: variance of estimates
        :return: RandomVariable
        """
        params = [x, variables, sigma]
        none = [i for i, param in enumerate(params) if param is None]
        not_none = [i for i, param in enumerate(params) if param is not None]

        def _p(x, *args):
            call_args = [None] * 3
            for i, arg in enumerate(args): call_args[none[i]] = arg
            for i in not_none: call_args[i] = params[i]

            return UniLinear.p(x, *call_args)

        def _sample(*args, size: int = 1):
            call_args = [None] * 3
            for i, arg in enumerate(args): call_args[none[i]] = arg
            for i in not_none: call_args[i] = params[i]

            return UniLinear.sample(*call_args, size=size)

        parameters = {
            UniLinear.x: Parameter((), x),
            UniLinear.variables: Parameter((), variables),
            UniLinear.sigma: Parameter((), sigma)
        }

        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(x: np.ndarray, variables: np.ndarray, sigma: np.float, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if x.ndim == 0: x = x.reshape(1, 1)
        if x.ndim == 1: x = x.reshape(-1, 1)
        return x @ variables[:-1] + variables[-1] + normal.sample(mu=0, sigma=sigma, size=x.shape[0])

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def p(y: Tuple[np.ndarray, np.ndarray], x: np.ndarray, variables: np.ndarray, sigma: np.float) -> np.ndarray:
        if x.ndim == 0: x = x.reshape(1, 1)
        if x.ndim == 1: x = x.reshape(-1, 1)

        # broadcasting over mu + data
        if variables.ndim == 2:
            return normal.p(y - (x @ variables[:, None, :-1] + variables[:, None, None, -1]).squeeze(axis=2),
                            mu=0.0, sigma=sigma)

        return normal.p(y - (x @ variables[:-1] + variables[-1]), mu=0.0, sigma=sigma)
