import numpy as np
import numba
from typing import Tuple

from probpy.core import Distribution, RandomVariable, Parameter
from probpy.distributions import normal

fast_p_normal = normal.fast_p

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
    @numba.jit(nopython=True, forceobj=False, fastmath=True)
    def fast_p(y: np.ndarray, x: np.ndarray, variables: np.ndarray, sigma: np.float):
        return fast_p_normal(y - (np.sum(x * variables[:-1]) + variables[-1]), mu=0.0, sigma=sigma)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def p(y: np.ndarray, x: np.ndarray, variables: np.ndarray, sigma: np.float) -> np.ndarray:
        if x.ndim == 0: x = x.reshape(1, 1)
        if x.ndim == 1: x = x.reshape(-1, 1)

        return UniLinear.fast_p(y, x, variables, sigma)

    @staticmethod
    def jit_probability(rv: RandomVariable):
        x = rv.parameters[UniLinear.x].value
        variables = rv.parameters[UniLinear.variables].value
        sigma = rv.parameters[UniLinear.sigma].value

        _fast_p = UniLinear.fast_p
        if x is None and variables is None and sigma is None:
            return _fast_p
        elif x is None and variables is None:
            def fast_p(y: np.ndarray, x: np.ndarray, variables: np.ndarray):
                return _fast_p(y, x, variables, sigma)
        elif x is None and sigma is None:
            def fast_p(y: np.ndarray, x: np.ndarray, sigma: np.float):
                return _fast_p(y, x, variables, sigma)
        elif x is None:
            def fast_p(y: np.ndarray, x: np.ndarray):
                return _fast_p(y, x, variables, sigma)
        elif variables is None and sigma is None:
            def fast_p(y: np.ndarray, variables: np.ndarray, sigma: np.float):
                return _fast_p(y, x, variables, sigma)
        elif variables is None:
            def fast_p(y: np.ndarray, variables: np.ndarray):
                return _fast_p(y, x, variables, sigma)
        elif sigma is None:
            def fast_p(y: np.ndarray, sigma: np.float):
                return _fast_p(y, x, variables, sigma)
        else:
            def fast_p(y: np.ndarray):
                return _fast_p(y, x, variables, sigma)

        fast_p = numba.jit(nopython=True, forceobj=False, fastmath=True)(fast_p)

        return fast_p
