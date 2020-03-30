import numpy as np
import numba
from typing import Tuple

from probpy.core import Distribution, RandomVariable, Parameter
from probpy.distributions import normal


class UniLinear(Distribution):
    x = "x"
    variables = "variables"
    sigma = "sigma"

    @classmethod
    def med(cls, x: np.ndarray = None, variables: np.ndarray = None, sigma: np.float = None, shape: int = None) -> RandomVariable:
        params = [x, variables, sigma]
        none = [i for i, param in enumerate(params) if param is None]
        not_none = [i for i, param in enumerate(params) if param is not None]

        def _p(x, *args):
            call_args = [None] * 3
            for i, arg in enumerate(args): call_args[none[i]] = arg
            for i in not_none: call_args[i] = params[i]

            return UniLinear.p(x, *call_args)

        def _sample(*args, shape=()):
            call_args = [None] * 3
            for i, arg in enumerate(args): call_args[none[i]] = arg
            for i in not_none: call_args[i] = params[i]

            return UniLinear.sample(*call_args, shape=shape)

        parameters = {
            UniLinear.x: Parameter((), x),
            UniLinear.variables: Parameter((), variables),
            UniLinear.sigma: Parameter((), sigma)
        }

        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(x: np.ndarray, variables: np.ndarray, sigma: np.float, shape: np.ndarray = ()) -> Tuple[np.ndarray, np.ndarray]:
        if x.ndim == 1:
            x = x[:, None]
        return x @ variables[:-1] + variables[-1] + normal.sample(mu=0, sigma=sigma, shape=x.shape[0])

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def p(y: Tuple[np.ndarray, np.ndarray], x: np.ndarray, variables: np.ndarray, sigma: np.float) -> np.ndarray:
        if x.ndim == 1: x = x.reshape(-1, 1)
        return normal.p(y - (x @ variables[:-1] + variables[-1]), mu=0.0, sigma=sigma)
