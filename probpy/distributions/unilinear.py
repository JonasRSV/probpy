import numpy as np
import numba
from typing import Tuple

from probpy.core import Distribution, RandomVariable, Parameter
from probpy.distributions import multivariate_uniform, normal


class UniLinear(Distribution):
    variables = "variables"
    sigma = "sigma"

    @classmethod
    def med(cls, variables: np.ndarray = None, sigma: np.float = None, shape: int = None) -> RandomVariable:
        if variables is None and sigma is None:
            _sample = UniLinear.sample
            _p = UniLinear.p
            shape = shape
        elif variables is None:
            def _sample(variables: np.ndarray, shape: np.ndarray = ()): return UniLinear.sample(variables, sigma, shape)
            def _p(x: np.ndarray, variables: np.ndarray): return UniLinear.p(x, variables, sigma)
            shape = shape
        elif sigma is None:
            def _sample(sigma: np.float, shape: np.ndarray = ()): return UniLinear.sample(variables, sigma, shape)
            def _p(x: np.ndarray, sigma: np.float): return UniLinear.p(x, variables, sigma)
            shape = variables.size
        else:
            def _sample(shape: np.ndarray = ()): return UniLinear.sample(variables, sigma, shape)
            def _p(x: np.ndarray): return UniLinear.p(x, variables, sigma)
            shape = variables.size

        parameters = {
            UniLinear.variables: Parameter(shape=shape, value=variables),
            UniLinear.sigma: Parameter(shape=(), value=sigma)
        }

        return RandomVariable(_sample, _p, shape=shape, parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(variables: np.ndarray, sigma: np.float, shape: np.ndarray = (),
               bounds: Tuple = None) -> Tuple[np.ndarray, np.ndarray]:
        # We will sample a line instead of just the difference between Y and X since this seems more useful
        if bounds is None:
            lower_bound = np.zeros(variables.size - 1)
            upper_bound = np.ones(variables.size - 1)
        else:
            lower_bound, upper_bound = bounds

        lower_bound, upper_bound = np.array(lower_bound), np.array(upper_bound)

        if lower_bound.shape == ():
            lower_bound = lower_bound.reshape(-1, 1)
            upper_bound = upper_bound.reshape(-1, 1)

        x = multivariate_uniform.sample(a=lower_bound, b=upper_bound, shape=shape)
        y = x @ variables[:-1] + variables[-1] + normal.sample(mu=0, sigma=sigma, shape=shape)
        return x, y

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def p(x: Tuple[np.ndarray, np.ndarray], variables: np.ndarray, sigma: np.float) -> np.ndarray:
        x, y = x

        if x.ndim == 1:
            x = x[:, None]
        return normal.p(y - (x @ variables[:-1] + variables[-1]), mu=0.0, sigma=sigma)
