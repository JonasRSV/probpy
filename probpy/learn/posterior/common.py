from typing import Tuple, Callable, Union
import numpy as np
import numba
from probpy.core import RandomVariable
from probpy.distributions.jit import jit_probability


def jitted_likelihood(likelihood: Union[RandomVariable, Callable[[np.ndarray], np.ndarray]]):
    if type(likelihood) == RandomVariable:
        return jit_probability(likelihood)

    return numba.jit(nopython=True, fastmath=True, forceobj=False)(likelihood)


def jitted_prior(rv: RandomVariable): return jit_probability(rv)


def jit_log_probabilities(data: Tuple[np.ndarray],
                          likelihood: Callable[[Tuple[np.ndarray]], np.ndarray],
                          prior: Callable[[np.ndarray], np.ndarray]):
    if len(data) > 2:
        raise Exception("Only supporting up to 2 data arguments at the moment")

    if len(data) == 2:
        y, x = data

        y, x = np.array(y), np.array(x)

        if y.ndim == 0: y = y.reshape(-1)
        if x.ndim == 0: x = x.reshape(-1)

        def _ll(prior_samples):
            n_prior_samples = len(prior_samples)
            n_data = len(x)
            result = np.zeros(n_prior_samples)

            for i in range(result.size):
                for j in range(n_data):
                    result[i] += np.log(likelihood(y[j], x[j], prior_samples[i]))

            return result

    else:
        y = data[0]

        y = np.array(y)
        if y.ndim == 0: y = y.reshape(-1)

        def _ll(prior_samples):
            n_prior_samples = len(prior_samples)
            n_data = len(y)
            result = np.zeros(n_prior_samples)

            for i in range(result.size):
                for j in range(n_data):
                    result[i] += np.log(likelihood(y[j], prior_samples[i]))

            return result

    def _lp(prior_samples):
        return np.log(prior(prior_samples))

    lp = numba.jit(nopython=True, fastmath=True, forceobj=False)(_lp)
    ll = numba.jit(nopython=True, fastmath=True, forceobj=False)(_ll)

    return ll, lp
