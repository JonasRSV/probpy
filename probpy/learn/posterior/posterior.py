from probpy.core import RandomVariable
import numpy as np
from typing import Callable
from probpy.learn.posterior.mcmc import mcmc
from probpy.learn.posterior.search import search
from probpy.learn import conjugate

from typing import Union, Tuple


def _standardize_arguments(data: Union[np.ndarray, Tuple[np.ndarray]],
                           likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]]):
    if type(data) != tuple: data = (data,)

    return data, likelihood


def parameter_posterior(data: Union[np.ndarray, Tuple[np.ndarray]],
                        likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                        prior: RandomVariable,
                        mode='mcmc',
                        **kwargs) -> RandomVariable:
    """
    Estimate the posterior distribution of some likelihood and priors. This function uses conjugate priors, mcmc or ga.
    If a likelihood is given conjugate priors then the mode argument will be ignored and a conjugate update will be done, because
    it is much faster.

    :param data: data for likelihood
    :param likelihood: likelihood function / distribution
    :param priors: prior or list of priors
    :param mode: mcmc or search
    :param kwargs: arguments passed to mcmc / ga
    :return: RandomVariable
    """
    data, likelihood = _standardize_arguments(
        data, likelihood
    )

    rv = conjugate.attempt(data, likelihood, prior)
    if rv is not None: return rv

    if mode == "search":
        return search(
            data, likelihood, prior, **kwargs
        )
    else:
        return mcmc(
            data, likelihood, prior, **kwargs
        )
