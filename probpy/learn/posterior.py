from probpy.core import RandomVariable
import numpy as np
from typing import Callable
from .mcmc import mcmc
from .ga import ga
from . import conjugate

from typing import Union, Tuple


def _standardize_arguments(data: Union[np.ndarray, Tuple[np.ndarray]],
                           likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                           priors: Union[RandomVariable, Tuple[RandomVariable]]):
    if type(priors) == RandomVariable: priors = (priors,)
    if type(data) != tuple: data = (data,)

    return data, likelihood, priors


def parameter_posterior(data: Union[np.ndarray, Tuple[np.ndarray]],
                        likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                        priors: Union[RandomVariable, Tuple[RandomVariable]],
                        mode='ga',
                        **kwargs) -> RandomVariable:
    """
    Estimate the posterior distribution of some likelihood and priors. This function uses conjugate priors, mcmc or ga.
    If a likelihood is given conjugate priors then the mode argument will be ignored and a conjugate update will be done, because
    it is much faster.

    :param data: data for likelihood
    :param likelihood: likelihood function / distribution
    :param priors: prior or list of priors
    :param mode: mcmc or ga
    :param kwargs: arguments passed to mcmc / ga
    :return: RandomVariable
    """
    data, likelihood, priors, = _standardize_arguments(
        data, likelihood, priors,
    )

    rv = conjugate.attempt(data, likelihood, priors)
    if rv is not None: return rv

    if mode == "ga":
        return ga(
            data, likelihood, priors, **kwargs
        )
    else:
        return mcmc(
            data, likelihood, priors, **kwargs
        )
