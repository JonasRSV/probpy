from probpy.core import RandomVariable
from typing import Union, Tuple, Callable
from probpy.integration import posterior_predictive_integration
import numpy as np
from . import conjugate


def _probabilities(data: Tuple[np.ndarray],
                   likelihood: Callable[[Tuple[np.ndarray]], np.ndarray],
                   priors: Tuple[RandomVariable]):
    def _likelihood(*theta):
        return likelihood(*data, *theta)

    return _likelihood, priors


def _integrate_probability(data: Tuple[np.ndarray],
                           likelihood: Union[RandomVariable, Callable],
                           priors: Tuple[RandomVariable],
                           size: int) -> float:
    likelihood = likelihood if type(likelihood) != RandomVariable else likelihood.p
    _likelihood, _priors = _probabilities(data, likelihood, priors)

    return posterior_predictive_integration(size=size, likelihood=_likelihood, priors=_priors)


def _standardize_arguments(likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                           priors: Union[RandomVariable, Tuple[RandomVariable]],
                           data: Tuple[np.ndarray],
                           size: int):
    if type(priors) == RandomVariable: priors = (priors,)
    if type(data) == np.ndarray: data = (data,)

    return likelihood, priors, data, size,


def predictive_posterior(likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                         priors: Union[RandomVariable, Tuple[RandomVariable]],
                         data: Tuple[np.ndarray] = None,
                         size: int = 1000) -> Union[RandomVariable, float]:
    likelihood, priors, data, size = _standardize_arguments(likelihood, priors, data, size)

    rv = conjugate.attempt(likelihood, priors)
    if rv is not None: return rv

    if data is not None: return _integrate_probability(data, likelihood, priors, size)

    raise NotImplementedError("For non-conjugate non-data is not implemented yet")
