from probpy.core import RandomVariable, Distribution
import numpy as np
from typing import Callable
from .classic_mcmc import classic_mcmc
from .almost_mcmc import almost_mcmc
from . import conjugate

from typing import Union, Tuple


def _standardize_arguments(data: Union[np.ndarray, Tuple[np.ndarray]],
                           likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                           priors: Union[RandomVariable, Tuple[RandomVariable]],
                           samples: int,
                           burn_in: int,
                           energies: Tuple[float],
                           batch: int,
                           match_moments_for: Union[Tuple[Distribution], Distribution],
                           normalize: bool):
    if type(priors) == RandomVariable: priors = (priors,)
    if type(data) != tuple: data = (data,)
    if type(energies) == float: energies = [energies for _ in range(len(priors))]
    if match_moments_for is not None and type(match_moments_for) != tuple: match_moments_for = (match_moments_for,)

    return data, likelihood, priors, samples, burn_in, energies, batch, match_moments_for, normalize


def parameter_posterior(data: Union[np.ndarray, Tuple[np.ndarray]],
                        likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                        priors: Union[RandomVariable, Tuple[RandomVariable]],
                        samples: int = 1000,
                        burn_in: int = 100,
                        energies: Tuple[float] = 0.5,
                        batch=5,
                        match_moments_for: Union[Tuple[Distribution], Distribution] = None,
                        normalize: bool = True,
                        mcmc: bool = True) -> RandomVariable:
    data, likelihood, priors, samples, burn_in, energies, batch, match_moments_for, normalize = _standardize_arguments(
        data, likelihood, priors, samples, burn_in, energies, batch, match_moments_for, normalize
    )

    rv = conjugate.attempt(data, likelihood, priors)
    if rv is not None: return rv

    if mcmc:
        return classic_mcmc(
            data, likelihood, priors, samples, burn_in, energies, batch, match_moments_for, normalize
        )
    else:
        return almost_mcmc(
            data, likelihood, priors, samples, burn_in, energies, batch, normalize
        )
