from probpy.core import RandomVariable
import numpy as np
from typing import Callable, Tuple, Union
from .classic_mcmc import log_probabilities
from probpy.sampling import (fast_almost_mcmc_parameter_posterior_estimation)
from probpy.distributions import points
from probpy.density import URBK


def _sample_posterior(data: Tuple[np.ndarray],
                      likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                      priors: Tuple[RandomVariable],
                      samples: int,
                      mixing: int,
                      energies: Tuple[float],
                      batch: int):
    likelihood = likelihood if type(likelihood) != RandomVariable else likelihood.p

    log_likelihood, log_priors = log_probabilities(data, likelihood, priors)

    initial = [
        prior.sample(size=batch)
        for prior in priors
    ]

    samples, densities = fast_almost_mcmc_parameter_posterior_estimation(
        size=samples,
        log_likelihood=log_likelihood,
        log_priors=log_priors,
        initial=initial,
        energies=energies)

    samples = samples[mixing:]
    densities = densities[mixing:]

    return samples, densities


def almost_mcmc(data: Tuple[np.ndarray],
                likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                priors: Union[RandomVariable, Tuple[RandomVariable]],
                samples: int = 1000,
                mixing: int = 100,
                energies: Tuple[float] = 0.5,
                batch=5,
                normalize: bool = False):
    samples, densities = _sample_posterior(data, likelihood, priors, samples, mixing, energies, batch)

    if normalize:
        raise NotImplementedError("almost mcmc cannot normalize atm")
    else:
        density = URBK(variance=2)
        density.fit(samples, densities)

    return points.med(points=samples, density=density)
