from probpy.core import RandomVariable, Density
import numpy as np
from typing import Callable, Tuple, Union
from .mcmc import log_probabilities
from probpy.sampling import (ga_posterior_estimation)
from probpy.distributions import generic
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

    samples, densities = ga_posterior_estimation(
        size=samples,
        log_likelihood=log_likelihood,
        log_priors=log_priors,
        initial=initial,
        energies=energies)

    samples = samples[mixing:]
    densities = densities[mixing:]

    return samples, densities


def _standardize_arguments(_: Union[np.ndarray, Tuple[np.ndarray]],
                           __: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                           priors: Union[RandomVariable, Tuple[RandomVariable]],
                           energies: Tuple[float]):
    if type(energies) == float: energies = [energies for _ in range(len(priors))]

    return energies


def _generic_from_density_samples_densities(density: Density, samples: np.ndarray, densities: np.ndarray):
    densities = densities / densities.sum()

    density.fit(samples, densities)

    def _p(x: np.ndarray):
        return density.p(x)

    indexes = np.arange(0, densities.size)

    def _sample(size: int = 1):
        return samples[np.random.choice(indexes, size=size, p=densities)]

    return generic.med(sampling=_sample, probability=_p)


def ga(data: Tuple[np.ndarray],
       likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
       priors: Union[RandomVariable, Tuple[RandomVariable]],
       samples: int = 1000,
       mixing: int = 100,
       energies: Tuple[float] = 0.5,
       batch=5,
       normalize: bool = False,
       density: Density = None,
       **ubrk_args):
    """
    Don't call this function directly, always use parameter_posterior with mode="ga"

    :param data: data passed to likelihood
    :param likelihood: likelihood function
    :param priors: prior / priors
    :param samples: samples in ga estimate
    :param mixing: number of initial samples to ignore
    :param energies: energies in exploration
    :param batch: number of samples run concurrently
    :param normalize: normalize posterior
    :param density: density estimator
    :param ubrk_args: arguments to ubrk (default density estimator)
    :return: RandomVariable
    """
    energies = _standardize_arguments(data, likelihood, priors, energies)
    samples, densities = _sample_posterior(data, likelihood, priors, samples, mixing, energies, batch)

    if density is None:
        if normalize:
            raise NotImplementedError("ga cannot normalize atm")
        else:
            density = URBK(**ubrk_args)

    return _generic_from_density_samples_densities(density, samples, densities)
