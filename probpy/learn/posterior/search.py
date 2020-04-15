from probpy.core import RandomVariable, Density
import numpy as np
from typing import Callable, Tuple, Union
from probpy.learn.posterior.common import jit_log_probabilities
from probpy.search import (search_posterior_estimation)
from probpy.distributions import generic
from probpy.density import URBK


def _search_posterior(data: Tuple[np.ndarray],
                      likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                      priors: Tuple[RandomVariable],
                      samples: int,
                      energies: Tuple[float],
                      batch: int,
                      volume: float):
    likelihood = likelihood if type(likelihood) != RandomVariable else likelihood.p

    log_likelihood, log_priors = jit_log_probabilities(data, likelihood, priors)

    initial = [
        prior.sample(size=batch)
        for prior in priors
    ]

    return search_posterior_estimation(
        size=samples,
        log_likelihood=log_likelihood,
        log_priors=log_priors,
        initial=initial,
        energies=energies,
        volume=volume)


def _standardize_arguments(_: Union[np.ndarray, Tuple[np.ndarray]],
                           __: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                           priors: Union[RandomVariable, Tuple[RandomVariable]],
                           energies: Tuple[float]):
    if type(energies) == float: energies = [energies for _ in range(len(priors))]

    return energies


def _generic_from_density_samples_densities(density: Density, samples: np.ndarray, densities: np.ndarray, volume: float):
    densities = densities / densities.sum()

    density.fit(samples, densities)

    def _p(x: np.ndarray):
        return density.p(x)

    indexes = np.arange(0, densities.size)

    def _sample(size: int = 1):
        s = samples[np.random.choice(indexes, size=size, p=densities)]
        return s + np.random.normal(0, 1 / volume, size=s.shape)

    return generic.med(sampling=_sample, probability=_p)


def search(data: Tuple[np.ndarray],
           likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
           priors: Union[RandomVariable, Tuple[RandomVariable]],
           samples: int = 1000,
           energies: Tuple[float] = 0.5,
           batch=5,
           volume=10.0,
           normalize: bool = False,
           density: Density = None,
           **ubrk_args):
    """
    Don't call this function directly, always use parameter_posterior with mode="search"

    :param volume: volume of elements
    :param data: data passed to likelihood
    :param likelihood: likelihood function
    :param priors: prior / priors
    :param samples: samples in search estimate
    :param energies: energies in exploration
    :param batch: number of samples run concurrently
    :param normalize: normalize posterior
    :param density: density estimator
    :param ubrk_args: arguments to ubrk (default density estimator)
    :return: RandomVariable
    """
    energies = _standardize_arguments(data, likelihood, priors, energies)
    samples, densities = _search_posterior(data, likelihood, priors, samples, energies, batch, volume)

    if density is None:
        if normalize:
            raise NotImplementedError("search cannot normalize atm")
        else:
            density = URBK(**ubrk_args)

    return _generic_from_density_samples_densities(density, samples, densities, volume)
