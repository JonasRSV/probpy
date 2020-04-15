import numpy as np
from typing import Callable, Tuple, Union
from probpy.sampling import (
    fast_metropolis_hastings_log_space_parameter_posterior_estimation)
from probpy.learn import moment_matching
from probpy.distributions import generic
from probpy.density import UCKD, RCKD
from probpy.core import RandomVariable, Distribution, Density
from .common import jit_log_probabilities, jitted_prior, jitted_likelihood


def _sample_posterior(data: Tuple[np.ndarray],
                      likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                      prior: RandomVariable,
                      size: int,
                      energy: float,
                      batch: int):
    fast_ll = jitted_likelihood(likelihood)
    fast_p = jitted_prior(prior)
    log_likelihood, log_prior = jit_log_probabilities(data, fast_ll, fast_p)

    initial = prior.sample(size=batch)

    samples = fast_metropolis_hastings_log_space_parameter_posterior_estimation(
        size=size,
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        initial=initial,
        energy=energy)

    return samples


def _generic_from_density_samples(density: Density, samples: np.ndarray):
    density.fit(samples)

    def _p(x: np.ndarray):
        return density.p(x)

    def _sample(size: int = 1):
        return samples[np.random.randint(low=0, high=samples.shape[0], size=size)]

    return generic.med(sampling=_sample, probability=_p, fast_p=density.get_fast_p())


def mcmc(data: Tuple[np.ndarray],
         likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
         prior: RandomVariable,
         samples: int = 1000,
         mixing: int = 0,
         energy: float = 0.5,
         batch: int = 5,
         match_moments_for: Distribution = None,
         normalize: bool = True,
         density: Density = None):
    """
    Don't call this function directly, always use parameter_posterior with mode="mcmc"

    :param data: data passed to likelihood
    :param likelihood: likelihood function / distribution
    :param prior: prior distribution
    :param samples: number of mcmc samples to generate
    :param mixing: number of initial samples to ignore
    :param energy: variance in exploration
    :param batch: number of particles to run concurrently
    :param match_moments_for: distributions to force posterior into using moment matching
    :param normalize: normalize the resulting density
    :param density: density estimator
    :return: RandomVariable
    """

    samples = _sample_posterior(
        data, likelihood, prior, samples, energy, batch
    )[mixing:]

    rvs = moment_matching.attempt(
        samples,
        match_moments_for
    )

    if rvs is not None: return rvs

    if density is None:
        if normalize:
            density = RCKD()
        else:
            density = UCKD()

    return _generic_from_density_samples(density, samples)
