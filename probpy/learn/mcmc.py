import numpy as np
from typing import Callable, List, Tuple, Union
from probpy.sampling import (
    fast_metropolis_hastings_log_space_parameter_posterior_estimation)
from . import moment_matching
from probpy.distributions import generic
from probpy.density import UCKD, RCKD
from probpy.core import RandomVariable, Distribution, Density


def _reshape_samples(samples: List[np.ndarray]):
    result = []
    for sample in samples:
        if sample.ndim == 1:
            result.append(sample.reshape(-1, 1))
        else:
            result.append(sample)
    return result


def log_probabilities(data: Union[np.ndarray, Tuple[np.ndarray]],
                      likelihood: Callable[[Tuple[np.ndarray]], np.ndarray],
                      priors: Tuple[RandomVariable]):
    def _log_likelihood(*args):
        ll = np.log(likelihood(*data, *args))

        if ll.ndim == 2:
            ll = ll.sum(axis=1)

        return np.nan_to_num(ll, copy=False, nan=-1e6,
                          neginf=-1e6, posinf=-1e6)

    log_priors = []
    for prior in priors:
        def _log_prior(x, prior=prior):
            return np.nan_to_num(np.log(prior.p(x)), copy=False, nan=-1e6, neginf=-1e6, posinf=-1e6)

        log_priors.append(_log_prior)

    return _log_likelihood, log_priors


def _sample_posterior(data: Tuple[np.ndarray],
                      likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                      priors: Tuple[RandomVariable],
                      size: int,
                      energies: Tuple[float],
                      batch: int):
    likelihood = likelihood if type(likelihood) != RandomVariable else likelihood.p

    log_likelihood, log_priors = log_probabilities(data, likelihood, priors)

    initial = [
        prior.sample(size=batch)
        for prior in priors
    ]

    samples = fast_metropolis_hastings_log_space_parameter_posterior_estimation(
        size=size,
        log_likelihood=log_likelihood,
        log_priors=log_priors,
        initial=initial,
        energies=energies)

    return _reshape_samples(samples)


def _standardize_arguments(_: Union[np.ndarray, Tuple[np.ndarray]],
                           __: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                           priors: Union[RandomVariable, Tuple[RandomVariable]],
                           energies: Tuple[float],
                           match_moments_for: Union[Tuple[Distribution], Distribution]):
    if type(energies) == float: energies = [energies for _ in range(len(priors))]
    if match_moments_for is not None and type(match_moments_for) != tuple: match_moments_for = (match_moments_for,)

    return energies, match_moments_for


def _generic_from_density_samples(density: Density, samples: np.ndarray):
    density.fit(samples)

    def _p(x: np.ndarray):
        return density.p(x)

    def _sample(size: int = 1):
        return samples[np.random.randint(low=0, high=samples.shape[0], size=size)]

    return generic.med(sampling=_sample, probability=_p)


def mcmc(data: Tuple[np.ndarray],
         likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
         priors: Union[RandomVariable, Tuple[RandomVariable]],
         samples: int = 1000,
         mixing: int = 100,
         energies: Tuple[float] = 0.5,
         batch=5,
         match_moments_for: Union[Tuple[Distribution], Distribution] = None,
         normalize: bool = True,
         density: Density = None):
    """
    Don't call this function directly, always use parameter_posterior with mode="mcmc"

    :param data: data passed to likelihood
    :param likelihood: likelihood function / distribution
    :param priors: prior / priors
    :param samples: number of mcmc samples to generate
    :param mixing: number of initial samples to ignore
    :param energies: energies in exploration (variance in normal distribution)
    :param batch: number of particles to run concurrently
    :param match_moments_for: distributions to force posterior into using moment matching
    :param normalize: normalize the resulting density
    :param density: density estimator
    :return: RandomVariable
    """

    energies, match_moments_for = _standardize_arguments(data, likelihood, priors, energies, match_moments_for)

    samples = [sample[mixing:] for sample in _sample_posterior(
        data, likelihood, priors, samples, energies, batch
    )]

    rvs = moment_matching.attempt(
        samples,
        match_moments_for
    )

    if rvs is not None: return rvs

    samples = np.concatenate(samples, axis=1)
    if density is None:
        if normalize:
            density = RCKD()
        else:
            density = UCKD()

    return _generic_from_density_samples(density, samples)
