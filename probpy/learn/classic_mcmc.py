import numpy as np
from typing import Callable, List, Tuple, Union
from probpy.sampling import (
    fast_metropolis_hastings_log_space_parameter_posterior_estimation)
from . import moment_matching
from probpy.distributions import points
from probpy.density import UCKD
from probpy.core import RandomVariable, Distribution


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
        return np.nan_to_num(np.log(likelihood(*data, *args)).sum(axis=1), copy=False, nan=-100000,
                             neginf=-1000000, posinf=-1000000)

    log_priors = []
    for prior in priors:
        def _log_prior(x, prior=prior):
            return np.log(prior.p(x))

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


def classic_mcmc(data: Tuple[np.ndarray],
                   likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                   priors: Union[RandomVariable, Tuple[RandomVariable]],
                   samples: int = 1000,
                   mixing: int = 100,
                   energies: Tuple[float] = 0.5,
                   batch=5,
                   match_moments_for: Union[Tuple[Distribution], Distribution] = None,
                   normalize: bool = True):
    samples = [sample[mixing:] for sample in _sample_posterior(
        data, likelihood, priors, samples, energies, batch
    )]

    rvs = moment_matching.attempt(
        samples,
        match_moments_for
    )

    if rvs is not None: return rvs

    if normalize:
        rv = points.med(points=np.concatenate(samples, axis=1))
    else:
        rv = points.med(points=np.concatenate(samples, axis=1), density_estimator=UCKD)

    return rv
