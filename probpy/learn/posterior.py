from probpy.core import RandomVariable, Distribution
import numpy as np
from typing import Callable, List
from probpy.sampling import fast_metropolis_hastings_log_space_parameter_posterior_estimation
from . import conjugate, moment_matching
from probpy.distributions import points

from typing import Union, Tuple


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


def _reshape_samples(samples: List[np.ndarray]):
    result = []
    for sample in samples:
        if sample.ndim == 1:
            result.append(sample.reshape(-1, 1))
        else:
            result.append(sample)
    return result


def _standardize_arguments(data: Union[np.ndarray, Tuple[np.ndarray]],
                           likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                           priors: Union[RandomVariable, Tuple[RandomVariable]],
                           size: int,
                           energies: Tuple[float],
                           batch: int,
                           match_moments_for: Union[Tuple[Distribution], Distribution]):
    if type(priors) == RandomVariable: priors = (priors,)
    if type(data) != tuple: data = (data,)
    if type(energies) == float: energies = [energies for _ in range(len(priors))]
    if match_moments_for is not None and type(match_moments_for) != tuple: match_moments_for = (match_moments_for,)

    return data, likelihood, priors, size, energies, batch, match_moments_for


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


def parameter_posterior(data: Union[np.ndarray, Tuple[np.ndarray]],
                        likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                        priors: Union[RandomVariable, Tuple[RandomVariable]],
                        size: int = 1000,
                        energies: Tuple[float] = 0.05,
                        batch=10,
                        match_moments_for: Union[Tuple[Distribution], Distribution] = None) -> RandomVariable:
    data, likelihood, priors, size, energies, batch, match_moments_for = _standardize_arguments(
        data, likelihood, priors, size, energies, batch, match_moments_for
    )

    rv = conjugate.attempt(data, likelihood, priors)
    if rv is not None: return rv

    samples = _sample_posterior(
        data, likelihood, priors, size, energies, batch
    )

    rvs = moment_matching.attempt(
        samples,
        match_moments_for
    )

    if rvs is not None: return rvs

    return points.med(points=np.concatenate(samples, axis=1))

