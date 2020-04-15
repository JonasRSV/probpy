from typing import Callable as F
from typing import List, Tuple
import numpy as np
from probpy.core import RandomVariable


def metropolis_hastings(size: int,
                        pdf: F[[np.ndarray], np.ndarray],
                        proposal: RandomVariable,
                        initial: np.ndarray = None) -> List[np.ndarray]:
    """

    :param size: number of samples
    :param pdf: pdf to sample from
    :param proposal: proposal distribution
    :param initial: starting point
    :return: array of samples
    """
    if initial is None:
        p = np.random.rand(*proposal.shape)
    else:
        p = initial

    samples = []
    while len(samples) < size:
        sample = proposal.sample(p)

        accept_rate = min([(pdf(sample) * proposal.p(p, sample)) / (pdf(p) * proposal.p(sample, p)), 1])
        if np.random.rand() < accept_rate:
            samples.append(sample)
            p = sample

    return np.array(samples)


def fast_metropolis_hastings(size: int,
                             pdf: F[[np.ndarray], np.ndarray],
                             initial: np.ndarray,
                             energy: float = 1.0):
    """

    :param size: number of samples
    :param pdf: pdf to sample from
    :param initial: initial points
    :param energy: variance of jumps
    :return: array of samples
    """
    batch = initial.shape[0]
    jump = lambda: np.random.normal(0, energy, size=initial.shape)

    p, result = initial, []
    while len(result) < size:
        samples = p + jump()
        accept_rate = np.minimum(pdf(samples) / pdf(p), 1.0)
        accept_rate = accept_rate.flatten()

        accepted = accept_rate >= np.random.rand(batch)
        rejected = False == accepted

        result.extend(samples[accepted])

        samples[rejected] = p[rejected]
        p = samples

    return np.array(result)


def fast_metropolis_hastings_log_space(size: int,
                                       log_pdf: F[[np.ndarray], np.ndarray],
                                       initial: np.ndarray,
                                       energy: float = 1.0):
    """

    :param size: number of samples
    :param log_pdf: log pdf
    :param initial: initial points
    :param energy: energy of estimate
    :return: samples
    """
    batch = initial.shape[0]
    jump = lambda: np.random.normal(0, energy, size=initial.shape)

    p, result = initial, []
    while len(result) < size:
        samples = p + jump()
        accept_rate = np.minimum(log_pdf(samples) - log_pdf(p), 0.0)
        accept_rate = accept_rate.flatten()

        accepted = accept_rate >= np.log(np.random.rand(batch))
        rejected = False == accepted
        result.extend(samples[accepted])

        samples[rejected] = p[rejected]
        p = samples

    return np.array(result)


def fast_metropolis_hastings_log_space_parameter_posterior_estimation(
        size: int,
        log_likelihood: F[[np.ndarray], np.ndarray],
        log_prior: F[[np.ndarray], np.ndarray],
        initial: np.ndarray,
        energy: float):
    batch = initial.shape[0]
    jump = lambda: np.random.normal(0, energy, size=initial.shape)

    def _probability(x: np.ndarray):
        log_prior_probability = np.nan_to_num(log_prior(x).flatten(), nan=-15000, posinf=-15000, neginf=-15000)
        log_likelihood_probability = np.nan_to_num(log_likelihood(x).flatten(), nan=-15000, posinf=-15000, neginf=-15000)

        return log_prior_probability + log_likelihood_probability

    p = initial
    results = []
    while len(results) < size:
        samples = p + jump()

        accept_rate = np.minimum(_probability(samples) - _probability(p), 0.0)

        accepted = accept_rate >= np.log(np.random.rand(batch))
        rejected = False == accepted

        results.extend(samples[accepted])
        samples[rejected] = p[rejected]

        p = samples

    return np.array(results)


def metropolis(size: int,
               pdf: F[[np.ndarray], np.ndarray],
               proposal: RandomVariable,
               M: float) -> np.ndarray:
    """

    :param size: number of samples
    :param pdf: pdf to sample from
    :param proposal: proposal distribution
    :param M: normalization constant
    :return: array of samples
    """
    samples = []
    while len(samples) < size:
        remainder = size - len(samples)
        sample = proposal.sample(size=remainder)

        accept_rate = pdf(sample) / (M * proposal.p(sample))

        max_rate = accept_rate.max()
        if max_rate > 1.0: raise Exception("M to small, accept rate %s > 1.0. m: " % (max_rate))

        rejection_probability = np.random.rand(remainder)
        accept_mask = accept_rate > rejection_probability

        samples.extend(sample[accept_mask])

    return np.array(samples)
