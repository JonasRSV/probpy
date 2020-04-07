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
        log_likelihood: F[[Tuple[np.ndarray]], np.ndarray],
        log_priors: Tuple[F[[np.ndarray], np.ndarray]],
        initial: Tuple[np.ndarray],
        energies: Tuple[float]):
    if len(log_priors) != len(initial): raise Exception(f"log_priors: {len(log_priors)} != initial: {len(initial)}")
    batch = initial[0].shape[0]

    for i in initial:
        if i.shape[0] != batch: raise Exception(f"batch mismatch {batch} != {i.shape[0]}")

    jumpers, n = [], len(initial)
    for i, e in zip(initial, energies): jumpers.append(lambda i=i, e=e: np.random.normal(0, e, size=i.shape))

    def _probability(x: Tuple[np.ndarray]):
        prior_log_probability = np.sum([log_priors[i](x[i]) for i in range(n)], axis=0).flatten()
        data_log_probability = log_likelihood(*x).flatten()
        return prior_log_probability + data_log_probability

    p = initial
    results = [[] for _ in range(n)]
    while len(results[0]) < size:
        samples = [
            p[i] + jumpers[i]()
            for i in range(n)
        ]

        accept_rate = np.minimum(_probability(samples) - _probability(p), 0.0)
        accept_rate = accept_rate

        accepted = accept_rate >= np.log(np.random.rand(batch))
        rejected = False == accepted

        for i in range(n):
            results[i].extend(samples[i][accepted])
            samples[i][rejected] = p[i][rejected]

        p = samples

    return [np.array(result) for result in results]


def metropolis(size: int,
               pdf: F[[np.ndarray], np.ndarray],
               proposal: RandomVariable,
               M: float) -> List[np.ndarray]:
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
        if max_rate > 1.0: raise Exception("M to small, accept rate %s > 1.0. m: " % (accept_rate, max_rate))

        rejection_probability = np.random.rand(remainder)
        accept_mask = accept_rate > rejection_probability

        samples.extend(sample[accept_mask])

    return np.array(samples)
