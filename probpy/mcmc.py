from typing import Callable as F
from typing import List
import numpy as np
import numba
from .core import RandomVariable


@numba.jit(nopython=False, forceobj=True)
def metropolis_hastings(size: int,
                        pdf: F[[np.ndarray], np.ndarray],
                        proposal: RandomVariable,
                        initial: np.ndarray = None) -> List[np.ndarray]:
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

    return samples


def fast_metropolis_hastings(size: int,
                             pdf: F[[np.ndarray], np.ndarray],
                             initial: np.ndarray,
                             energy: float = 1.0):
    parallel_samples = initial.shape[0]
    dim = initial.ndim
    if dim == 1:
        jump = lambda: np.random.normal(0, energy, size=parallel_samples)
    else:
        jump = lambda: np.random.multivariate_normal(np.zeros(initial.shape[1]),
                                                     np.eye(initial.shape[1]) * energy, size=parallel_samples)

    p = initial
    j = 0
    result = []
    while len(result) < size:
        samples = p + jump()
        accept_rate = np.minimum(pdf(samples) / pdf(p), 1.0)
        accept_rate = accept_rate.flatten()

        accepted = accept_rate >= np.random.rand(parallel_samples)
        rejected = False == accepted

        result.extend(samples[accepted])

        samples[rejected] = p[rejected]
        p = samples
        j += 1

    return np.array(result)


def fast_metropolis_hastings_log_space(size: int,
                                       log_pdf: F[[np.ndarray], np.ndarray],
                                       initial: np.ndarray,
                                       energy: float = 1.0):
    parallel_samples = initial.shape[0]
    dim = initial.ndim
    if dim == 1: jump = lambda: np.random.normal(0, energy, size=parallel_samples)
    else: jump = lambda: np.random.multivariate_normal(np.zeros(initial.shape[1]),
                                                       np.eye(initial.shape[1]) * energy, size=parallel_samples)

    p = initial
    j = 0
    result = []
    while len(result) < size:
        samples = p + jump()
        accept_rate = np.minimum(log_pdf(samples) - log_pdf(p), 0.0)
        accept_rate = accept_rate.flatten()

        accepted = accept_rate >= np.log(np.random.rand(parallel_samples))
        rejected = False == accepted
        result.extend(samples[accepted])

        samples[rejected] = p[rejected]
        p = samples
        j += 1

    return np.array(result)


def metropolis(size: int,
               pdf: F[[np.ndarray], np.ndarray],
               proposal: RandomVariable,
               M: float) -> List[np.ndarray]:
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

    return samples
