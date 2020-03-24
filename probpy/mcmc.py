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


@numba.jit(nopython=False, forceobj=True)
def fast_metropolis_hastings(size: int,
                             pdf: F[[np.ndarray], np.ndarray],
                             initial: np.ndarray,
                             energy: float = 1.0):
    initial = np.array(initial)
    dim = initial.size
    if dim == 1: jumps = np.random.normal(0, energy, size=size)
    else: jumps = np.random.multivariate_normal(np.zeros(dim), np.eye(dim) * energy, size=size)

    barriers = np.random.rand(size)
    p = initial
    j = 0
    result = np.zeros(size, dtype=np.float32)
    for i in range(size):
        while True:
            sample = p + jumps[j % size]
            accept_rate = np.minimum(pdf(sample) / pdf(p), 1.0)
            if accept_rate >= barriers[j % size]:
                break
            j += 1
        j += 1
        result[i], p = sample, sample
    return result


def metropolis(size: int,
               pdf: F[[np.ndarray], np.ndarray],
               proposal: RandomVariable,
               M: float) -> List[np.ndarray]:
    samples = []
    while len(samples) < size:
        remainder = size - len(samples)
        sample = proposal.sample(shape=remainder)

        accept_rate = pdf(sample) / (M * proposal.p(sample))

        max_rate = accept_rate.max()
        if max_rate > 1.0: raise Exception("M to small, accept rate %s > 1.0. m: " % (accept_rate, max_rate))

        rejection_probability = np.random.rand(remainder)
        accept_mask = accept_rate > rejection_probability


        samples.extend(sample[accept_mask])

    return samples
